#include <tskit.h>

#include <vector>
#include <utility>
#include <iostream>
#include <sstream>
#include <memory>

void
handle_tsk_error(int ret, std::string message)
{
    if (ret != 0) {
        std::ostringstream o;
        o << message << "::" << tsk_strerror(ret);
        throw std::runtime_error(o.str());
    }
}

struct TskTreeSeq {
    tsk_treeseq_t ts;

    TskTreeSeq(const char *treefile)
    {
        auto ret = tsk_treeseq_load(&ts, treefile, 0);
        if (ret != 0) {
            handle_tsk_error(ret, "error reading treefile");
        }
    }

    ~TskTreeSeq() { tsk_treeseq_free(&ts); }
};

struct TskTree {
    tsk_tree_t tree;

    TskTree(const TskTreeSeq &ts) : tree{}
    {
        auto ret = tsk_tree_init(&tree, &ts.ts, 0);
        if (ret < 0) {
            handle_tsk_error(ret, "error initializing tree");
        }
        ret = tsk_tree_first(&tree);
        if (ret < 0) {
            handle_tsk_error(ret, "error getting first tree");
        }
    }

    ~TskTree() { tsk_tree_free(&tree); }
};

struct Node {
    int id;
    std::vector<Node> children;

    Node() : id{ -1 }, children{} {}
};

static tsk_size_t
run_parsimony_library(tsk_tree_t *tree, tsk_variant_t *var)
{
    int8_t ancestral_state;
    tsk_size_t num_transitions;
    tsk_state_transition_t *transitions;

    int ret = tsk_tree_map_mutations(tree, var->genotypes.i8, NULL, 0, &ancestral_state,
        &num_transitions, &transitions);
    if (ret < 0) {
        handle_tsk_error(ret, "error from tsk_tree_map_mutations");
    }
    if (num_transitions > var->site->mutations_length) {
        throw std::runtime_error("This shouldn't happen");
    }
    free(transitions);
    return num_transitions;
}

int
count_roots(const tsk_tree_t *tree)
{
    int nroots = 0;
    for (auto root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        nroots += 1;
    }
    return nroots;
}

void
build_node_stack_recursive(const tsk_tree_t *tree, tsk_id_t u, tsk_id_t parent,
    std::vector<std::pair<int, Node> > &node_stack)
{
    node_stack[u].first = parent;
    node_stack[u].second.id = u;

    for (auto c = tree->left_child[u]; c != TSK_NULL; c = tree->right_sib[c]) {
        build_node_stack_recursive(tree, c, u, node_stack);
    }
}

void
build_recursive(const tsk_tree_t *tree, std::vector<std::pair<int, Node> > &node_stack)
{
    node_stack.resize(tree->left_root + 1);
    build_node_stack_recursive(tree, tree->left_root, -1, node_stack);
}

static Node
build_tree(tsk_tree_t *tree)
{
    std::vector<std::pair<int, Node> > node_stack;

    build_recursive(tree, node_stack);

    if (node_stack.empty()) {
        throw std::runtime_error("node stack is empty");
    }

    for (std::size_t i = 0; i < node_stack.size(); ++i) {
        if (node_stack[i].first != TSK_NULL) {
            // node is not a root node
            node_stack[node_stack[i].first].second.children.push_back(
                node_stack[i].second);
        }
    }

    Node root{ node_stack.back().second };
    return root;
}

void
_traverse_confirm(const Node &node, int parent, int depth)
{
    for (int d = 0; d < depth; ++d) {
        std::cout << "    ";
    }
    std::cout << "Visit confirm " << node.id << ' ' << parent << ' ' << depth << '\n';
    for (auto &c : node.children) {
        _traverse_confirm(c, node.id, depth + 1);
    }
}

void
traverse_recursive_confirm(const Node &node)
{
    _traverse_confirm(node, -1, 0);
}

static int
argmax(tsk_size_t n, const int8_t *values)
{
    tsk_size_t k;
    int8_t max = INT8_MIN;
    tsk_size_t k_max = 0;

    for (k = 0; k < n; k++) {
        if (values[k] > max) {
            max = values[k];
            k_max = k;
        }
    }
    return k_max;
}

void
_hartigan_postorder(const Node &parent, tsk_size_t num_nodes, tsk_size_t num_alleles,
    int8_t *optimal_set)
{
    int allele_count[num_alleles]; /* This isn't portable, and it's a bad idea */
    int max_allele_count;
    tsk_size_t k;

    for (k = 0; k < num_alleles; k++) {
        allele_count[k] = 0;
    }

    for (auto &child : parent.children) {
        _hartigan_postorder(child, num_nodes, num_alleles, optimal_set);
        for (k = 0; k < num_alleles; k++) {
            allele_count[k] += optimal_set[child.id * num_alleles + k];
        }
    }

    /* Assuming all leaf nodes are samples */
    if (parent.children.size() != 0) {
        max_allele_count = 0;
        for (k = 0; k < num_alleles; k++) {
            max_allele_count = TSK_MAX(max_allele_count, allele_count[k]);
        }
        for (k = 0; k < num_alleles; k++) {
            if (allele_count[k] == max_allele_count) {
                optimal_set[parent.id * num_alleles + k] = 1;
            }
        }
    }
}

static tsk_size_t
_hartigan_preorder(const Node &parent, int8_t state, tsk_size_t num_nodes,
    tsk_size_t num_alleles, const int8_t *optimal_set)
{
    tsk_size_t num_mutations = 0;
    const int8_t *parent_optimal_set = &optimal_set[parent.id * num_alleles];

    if (parent_optimal_set[state] == 0) {
        /* Choose a new state */
        state = argmax(num_alleles, parent_optimal_set);
        num_mutations++;
    }
    for (auto &child : parent.children) {
        num_mutations
            += _hartigan_preorder(child, state, num_nodes, num_alleles, optimal_set);
    }
    return num_mutations;
}

static tsk_size_t
run_parsimony_recursive(const Node &root, const TskTreeSeq &ts, const tsk_variant_t *var)
{
    const auto *genotypes = var->genotypes.i8;
    const auto num_alleles = var->num_alleles;
    const auto num_nodes = tsk_treeseq_get_num_nodes(&ts.ts);
    const auto num_samples = tsk_treeseq_get_num_samples(&ts.ts);
    const auto *samples = tsk_treeseq_get_samples(&ts.ts);
    tsk_size_t j, num_mutations;
    tsk_id_t u;
    int8_t ancestral_state;
    /* TODO Do this in a C++ way */
    int8_t *optimal_set
        = (int8_t *) calloc((num_nodes + 1) * num_alleles, sizeof(*optimal_set));

    if (optimal_set == NULL) {
        throw std::runtime_error("Memory allocation failure");
    }

    for (j = 0; j < num_samples; j++) {
        u = samples[j];
        optimal_set[u * num_alleles + genotypes[j]] = 1;
    }
    _hartigan_postorder(root, num_nodes, num_alleles, optimal_set);
    ancestral_state = argmax(num_alleles, &optimal_set[root.id * num_alleles]);

    num_mutations
        = _hartigan_preorder(root, ancestral_state, num_nodes, num_alleles, optimal_set);

    /* num_mutations = _hartigan_preorder(tree->left_root, ancestral_state, num_nodes, */
    /* num_alleles, optimal_set, tree->right_child, tree->left_sib); */

    free(optimal_set);
    return num_mutations;
}

static TskTreeSeq
load_treeseq(const char *treefile)
{
    return TskTreeSeq(treefile);
}

int
main(int argc, char **argv)
{
    if (argc != 3) {
        throw std::invalid_argument("usage: <tree sequence file> <max_sites>");
    }
    auto ts = load_treeseq(argv[1]);
    tsk_id_t max_sites = std::atoi(argv[2]);
    TskTree tree(ts);
    if (count_roots(&tree.tree) != 1) {
        throw std::invalid_argument("tree must have a single root");
    }
    auto cpp_tree = build_tree(&tree.tree);
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    auto ret = tsk_vargen_init(&vargen, &ts.ts, NULL, 0, NULL, 0);
    if (ret < 0) {
        handle_tsk_error(ret, "error initializing tsk_variant_t");
    }

    double lib_total_time = 0;
    double recursive_total_time = 0;
    while ((ret = tsk_vargen_next(&vargen, &var)) == 1) {
        if (var->site->id >= max_sites) {
            break;
        }
        auto before = clock();
        auto score_lib = run_parsimony_library(&tree.tree, var);
        auto duration = clock() - before;
        lib_total_time += ((double) duration) / CLOCKS_PER_SEC;

        before = clock();
        auto score_cpp = run_parsimony_recursive(cpp_tree, ts, var);
        duration = clock() - before;
        recursive_total_time += ((double) duration) / CLOCKS_PER_SEC;
        if (score_cpp != score_lib) {
            throw std::runtime_error("Error in parsimony implementation");
        }
    }
    if (ret < 0) {
        handle_tsk_error(ret, "vargen");
    }
    std::cout << "lib\t" << std::scientific << lib_total_time / max_sites << "\n";
    std::cout << "recursive_pre_alloc\t" << std::scientific
              << recursive_total_time / max_sites << "\n";
    return 0;
}
