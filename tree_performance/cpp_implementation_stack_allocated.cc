#include <tskit.h>

#include <vector>
#include <utility>
#include <iostream>
#include <memory>

struct TskTreeSeq {
    tsk_treeseq_t ts;

    TskTreeSeq(const char *treefile)
    {
        auto ret = tsk_treeseq_load(&ts, treefile, 0);
        if (ret != 0) {
            throw std::runtime_error("error reading treefile");
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
            throw std::runtime_error("error initializing tree");
        }
        ret = tsk_tree_first(&tree);
        if (ret < 0) {
            throw std::runtime_error("error building first tree");
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
        throw std::runtime_error("error from tsk_tree_map_mutations");
    }
    if (num_transitions > var->site->mutations_length) {
        throw std::runtime_error("This shouldn't happen");
    }
    free(transitions);
    return num_transitions;
}

static void
_traverse(const tsk_tree_t *tree, tsk_id_t u, tsk_id_t parent, int depth)
{
    tsk_id_t v;
    int j;

    for (j = 0; j < depth; j++) {
        printf("    ");
    }
    printf("Visit recursive %lld %d %d\n", (long long) u, parent, depth);
    for (v = tree->left_child[u]; v != TSK_NULL; v = tree->right_sib[v]) {
        _traverse(tree, v, u, depth + 1);
    }
}

static void
traverse_recursive(const tsk_tree_t *tree)
{
    tsk_id_t root;

    for (root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root]) {
        _traverse(tree, root, -1, 0);
    }
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

static TskTreeSeq
load_treeseq(const char *treefile)
{
    return TskTreeSeq(treefile);
}

int
main(int argc, char **argv)
{
    auto ts = load_treeseq(argv[1]);
    tsk_id_t max_sites = std::atoi(argv[2]);
    TskTree tree(ts);
    if (count_roots(&tree.tree) != 1) {
        throw std::invalid_argument("tree must have a single root");
    }
    traverse_recursive(&tree.tree);

    auto cpp_tree = build_tree(&tree.tree);
    traverse_recursive_confirm(cpp_tree);

    tsk_vargen_t vargen;
    tsk_variant_t *var;
    auto ret = tsk_vargen_init(&vargen, &ts.ts, NULL, 0, NULL, 0);
    if (ret < 0) {
        throw std::runtime_error("error initializing tsk_variant_t");
    }

    while ((ret = tsk_vargen_next(&vargen, &var)) == 1) {
        if (var->site->id >= max_sites) {
            break;
        }
        auto before = clock();
        auto score_lib = run_parsimony_library(&tree.tree, var);
        auto duration = clock() - before;
        auto lib_total_time = ((double) duration) / CLOCKS_PER_SEC;

        before = clock();
        duration = clock() - before;
        auto recursive_total_time = ((double) duration) / CLOCKS_PER_SEC;
    }
}
