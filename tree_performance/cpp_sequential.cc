#include <tskit.h>

#include <vector>
#include <algorithm>
#include <stack>
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

    explicit Node(tsk_id_t id) : id{ id }, children{} {}
};

struct HeapNode {
    int id;
    std::vector<std::unique_ptr<HeapNode> > children;

    HeapNode() : id{ -1 }, children{} {}

    explicit HeapNode(tsk_id_t id) : id{ id }, children{} {}
};

inline const auto &
dispatch_node(const Node &n)
{
    return n;
}

inline const auto &
dispatch_node(const std::unique_ptr<HeapNode> &n)
{
    return *n;
}

inline const auto
get_node_id(const Node &n)
{
    return n.id;
}

inline const auto
get_node_id(const HeapNode &n)
{
    return n.id;
}

inline const auto
get_node_id(const std::unique_ptr<HeapNode> &n)
{
    return n->id;
}

template <typename NodeType>
static void
print_tree(const NodeType &parent, int depth)
{
    for (int d = 0; d < depth; d++) {
        std::cout << "  ";
    }
    std::cout << parent.id << "\n";
    for (auto &child : parent.children) {
        print_tree(dispatch_node(child), depth + 1);
    }
}

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

// Reason 43,124,151 why we can hate C++:
// returning Node & here prevented copy-elision,
// resulting in n^2 memory needs.
inline Node 
build_tree_pre_allocated_dispatch(Node &&n)
{
    return Node(std::move(n));
}

inline std::unique_ptr<HeapNode>
build_tree_pre_allocated_dispatch(HeapNode &&n)
{
    return std::make_unique<HeapNode>(std::move(n));
}

template <typename NodeType>
static NodeType
build_tree_contiguous(const tsk_tree_t *tree)
{
    std::stack<tsk_id_t, std::vector<tsk_id_t> > node_stack;

    std::vector<NodeType> nodes, nodes_input_order;
    std::size_t num_nodes = tsk_treeseq_get_num_nodes(tree->tree_sequence);
    nodes.reserve(num_nodes);

    std::vector<std::vector<tsk_id_t> > child_map(num_nodes);
    std::vector<int> order_seen(num_nodes, -1);

    node_stack.push(tree->left_root);

    int order = 0;

    while (!node_stack.empty()) {
        auto parent = node_stack.top();
        node_stack.pop();

        nodes.emplace_back(parent);
        nodes_input_order.emplace_back(parent);
        order_seen[parent] = order++;

        auto child = tree->left_child[parent];
        for (; child != TSK_NULL; child = tree->right_sib[child]) {
            child_map[parent].push_back(child);
            node_stack.push(child);
        }
    }

    std::sort(
        begin(nodes), end(nodes), [&order_seen](const auto &left, const auto &right) {
            return order_seen[left.id] < order_seen[right.id];
        });
    if (nodes[0].id != tree->left_root) {
        throw std::runtime_error("something bad is going on...");
    }

    for (auto r = std::rbegin(nodes); r != std::rend(nodes); ++r) {
        for (auto c : child_map[r->id]) {
            nodes_input_order[order_seen[r->id]].children.emplace_back(
                build_tree_pre_allocated_dispatch(
                    std::move(nodes_input_order[order_seen[c]])));
        }
    }

    auto x{ std::move(nodes_input_order[0]) }; // subtle...
    return x;
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

template <typename NodeType>
static void
_hartigan_postorder(const NodeType &parent, tsk_size_t num_nodes, tsk_size_t num_alleles,
    int8_t *optimal_set)
{
    int allele_count[num_alleles]; /* This isn't portable, and it's a bad idea */
    int max_allele_count;
    tsk_size_t k;

    for (k = 0; k < num_alleles; k++) {
        allele_count[k] = 0;
    }

    for (auto &child : parent.children) {
        _hartigan_postorder(dispatch_node(child), num_nodes, num_alleles, optimal_set);
        for (k = 0; k < num_alleles; k++) {
            allele_count[k] += optimal_set[get_node_id(child) * num_alleles + k];
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
                optimal_set[get_node_id(parent) * num_alleles + k] = 1;
            }
        }
    }
}

template <typename NodeType>
static tsk_size_t
_hartigan_preorder(const NodeType &parent, int8_t state, tsk_size_t num_nodes,
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
        num_mutations += _hartigan_preorder(
            dispatch_node(child), state, num_nodes, num_alleles, optimal_set);
    }
    return num_mutations;
}

template <typename NodeType>
static tsk_size_t
run_parsimony_recursive(
    const NodeType &root, const TskTreeSeq &ts, const tsk_variant_t *var)
{
    const auto *genotypes = var->genotypes.i8;
    const auto num_alleles = var->num_alleles;
    const auto num_nodes = tsk_treeseq_get_num_nodes(&ts.ts);
    const auto num_samples = tsk_treeseq_get_num_samples(&ts.ts);
    const auto *samples = tsk_treeseq_get_samples(&ts.ts);
    tsk_size_t j, num_mutations;
    tsk_id_t u;
    int8_t ancestral_state;
    std::vector<std::int8_t> optimal_set((num_nodes + 1) * num_alleles, 0);

    for (j = 0; j < num_samples; j++) {
        u = samples[j];
        optimal_set[u * num_alleles + genotypes[j]] = 1;
    }
    _hartigan_postorder(root, num_nodes, num_alleles, optimal_set.data());
    ancestral_state = argmax(num_alleles, &optimal_set[root.id * num_alleles]);
    num_mutations = _hartigan_preorder(
        root, ancestral_state, num_nodes, num_alleles, optimal_set.data());
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
    auto cpp_tree_prealloc = build_tree_contiguous<Node>(&tree.tree);
    auto cpp_tree_heapalloc = build_tree_contiguous<HeapNode>(&tree.tree);
    tsk_vargen_t vargen;
    tsk_variant_t *var;

    /* print_tree(cpp_tree_prealloc, 0); */
    /* print_tree(dispatch_node(&cpp_tree_heapalloc), 0); */

    auto ret = tsk_vargen_init(&vargen, &ts.ts, NULL, 0, NULL, 0);
    if (ret < 0) {
        handle_tsk_error(ret, "error initializing tsk_variant_t");
    }

    double lib_total_time = 0;
    double prealloc_total_time = 0;
    double heapalloc_total_time = 0;
    while ((ret = tsk_vargen_next(&vargen, &var)) == 1) {
        if (var->site->id >= max_sites) {
            break;
        }
        auto before = clock();
        auto score_lib = run_parsimony_library(&tree.tree, var);
        auto duration = clock() - before;
        lib_total_time += ((double) duration) / CLOCKS_PER_SEC;

        before = clock();
        auto score_cpp = run_parsimony_recursive<Node>(cpp_tree_prealloc, ts, var);
        duration = clock() - before;
        prealloc_total_time += ((double) duration) / CLOCKS_PER_SEC;

        /* std::cout << "score = " << score_lib << ": " << score_cpp << "\n"; */
        if (score_cpp != score_lib) {
            throw std::runtime_error("Error in parsimony implementation (Node)");
        }

        before = clock();
        score_cpp = run_parsimony_recursive<HeapNode>(cpp_tree_heapalloc, ts, var);
        duration = clock() - before;
        heapalloc_total_time += ((double) duration) / CLOCKS_PER_SEC;

        if (score_cpp != score_lib) {
            throw std::runtime_error("Error in parsimony implementation (HeapNode)");
        }
    }
    if (ret < 0) {
        handle_tsk_error(ret, "vargen");
    }
    std::cout << "cpp_lib         " << lib_total_time / max_sites << "\n";
    std::cout << "cpp_pre_alloc   " << prealloc_total_time / max_sites << "\n";
    std::cout << "cpp_heap_alloc  " << heapalloc_total_time / max_sites << "\n";
    tsk_vargen_free(&vargen);
    return 0;
}
