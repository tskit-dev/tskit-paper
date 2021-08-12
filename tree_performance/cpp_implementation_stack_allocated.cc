#include <tskit.h>

#include <vector>
#include <memory>

struct TskTreeSeq
{
    tsk_treeseq_t ts;

    TskTreeSeq(const char *treefile)
    {
        auto ret = tsk_treeseq_load(&ts, treefile, 0);
        if (ret != 0)
            {
                throw std::runtime_error("error reading treefile");
            }
    }

    ~TskTreeSeq()
    {
        tsk_treeseq_free(&ts);
    }
};

struct TskTree
{
    tsk_tree_t tree;

    TskTree(const TskTreeSeq &ts) : tree{}
    {
        auto ret = tsk_tree_init(&tree, &ts.ts, 0);
        if (ret < 0)
            {
                throw std::runtime_error("error initializing tree");
            }
        ret = tsk_tree_first(&tree);
        if (ret < 0)
            {
                throw std::runtime_error("error building first tree");
            }
    }

    ~TskTree()
    {
        tsk_tree_free(&tree);
    }
};

struct Node
{
    std::vector<Node> children;
};

static void
_traverse(tsk_tree_t *tree, tsk_id_t u, int depth)
{
    tsk_id_t v;
    int j;

    for (j = 0; j < depth; j++)
        {
            printf("    ");
        }
    printf("Visit recursive %lld %d\n", (long long)u, depth);
    for (v = tree->left_child[u]; v != TSK_NULL; v = tree->right_sib[v])
        {
            _traverse(tree, v, depth + 1);
        }
}

static void
traverse_recursive(tsk_tree_t *tree)
{
    tsk_id_t root;

    for (root = tree->left_root; root != TSK_NULL; root = tree->right_sib[root])
        {
            _traverse(tree, root, 0);
        }
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
    TskTree tree(ts);
    traverse_recursive(&tree.tree);
}
