#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <time.h>

#include <tskit.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));                 \
    }


static tsk_size_t
run_parsimony_library(tsk_tree_t *tree, tsk_variant_t *var)
{
    int8_t ancestral_state;
    tsk_size_t num_transitions;
    tsk_state_transition_t *transitions;

    int ret = tsk_tree_map_mutations(tree, var->genotypes.i8, NULL, 0, &ancestral_state,
            &num_transitions, &transitions);
    check_tsk_error(ret);
    if (num_transitions > var->site->mutations_length) {
        errx(EXIT_FAILURE, "This shouldn't happen");
    }
    free(transitions);
    return num_transitions;
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

static tsk_size_t
_hartigan_preorder(tsk_id_t parent, int8_t state,
        tsk_size_t num_nodes, tsk_size_t num_alleles,
        const int8_t *optimal_set, const tsk_id_t *right_child, const tsk_id_t *left_sib)
{
    tsk_id_t child;
    tsk_size_t num_mutations = 0;
    const int8_t *parent_optimal_set = &optimal_set[parent * num_alleles];

    if (parent_optimal_set[state] == 0) {
        /* Choose a new state */
        state = argmax(num_alleles, parent_optimal_set);
        num_mutations++;
    }
    for (child = right_child[parent]; child != TSK_NULL; child = left_sib[child]) {
        num_mutations += _hartigan_preorder(child, state, num_nodes, num_alleles,
                optimal_set, right_child, left_sib);
    }
    return num_mutations;
}

static void
_hartigan_postorder(tsk_id_t parent, tsk_size_t num_nodes, tsk_size_t num_alleles,
        int8_t *optimal_set, const tsk_id_t *right_child, const tsk_id_t *left_sib)
{
    int allele_count[num_alleles]; /* This isn't portable, and it's a bad idea */
    int max_allele_count;
    tsk_size_t k;
    tsk_id_t child;

    for (k = 0; k < num_alleles; k++) {
        allele_count[k] = 0;
    }
    for (child = right_child[parent]; child != TSK_NULL; child = left_sib[child]) {
        _hartigan_postorder(child, num_nodes, num_alleles, optimal_set,
                right_child, left_sib);
        for (k = 0; k < num_alleles; k++) {
            allele_count[k] += optimal_set[child * num_alleles + k];
        }
    }
    /* Assuming all leaf nodes are samples */
    if (right_child[parent] != TSK_NULL) {
        max_allele_count = 0;
        for (k = 0; k < num_alleles; k++) {
            max_allele_count = TSK_MAX(max_allele_count, allele_count[k]);
        }
        for (k = 0; k < num_alleles; k++) {
            if (allele_count[k] == max_allele_count) {
                optimal_set[parent * num_alleles + k] = 1;
            }
        }
    }
}

static tsk_size_t
run_parsimony_recursive(tsk_tree_t *tree, tsk_variant_t *var)
{
    const int8_t *genotypes = var->genotypes.i8;
    const tsk_size_t num_alleles = var->num_alleles;
    const tsk_size_t num_nodes = tsk_treeseq_get_num_nodes(tree->tree_sequence);
    const tsk_size_t num_samples = tsk_treeseq_get_num_samples(tree->tree_sequence);
    const tsk_id_t *samples = tsk_treeseq_get_samples(tree->tree_sequence);
    tsk_size_t j, num_mutations;
    tsk_id_t u;
    int8_t ancestral_state;
    int8_t *optimal_set = calloc((num_nodes + 1) * num_alleles, sizeof(*optimal_set));

    if (optimal_set == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }

    for (j = 0; j < num_samples; j++) {
        u = samples[j];
        optimal_set[u * num_alleles + genotypes[j]] = 1;
    }
    /* Assuming a single root */
    _hartigan_postorder(tree->left_root, num_nodes, num_alleles, optimal_set,
            tree->right_child, tree->left_sib);
    ancestral_state = argmax(num_alleles, &optimal_set[tree->left_root * num_alleles]);
    num_mutations = _hartigan_preorder(tree->left_root, ancestral_state,
            num_nodes, num_alleles, optimal_set, tree->right_child, tree->left_sib);

    free(optimal_set);

    return num_mutations;

}

int
main(int argc, char **argv)
{
    int ret;
    tsk_treeseq_t ts;
    tsk_tree_t tree;
    tsk_vargen_t vargen;
    tsk_variant_t *var;
    tsk_id_t max_sites;
    tsk_size_t score_lib, score_recursive;
    clock_t before, duration;
    double lib_total_time = 0.0;
    double recursive_total_time = 0.0;

    /* When we add the recursive implementation we can add another argument here
     * to decide which version to run. */
    if (argc != 3) {
        errx(EXIT_FAILURE, "usage: <tree sequence file> <max_sites>");
    }
    max_sites = atoi(argv[2]);
    if (max_sites <= 0) {
        errx(EXIT_FAILURE, "bad max_sites value");
    }
    ret = tsk_treeseq_load(&ts, argv[1], 0);
    check_tsk_error(ret);
    ret = tsk_tree_init(&tree, &ts, 0);
    check_tsk_error(ret);
    ret = tsk_tree_first(&tree);
    check_tsk_error(ret);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, 0);
    check_tsk_error(ret);

    while ((ret = tsk_vargen_next(&vargen, &var)) == 1) {
        if (var->site->id >= max_sites) {
            break;
        }
        before = clock();
        score_lib = run_parsimony_library(&tree, var);
        duration = clock() - before;
        lib_total_time += ((double) duration) / CLOCKS_PER_SEC;

        before = clock();
        score_recursive = run_parsimony_recursive(&tree, var);
        duration = clock() - before;
        recursive_total_time += ((double) duration) / CLOCKS_PER_SEC;
        if (score_lib != score_recursive) {
            errx(EXIT_FAILURE, "Score mismatch");
        }

    }
    check_tsk_error(ret);
    printf("lib\t%g\n", lib_total_time / max_sites);
    printf("recursive\t%g\n", recursive_total_time / max_sites);

    tsk_vargen_free(&vargen);
    tsk_tree_free(&tree);
    tsk_treeseq_free(&ts);
    return 0;
}
