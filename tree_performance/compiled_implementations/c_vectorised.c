
#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <time.h>
#include <string.h>

#include <tskit.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));                 \
    }

static int
argmax(tsk_size_t n, const int8_t *restrict values)
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

#define NODE_OFFSET(num_sites, num_alleles, node)                                       \
    ((size_t) num_sites * (size_t) num_alleles * (size_t) node)
#define SITE_OFFSET(num_alleles, site) ((size_t) num_alleles * (size_t) site)

static void
_hartigan_preorder(tsk_id_t parent, const int8_t *restrict state, tsk_size_t num_nodes,
    tsk_size_t num_sites, tsk_size_t num_alleles, const int8_t *restrict optimal_set,
    int32_t *restrict score, const tsk_id_t *restrict right_child,
    const tsk_id_t *restrict left_sib)
{
    tsk_id_t child;
    tsk_size_t j;
    const int8_t *parent_optimal_set
        = optimal_set + NODE_OFFSET(num_sites, num_alleles, parent);
    const int8_t *site_optimal_set;
    int8_t *node_state = malloc(num_sites * sizeof(*node_state));

    if (node_state == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }

    for (j = 0; j < num_sites; j++) {
        site_optimal_set = parent_optimal_set + SITE_OFFSET(num_alleles, j);
        node_state[j] = state[j];
        if (site_optimal_set[state[j]] == 0) {
            /* Choose a new state */
            node_state[j] = argmax(num_alleles, site_optimal_set);
            score[j]++;
        }
    }
    for (child = right_child[parent]; child != TSK_NULL; child = left_sib[child]) {
        _hartigan_preorder(child, node_state, num_nodes, num_sites, num_alleles,
            optimal_set, score, right_child, left_sib);
    }
    free(node_state);
}

static void
_hartigan_postorder(tsk_id_t parent, tsk_size_t num_nodes, tsk_size_t num_sites,
    tsk_size_t num_alleles, int8_t *restrict optimal_set,
    const tsk_id_t *restrict right_child, const tsk_id_t *restrict left_sib)
{
    tsk_size_t j, k;
    tsk_id_t child;
    int32_t max_allele_count, *allele_count, *site_allele_count;
    int8_t *node_optimal_set, *site_optimal_set;

    allele_count = calloc(num_sites * num_alleles, sizeof(*allele_count));
    if (allele_count == NULL) {
        errx(EXIT_FAILURE, "No memory");
    }

    for (child = right_child[parent]; child != TSK_NULL; child = left_sib[child]) {
        _hartigan_postorder(child, num_nodes, num_sites, num_alleles, optimal_set,
            right_child, left_sib);
        node_optimal_set = optimal_set + NODE_OFFSET(num_sites, num_alleles, child);
        /* Unroll the inner num_alleles loop */
        for (j = 0; j < num_sites * num_alleles; j++) {
            allele_count[j] += node_optimal_set[j];
        }
    }
    if (right_child[parent] != TSK_NULL) {
        node_optimal_set = optimal_set + NODE_OFFSET(num_sites, num_alleles, parent);
        for (j = 0; j < num_sites; j++) {
            site_optimal_set = node_optimal_set + SITE_OFFSET(num_alleles, j);
            site_allele_count = allele_count + SITE_OFFSET(num_alleles, j);
            max_allele_count = 0;
            for (k = 0; k < num_alleles; k++) {
                max_allele_count = TSK_MAX(max_allele_count, site_allele_count[k]);
            }
            for (k = 0; k < num_alleles; k++) {
                if (site_allele_count[k] == max_allele_count) {
                    site_optimal_set[k] = 1;
                }
            }
        }
    }
    free(allele_count);
}

static tsk_size_t
run_parsimony_library(tsk_tree_t *tree, const int8_t *genotypes)
{
    int8_t ancestral_state;
    tsk_size_t num_transitions;
    tsk_state_transition_t *transitions;

    int ret = tsk_tree_map_mutations(tree, (int8_t *) genotypes, NULL, 0,
        &ancestral_state, &num_transitions, &transitions);
    check_tsk_error(ret);
    free(transitions);
    return num_transitions;
}

static void
run_parsimony(tsk_tree_t *tree, tsk_size_t num_sites, const int8_t *genotypes_chunk,
    bool check_score)
{
    /* tsk_size_t is current 32 bit, which leads to overflows with large values */
    const size_t num_nodes = tsk_treeseq_get_num_nodes(tree->tree_sequence);
    const size_t num_samples = tsk_treeseq_get_num_samples(tree->tree_sequence);
    const tsk_id_t *samples = tsk_treeseq_get_samples(tree->tree_sequence);
    tsk_size_t j, k, lib_score;
    tsk_id_t u;
    const int8_t *site_genotypes;
    int8_t allele, *optimal_set, *ancestral_state, *node_optimal_set, *site_optimal_set;
    int32_t *score;
    size_t num_alleles = 0;

    for (j = 0; j < num_sites * num_samples; j++) {
        num_alleles = TSK_MAX(num_alleles, genotypes_chunk[j]);
    }
    num_alleles++;

    optimal_set = calloc(num_nodes * num_sites * num_alleles, sizeof(*optimal_set));
    ancestral_state = calloc(num_sites, sizeof(*ancestral_state));
    score = calloc(num_sites, sizeof(*score));
    if (optimal_set == NULL || ancestral_state == NULL || score == NULL) {
        errx(EXIT_FAILURE, "Out of memory");
    }

    for (k = 0; k < num_samples; k++) {
        u = samples[k];
        node_optimal_set = optimal_set + NODE_OFFSET(num_sites, num_alleles, u);
        for (j = 0; j < num_sites; j++) {
            site_genotypes = genotypes_chunk + num_samples * j;
            allele = site_genotypes[k];
            site_optimal_set = node_optimal_set + SITE_OFFSET(num_alleles, j);
            site_optimal_set[allele] = 1;
        }
    }
    _hartigan_postorder(tree->left_root, num_nodes, num_sites, num_alleles, optimal_set,
        tree->right_child, tree->left_sib);
    node_optimal_set
        = optimal_set + NODE_OFFSET(num_sites, num_alleles, tree->left_root);
    for (j = 0; j < num_sites; j++) {
        site_optimal_set = node_optimal_set + SITE_OFFSET(num_alleles, j);
        ancestral_state[j] = argmax(num_alleles, site_optimal_set);
    }
    _hartigan_preorder(tree->left_root, ancestral_state, num_nodes, num_sites,
        num_alleles, optimal_set, score, tree->right_child, tree->left_sib);

    if (check_score) { /* TODO add some flags instead? */
        for (j = 0; j < num_sites; j++) {
            site_genotypes = genotypes_chunk + j * num_samples;
            lib_score = run_parsimony_library(tree, site_genotypes);
            if (lib_score != score[j]) {
                printf("error at chunk offset %d: %d != %d\n", j, score[j], lib_score);
                errx(EXIT_FAILURE, "Score mismatch!");
            }
        }
    }

    free(optimal_set);
    free(ancestral_state);
    free(score);
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
    tsk_size_t j, chunk_size, num_samples;
    int8_t *genotypes_chunk = NULL;
    int8_t *site_genotypes;
    double total_time = 0;
    bool check_score = false;
    clock_t before, duration;

    if (argc != 4) {
        errx(EXIT_FAILURE, "usage: <tree sequence file> <max_sites> <chunk_size>");
    }
    max_sites = atoi(argv[2]);
    chunk_size = (tsk_size_t) atoi(argv[3]);
    if (max_sites <= 0 || chunk_size == 0) {
        errx(EXIT_FAILURE, "bad max_sites or chunk_size value");
    }
    ret = tsk_treeseq_load(&ts, argv[1], 0);
    check_tsk_error(ret);
    ret = tsk_tree_init(&tree, &ts, 0);
    check_tsk_error(ret);
    ret = tsk_tree_first(&tree);
    check_tsk_error(ret);

    ret = tsk_vargen_init(&vargen, &ts, NULL, 0, NULL, 0);
    check_tsk_error(ret);

    num_samples = tsk_treeseq_get_num_samples(&ts);
    genotypes_chunk = malloc(chunk_size * num_samples * sizeof(*site_genotypes));
    if (genotypes_chunk == NULL) {
        errx(EXIT_FAILURE, "No memory");
    }
    j = 0;
    while ((ret = tsk_vargen_next(&vargen, &var)) == 1) {
        if (var->site->id >= max_sites) {
            break;
        }
        site_genotypes = genotypes_chunk + num_samples * j;
        memcpy(site_genotypes, var->genotypes.i8, num_samples * sizeof(*site_genotypes));
        j++;
        if (j == chunk_size) {
            j = 0;
            before = clock();
            run_parsimony(&tree, chunk_size, genotypes_chunk, check_score);
            duration = clock() - before;
            total_time += ((double) duration) / CLOCKS_PER_SEC;
        }
    }
    if (j != 0) {
        before = clock();
        run_parsimony(&tree, j, genotypes_chunk, check_score);
        duration = clock() - before;
        total_time += ((double) duration) / CLOCKS_PER_SEC;
    }
    check_tsk_error(ret);
    printf("c_vectorised  %g\n", total_time / max_sites);

    tsk_vargen_free(&vargen);
    tsk_tree_free(&tree);
    tsk_treeseq_free(&ts);
    free(genotypes_chunk);
    return 0;
}
