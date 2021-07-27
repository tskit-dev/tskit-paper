#include <stdio.h>
#include <stdlib.h>
#include <err.h>
#include <time.h>

#include <tskit.h>

#define check_tsk_error(val)                                                            \
    if (val < 0) {                                                                      \
        errx(EXIT_FAILURE, "line %d: %s", __LINE__, tsk_strerror(val));                 \
    }


static void
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
    clock_t before, duration;
    double total_time = 0.0;

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
        run_parsimony_library(&tree, var);
        duration = clock() - before;
        total_time += ((double) duration) / CLOCKS_PER_SEC;
    }
    check_tsk_error(ret);
    printf("%g\n", total_time / max_sites);

    tsk_vargen_free(&vargen);
    tsk_tree_free(&tree);
    tsk_treeseq_free(&ts);
    return 0;
}
