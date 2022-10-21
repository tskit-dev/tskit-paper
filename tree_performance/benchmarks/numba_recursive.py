import tskit
import util
import functools
import numba
import numpy as np
import msprime

VECTORISED = False


@numba.njit()
def _hartigan_initialise(optimal_set, genotypes, samples):
    for j, u in enumerate(samples):
        optimal_set[u, genotypes[j]] = 1


@numba.njit()
def _hartigan_preorder(node, state, optimal_set, right_child, left_sib):
    mutations = 0
    if optimal_set[node, state] == 0:
        state = np.argmax(optimal_set[node])
        mutations = 1

    v = right_child[node]
    while v != tskit.NULL:
        v_muts = _hartigan_preorder(v, state, optimal_set, right_child, left_sib)
        mutations += v_muts
        v = left_sib[v]
    return mutations


@numba.njit()
def _hartigan_postorder(parent, optimal_set, right_child, left_sib):
    num_alleles = optimal_set.shape[1]
    allele_count = np.zeros(num_alleles, dtype=np.int32)
    child = right_child[parent]
    while child != tskit.NULL:
        _hartigan_postorder(child, optimal_set, right_child, left_sib)
        allele_count += optimal_set[child]
        child = left_sib[child]
    # NOTE: taking a short-cut here as assuming we only have external samples
    # Fine for this analysis.
    if right_child[parent] != tskit.NULL:
        max_allele_count = np.max(allele_count)
        for j in range(num_alleles):
            if allele_count[j] == max_allele_count:
                optimal_set[parent, j] = 1


def numba_hartigan_parsimony(tree, genotypes, alleles):
    right_child = tree.right_child_array
    left_sib = tree.left_sib_array

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_nodes = tree.tree_sequence.num_nodes

    optimal_set = np.zeros((num_nodes, num_alleles), dtype=np.int8)
    _hartigan_initialise(optimal_set, genotypes, tree.tree_sequence.samples())
    _hartigan_postorder(tree.root, optimal_set, right_child, left_sib)
    ancestral_state = np.argmax(optimal_set[tree.root])
    # Because we're not constructing the mutations we can just return
    # the count directly. It's straightforward to do, though, and
    # doesn't impact performance much because mismatches from the
    # optimal state are rare in the preorder phase
    return _hartigan_preorder(
        tree.root, ancestral_state, optimal_set, right_child, left_sib
    )


def run(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return util.benchmark_python(
        ts,
        functools.partial(numba_hartigan_parsimony, tree),
        "py_numba_recursive",
        max_sites=max_sites,
    )


def warmup():
    ts = msprime.sim_ancestry(100, sequence_length=100000, random_seed=43)
    genotypes = np.zeros(ts.num_samples, dtype=np.int8)
    numba_hartigan_parsimony(ts.first(), genotypes, ["0"])
