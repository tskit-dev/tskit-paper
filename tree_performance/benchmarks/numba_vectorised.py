import tskit
import time
import util
import numpy as np
import numba
import msprime

VECTORISED = True


@numba.njit()
def _hartigan_initialise_vectorised(optimal_set, genotypes, samples):
    for k, site_genotypes in enumerate(genotypes):
        for j, u in enumerate(samples):
            optimal_set[u, k, site_genotypes[j]] = 1


@numba.njit()
def _hartigan_preorder_vectorised(node, state, optimal_set, right_child, left_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    mutations = np.zeros(num_sites, dtype=np.int32)
    # Strictly speaking we only need to do this if we mutate it. Might be worth
    # keeping track of - but then that would complicate the inner loop, which
    # could hurt vectorisation/pipelining/etc.
    state = state.copy()
    for j in range(num_sites):
        site_optimal_set = optimal_set[node, j]
        if site_optimal_set[state[j]] == 0:
            # state[j] = np.argmax(site_optimal_set)
            maxval = -1
            argmax = -1
            for k in range(num_alleles):
                if site_optimal_set[k] > maxval:
                    maxval = site_optimal_set[k]
                    argmax = k
            state[j] = argmax
            mutations[j] = 1

    v = right_child[node]
    while v != tskit.NULL:
        v_muts = _hartigan_preorder_vectorised(
            v, state, optimal_set, right_child, left_sib
        )
        mutations += v_muts
        v = left_sib[v]
    return mutations


@numba.njit()
def _hartigan_postorder_vectorised(parent, optimal_set, right_child, left_sib):
    num_sites, num_alleles = optimal_set.shape[1:]

    allele_count = np.zeros((num_sites, num_alleles), dtype=np.int32)
    child = right_child[parent]
    while child != tskit.NULL:
        _hartigan_postorder_vectorised(child, optimal_set, right_child, left_sib)
        allele_count += optimal_set[child]
        child = left_sib[child]

    if right_child[parent] != tskit.NULL:
        for j in range(num_sites):
            site_allele_count = allele_count[j]
            # max_allele_count = np.max(site_allele_count)
            max_allele_count = 0
            for k in range(num_alleles):
                if site_allele_count[k] > max_allele_count:
                    max_allele_count = site_allele_count[k]
            for k in range(num_alleles):
                if site_allele_count[k] == max_allele_count:
                    optimal_set[parent, j, k] = 1


def numba_hartigan_parsimony_vectorised(tree, genotypes, alleles):

    right_child = tree.right_child_array
    left_sib = tree.left_sib_array

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_sites = genotypes.shape[0]
    num_nodes = tree.tree_sequence.num_nodes
    samples = tree.tree_sequence.samples()

    optimal_set = np.zeros((num_nodes, num_sites, num_alleles), dtype=np.int8)
    _hartigan_initialise_vectorised(optimal_set, genotypes, samples)
    _hartigan_postorder_vectorised(tree.root, optimal_set, right_child, left_sib)
    ancestral_state = np.argmax(optimal_set[tree.root], axis=1)
    return _hartigan_preorder_vectorised(
        tree.root, ancestral_state, optimal_set, right_child, left_sib
    )


def run(ts_path, max_sites, chunk_size):

    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()

    alleles = ("A", "C", "G", "T")
    total_time = 0
    for chunk in util.variant_chunks(
        ts, chunk_size, max_sites=max_sites, alleles=alleles
    ):
        genotypes = np.array([var.genotypes for var in chunk])
        before = time.perf_counter()
        numba_hartigan_parsimony_vectorised(tree, genotypes, alleles)
        duration = time.perf_counter() - before
        total_time += duration
    return [
        {
            "implementation": "py_numba_vect",
            "sample_size": ts.num_samples,
            "time_mean": total_time / max_sites,
        }
    ]


def warmup():
    ts = msprime.sim_ancestry(100, sequence_length=100000, random_seed=43)
    genotypes = np.zeros(ts.num_samples, dtype=np.int8)
    numba_hartigan_parsimony_vectorised(ts.first(), np.array([genotypes]), ["0"])
