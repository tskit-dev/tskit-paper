import tskit
import util
import functools
import numba
import numpy as np
import msprime

VECTORISED = False

@numba.njit()
def _iterative_hartigan_parsimony(genotypes, samples, left_child, parent, postorder, preorder, root):
    num_nodes = len(postorder)
    num_alleles = np.max(genotypes) + 1
    optimal_set = np.zeros((num_nodes, num_alleles), dtype=np.int8)
    # Simple version assuming non missing data and one root
    for allele, u in zip(genotypes, samples):
        optimal_set[u, allele] = 1

    allele_count = np.zeros((num_nodes, num_alleles), dtype=np.int32)
    for node_j in postorder:
        if left_child[node_j] != tskit.NULL:
            max_allele_count = 0
            for allele_k in range(num_alleles):
                if allele_count[node_j, allele_k] > max_allele_count:
                    max_allele_count = allele_count[node_j, allele_k]
            for allele_k in range(num_alleles):
                if allele_count[node_j, allele_k] == max_allele_count:
                    optimal_set[node_j, allele_k] = 1
        for allele_k in range(num_alleles):
            allele_count[parent[node_j], allele_k] += optimal_set[node_j, allele_k]

    anc_index, max_val = -1, -1
    for i in range(len(optimal_set[root])):
        if optimal_set[root, i] > max_val:
            anc_index, max_val = i, optimal_set[root, i]

    state = np.zeros((num_nodes), dtype=np.int32)
    state[:] = anc_index
    mutations = 0
    for node_j in preorder:
        state[node_j] = state[parent[node_j]]
        if optimal_set[node_j, state[node_j]] == 0:
            maxval, argmax = -1, -1
            for k in range(num_alleles):
                if optimal_set[node_j, k] > maxval:
                    maxval = optimal_set[node_j, k]
                    argmax = k
            state[node_j] = argmax
            mutations += 1
    return mutations

def iterative_hartigan_parsimony(tree, genotypes, alleles):
    samples = tree.tree_sequence.samples()
    left_child = tree.left_child_array
    parent = tree.parent_array
    postorder = tree.postorder()
    preorder = tree.preorder()
    return _iterative_hartigan_parsimony(genotypes, samples, left_child, parent, postorder, preorder, tree.root)

def run(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return util.benchmark_python(
        ts,
        functools.partial(iterative_hartigan_parsimony, tree),
        "py_numba_iterative",
        max_sites=max_sites,
    )

def warmup():
    ts = msprime.sim_ancestry(100, sequence_length=100000, random_seed=43)
    genotypes = np.zeros(ts.num_samples, dtype=np.int8)
    iterative_hartigan_parsimony(ts.first(), genotypes, ["0"])