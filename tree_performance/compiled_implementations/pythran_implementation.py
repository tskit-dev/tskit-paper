"""
Implementations of parsimony algorithms using Pythran.
"""
import numpy as np

# pythran export _hartigan_initialise(int8[:,:], int8[:], int32[:])
def _hartigan_initialise(optimal_set, genotypes, samples):
    for j, u in enumerate(samples):
        optimal_set[u, genotypes[j]] = 1


# pythran export _hartigan_postorder(int32, int64, int8[:, :], int32[:], int32[:])
def _hartigan_postorder(parent, num_alleles, optimal_set, right_child, left_sib):
    allele_count = np.zeros(num_alleles, dtype=np.int32)
    child = right_child[parent]
    while child != -1:
        _hartigan_postorder(child, num_alleles, optimal_set, right_child, left_sib)
        allele_count += optimal_set[child]
        child = left_sib[child]

    if right_child[parent] != -1:
        max_allele_count = np.max(allele_count)
        for j in range(num_alleles):
            if allele_count[j] == max_allele_count:
                optimal_set[parent, j] = 1


# pythran export _hartigan_preorder(int32, int8, int8[:, :], int32[:], int32[:])
def _hartigan_preorder(node, state, optimal_set, right_child, left_sib):
    mutations = 0
    if optimal_set[node, state] == 0:
        state = np.argmax(optimal_set[node])
        mutations = 1

    v = right_child[node]
    while v != -1:
        v_muts = _hartigan_preorder(v, state, optimal_set, right_child, left_sib)
        mutations += v_muts
        v = left_sib[v]
    return mutations
