"""
Implementations of parsimony algorithms using Pythran.
"""
import numpy as np


# pythran export _hartigan_postorder(int32, int8[:, :], int32[:], int32[:])
def _hartigan_postorder(parent, optimal_set, right_child, left_sib):
    num_alleles = optimal_set.shape[1]
    allele_count = np.zeros(num_alleles, dtype=np.int32)
    child = right_child[parent]
    while child != -1:
        _hartigan_postorder(child, optimal_set, right_child, left_sib)
        allele_count += optimal_set[child]
        child = left_sib[child]

    # NOTE: taking a short-cut here as assuming we only have external samples
    # Fine for this analysis.
    if right_child[parent] != -1:
        max_allele_count = np.max(allele_count)
        for j in range(num_alleles):
            if allele_count[j] == max_allele_count:
                optimal_set[parent, j] = 1


# pythran export _hartigan_preorder(int, int, int8[:, :], int32[:], int32[:])
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
