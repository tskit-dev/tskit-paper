import util
import tskit
import functools
import numpy as np
from compiled_implementations import pythran_implementation
import numba

VECTORISED = False


@numba.njit()
def _hartigan_initialise(optimal_set, genotypes, samples):
    for j, u in enumerate(samples):
        optimal_set[u, genotypes[j]] = 1


def pythran_hartigan_parsimony(tree, genotypes, alleles):
    # Basically same implementation as numba method above.
    right_child = tree.right_child_array
    left_sib = tree.left_sib_array

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_nodes = tree.tree_sequence.num_nodes

    optimal_set = np.zeros((num_nodes + 1, num_alleles), dtype=np.int8)
    _hartigan_initialise(optimal_set, genotypes, tree.tree_sequence.samples())
    pythran_implementation._hartigan_postorder(
        tree.root, num_alleles, optimal_set, right_child, left_sib
    )
    ancestral_state = np.argmax(optimal_set[tree.root]).astype(np.int8)
    return pythran_implementation._hartigan_preorder(
        tree.root, ancestral_state, optimal_set, right_child, left_sib
    )


def run(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return util.benchmark_python(
        ts,
        functools.partial(pythran_hartigan_parsimony, tree),
        "py_pythran",
        max_sites=max_sites,
    )
