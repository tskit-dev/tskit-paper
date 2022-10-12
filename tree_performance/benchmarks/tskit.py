import tskit
import util

VECTORISED = False


def run(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()

    def f(genotypes, alleles):
        _, mutations = tree.map_mutations(genotypes, alleles)
        return len(mutations)

    return util.benchmark_python(ts, f, "py_tskit", max_sites=max_sites)
