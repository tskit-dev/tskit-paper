"""
Benchmarking script. Should be run in the current working directory.
"""
import importlib
import sys
import os
import pathlib
import benchmarks.numba
import benchmarks.numba_vectorised
import benchmarks.pythran
import pandas as pd
import numpy as np
import click
import msprime
import tskit
import util


def import_bench_source_file(name):
    spec = importlib.util.spec_from_file_location(name, f"benchmarks/{name}")
    if spec is None:
        raise ImportError(f"Could not load spec for module '{name}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except FileNotFoundError as e:
        raise ImportError(f"{e.strerror}: {name}") from e
    return module


@click.command()
@click.option("--max-sites", type=int, default=100)
@click.option("--chunk-size", type=int, default=1000)
def benchmark(max_sites, chunk_size):
    """
    Run all benchmarks
    """
    modules = [
        import_bench_source_file(module)
        for module in os.listdir("benchmarks")
        if module.endswith(".py")
    ]
    for benchmark in modules:
        try:
            benchmark.warmup()
            print(f"Warmed up {benchmark.__name__}")
        except AttributeError:
            pass

    datapath = pathlib.Path("data")
    s_perf_data = []
    v_perf_data = []
    for path in sorted(datapath.glob("*.trees")):
        ts = tskit.load(path)
        order = "preorder" if "preorder" in str(path) else "msprime"
        assert ts.num_trees == 1
        for benchmark in modules:
            print("running", benchmark.__name__)
            if benchmark.VECTORISED:
                for datum in benchmark.run(
                    path, max_sites=max_sites, chunk_size=min(chunk_size, max_sites)
                ):
                    v_perf_data.append({"order": order, **datum})
                    print(datum)
            else:
                for datum in benchmark.run(path, max_sites=max_sites):
                    s_perf_data.append({"order": order, **datum})
                    print(datum)
            df = pd.DataFrame(s_perf_data)
            df.to_csv("../data/tree-performance-sequential.csv")
            df = pd.DataFrame(v_perf_data)
            df.to_csv("../data/tree-performance-vectorised.csv")


def run_simulation(
    sample_size, mutation_rate=1e-2, sequence_length=100000, random_seed=42
):
    ts = msprime.sim_ancestry(
        sample_size, ploidy=1, random_seed=random_seed, sequence_length=sequence_length
    )
    return msprime.sim_mutations(ts, rate=mutation_rate)


def to_preorder(ts, verify=False):
    """
    Returns a copy of the specified tree sequence in which the nodes are listed
    in preorder, such that the first root is node 0, its left-most child is node 1,
    etc.
    """
    if ts.num_trees != 1:
        raise ValueError("Only applicable for tree sequences containing one tree")
    node_map = np.zeros(ts.num_nodes, dtype=np.int32) - 1
    tables = ts.dump_tables()
    tables.nodes.clear()
    tree = ts.first()
    for u in tree.nodes():
        node_map[u] = tables.nodes.append(ts.node(u))
    tables.edges.parent = node_map[tables.edges.parent]
    tables.edges.child = node_map[tables.edges.child]
    tables.mutations.node = node_map[tables.mutations.node]
    new_ts = tables.tree_sequence()
    if verify:
        # This isn't really necessary, but it doesn't take long and just
        # to reassure ourselves it's working correctly in the absence of
        # unit tests
        zipped = zip(new_ts.variants(samples=node_map[ts.samples()]), ts.variants())
        with click.progressbar(
            zipped, length=ts.num_sites, label=f"Verify preorder"
        ) as bar:
            for v1, v2 in bar:
                assert np.array_equal(v1.genotypes, v2.genotypes)
    return new_ts


@click.command()
def generate_data():
    """
    Generate the data used in the benchmarks. Saved in the "data" directory.
    """
    for k in range(1, 8):
        n = 10**k
        ts = run_simulation(n)
        print(n, ":", ts.num_mutations, "at", ts.num_sites, "sites")
        ts.dump(f"data/n_1e{k}.trees")
        ts_preorder = to_preorder(ts, verify=k < 6)
        ts_preorder.dump(f"data/n_1e{k}_preorder.trees")


@click.group()
def cli():
    pass


@click.command()
@click.argument("filename")
def verify(filename):
    """
    Verify the Python implementations to make sure they are correct.
    """

    ts = tskit.load(filename)
    tree = ts.first()

    chunk_size = 6
    alleles = ("A", "C", "G", "T")
    chunks = util.variant_chunks(ts, chunk_size, alleles=alleles)
    with click.progressbar(
        chunks, length=ts.num_sites // chunk_size, label="verify chunkwise"
    ) as bar:
        for chunk in bar:
            genotypes = np.array([var.genotypes for var in chunk])
            chunk_score = (
                benchmarks.numba_vectorised.numba_hartigan_parsimony_vectorised(
                    tree, genotypes, alleles
                )
            )
            # print(chunk_score)
            # print(len(chunk), len(chunk_score))
            assert len(chunk_score) == len(chunk)
            for var, score in zip(chunk, chunk_score):
                _, mutations = tree.map_mutations(var.genotypes, var.alleles)
                lib_score = len(mutations)
                # print(score, lib_score)
                assert score == lib_score

    with click.progressbar(
        ts.variants(), length=ts.num_sites, label=f"Verify parsimony"
    ) as bar:
        for variant in bar:
            _, mutations = tree.map_mutations(variant.genotypes, variant.alleles)
            lib_score = len(mutations)
            pythran_score = benchmarks.pythran.pythran_hartigan_parsimony(
                tree, variant.genotypes, variant.alleles
            )
            assert pythran_score == lib_score
            numba_score = benchmarks.numba.numba_hartigan_parsimony(
                tree, variant.genotypes, variant.alleles
            )
            assert numba_score == lib_score


@click.command()
@click.argument("filename")
def quickbench(filename):
    """
    Run a quick benchmark on the specified file.
    """
    warmup_jit()

    ts = tskit.load(filename)
    assert ts.num_trees == 1
    for impl in [
        # benchmark_biopython,
        benchmark_R,
        # benchmark_pythran,
        # benchmark_numba,
        benchmark_numba_vectorised,
        # benchmark_tskit,
        benchmark_c_library,
    ]:
        # m = 100 if ts.num_samples < 10 ** 6 else 10
        m = 10000
        for datum in impl(filename, max_sites=m):
            print(datum["implementation"], datum["time_mean"], sep="\t")
    # _hartigan_postorder.inspect_types()
    # _hartigan_preorder.inspect_types()


cli.add_command(generate_data)
cli.add_command(verify)
cli.add_command(quickbench)
cli.add_command(benchmark)


if __name__ == "__main__":
    cli()
