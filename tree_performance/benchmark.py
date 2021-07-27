"""
Benchmarking script. Should be run in the current working directory.
"""
import pathlib
import time
import itertools
import functools
import subprocess

import pandas as pd
import numpy as np
import click
import msprime
import tskit
import numba


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


@numba.njit()
def _hartigan_preorder(node, state, optimal_set, right_child, left_sib):
    # Note: simplifying this to just return a count rather than the actual
    # mutations to make it comparable with local C recursive impl
    # mutations = []
    # if optimal_set[node, state] == 0:
    #     state = np.argmax(optimal_set[node])
    #     mutations.append((node, state))
    mutations = 0
    if optimal_set[node, state] == 0:
        state = np.argmax(optimal_set[node])
        mutations = 1

    v = right_child[node]
    while v != tskit.NULL:
        v_muts = _hartigan_preorder(v, state, optimal_set, right_child, left_sib)
        # mutations.extend(v_muts)
        mutations += v_muts
        v = left_sib[v]
    return mutations


def numba_map_mutations(tree, genotypes, alleles):

    right_child = tree.right_child_array
    left_sib = tree.left_sib_array

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_nodes = tree.tree_sequence.num_nodes

    optimal_set = np.zeros((num_nodes + 1, num_alleles), dtype=np.int8)
    for allele, u in zip(genotypes, tree.tree_sequence.samples()):
        optimal_set[u, allele] = 1

    _hartigan_postorder(tree.root, optimal_set, right_child, left_sib)
    ancestral_state = np.argmax(optimal_set[tree.root])
    # Because we're not constructing the mutations we can just return
    # the count directly. It's straightforward to do, though, and
    # doesn't impact performance much because mismatches from the
    # optimal state are rare in the preorder phase
    return _hartigan_preorder(
        tree.root, ancestral_state, optimal_set, right_child, left_sib
    )


def benchmark_python(ts, func, implementation, max_sites=1000):
    assert ts.num_sites >= max_sites
    variants = itertools.islice(ts.variants(), max_sites)
    times = np.zeros(max_sites)

    with click.progressbar(
        variants, length=max_sites, label=f"{implementation}:n={ts.num_samples}"
    ) as bar:
        for j, variant in enumerate(bar):
            before = time.perf_counter()
            num_mutations = func(variant.genotypes, variant.alleles)
            duration = time.perf_counter() - before
            times[j] = duration
            site = ts.site(j)
            assert num_mutations <= len(site.mutations)
    return [
        {
            "implementation": f"{implementation}",
            "sample_size": ts.num_samples,
            "time_mean": np.mean(times),
            "time_var": np.var(times),
        }
    ]


def benchmark_tskit(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()

    def f(genotypes, alleles):
        _, mutations = tree.map_mutations(genotypes, alleles)
        return len(mutations)

    return benchmark_python(ts, f, "tskit", max_sites=max_sites)


def benchmark_numba(ts_path, max_sites):

    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return benchmark_python(
        ts,
        functools.partial(numba_map_mutations, tree),
        "numba",
        max_sites=max_sites,
    )


def benchmark_external(command, ts_path, max_sites):
    result = subprocess.run(
        command + f" {ts_path} {max_sites}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    ret = []
    ts = tskit.load(ts_path)
    for line in result.stdout.splitlines():
        splits = line.split()
        implementation = splits[0]
        time = float(splits[1])
        ret.append(
            {
                "implementation": implementation,
                "sample_size": ts.num_samples,
                "time_mean": time,
                "time_var": 0,  # TODO Probably not worth bothering with this
            }
        )
    return ret


def benchmark_c_library(ts_path, max_sites):
    return benchmark_external(f"./c_implementation", ts_path, max_sites)


@click.command()
@click.option("--max-sites", type=int, default=1000)
def run_benchmarks(max_sites):
    """
    Run the benchmarks and save the data.
    """
    # Warm up the jit
    ts = msprime.sim_ancestry(100, sequence_length=100000, random_seed=43)
    numba_map_mutations(ts.first(), np.zeros(ts.num_samples, dtype=np.int8), ["0"])

    datapath = pathlib.Path("data")
    perf_data = []
    for path in sorted(datapath.glob("*.trees")):
        ts = tskit.load(path)
        assert ts.num_trees == 1
        for impl in [benchmark_numba, benchmark_tskit, benchmark_c_library]:
            perf_data.extend(impl(path, max_sites=max_sites))
            print(perf_data[-3:])
            df = pd.DataFrame(perf_data)
            df.to_csv("../data/tree-performance.csv")


def run_simulation(
    sample_size, mutation_rate=1e-2, sequence_length=100000, random_seed=42
):
    ts = msprime.sim_ancestry(
        sample_size, ploidy=1, random_seed=random_seed, sequence_length=sequence_length
    )
    return msprime.sim_mutations(ts, rate=mutation_rate)


@click.command()
def generate_data():
    """
    Generate the data used in the benchmarks. Saved in the "data" directory.
    """
    for k in range(1, 7):
        n = 10 ** k
        ts = run_simulation(n)
        print(n, ":", ts.num_mutations, "at", ts.num_sites, "sites")
        ts.dump(f"data/n_1e{k}.trees")


@click.group()
def cli():
    pass


cli.add_command(generate_data)
cli.add_command(run_benchmarks)


if __name__ == "__main__":
    cli()
