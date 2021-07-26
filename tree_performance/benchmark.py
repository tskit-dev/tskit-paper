"""
Benchmarking script. Should be run in the current working directory.
"""
import pathlib
import time
import itertools

import pandas as pd
import numpy as np
import click
import msprime
import tskit


def benchmark_tskit(ts, max_sites=1000):
    tree = ts.first()
    assert ts.num_sites >= max_sites
    variants = itertools.islice(ts.variants(), max_sites)
    times = np.zeros(max_sites)

    with click.progressbar(
        variants, length=max_sites, label=f"tskit:n={ts.num_samples}"
    ) as bar:
        for j, variant in enumerate(bar):
            before = time.perf_counter()
            tree.map_mutations(variant.genotypes, variant.alleles)
            duration = time.perf_counter() - before
            times[j] = duration
    return {
        "implementation": "tskit",
        "sample_size": ts.num_samples,
        "time_mean": np.mean(times),
        "time_var": np.var(times),
    }


@click.command()
@click.option("--max-sites", type=int, default=1000)
def run_benchmarks(max_sites):
    """
    Run the benchmarks and save the data.
    """
    datapath = pathlib.Path("data")
    perf_data = []
    for path in sorted(datapath.glob("*.trees")):
        ts = tskit.load(path)
        assert ts.num_trees == 1
        perf_data.append(benchmark_tskit(ts, max_sites=max_sites))
        print(perf_data[-1])
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
