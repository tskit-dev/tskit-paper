"""
Make the data for the running the performance comparisons.
"""
import msprime


def run_simulation(
    sample_size, mutation_rate=1e-2, sequence_length=100000, random_seed=42
):
    ts = msprime.sim_ancestry(
        sample_size, ploidy=1, random_seed=random_seed, sequence_length=sequence_length
    )
    return msprime.sim_mutations(ts, rate=mutation_rate)


if __name__ == "__main__":
    for k in range(1, 7):
        n = 10 ** k
        ts = run_simulation(n)
        print(n, ":", ts.num_mutations, "at", ts.num_sites, "sites")
        ts.dump(f"data/n_1e{k}.trees")
