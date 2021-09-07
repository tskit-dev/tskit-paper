"""
Benchmarking script. Should be run in the current working directory.
"""
import pathlib
import time
import itertools
import functools
import subprocess
import io
import textwrap

import pandas as pd
import numpy as np
import click
import msprime
import tskit
import numba
import Bio.Phylo.TreeConstruction

import pythran_implementation


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
    mutations = 0
    if optimal_set[node, state] == 0:
        state = np.argmax(optimal_set[node])
        mutations = 1

    v = right_child[node]
    while v != tskit.NULL:
        v_muts = _hartigan_preorder(v, state, optimal_set, right_child, left_sib)
        mutations += v_muts
        v = left_sib[v]
    return mutations


@numba.njit()
def _hartigan_initialise(optimal_set, genotypes, samples):
    for j, u in enumerate(samples):
        optimal_set[u, genotypes[j]] = 1


def numba_hartigan_parsimony(tree, genotypes, alleles):

    right_child = tree.right_child_array
    left_sib = tree.left_sib_array

    # Simple version assuming non missing data and one root
    num_alleles = np.max(genotypes) + 1
    num_nodes = tree.tree_sequence.num_nodes

    optimal_set = np.zeros((num_nodes, num_alleles), dtype=np.int8)
    _hartigan_initialise(optimal_set, genotypes, tree.tree_sequence.samples())
    _hartigan_postorder(tree.root, optimal_set, right_child, left_sib)
    ancestral_state = np.argmax(optimal_set[tree.root])
    # Because we're not constructing the mutations we can just return
    # the count directly. It's straightforward to do, though, and
    # doesn't impact performance much because mismatches from the
    # optimal state are rare in the preorder phase
    return _hartigan_preorder(
        tree.root, ancestral_state, optimal_set, right_child, left_sib
    )


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
def _hartigan_initialise_vectorised(optimal_set, genotypes, samples):
    for k, site_genotypes in enumerate(genotypes):
        for j, u in enumerate(samples):
            optimal_set[u, k, site_genotypes[j]] = 1


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
        }
    ]


def benchmark_biopython(ts_path, max_sites=1000):
    ts = tskit.load(ts_path)
    if ts.num_samples >= 10 ** 5:
        return []
    assert ts.num_sites >= max_sites
    variants = itertools.islice(ts.variants(), max_sites)
    times = np.zeros(max_sites)

    tree = ts.first()
    # THIS IS SLOOOOOOOW!
    newick = tree.newick(node_labels={u: str(u) for u in ts.samples()})
    bp_tree = Bio.Phylo.read(io.StringIO(newick), "newick")
    ps = Bio.Phylo.TreeConstruction.ParsimonyScorer()

    with click.progressbar(
        variants, length=max_sites, label=f"BioPython:n={ts.num_samples}"
    ) as bar:
        for j, variant in enumerate(bar):
            records = [
                Bio.SeqRecord.SeqRecord(
                    Bio.Seq.Seq(str(variant.genotypes[k])), id=str(u)
                )
                for k, u in enumerate(ts.samples())
            ]
            alignment = Bio.Align.MultipleSeqAlignment(records)

            before = time.perf_counter()
            bp_score = ps.get_score(bp_tree, alignment)
            duration = time.perf_counter() - before
            times[j] = duration
            _, mutations = tree.map_mutations(variant.genotypes, variant.alleles)
            assert bp_score == len(mutations)
    return [
        {
            "implementation": f"BioPython",
            "sample_size": ts.num_samples,
            "time_mean": np.mean(times),
        }
    ]


def benchmark_tskit(ts_path, max_sites):
    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()

    def f(genotypes, alleles):
        _, mutations = tree.map_mutations(genotypes, alleles)
        return len(mutations)

    return benchmark_python(ts, f, "py_tskit", max_sites=max_sites)


def benchmark_numba(ts_path, max_sites):

    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return benchmark_python(
        ts,
        functools.partial(numba_hartigan_parsimony, tree),
        "py_numba",
        max_sites=max_sites,
    )


def benchmark_pythran(ts_path, max_sites):

    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()
    return benchmark_python(
        ts,
        functools.partial(pythran_hartigan_parsimony, tree),
        "py_pythran",
        max_sites=max_sites,
    )


def benchmark_numba_vectorised(ts_path, max_sites, chunk_size):

    ts = tskit.load(ts_path)
    assert ts.num_trees == 1
    tree = ts.first()

    alleles = ("A", "C", "G", "T")
    total_time = 0
    for chunk in variant_chunks(ts, chunk_size, max_sites=max_sites, alleles=alleles):
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


def benchmark_external(command, ts_path, max_sites, chunk_size=None):
    cmd = command + f" {ts_path} {max_sites}"
    if chunk_size is not None:
        cmd += f" {chunk_size}"
    result = subprocess.run(
        cmd,
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
            }
        )
    return ret


def write_fasta(ts, num_sites, fasta_path):

    tree = ts.first()
    H = np.empty((ts.num_samples, num_sites), dtype=np.int8)
    for var in ts.variants():
        if var.site.id >= num_sites:
            break
        alleles = np.full(len(var.alleles), 0, dtype=np.int8)
        for i, allele in enumerate(var.alleles):
            ascii_allele = allele.encode("ascii")
            allele_int8 = ord(ascii_allele)
            alleles[i] = allele_int8
        H[:, var.site.id] = alleles[var.genotypes]

    with open(fasta_path, "w") as f:
        # Sample are labelled 1,.., n in the newick
        for u, h in zip(ts.samples(), H):
            print(f">{u}", file=f)
            sequence = h.tobytes().decode("ascii")
            for line in textwrap.wrap(sequence):
                print(line, file=f)


def benchmark_R(ts_path, max_sites, chunk_size):

    assert max_sites == chunk_size
    ts_path = pathlib.Path(ts_path)
    ts = tskit.load(ts_path)
    if ts.num_samples >= 10 ** 7:
        # We seem to run out of memory at 10^7 samples, even though it's only
        # 10 sites.
        return []
    newick_path = ts_path.with_suffix(".nwk")
    if not newick_path.exists():
        print("writing", newick_path)
        tree = ts.first()
        with open(newick_path, "w") as f:
            f.write(tree.newick(node_labels={u: str(u) for u in ts.samples()}))
    fasta_path = ts_path.with_suffix(f".m_{max_sites}.fasta")
    if not fasta_path.exists():
        write_fasta(ts, max_sites, fasta_path)

    result = subprocess.run(
        f"Rscript R_implementations.R {ts_path} {max_sites} {newick_path} {fasta_path}",
        shell=True,
        check=True,
        capture_output=True,
        text=True,
    )
    ret = []
    for line in result.stdout.splitlines():
        splits = line.split()
        implementation = splits[0]
        time = float(splits[1])
        ret.append(
            {
                "implementation": implementation,
                "sample_size": ts.num_samples,
                "time_mean": time,
            }
        )
    return ret


def benchmark_c_vectorised(ts_path, max_sites, chunk_size):
    return benchmark_external(f"./c_vectorised", ts_path, max_sites, chunk_size)


def benchmark_c_sequential(ts_path, max_sites):
    return benchmark_external(f"./c_sequential", ts_path, max_sites)


def benchmark_cpp_sequential(ts_path, max_sites):
    return benchmark_external(f"./cpp_sequential", ts_path, max_sites)


def warmup_jit():
    ts = msprime.sim_ancestry(100, sequence_length=100000, random_seed=43)
    genotypes = np.zeros(ts.num_samples, dtype=np.int8)
    numba_hartigan_parsimony(ts.first(), genotypes, ["0"])
    numba_hartigan_parsimony_vectorised(ts.first(), np.array([genotypes]), ["0"])


@click.command()
@click.option("--max-sites", type=int, default=1000)
@click.option("--chunk-size", type=int, default=1000)
def benchmark_vectorised(max_sites, chunk_size):
    """
    Benchmark the vectorised implementations
    """
    warmup_jit()

    datapath = pathlib.Path("data")
    perf_data = []
    for path in sorted(datapath.glob("*.trees")):
        ts = tskit.load(path)
        order = "preorder" if "preorder" in str(path) else "msprime"
        assert ts.num_trees == 1
        for impl in [
            benchmark_numba_vectorised,
            benchmark_c_vectorised,
            benchmark_R,
        ]:
            m = max_sites if ts.num_samples < 10 ** 6 else 10
            for datum in impl(path, max_sites=m, chunk_size=min(chunk_size, m)):
                perf_data.append({"order": order, **datum})
                print(datum)
            df = pd.DataFrame(perf_data)
            df.to_csv("../data/tree-performance-vectorised.csv")


@click.command()
@click.option("--max-sites", type=int, default=100)
def benchmark_sequential(max_sites):
    """
    Benchmark the sequential implementations
    """
    warmup_jit()

    datapath = pathlib.Path("data")
    perf_data = []
    for path in sorted(datapath.glob("*.trees")):
        ts = tskit.load(path)
        order = "preorder" if "preorder" in str(path) else "msprime"
        assert ts.num_trees == 1
        for impl in [
            benchmark_pythran,
            benchmark_numba,
            benchmark_tskit,
            benchmark_c_sequential,
            benchmark_cpp_sequential,
            benchmark_biopython,
        ]:
            print("running", impl)
            m = max_sites if ts.num_samples < 10 ** 6 else 10
            for datum in impl(path, max_sites=m):
                perf_data.append({"order": order, **datum})
                print(datum)
            df = pd.DataFrame(perf_data)
            df.to_csv("../data/tree-performance-sequential.csv")


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


def variant_chunks(ts, chunk_size, max_sites=None, **kwargs):
    chunk = []
    for var in ts.variants(**kwargs):
        if max_sites is not None and var.site.id >= max_sites:
            break
        chunk.append(var)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


@click.command()
def generate_data():
    """
    Generate the data used in the benchmarks. Saved in the "data" directory.
    """
    for k in range(1, 8):
        n = 10 ** k
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
    chunks = variant_chunks(ts, chunk_size, alleles=alleles)
    with click.progressbar(
        chunks, length=ts.num_sites // chunk_size, label="verify chunkwise"
    ) as bar:
        for chunk in bar:
            genotypes = np.array([var.genotypes for var in chunk])
            chunk_score = numba_hartigan_parsimony_vectorised(tree, genotypes, alleles)
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
            pythran_score = pythran_hartigan_parsimony(
                tree, variant.genotypes, variant.alleles
            )
            assert pythran_score == lib_score
            numba_score = numba_hartigan_parsimony(
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


@click.command()
def benchmark_libs():
    """
    Runs benchmarks on available libraries.
    """
    ts = tskit.load("data/n_1e1.trees")
    tree = ts.first()

    # chunk_size = 10
    # chunk = []
    # score = []
    # for var in ts.variants(alleles=("A", "C", "G", "T")):
    #     chunk.append(var.genotypes)
    #     if len(chunk) == chunk_size:
    #         chunk = np.array(chunk)
    #         chunk_score = numba_hartigan_parsimony_vectorised(tree, chunk, var.alleles)
    #         chunk = []
    #         score.extend(chunk_score)
    # score = np.array(chunk_score, dtype=np.int32)
    # # print(score)
    # # _hartigan_postorder_vectorised.inspect_types()
    # # _hartigan_preorder_vectorised.inspect_types()

    # warmup_jit()

    # benchmark_biopython("data/n_1e3.trees")

    # x = benchmark_R("data/n_1e3.trees")
    # print(x)

    # ts = tskit.load("data/n_1e1.trees")
    # print(ts)

    # assert ts.num_trees == 1
    # for impl in [
    #     benchmark_pythran,
    #     benchmark_numba,
    #     benchmark_tskit,
    #     benchmark_c_library,
    # ]:
    #     m = 1000 if ts.num_samples < 10 ** 6 else 10
    #     for datum in impl(filename, max_sites=m):
    #         print(datum["implementation"], datum["time_mean"], sep="\t")
    # _hartigan_postorder.inspect_types()
    # _hartigan_preorder.inspect_types()


cli.add_command(generate_data)
cli.add_command(benchmark_sequential)
cli.add_command(benchmark_vectorised)
cli.add_command(verify)
cli.add_command(quickbench)
cli.add_command(benchmark_libs)


if __name__ == "__main__":
    cli()
