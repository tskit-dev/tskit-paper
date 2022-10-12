import itertools
import subprocess
import time
import numpy as np
import click
import tskit
import textwrap


def benchmark_external(command, ts_path, max_sites=1000, chunk_size=None):
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


def get_newick_path(ts_path, ts):
    # Make a newick file for the specified ts as it's very slow
    newick_path = ts_path.with_suffix(".nwk")
    if not newick_path.exists():
        print("writing", newick_path)
        tree = ts.first()
        with open(newick_path, "w") as f:
            f.write(tree.newick(node_labels={u: str(u) for u in ts.samples()}))
    return newick_path


def write_fasta(ts, num_sites, fasta_path):
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
