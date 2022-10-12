import subprocess
import pathlib
import tskit
import util

VECTORISED = True


def run(ts_path, max_sites, chunk_size):

    ts_path = pathlib.Path(ts_path)
    ts = tskit.load(ts_path)
    if ts.num_samples >= 10**7:
        # We seem to run out of memory at 10^7 samples, even though it's only
        # 10 sites.
        return []
    newick_path = util.get_newick_path(ts_path, ts)
    fasta_path = ts_path.with_suffix(f".m_{max_sites}.fasta")
    if not fasta_path.exists():
        util.write_fasta(ts, max_sites, fasta_path)

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
