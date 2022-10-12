import numpy as np
import click
import tskit
import itertools
import time
import Bio.Phylo.TreeConstruction
import util

VECTORISED = False


def run(ts_path, max_sites):
    ts = tskit.load(ts_path)
    if ts.num_samples >= 10**5:
        return []
    assert ts.num_sites >= max_sites
    variants = itertools.islice(ts.variants(), max_sites)
    times = np.zeros(max_sites)

    tree = ts.first()
    newick_path = util.get_newick_path(ts_path, ts)
    # This bit is very very slow
    bp_tree = Bio.Phylo.read(newick_path, "newick")
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
