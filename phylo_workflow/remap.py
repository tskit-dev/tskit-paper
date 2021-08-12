import tskit
import tqdm

ts = tskit.load("out.trees")
tree = ts.first()
for v in tqdm.tqdm(ts.variants(), total=ts.num_sites):
    pass
for v in tqdm.tqdm(ts.variants(), total=ts.num_sites):
    tree.map_mutations(v.genotypes, v.alleles)
