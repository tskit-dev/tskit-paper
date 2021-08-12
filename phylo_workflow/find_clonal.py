import time

import tskit
import numpy as np
import msprime
import collections


# All this code assumes a single tree

def num_mutations_above_node(ts):
    return np.bincount(ts.tables.mutation.node)

def sum_mutations_below(ts):
    tree = ts.first()
    num_above_node = num_mutations_above_node(ts)
    # +1 to deal with the -1 case at the root
    out = np.zeros(dtype=np.int32, shape=ts.num_nodes+1)
    parent = tree.parent_array
    for node in tree.nodes(order="postorder"):
        out[parent[node]] += num_above_node[node] + out[node]
    return out

def clonal_groups(ts):
    tree = ts.first()
    muts_below = sum_mutations_below(ts)
    stack = collections.deque(tree.roots)
    while len(stack)>0:
        v = stack.pop()
        if muts_below[v] == 0: #Clonal below this node
            for i, _ in enumerate(tree.samples(v)):
                if i > 0:
                    #More than one sample in this subtree
                    yield tree.samples(v)
                    break
        else:
            stack.extend(reversed(tree.children(v)))

ts = msprime.sim_ancestry(10, random_seed=42)
ts = msprime.sim_mutations(ts, rate=5, random_seed=42)
# with open("out.svg", "w") as f:
#     f.write(ts.draw_svg(size=(1000,1000)))
assert [[n for n in g] for g in clonal_groups(ts) ] == [
    [8, 17, 19],
    [5, 7],
    [2, 11],
    [0, 4, 10, 18]
]

ts = tskit.load("out.trees")
t = time.perf_counter()
out = clonal_groups(ts)
for group in out:
    for n in group:
        pass
t2 = time.perf_counter()
print(t2-t)