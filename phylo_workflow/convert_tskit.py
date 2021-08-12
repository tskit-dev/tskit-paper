import parsimony_pb2
import dendropy
import tskit
import sys
import collections
import numpy as np



def _convert(node, tables, node_map):
    depth = 0
    for child in node.child_nodes():
        depth = max(depth, _convert(child, tables, node_map))
    flags = 0
    metadata = {}
    if node.taxon is not None:
        node_name = str(node.taxon)
        flags = 0 if "condensed" in node_name else tskit.NODE_IS_SAMPLE
        metadata = parse_metadata(node_name)
    node_map[node] = tables.nodes.add_row(flags, time=depth, metadata=metadata)
    for child in node.child_nodes():
        tables.edges.add_row(0, 1, node_map[node], node_map[child])
    return depth + 1

def define_metadata(tables):
    tables.nodes.metadata_schema = tskit.MetadataSchema(
        {
            "codec": "json",
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "date": {"type": ["string", "null"]},
                "accession": {"type": ["string", "null"]},
            },
            "additionalProperties": True,
        }
    )

def parse_metadata(name):
    toks = name.split("|")
    name = toks[0]
    metadata = {"date": toks[-1]}
    if len(toks) > 2:
        assert len(toks) == 3
        metadata["accession"] = toks[1]
    # There is definitely some extra location coding in the names,
    # but there's nothing that's uniform across all samples.
    metadata["name"] = name
    return metadata

def convert(mat):
    tree = dendropy.Tree.get(data=mat.newick, schema="newick")
    tables = tskit.TableCollection(1)
    define_metadata(tables)
    node_map = {}
    _convert(tree.seed_node, tables, node_map)
    # print(tables.nodes)
    tables.sort()
    ts = tables.tree_sequence()
    # print(ts.draw_text())
    # print(node)
    return ts

if __name__ == "__main__":
    mat = parsimony_pb2.data()
    with open(sys.argv[1], "rb") as pb_file:
        mat.ParseFromString(pb_file.read())

    condensed_nodes = {}
    for val in mat.condensed_nodes:
        condensed_nodes[val.node_name] = val.condensed_leaves
    print("WTF")
    print(sum(len(g) for g in condensed_nodes.values()))
    ts = convert(mat)
    tables = ts.dump_tables()
    # Let's assume the mutations are in depth first traversal order.
    tree = ts.first()
    assert ts.num_nodes == len(mat.node_mutations)

    print("Making sites")
    sites = {}
    acgt = "ACGT"
    for j, u in enumerate(tree.nodes()):
        # print("node", j, u)
        # print(u, mat_mutations)
        mat_mutations = mat.node_mutations[j]
        # print(type(mat_mutations))
        # print(mat_mutations.mutation[0])
        # print(dir(mat_mutations))
        # print(len(mat_mutations))
        for mut in mat_mutations.mutation:
            pos = mut.position
            if pos not in sites:
                sites[pos] = acgt[mut.ref_nuc], list()
            ancestral_state, mutations = sites[pos]
            assert ancestral_state == acgt[mut.ref_nuc]
            assert len(mut.mut_nuc) == 1
            mutations.append((u, acgt[mut.mut_nuc[0]]))

    for pos in sorted(sites.keys()):
        ancestral_state, mutations = sites[pos]
        site = tables.sites.add_row(pos, ancestral_state)
        for node, derived_state in mutations:
            tables.mutations.add_row(site=site, node=node, derived_state=derived_state)
    tables.sequence_length = pos + 1

    #Expand condensed nodes, we have to do this last so mutations match above
    count = 0
    for node in ts.nodes():
        if 'name' in node.metadata and "condensed" in node.metadata['name']:
            for child in condensed_nodes[node.metadata['name'].replace(" ", "_")[1:-1]]:
                count += 1
                child_id = tables.nodes.add_row(tskit.NODE_IS_SAMPLE, time=node.time-1, metadata=parse_metadata(str(child)))
                tables.edges.add_row(0, 1, node.id, child_id)
    print(count)
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()

    tables.edges.right = np.zeros_like(tables.edges.right) + tables.sequence_length

    ts = tables.tree_sequence()
    print(ts)
    ts.dump(sys.argv[2])
    # ts = tskit.load(sys.argv[1])
    # for node in ts.nodes():
    #     # print(node.metadata)
    #     if "name" in node.metadata:
    #         name = node.metadata["name"]
    #         if "condensed" not in name:
    #             parse_metadata(name)



