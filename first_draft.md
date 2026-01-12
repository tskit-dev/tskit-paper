# Population scale Ancestral Recombination Graphs with tskit 1.0

Ancestral recombination graphs (ARGs) are a natural representation of
genome-wide evolutionary history: they connect haplotypes through the sequence
of local genealogies induced by recombination. For decades, ARGs were
conceptually central but practically awkward—too large to store, too slow to
traverse, and too fragile as a software substrate for an expanding ecosystem of
simulators, inference methods, and downstream statistics. Recent progress in
inference and simulation has changed that landscape, and the tskit project has
been a key enabling technology. The release of tskit 1.0 (PyPI 1.0.0 released
Nov 27, 2025) formalises a long-term stability guarantee for the core
representation and APIs, turning succinct genealogical data structures into
durable infrastructure rather than a moving target.

At the heart of tskit is a simple tabular data model: an ARG is stored as nodes
(ancestral genomes at particular times) and edges (parent–child inheritance
relationships spanning genomic intervals). This encoding yields a “tree
sequence”: a sequence of correlated marginal trees along the genome, stored
compactly by exploiting the fact that adjacent trees share most of their
structure. Rather than representing each tree independently, tskit stores the
differences between neighbouring trees, enabling algorithms that stream along
the genome and update state incrementally. This combination—compact storage
plus incremental traversal—has repeatedly shifted what is feasible at
population scale.

Case study 1: population genetic statistics at scale. A major early
demonstration of the tree-sequence advantage was the general framework for
computing single-site statistics based on a duality between mutations on
genomes and branch lengths on genealogies. The resulting tskit statistics API
can compute a broad family of familiar summaries (and their tree-based
analogues) by iterating through trees and updating accumulators, avoiding
repeated scanning over variants. This is not only faster; it makes the
definition of statistics modular and composable, encouraging method development
that stays close to theory while remaining performant in practice. In a
Correspondence context, the key point is not one particular statistic, but that
the same representation supports a general pattern: “write once, run at
population scale,” because the underlying iteration is proportional to
genealogical change along the genome, not to the raw size of dense genotype
matrices.

Case study 2: whole-genome simulation through a real pedigree. The recent
Science study “On the genes, genealogies, and geographies of Quebec” simulated
genetic transmission through a pedigree built from ~4 million genealogical
records and released whole-genome simulated data on nearly 1.5 million
individuals—numbers that would be implausible without a representation that can
compress shared ancestry and be traversed efficiently. This example highlights
a second “scale leap”: tree sequences are not only a storage format, but a
computational interface for forward- and pedigree-based simulation workflows.
The same node/edge tables used for coalescent simulations naturally encode
inheritance through recorded pedigrees, and can be simplified down to the
subset of samples or genomic regions relevant to an analysis while preserving
exact genealogical relationships. That ability—simulate richly, then simplify
aggressively—changes the ergonomics of large simulation studies, enabling
iterative model-building, rapid quality control, and realistic benchmarking.

Case study 3: pandemic-scale inference from viral genomes. ARG and
tree-sequence methods are often mentally compartmentalised: “phylogenetics for
pathogens” and “population genetics for recombining genomes.” The sc2ts project
(SARS-CoV-2 to tree sequence) usefully breaks that dichotomy by inferring
ARG-like structures for SARS-CoV-2 at pandemic scale in tree-sequence form,
with tooling built around tskit and storage layers such as Zarr. While
SARS-CoV-2 is not the archetypal recombining diploid dataset, the example is a
powerful stress test for generality: it shows that a single, stable
representation can support workflows spanning traditional phylogenetic and
population-genetic divides, and that “ARG infrastructure” need not be
restricted to one community’s conventions. In practical terms, it also
underscores a key virtue of the table-based model: node metadata, mutations,
and tree topology can be accessed without pointer-heavy object graphs, enabling
fast queries and scalable storage even when the number of genomes is enormous.

Under the hood, tskit is implemented in C, with widely used frontends in Python
(and bindings in other languages). Its vectorised, table-first design makes it
straightforward to expose zero-copy views into the underlying arrays (for
example through NumPy), supporting high-performance analysis pipelines that
avoid unnecessary memory duplication. This is a pragmatic but consequential
design choice: it lowers the cost of interoperability, makes composable
pipelines easier to build, and helps ensure that downstream tools inherit
performance and correctness properties from a shared core.

The 1.0 release matters because it makes these advantages dependable. Long-term
stability in the core model and APIs reduces “dependency risk” for downstream
developers, supports archival of inferred genealogies as durable scientific
objects, and enables independent tools to interoperate through a common
exchange format. In combination with an active ecosystem—spanning simulation,
inference, statistics, and visualisation—tskit 1.0 marks a transition from a
successful library to stable infrastructure for population-scale genealogical
analysis.

