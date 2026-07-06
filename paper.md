**Population-scale Ancestral Recombination Graphs with tskit 1.0**

Ben Jeffery^1,\*^, Yan Wong^1,\*^, Kevin Thornton^2,\*^, Georgia Tsambos^3,4,\*^, Gertjan Bisschop^1,†^, Yun Deng^5,†^, E. Castedo Ellerman^6,†^, Thomas B. Forest^7,†^, Halley Fritze^8,†^, Daniel Goldstein^9,†^, Gregor Gorjanc^10,†^, Graham Gower^11,†^, Simon Gravel^12,†^, Jeremy Guez^13,14,†^, Benjamin C. Haller^15,†^, Andrew D. Kern^7,†^, Lloyd Kirk^16,†^, Ivan Krukov^12,†^, Hanbin Lee^8,†^, Brieuc Lehmann^17,†^, Hossameldin Loay^1,†^, Matthew M. Osmond^18,†^, Duncan S. Palmer^19,20,21,†^, Nathaniel S. Pope^7,†^, Aaron P. Ragsdale^16,†^, Duncan Robertson^1,†^, Murillo F. Rodrigues^7,†^, Hugo van Kemenade^22,†^, Clemens L. Weiß^23,24,†^, Anthony Wilder Wohns^23,1,†^, Shing H. Zhan^1,25,†^, Brian C. Zhang^19,†^, Marianne Aspbury^26^, Nikolas A. Baya^1^, Saurabh Belsare^7^, Arjun Biddanda^27^, Francisco Campuzano Jiménez^28^, Ariella Gladstein^29^, Bing Guo^30,31^, Savita Karthikeyan^1^, Warren W. Kretzschmar^32^, Inés Rebollo^33,34^, Kumar Saunack^22^, Ruhollah Shemirani^35^, Alexis Simon^36^, Chris Smith^7^, Jeet Sukumaran^37^, Jonathan Terhorst^8^, Per Unneberg^38^, Ao Zhang^1^, Peter Ralph^7,39,‡^, Jerome Kelleher^1,‡^

^1^Big Data Institute, Li Ka Shing Centre for Health Information and Discovery, University of Oxford, OX3 7LF, UK  
^2^Department of Systems Biology, University of California, Irvine, CA 92697, USA  
^3^Melbourne Integrative Genomics, School of Mathematics and Statistics, University of Melbourne, Victoria, 3010, Australia  
^4^Department of Genome Sciences, University of Washington, Seattle, WA 98195, USA  
^5^Department of Genetics, Stanford University, Stanford, CA 94305, USA  
^6^Fresh Pond Research Institute, Cambridge, MA 02140, USA  
^7^Institute of Ecology and Evolution, University of Oregon, Eugene OR 97402, USA  
^8^Department of Statistics, University of Michigan, Ann Arbor, MI, 48109, USA  
^9^Khoury College of Computer Sciences, Northeastern University, MA 02115, USA  
^10^The Roslin Institute and Royal (Dick) School of Veterinary Studies, University of Edinburgh, EH25 9RG, UK  
^11^Microbiology and Infectious Diseases, SA Pathology, Adelaide, SA 5000, Australia  
^12^Department of Human Genetics, McGill University, Montreal, QC H3A 0C7, Canada  
^13^UMR 7206 Eco-Anthropologie, CNRS, MNHN, Université Paris Cité, 75116 Paris, France  
^14^Université Paris-Saclay, CNRS, INRIA, Laboratoire Interdisciplinaire des Sciences du Numérique, 91400, Orsay, France  
^15^Dept. of Computational Biology, Cornell University, Ithaca, NY 14853, USA  
^16^Department of Integrative Biology, University of Wisconsin--Madison, WI 53706, USA  
^17^Department of Statistical Science, University College London, London, WC1E 7HB, UK  
^18^Department of Ecology and Evolutionary Biology, University of Toronto, Toronto, Ontario, M5S 3B2, Canada  
^19^Department of Statistics, University of Oxford, OX1 3LB, UK  
^20^The Pioneer Centre for SMARTbiomed, Big Data Institute, Li Ka Shing Centre for Health Information and Discovery, University of Oxford, OX3 7LF, UK  
^21^Program in Medical and Population Genetics, Broad Institute of MIT and Harvard, Cambridge, Massachusetts 02142, USA  
^22^Independent researcher  
^23^Department of Genetics, Stanford University School of Medicine, Stanford, CA 94305, USA  
^24^Stanford Cancer Institute, Stanford School of Medicine, Stanford, CA 94305, USA  
^25^Infectious Disease Epidemiology Unit (IDEU), Nuffield Department of Population Health, University of Oxford, OX3 7LF, UK  
^26^Department of Paediatrics, University of Oxford, Oxford, UK  
^27^Department of Biology, Johns Hopkins University, Baltimore, MD, 21218, USA  
^28^Department of Biology, University of Antwerp, Antwerp, 2610, Belgium  
^29^Department of Human Genetics, University of California, Los Angeles, CA 90095, USA  
^30^Center for Vaccine Development and Global Health, University of Maryland School of Medicine, Baltimore, MD, 21201, USA  
^31^Institute for Genome Sciences, University of Maryland School of Medicine, Baltimore, MD, 21201, USA  
^32^Center for Hematology and Regenerative Medicine, Karolinska Institute, 141 83 Huddinge, Sweden  
^33^Instituto Nacional de Investigación Agropecuaria (INIA), Estación Experimental Las Brujas, Ruta 48 km 10, Canelones, Uruguay  
^34^Department of Statistics, Universidad de la República, College of Agriculture, Garzón 780, Montevideo, Uruguay  
^35^Institute for Genomic Health, Icahn School of Medicine at Mount Sinai, NY, 10029, USA  
^36^Sorbonne Université, CNRS, UMR 7144 AD2M, DiSEEM, Station Biologique de Roscoff, France  
^37^Biology Department, San Diego State University, San Diego, CA 92182-4614, USA  
^38^Department of Cell and Molecular Biology, National Bioinformatics Infrastructure Sweden, Science for Life Laboratory, Uppsala University, Husargatan 3, SE-752 37 Uppsala, Sweden  
^39^Department of Data Science, University of Oregon, Eugene OR 97402, USA  

\*Joint first author; †Joint second author; ‡Joint senior author

## Main text

Ancestral recombination graphs (ARGs) capture the full genetic history of
samples from a recombining species. Although ARGs have been a central
theoretical object in population genetics for decades, their practical use was
constrained by the lack of scalable inference methods, standard interchange
formats, and software infrastructure. Recent breakthroughs in simulation and
inference have substantially changed this landscape, leading to renewed
interest in ARG-based analyses across population and statistical
genetics^1--3^. The tskit library has played a key enabling role in this shift
and has become foundational infrastructure for working with ARGs. This paper
marks the release of tskit 1.0, which formalises long-term stability guarantees
for its data formats and APIs.

At the core of tskit is the succinct tree sequence data model which defines a
set of nodes (genomes at particular times) and edges (inheritance relationships
between nodes spanning genomic intervals) in a simple tabular form^4^. This
encoding provides a lossless representation of a general class of ARGs suitable
for large-scale computation^5^. The data model also incorporates site,
mutation, population, and pedigree information and supports arbitrary metadata
associated with each of these components. Provenance information is recorded
natively, enhancing reproducibility and transparency. These features make the
tskit data model a semantically complete and interoperable representation of
ARGs that serves as a common foundation across diverse analytical workflows
(Figure 1).

Simulation is a fundamental tool in population genomics, and was the first
domain in which the tskit data model demonstrated its impact. Introduced
initially as part of the msprime simulator, the tskit data model enabled
performance improvements of several orders of magnitude over previous
coalescent simulation approaches^6^. The same representation later enabled
efficient forward-time simulation of ARGs and yielded substantial speedups by
avoiding explicit simulation of neutral mutations^4^. Because these
forward-time and coalescent simulators share this common representation, their
complementary strengths can be combined within a single workflow. This has made
it possible to simulate ARGs under complex demographic scenarios involving
geography and selection that were previously infeasible. Simulation
capabilities have continued to expand, including whole-autosome ARG simulations
for nearly 1.5 million individuals based on a large human pedigree^7^.

The lack of scalable inference methods has been a major obstacle to empirical
application of ARGs. Although there are many inference methods^5^, tsinfer was
the first to scale to hundreds of thousands of samples, directly leveraging the
tskit data model^8^. Many recent ARG inference methods have chosen to support
tskit as an output format in addition to their own native representations
(Table S1). This shared output layer enables inferred ARGs to interoperate
directly with simulators, facilitating systematic evaluation and benchmarking
against known ground truth. It also shifts the burden of format conversion away
from downstream users, who can instead rely on inference tools to emit results
in a common, well-defined representation. The scalability and flexibility of
this approach are illustrated by the recent inference of an ARG for 2.48
million SARS-CoV-2 whole genomes, which occupies 32 MiB of storage and can be
loaded into memory in under a second^9^.

Efficient storage and analysis of large genetic datasets is a central design
goal of tskit, and the data model has enabled substantial performance gains in
downstream analyses. For example, single-site population genetic statistics can
be computed orders of magnitude faster than from genotype matrices while using
far less memory by operating on the underlying ARG structure^10^. Tskit exposes
a large API with a performance-critical core implemented in C and bindings
available for Python, Rust, and R. Its vectorised, table-first design allows
zero-copy access to underlying arrays, supporting high-performance analysis
pipelines. As a result, downstream tools inherit performance and correctness
properties from a shared, well-tested core.

The goal of tskit is to provide a shared technical foundation, centred on
efficient, well-tested, and thoroughly documented primitive operations on ARGs,
rather than to directly implement end-user workflows. This design principle has
enabled a broad ecosystem of downstream software---spanning simulation, ARG
inference, population and statistical genetic inference, analysis, and
visualisation---with 64 published tools now using tskit as a core dependency
(Table S1). Building on the initial introduction of the succinct tree sequence
data model^6^ and its formalisation as a general ARG representation^5^, tskit
1.0 marks the maturity of the software library and data model for scalable ARG
analysis (see Supplementary Information). By focusing on stable primitives
rather than prescribing analytical pipelines, tskit enables methodological
innovation to concentrate on modelling, inference, and interpretation rather
than bespoke data formats and tooling. In this way, tskit provides a common and
extensible foundation that supports the further expansion of ARG-based analyses
as datasets, methods, and applications grow. As tskit is applied to a wider
range of biological applications, future development is likely to address
additional complexities such as supporting multiple chromosomes and structural
variants. Extensive documentation, tutorials, and other information are
available at <https://tskit.dev>.

## Data and Code Availability

tskit is free and open-source software. Documentation, tutorials, and
installation instructions are available at <https://tskit.dev>, and the source
code is maintained at <https://github.com/tskit-dev/tskit>. The code and data
used to produce this manuscript are available at
<https://github.com/tskit-dev/tskit-paper>.

## Acknowledgements 

We gratefully acknowledge funding from the Robertson
Foundation, the NIH (research grants HG011395 and HG012473), and the NSF
(research grant OAC-2104115), supporting core tskit development.

## Author Contributions

Authors contributed to tskit through software design, development, testing,
documentation, and sustained contributions to the tskit-dev community. 
J.K., P.R., B.J., Y.W., K.T., and G.T. made major contributions;
the joint second authors made substantial contributions; 
and the remaining authors made minor contributions. 
J.K. and P.R. wrote the manuscript with input from all authors.

## Competing Interests

The authors declare no competing interests.

## References
1. Brandt, D. Y. C., Huber, C. D., Chiang, C. W. K. & Ortega-Del Vecchyo, D. The promise of inferring the past using the Ancestral Recombination Graph (ARG). *Genome Biology and Evolution* **16**, evae005 (2024).

2. Lewanski, A. L., Grundler, M. C. & Bradburd, G. S. The era of the ARG: An introduction to ancestral recombination graphs and their significance in empirical evolutionary genomics. *PLOS Genetics* **20**, 1--24 (2024).

3. Nielsen, R., Vaughn, A. H. & Deng, Y. Inference and applications of ancestral recombination graphs. *Nature Reviews Genetics* **26**, 47--58 (2024).

4. Kelleher, J., Thornton, K. R., Ashander, J. & Ralph, P. L. Efficient pedigree recording for fast population genetics simulation. *PLoS Computational Biology* **14**, 1--21 (2018).

5. Wong, Y. *et al.* A general and efficient representation of ancestral recombination graphs. *Genetics* **228**, iyae100 (2024).

6. Kelleher, J., Etheridge, A. M. & McVean, G. Efficient coalescent simulation and genealogical analysis for large sample sizes. *PLOS Computational Biology* **12**, e1004842 (2016).

7. Anderson-Trocmé, L. *et al.* On the genes, genealogies, and geographies of Quebec. *Science* **380**, 849--855 (2023).

8. Kelleher, J. *et al.* Inferring whole-genome histories in large population datasets. *Nature Genetics* **51**, 1330--1338 (2019).

9. Zhan, S. H. *et al.* A pandemic-scale ancestral recombination graph for SARS-CoV-2. *bioRxiv* (2025).

10. Ralph, P., Thornton, K. & Kelleher, J. Efficiently summarizing relationships in large samples: A general duality between statistics of genealogies and genomes. *Genetics* **215**, 779--797 (2020).

## Figure Legend

**Figure 1.** Tskit enables an interoperable ARG software ecosystem. ARGs produced by simulation or inference tools can be analysed by diverse downstream applications via tskit's well-defined tabular data model, C library and Python/Rust/R bindings. Tools shown are representative examples from Table S1 (three per category; ordered by citation count).
