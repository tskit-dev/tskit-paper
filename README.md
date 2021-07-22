# tskit-paper

This repo contains the manuscript for the publication describing tskit.

## Abstract

The ability to store and analyse related genetic sequences is an
essential requirement for simulation, inference and analysis in both
population genetics and phlyogenetics. The recent introduction of the
succinct tree sequence data structure has provided a way to achieve
this at population scale. Here we present the `tskit` software,
a high-performance, portable, open-source, community-driven library.
`tskit` allows the creation, manipulation and analysis of succinct tree
sequences, with first-class support for provenance and user-defined metadata.
`tskit` enables a common foundation across software projects that use
succinct tree sequences, which results in unprecedented interoprability,
reproducibility and maintainability.

## Writing process

Like the development of tskit itself, the process of writing this 
article is managed through an open source development model via
GitHub. Those responsible for particular features will be 
assigned sections of writing via issues. Those wishing to make 
edits should also follow these guidelines:

- For significant edits, please use the GitHub workflow described 
  [here](https://stdpopsim.readthedocs.io/en/latest/development.html#github-workflow)
  to create a fork, feature branches and make pull requests.

- For smaller edits, using the 
  [edit file](https://help.github.com/en/github/managing-files-in-a-repository/editing-files-in-your-repository) 
  button on the paper.tex file in GitHub may be simpler than checking out 
  the repo locally, etc.

- Please keep edits localised to a single section to make the 
  review process simpler.

- Please insert line-breaks before 90-100 characters, ideally at semantic
  break points.
  
- For high-level/structural issues, please open an issue to discuss before making changes.


