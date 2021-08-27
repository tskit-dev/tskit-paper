# install.packages("reticulate")
# install.packages("phangorn")
library(reticulate)
library(ape)
library(phangorn)

# The num_sites argument here should be the same number of sites in the FASTA.
# I can't see now to get the number of actual sites out of phyDat - it only
# seems to care about the unique sites.
benchmark_phangorn <- function(ts_path, num_sites, tree_path, fasta_path) {

    tree <- read.tree(tree_path)
    # print("read tree")
    data <- read.phyDat(fasta_path, format="fasta", type="DNA")
    # print("data")

    num_distinct_sites <- attr(data, "nr")

    # This method is fast, but seems to use a *lot* of memory. Won't 
    # run for 10^6 samples
    before <- proc.time()
    score <- sankoff(tree, data)
    duration <- proc.time() - before
    
    time_per_site <- (duration[1] + duration[2]) / num_distinct_sites
    # Sum of user and system time
    cat("phangorn", time_per_site, "\n")
        
    # Check the parsimony result against tskit
    tskit <- reticulate::import("tskit")
    ts <- tskit$load(ts_path)
    variants <- ts$variants()
    tsk_tree <- ts$first()

    py_score <- 0
    for (j in 1:num_sites) {
        var <- iter_next(variants)
        ret <- tsk_tree$map_mutations(var$genotypes, var$alleles)
        py_score <- py_score + length(ret[[2]])
    }
    if (py_score != score) {
        stop("mismatch score")
    }

}

args = commandArgs(trailingOnly=TRUE)

benchmark_phangorn(args[1], as.integer(args[2]), args[3], args[4])
