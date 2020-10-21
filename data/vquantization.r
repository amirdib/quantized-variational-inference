#! /usr/bin/env Rscript

library(muHVT)
library(MASS)



args <- commandArgs(trailing = TRUE)
D <- as.integer(args[1])
n <- as.integer(args[2])
N <- as.integer(args[3])
print(D)
print(n)
print(N)

generate_standard_normal_voronoi <- function(N, d, n_clusters)
    {
   
    mu <- c(rep(0, d)) 
    sigma <-  diag(d) 
    samples <- mvrnorm(N, mu = mu, Sigma = sigma ) 

    hvt.results <- list()
    hvt.results <- muHVT::HVT(samples,
                          nclust = n_clusters,
                          depth = 1,
                          quant.err = 0.2,
                          projection.scale = 10,
                          normalize = T,
                          distance_metric = "L2_Norm",
                          error_metric = "mean")
    hvt.results[[3]][['summary']]['n'] <- hvt.results[[3]][['summary']]['n']/N
    list(hvt.results[[3]][['summary']], samples)
}

set.seed(Sys.time())
result = generate_standard_normal_voronoi(N=N, d=D, n_clusters=n)
rsamples <- result[[2]]
quantized_samples <- result[[1]]


write.csv(quantized_samples, 'grid.dat',row.names = FALSE)