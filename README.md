# Using Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations

We provide the `code` and the `data` used for the numerical experiments in:

[1] "Using Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations" arXiv preprint 

#

The `code/` folder provides the experimental results on various benchmarks, implemented in [Julia](https://julialang.org).

The folder `data/` contains all the necessary information to execute the notebooks. It contains various Polynomial Optimization (POP) instances, with initial relaxation bounds, available local solutions, as well as the SDP solver logs. For more details, download and consult associated `_EXPLANATION.txt` files. The data is always stored in `.jld2` format.


## Getting started

The code requires Julia  1.10.5+ version, as well as the following packages/libraries:

- Linear Algebra
- Random
- SparseArrays
- IJulia and Jupyter (for running the notebooks)
- DynamicPolynomials
- JLD2
- [TSSOS](https://github.com/wangjie212/TSSOS/) (Polynomial optimization library based on the sparsity adapted Moment-SOS hierarchies)
  
Our optimization problems were solved using the SDP solver [Mosek](https://www.mosek.com/) (licence required).

All the experiments can be reproduced by running `code/notebook_name.ipynb`, where `notebook_name` is reflective of the experimental tables from the paper. All instructions on how to run the experiments are detailed in the notebook.

## Contents



The code is organized as follows:

-
- 
