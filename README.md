# Using Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations

We provide the `code` and the `data` used for the numerical experiments in:

[1] "Using Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations" arXiv preprint 

#

The `code/` directory includes all the algorithms and functions required to run the notebooks and evaluate various benchmarks. The implementation is written in [Julia](https://julialang.org).

The `data/` directory contains all the datasets necessary to replicate the experimental results presented in the paper. It includes Polynomial Optimization (POP) instances, initial relaxation bounds, known local solutions, and logs from the SDP solver. For detailed descriptions, refer to the accompanying `_EXPLANATION.txt` files. All data is provided in `.jld2` format.


## Getting started

The code requires Julia  1.10.5+ version, as well as the following packages/libraries:

- Linear Algebra
- Random
- Printf,
- IOLogging
- SparseArrays
- IJulia and Jupyter (for running the notebooks)
- DynamicPolynomials
- JLD2
- [TSSOS](https://github.com/wangjie212/TSSOS/) (polynomial optimization library based on the sparsity adapted Moment-SOS hierarchies)
  
Our optimization problems were solved using the SDP solver [Mosek](https://www.mosek.com/) (licence required).

All the experiments can be reproduced by running `code/notebook_name.ipynb`, where `notebook_name` is reflective of the experimental tables from the paper. All instructions on how to run the experiments are detailed in the notebooks and associated `.jl` function files.

## Examples

In order solve the POP instances corresponding to the 3rd row of `Table 1` (where $n=20, s= 0\\% $), execute the following lines:

```julia
using DynamicPolynomials, TSSOS, Random, LinearAlgebra, Statistics, JLD2,Printf, IOLogging, SparseArrays
include("path/Functions-H1.jl")   

n = 20                # set the problem dimension
@polyvar x[1:n]       # define decision variables

N = 15                # maximum number of iterations
eps = 0.1             # threshold penalization factor

seeds = data50_20_random[7]            # random seeds used to generate different QCQP instances
                                       # download 'data50_20_random.jld2 from the `data/` folder

algorithm_logs_file = "n=20_eps=0pt1.txt"   # initialize a .txt file where the algorithm logs will be saved

running_time = true                    # specify whether SDP solving time from each iteration should be recovered
delta = 0.5                            # set the gap tolerance 

beta = 1e-5                            # Regularization parameter used for defining Christoffel sublevel sets.

res_20_0pt1 = Iterate_Ly_random_Multiple_Instances(x, n, N, eps, seeds, algorithm_logs_file, running_time, delta, beta)
```
Note: Ensure the file `Functions-H1.jl` is correctly included and accessible in the specified path. 

- Then, `res_20_0pt1[1]` will contain the sequences of bound improvements for each of `length(seeds)` instances.
  In particular, `res_20_0pt1[1][9]` will output bound trajectories of `H1` for the instance corresponding to the random seed `seeds[9]`, i.e:
  
  ```
  
  3-element Vector{Float64}:
   -51.413604
   -50.994815
   -50.570687
  
 - while `res_20_0pt1[7][9]` will output the combined SDP solving time (in seconds) of all 3 iterations (may vary depending on your hardware):
    ```
      0.06378790000000001
    ```

    
Read the description of `Iterate_Ly_random_Multiple_Instances()` to fully understand the structure of the output.
#

- In order to apply `H1` to a specific POP instance, unrelated to our experimental setup, modify the problem definition lines of the `iterate_Ly_random()` function.
- Analougous manipulations should be done when applying our second bound strengthening approach `H2`, 
or when using correlative sparsity adapted versions `H1CS` and `H2CS`.

## Main references:
- [TSSOS: a Julia library to exploit sparsity for large-scale polynomial optimization](https://arxiv.org/abs/2103.00915)
- [Sparse Polynomial Optimization: Theory and Practice](https://arxiv.org/abs/2208.11158)


## Contact 
[Srećko Ðurašinović](https://www.linkedin.com/in/srecko-durasinovic-29b5921ba?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BdEqNOBumRMmZlqEysNiMdg%3D%3D): srecko001@e.ntu.edu.sg
