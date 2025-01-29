# Leveraging Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations

We provide the `code` and the `data` used for the numerical experiments in:

[1] Srećko Ðurašinović, Perla Azzi, Jean-Bernard Lasserre, Victor Magron, Olga Mula, and Jun Zhao. "*Leveraging Christoffel-Darboux Kernels to Strengthen Moment-SOS Relaxations*" arXiv, https://arxiv.org/abs/2501.14281, 2025. 

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

## Reproducing our experimental results

For example, in order solve the POP instances corresponding to the 3rd row of `Table 1` (where $n=20, s= 0\\% $), execute the following lines:

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
  
  ```julia
  
  3-element Vector{Float64}:
   -51.413604
   -50.994815
   -50.570687
  
 - while `res_20_0pt1[7][9]` will output the combined SDP solving time (in seconds) of all 3 iterations (may vary depending on your hardware):
    ```julia
      0.06378790000000001
    ```

    
Read the description of `Iterate_Ly_random_Multiple_Instances()` to fully understand the structure of the output.

## General cases

We provide the code designed to construct (marginal) Christoffel polynomials of a specified order for any given POP instance. These Christoffel polynomials are built using moment matrices derived from either the dense or sparse moment-SOS hierarchy. Moreover, in `CDK-TSSOS.jl`, we provide the code capable of applying **H1 and H2** (or their correlatively-sparse versions **H1CS and H2CS**) to any given POP.

- We emphasize that all functionalities have also been integrated into the [TSSOS](https://github.com/wangjie212/TSSOS/) library.

### Illustration: 
Let us consider the following POP (Example 3.2 from [[2]](https://arxiv.org/abs/2208.11158)):
```julia
n = 6
@polyvar x[1:n]
f = x[2]*x[5] + x[3]*x[6] - x[2]*x[3] - x[5]*x[6] + x[1]*(-x[1] + x[2] + x[3] - x[4] + x[5] + x[6])
g = [(6.36 - x[i]) * (x[i] - 4) for i in 1:6]
pop = [f, g...]
d = 1
```
We can solve the problem using the dense moment-SOS hierarchy:
```julia
opt, sol, data = cs_tssos_first(pop, x, d, TS=false, CS=false, solution=true)
```
Afterwards, one can try strenghtening the bound via **H1**:
```julia
N = 5
eps = 0.05
dc = 1
gap_tol = 0.1
resultH1 = run_H1(pop,x,d,dc,N,eps,gap_tol)
```
or via **H2**
```julia
dc = 1
local_sol = sol
tau = 1.1
resultH2 = run_H2(pop,x,d,dc,local_sol,tau)
```
Alternatively, the problem can be solved using the sparse moment-SOS hierarchy:
```julia
opt, sol, data = cs_tssos_first(pop, x, d, TS=false, CS=false, solution=true)
```
Afterwards, the bound can be strengthened either via **H1CS**:
```julia
N = 5
eps = 0.05
dc = 1
gap_tol = 0.1
resultH1CS = run_H1CS(pop,x,d,dc,N,eps,gap_tol)
```
or via **H2CS**
```julia
dc = 1
local_sol = sol
tau = 1.1
resultH2CS = run_H2CS(pop,x,d,dc,local_sol,tau)
```

For the sake of completeness, let us demonstrate how different Christoffel polynomials can be constructed from the output of the: 
- dense Moment-SOS hierarchy of order 2
```julia
d = 2
opt, sol, data = cs_tssos_first(pop, x, d, TS=false, CS=false, solution=true, Mommat=true);

k = 4
dc = 1
CDK_order1 = construct_CDK(x, dc, data.moment[1])  # Constructs multivariate Christoffel polynomial of order dc=1 (quadratic CDK)
CDK_4_order1 = construct_marginal_CDK(x, k, dc, data.moment[1])  # Constructs marginal Christoffel polynomial, associated to x_4, of order dc=1 

dc = 2
CDK_order2 = construct_CDK(x, dc, data.moment[1])  # Construct multivariate Christoffel polynomial of order dc=2 (quartic CDK)
CDK_4_order2 = construct_marginal_CDK(x, k, dc, data.moment[1])  # Constructs marginal Christoffel polynomial, associated to x_4, of order dc=2 

```
- sparse Moment-SOS hierarchy of order 2
```julia
d = 2
opt, sol, data = cs_tssos_first(pop, x, d, TS=false, solution=true, Mommat=true);

k = 4
dc = 1
CDK_sparse_order1 = construct_CDK_cs(x, dc, data.moment, data.cliques)  # Constructs multivariate Christoffel polynomial of order dc=1 for each identified clique.
CDK_sparse_4_order1 = construct_marginal_CDK_cs(x, k, dc, data.moment, data.cliques)  # Constructs marginal Christoffel polynomial, associated to x_4, of order dc=1 

dc = 2
CDK_sparse_order2 = construct_CDK_cs(x, dc, data.moment, data.cliques)  # Constructs multivariate Christoffel polynomial of order dc=2 for each identified clique.
CDK_sparse_4_order2 = construct_marginal_CDK_cs(x, k, dc, data.moment, data.cliques)  # Constructs marginal Christoffel polynomial, associated to x_4, of order dc=2

```



## Main references:
- [1] [TSSOS: a Julia library to exploit sparsity for large-scale polynomial optimization](https://arxiv.org/abs/2103.00915)
- [2] [Sparse Polynomial Optimization: Theory and Practice](https://arxiv.org/abs/2208.11158)
- [3] [The Christoffel–Darboux Kernel for Data Analysis](https://www.cambridge.org/core/books/christoffeldarboux-kernel-for-data-analysis/CFA5119ADEA8671297D89F08C21ACF98)


## Contact 
[Srećko Ðurašinović](https://www.linkedin.com/in/srecko-durasinovic-29b5921ba?lipi=urn%3Ali%3Apage%3Ad_flagship3_profile_view_base_contact_details%3BdEqNOBumRMmZlqEysNiMdg%3D%3D): srecko001@e.ntu.edu.sg
