function generate_Q_and_q(n,seed)

    """
    Generates randomly a symmetric matrix `Q` and a vector `q` for an optimization problem. 
    
    # Arguments
    - `n::Int`: The dimension of the matrix `Q` and the vector `q`.
    - `seed::Int`: Seed value for randomness, ensuring reproducibility.
    
    # Returns
    - `Q::Matrix{Float64}`: A symmetric `n × n` matrix generated from a random normal distribution.
    - `q::Vector{Float64}`: A vector of length `n` with entries sampled from a standard normal distribution.
    
    """
    Random.seed!(seed)
    
    A = randn(n, n)
    
    Q = (A + A') / 2
    
    q = randn(n)
    
    return Q, q
    
end



function generate_Q_and_q_rsparse(n,seed,percentage_dense=0.8)
    """
    Generates a sparse symmetric matrix `Q` and a sparse vector `q` for an optimization problem. 
    The sparsity level of the matrix and vector is controlled by `percentage_dense`.
    
    # Arguments
    - `n::Int`: The dimension of the matrix `Q` and the vector `q`.
    - `seed::Int`: Seed value for randomness, ensuring reproducibility.
    - `percentage_dense::Float64` (optional): The proportion of non-zero entries in the matrix and vector. Default is `0.8`.
    
    # Returns
    - `Q::SparseMatrixCSC{Float64}`: A symmetric sparse `n × n` matrix with `percentage_dense` proportion of non-zero entries.
    - `q::SparseVector{Float64}`: A sparse vector of length `n` with `percentage_dense` proportion of non-zero entries.
    
    """
    Random.seed!(seed)
    Q1 = sprandn(n, n, percentage_dense)  # percentage_dense% non-zero entries, generated from N(0,1)
    Q = (Q1 + Q1') / 2
    
    q = sprandn(n, percentage_dense)
    
    return Q, q
    
end




function construct_CDKmarg_SVD(Mm, k, threshold=0.001)

    """
    Constructs marginal Christoffel polynomials (CDK) using Singular Value Decomposition (SVD) for the input matrix `Mm`. 
    
    # Arguments
    - `Mm::Matrix`: The input moment matrix from which which the marginal CDK is constructed.
    - `k::Int`: Specifies the index of the decision variable for which CDK should be constructed .
    - `threshold::Float64` (optional): A threshold value for filtering eigenvalue (i.e., deciding the numerical rank of `Mm`) . Default is `0.001`.
    
    # Returns
    - `cdk`: The constructed marginal CDK.
    - `p_alpha_squared[1:negativeEV]`: Polynomials in the kernel of the marginal moment matrix.
    - `positiveEV`: The set of positive eigenvalues of the marginal moment matrix.
    - `negativeEV`: The set of negative eigenvalues of the marginal moment matrix.
    - `minimum(Qval)`: The minimum eigenvalue of the marginal moment matrix.
    
    """
    eigen_result = eigen(Mm[[1,k+1],[1,k+1]])

    Qvec = eigen_result.vectors
    Qval = eigen_result.values

    p_alpha_squared=[]   #Vector of orhonormal polynomials 
    
    for j=1:length(Qval)
        push!(p_alpha_squared, (vcat(1,x[k])'*Qvec[:,j])^2)
    end

    positiveEV = count(x -> abs(x) > threshold, Qval)
    negativeEV=count(x -> abs(x) <= threshold, Qval)

    if positiveEV==length(Qval)
        cdk=sum(p_alpha_squared[i]/(Qval[i]) for i=1:length(Qval))
    else
        cdk=sum(p_alpha_squared[i]/(Qval[i]) for i=(length(Qval)-positiveEV+1):length(Qval))
    end 

    return cdk, p_alpha_squared[1:negativeEV], positiveEV, negativeEV, minimum(Qval)
        
end


function Run_H2(x,n,seed,solIpopt,tau,running_time=false,kernel_cutoff=0.001)            #regular random cases

    """`
    Applies H2 algorithm to a particular QCQP instance generated using `seed` (instance is generated randomly, having 100% dense entries)
    
    # Arguments
    - `x::Vector`: Initial vector of decision variables 
    - `n::Int`: Dimension of the problem.
    - `seed::Int`: Seed value for randomness, ensuring reproducibility.
    - `solIpopt`: Available solution (here obtained from an external local optimizer IPOPT).
    - `tau::Float64`: Filtering parameter 
    - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of the H2. Default is `false`.
    - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controling the strength of the Tikhonov regularization of marginal Christoffel polynomials. Default is `0.001`.
    
    # Returns
    A tuple of the following:
    1. `[opt, optcdk]`: A pair containing the optimal solutions of moment relaxation before and after CDK strengthening via H2.
    2. `[gap, gapcdk]`: Relative optimality gaps before and after CDK strengthening via H2.
    3. `Gammas`: Thresholds used to construct sublevel sets of marginal Christoffel polynomials (CDK).
    4. `[ubRef, f(solcdk)]`: The available upper bound before and after CDK strengthening via H2.
    5. `data.moment[1]`: The moment matrix from which marginal CDK polynomials were constructed
    6. `total_time`: The total runtime of H2 `running_time` is `true`.
    
    """

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS_____________________ ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q(n,seed)

    println("Eigvals of Q = ", eigvals(Q)')
    println()
    
    f=x'*Q*x+x'q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    println()
    println()
    println("_______________ SOLVING THE INITIAL RELAXATION _____________________ ")
    println()
    println()

    Random.seed!(seed)
    log_file1 = "solver_log1.txt"

    if running_time

        open(log_file1, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
    else
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
    end

    println()
    println("\e[1mINITIAL BOUND\e[0m = ", opt)
    println()
        


    println()
    
    ubRef=f(solIpopt)
    gap=abs((ubRef-opt)/ubRef)*100
    
    println()
    println(" _________________******_________________******_________________********_________________ ")
    println(" _________________******_________________******_________________********_________________ ")
    println()


    CDK_marg_SVD=[]
    for i=1:n
        push!(CDK_marg_SVD,construct_CDKmarg_SVD(data.moment[1],i))
    end
    
    popcdk = Vector{Polynomial{true, Float64}}(pop)
    Gammas=[]

    for i=1:n
        temp=CDK_marg_SVD[i]
        gamma_i=temp[1](solIpopt[i])+1e-6 # to avoid numerical issues
        push!(Gammas,gamma_i)   

        if gamma_i < tau
            if temp[4] > 0
                push!(popcdk,kernel_cutoff-sum(temp[2]), gamma_i-temp[1])
            else
                push!(popcdk, gamma_i-temp[1])
            end
        end
        
                            
    end

    for i=1:length(popcdk)
        popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))
    end

    
            
        
    println()
    println("__________________ APPLYING THE HEURISTIC H2________________________________ ")
    println()
    println("___________ New total number of constraints: ", length(popcdk)-1)
    println()

    Random.seed!(seed)

    log_file2 = "solver_log2.txt"

    
    if running_time

        open(log_file2, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
                end
            end
        end
        
    # Read the log file to extract the required information
        log_content2 = read(log_file2, String)
        time_match2 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content2)
        sdp_solving_time2 = parse(Float64, time_match2.captures[1])
    else
        optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
    end

        
    println()
    println("Summary: ____________________________________________________________________ ")
    
    optcdk=optcdk*maximum(abs.(coefficients(pop[1])))
    gapcdk= abs((ubRef-optcdk)/ubRef)*100
    println("Gammas= ",Gammas)
    println()
    println("\e[1mNEW BOUND\e[0m = ", optcdk)
    println()

    if running_time
        total_time=sdp_solving_time1+sdp_solving_time2
    else
        total_time=""
    end
    return [opt,optcdk],[gap,gapcdk],Gammas,[ubRef,f(solcdk)],data.moment[1],total_time       
            
end 




function Run_H2_rsparse(x,n,seed,percentage_dense,solIpopt,tau,running_time=false,kernel_cutoff=0.001)    #random cases with some degree of unstructured sparsity

    
    """`
    Applies H2 algorithm to a particular QCQP instance generated using `seed` (instance is generated randomly, having percentage_dense% dense entries)
    
    # Arguments
    - `x::Vector`: Initial vector of decision variables 
    - `n::Int`: Dimension of the problem.
    - `seed::Int`: Seed value for randomness, ensuring reproducibility.
    - `percentage_dense::Float64`: Determines the degree of sparsity of the matrix Q used to construct the QCQP instance. 
    - `solIpopt::Vector`: Available solution (here obtained from an external local optimizer IPOPT).
    - `tau::Float64`: Filtering parameter 
    - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of the H2. Default is `false`.
    - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controling the strength of the Tikhonov regularization of marginal Christoffel polynomials. Default is `0.001`.
    
    # Returns
    A tuple of the following:
    1. `[opt, optcdk]`: A pair containing the optimal solutions of moment relaxation before and after CDK strengthening via H2.
    2. `[gap, gapcdk]`: Relative optimality gaps before and after CDK strengthening via H2.
    3. `Gammas`: Thresholds used to construct sublevel sets of marginal Christoffel polynomials (CDK).
    4. `[ubRef, f(solcdk)]`: The available upper bound before and after CDK strengthening via H2.
    5. `data.moment[1]`: The moment matrix from which marginal CDK polynomials were constructed
    6. `total_time`: The total runtime of H2 `running_time` is `true`.
    
    """

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS_____________________ ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q_rsparse(n,seed,percentage_dense)

    println("Eigvals of Q = ", eigvals(Matrix(Q))')
    println()
    
    f=x'*Q*x+x'q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    println()
    println()
    println("_______________ SOLVING THE INITIAL RELAXATION _____________________ ")
    println()
    println()

    Random.seed!(seed)
    log_file1 = "solver_log1.txt"

    if running_time

        open(log_file1, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
    else
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
    end

    println()
    println("\e[1mINITIAL BOUND\e[0m = ", opt)
    println()
        


    println()
    
    ubRef=f(solIpopt)
    gap=abs((ubRef-opt)/ubRef)*100
    
    println()
    println(" _________________******_________________******_________________********_________________ ")
    println(" _________________******_________________******_________________********_________________ ")
    println()


    CDK_marg_SVD=[]
    for i=1:n
        push!(CDK_marg_SVD,construct_CDKmarg_SVD(data.moment[1],i))
    end
    
    popcdk = Vector{Polynomial{true, Float64}}(pop)
    Gammas=[]

    for i=1:n
        temp=CDK_marg_SVD[i]
        gamma_i=temp[1](solIpopt[i])+1e-6 # to avoid numerical issues
        push!(Gammas,gamma_i)   

        if gamma_i < tau
            if temp[4] > 0
                push!(popcdk,kernel_cutoff-sum(temp[2]), gamma_i-temp[1])
            else
                push!(popcdk, gamma_i-temp[1])
            end
        end
        
                            
    end

    for i=1:length(popcdk)
        popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))
    end

    
            
        
    println()
    println("__________________ APPLYING THE HEURISTIC H2________________________________ ")
    println()
    println("___________ New total number of constraints: ", length(popcdk)-1)
    println()

    Random.seed!(seed)

    log_file2 = "solver_log2.txt"

    
    if running_time

        open(log_file2, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
                end
            end
        end
        
    # Read the log file to extract the required information
        log_content2 = read(log_file2, String)
        time_match2 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content2)
        sdp_solving_time2 = parse(Float64, time_match2.captures[1])
    else
        optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
    end

        
    println()
    println("Summary: ____________________________________________________________________ ")
    
    optcdk=optcdk*maximum(abs.(coefficients(pop[1])))
    gapcdk= abs((ubRef-optcdk)/ubRef)*100
    println("Gammas= ",Gammas)
    println()
    println("\e[1mNEW BOUND\e[0m = ", optcdk)
    println()

    if running_time
        total_time=sdp_solving_time1+sdp_solving_time2
    else
        total_time=""
    end
    return [opt,optcdk],[gap,gapcdk],Gammas,[ubRef,f(solcdk)],data.moment[1],total_time       
            
end 


function Run_H2_Multiple_Instances(x, n, seed, solIpopt, tau, filename, running_time=false,kernel_cutoff=0.001)  #for regular random cases

     """`
        Applies H2 algorithm to a multiples QCQP instance generated using different `seed` (instances are generated randomly, having 100% dense entries)
        
        # Arguments
        - `x::Vector`: Initial vector of decision variables 
        - `n::Int`: Dimension of the problem.
        - `seed::Vector`: Vector of seeds allowing to solve different instances corresponding to different random seeds.
        - `solIpopt`: Vector of available local solution for each seed (here obtained from an external local optimizer IPOPT).
        - `tau::Float64`: Filtering parameter.
        - `filename::String`: Name of the .txt file that will contain solving logs for each instance.
        - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of the H2. Default is `false`.
        - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controling the strength of the Tikhonov regularization of marginal Christoffel polynomials. Default is `0.001`.
        
        # Returns
        Following lists:
        1. `All_opts`: A list of all strengthened bounds for each QCQP instance
        2. `ALL_gaps`: A list of all post-H2 gaps for each QCQP instance
        3. `ALL_Gammas`: A list of all marginal CDK thresholds for each QCQP instance
        4. `ALL_UBS`: A list of upper bounds for each instance identified during the optimization process.
        5. `All_MM`: A list of moment matrices for each instance from which marginal CDKs were constructed
        6. `All_time`: A collection of runtimes for each instance, if `running_time` is `true`.

    
    """
    
        All_opts = Any[]
        ALL_gaps = Any[]
        ALL_Gammas = Any[]
        ALL_UBS = Any[]
        All_MM = Any[]
        All_time = Any[]
    open(filename, "w") do io
        
        for i = 1:length(seed)
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()______________ \e[1mSOLVING THE INSTANCE $i\e[0m _______________()()()()()")
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io, "()()()()()______________ SOLVING THE INSTANCE $i _______________()()()()()")
            println(io)

            opts, gaps, gammas, UBS, mm, r_time = Run_H2(x, n, seed[i], solIpopt[i], tau,running_time,kernel_cutoff)
            
            push!(All_opts, opts)
            push!(ALL_Gammas, gammas)
            push!(ALL_UBS, UBS)
            push!(ALL_gaps, gaps)
            push!(All_MM, mm)
            if running_time
                push!(All_time,r_time)
            end
                
            
            println(io, "Instance $i Results:")
            println(io, "Computed bounds: ", opts)
            println(io, "Gaps: ", gaps)
            println(io, "Gammas: ", gammas)
            println(io, "Known upper bounds: ", UBS)
            println(io, "Moment Matrix: ", mm)
            if running_time
                println(io, "Total Running Time: ", r_time)
            end
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, ALL_Gammas, ALL_UBS, All_MM, All_time
end



function Run_H2_Multiple_Instances_rsparse(x, n, seed, percentage_dense, solIpopt, tau, filename, running_time=false,kernel_cutoff=0.001)

    """`
        Applies H2 algorithm to a multiples QCQP instance generated using different `seed` (instances are generated randomly, having percentage_dense% dense entries)
        
        # Arguments
        - `x::Vector`: Initial vector of decision variables 
        - `n::Int`: Dimension of the problem.
        - `seed::Vector`: Vector of seeds allowing to solve different instances corresponding to different random seeds.
        - `percentage_dense::Float64`: Determines the degree of sparsity of the matrix Q used to construct the QCQP instance. 
        - `solIpopt`: Vector of available local solution for each seed (here obtained from an external local optimizer IPOPT).
        - `tau::Float64`: Filtering parameter 
        - `filename::String`: Name of the .txt file that will contain solving logs for each instance 
        - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of the H2. Default is `false`.
        - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controling the strength of the Tikhonov regularization of marginal Christoffel polynomials. Default is `0.001`.
        
        # Returns
        Following lists:
        1. `All_opts`: A list of all strengthened bounds for each QCQP instance
        2. `ALL_gaps`: A list of all post-H2 gaps for each QCQP instance
        3. `ALL_Gammas`: A list of all marginal CDK thresholds for each QCQP instance
        4. `ALL_UBS`: A list of upper bounds for each instance identified during the optimization process.
        5. `All_MM`: A list of moment matrices for each instance from which marginal CDKs were constructed
        6. `All_time`: A collection of runtimes for each instance, if `running_time` is `true`.

    
    """
    
    
        All_opts = Any[]
        ALL_gaps = Any[]
        ALL_Gammas = Any[]
        ALL_UBS = Any[]
        All_MM = Any[]
        All_time = Any[]
    open(filename, "w") do io
        
        for i = 1:length(seed)
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()______________ \e[1mSOLVING THE INSTANCE $i\e[0m _______________()()()()()")
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io, "()()()()()______________ SOLVING THE INSTANCE $i _______________()()()()()")
            println(io)
            opts, gaps, gammas, UBS, mm, r_time = Run_H2_rsparse(x, n, seed[i], percentage_dense, solIpopt[i], tau,running_time,kernel_cutoff)
            
            push!(All_opts, opts)
            push!(ALL_Gammas, gammas)
            push!(ALL_UBS, UBS)
            push!(ALL_gaps, gaps)
            push!(All_MM, mm)
            if running_time
                push!(All_time,r_time)
            end
                
            
            println(io, "Instance $i Results:")
            println(io, "Computed bounds: ", opts)
            println(io, "Gaps: ", gaps)
            println(io, "Gammas: ", gammas)
            println(io, "Known upper bounds: ", UBS)
            println(io, "Moment Matrix: ", mm)
            if running_time
                println(io, "Total Running Time: ", r_time)
            end
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, ALL_Gammas, ALL_UBS, All_MM, All_time
end





function AnalyseResults_H2(results,data,delta=0.5)              #Analysing the performace of H2
   
    """`

    Analyzes the performance of H2

    # Arguments
    - `results`: A data structure containing the output of the H2 optimization process. It is expected to include bounds, gaps, and timing information for each test case.
    - `data`: A data structure containing TRUE reference values and problem-specific details (in form of a list), such as:
        - `data[3]`: Upper bounds for each test case.
        - `data[4]`: True optimal values.
        - `data[6]`: Timing information for the second-order moment relaxation.
        - `data[7]`: Seeds or identifiers from which POP/QCQP instances were generated.
    - `delta::Float64`: Desired gap tolerance (in %)

    
    # Returns
    A tuple with the following elements:
    1. **Average gaps**: 
        - `[PREavg, POSTavg]`: Mean gaps (in percentage) before and after applying H2, for non-failed cases.
    2. **Maximum gaps**: 
        - `[PREmax, POSTmax]`: Maximum gaps (in percentage) before and after applying H2, for non-failed  cases.
    3. **Timing information**:
        - `[H2_time, order2_time]`: Mean runtime for H2 and second-order relaxation, for non-failed  cases.
    4. **Failure analysis**:
        - `[length(failed_seeds), count_gaps_below_threshold]`: Number of failed cases (failed = H2 produced an upper bound) and count of Solved cases (Solved = post-H2 gaps are 
          below a delta% threshold).
    5. **Details of failed cases**:
        - `[failed, failed_seeds]`: Indices of failed cases and their corresponding seeds.
    6. **Accuracy of IPOPT results**:
        - `[Ipopt_not_exact, Ipopt_not_exact_seeds, length(Ipopt_not_exact)]`: Indices of cases where IPOPT bounds are not exact, corresponding seeds, and the total count of such 
        cases.
    
    # Notes
    - A case is considered "failed" if the lower bound exceeds the upper bound by more than `1e-6`.
    - The function computes gaps as percentages of the true values.
    - Seeds provide a means of identifying and analyzing specific cases.

    """

    tot_number_cases=length(data[7])

    failed=[i for i=1:tot_number_cases if results[1][i][2]>data[4][i]+1e-6]  # failed cases - lb > ub

    POSTvsTRUE=[[results[1][i][2],data[4][i]] for i=1:tot_number_cases] #bounds
    PREvsTRUE=[[results[1][i][1],data[4][i]] for i=1:tot_number_cases]  #bounds

    POSTGaps =[abs(POSTvsTRUE[i][2] - POSTvsTRUE[i][1]) / abs(POSTvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100
    PREGaps = [abs(PREvsTRUE[i][2] - PREvsTRUE[i][1]) / abs(PREvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100


    POSTavg = mean(POSTGaps)  # post H2 mean gap
    PREavg = mean(PREGaps)  # pre H2 mean gap

    POSTmax = maximum(POSTGaps)   # post H2 mmax  gap
    PREmax = maximum(PREGaps)   # pre H2 max gap
    
    count_gaps_below_threshold = count(x -> x < delta, POSTGaps)  #solved cases

    order2_time=mean([data[6][i] for i in 1:tot_number_cases if !(i in failed)])
    H2_time=mean([results[end][i] for i in 1:tot_number_cases if !(i in failed)])

 
    Ipopt_not_exact = findall(i -> abs(data[3][i] - data[4][i]) > 1e-5, 1:tot_number_cases)

    failed_seeds = data[7][failed]
    Ipopt_not_exact_seeds=data[7][Ipopt_not_exact]

    return [[PREavg,POSTavg],[PREmax,POSTmax], [H2_time,order2_time],[length(failed_seeds),count_gaps_below_threshold],[failed,failed_seeds],[Ipopt_not_exact,Ipopt_not_exact_seeds,length(Ipopt_not_exact)]]

end


#______________________________________________________________________________________________________________________

########################################## Some other helper functions ################################################
#______________________________________________________________________________________________________________________


function get_basis(n::Int64,d::Int64) # Monomial basis computation
lb=binomial(n+d,d)
basis=zeros(n,lb)
i=UInt64(0)
t=UInt64(1)
while i<d+1
if basis[n,t]==i
if i<d
@inbounds t+=1
@inbounds basis[1,t]=i+1
@inbounds i+=1
else
@inbounds i+=1
end
else
j=UInt64(1)
while basis[j,t]==0
@inbounds j+=1
end
if j==1
@inbounds t+=1
@inbounds basis[:,t]=basis[:,t-1]
@inbounds basis[1,t]=basis[1,t]-1
@inbounds basis[2,t]=basis[2,t]+1
else t+=1
@inbounds basis[:,t]=basis[:,t-1]
@inbounds basis[1,t]=basis[j,t]-1
@inbounds basis[j,t]=0
@inbounds basis[j+1,t]=basis[j+1,t]+1
end
end
end
return basis
end


function monomialvector(vector,d)
n=length(vector)
sd=binomial(n+d,d)
monobasis=get_basis(n,d)
monom=[]
monomial_vector=Vector{Any}(undef,sd)
for i in 1:sd
    monomial_vector[i]=prod(vector[j]^Int(monobasis[:,i][j]) for j=1:n)      
end
push!(monom,monomial_vector) 
end

    
function indices(v,B)
    list_indices=[]
    for i=1:length(v)
    for j=1:length(B)
            if v[i]==B[j] push!(list_indices,j)
                break
            end
       end
    end
        return list_indices
 end
    






