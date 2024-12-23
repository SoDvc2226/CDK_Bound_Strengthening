

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





function construct_CDK(Mm, threshold=0.001)

    """
    Constructs Christoffel polynomial of order d (CDK) using Singular Value Decomposition (SVD) for the input moment matrix `Mm` of order d. 
    
    # Arguments
    - `Mm::Matrix`: The input moment matrix from which which the CDK is constructed.
    - `threshold::Float64` (optional): A threshold value for filtering eigenvalues (i.e., deciding the numerical rank of `Mm`) . Default is `0.001`.
    
    # Returns
    - `p_alpha_squared[1:negativeEV]`: Polynomials in the kernel of the moment matrix.
    - `cdk`: The constructed CDK.
    - `positiveEV`: The set of positive eigenvalues of the moment matrix.
    - `minimum(Qval)`: The minimum eigenvalue of the moment matrix.
    - `negativeEV`: The set of negative eigenvalues of the moment matrix.
    
    """

    eigen_result = eigen(Mm)

    Qvec = eigen_result.vectors
    Qval = eigen_result.values

    p_alpha_squared=[]   #Vector of orhonormal polynomials 
    
    for i=1:length(Qval)
        push!(p_alpha_squared, (vcat(1,x)'*Qvec[:,i])^2)
    end

    threshold=0.001

    positiveEV = count(x -> abs(x) > threshold, Qval)
    negativeEV=count(x -> abs(x) <= threshold, Qval)

    if positiveEV==length(Qval)
        cdk=sum(p_alpha_squared[i]/(Qval[i]) for i=1:length(Qval))
    else
        cdk=sum(p_alpha_squared[i]/(Qval[i]) for i=(length(Qval)-positiveEV+1):length(Qval))
    end 

    return p_alpha_squared[1:negativeEV], cdk, positiveEV,minimum(Qval),negativeEV
        
end


function iterate_Ly_random(x,n,N,eps,seed,running_time=false,gap_tol=0.5,kernel_cutoff=0.001)

    
    """

    Performs iterative bound strengthening (i.e, implements H1) for  a particular randomly generated instance of QCQP. 
    
    # Arguments
    - `x::Vector`: Initial vector of decision variables.
    - `n::Int`: Dimension or size of the problem.
    - `N::Int`: Maximum number of iterations to perform.
    - `eps::Float64`: Perturbation factor deciding the volume of CDK sublevel sets.
    - `seed::Int`: Seed value that generates a particular POP instance
    - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of H1. Default is `false`.
    - `gap_tol::Float64` (optional): Threshold for optimality gap tolerance. Default is `0.5`.
    - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controlling the strength of the Tikhonov regularization of CDK. 
       Default is `0.001`.
    
    # Returns
    A list with the following elements:
    1. `OPT`: Relaxation bound from each iteration.
    2. `gap_history`: A history of the optimality gaps at each iteration.
    3. `Moment_matrices`: Moment matrices computed during each iterations used to construct CDK sublevel constraints.
    4. `K`: Parameter to keep track of the rank of the moment matrices for each iteration.
    5. `Certain_overrestriction`: A measure or flag for over-restriction (when H1 is sure that further feasible set reductions would yield an invalid bound).
    6. `ubRef`: Upper bounds with respect to which the optimality gap is measured
    7. `Times`: SDP solving time of each iteration, if `running_time` is `true`.

    """

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS: ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q(n,seed)

    println("Dimension n = ",n)
    println("Threshold reduction coefficient  = ", 1-eps)
    println("Eigvals of Q = ", eigvals(Q)')
    println()
    
    f=x'*Q*x+x'q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    Times = []
    Moment_matrices=[]
    Certain_overrestriction=false
    normalization_factor=maximum(abs.(coefficients(f)))

    println()
    println()
    println("_______________ SOLVING THE INITIAL RELAXATION: ")
    println()

    log_file1 = "solver_log1.txt"

    if running_time
        open(log_file1, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    Random.seed!(seed)
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
        push!(Times, sdp_solving_time1)
    else
        Random.seed!(seed)
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
    end
    println()
    println(" ________________________________________________________________________________________________ ")

    push!(Moment_matrices,data.moment[1])

    ubRef=f(sol)

    initial_gap=(abs((ubRef-opt))/abs(ubRef))*100

    current_gap=initial_gap
    current_opt=opt

    println()
    println("Value of the initial relaxation = ", opt)
    println("Global upper bound = ", f(sol))
    println("Initial optimality gap = ", round(initial_gap,digits=5), "%")
    println()

    
    OPT=[round(opt,digits=6)]
    gap_history=[round(current_gap,digits=4)]

    i=0

    k=1                                         #To keep track of the rank reduction of the moment matrix
    K=Int.(ones(size(data.moment[1], 1)))



    CDK_positive=Any[]
    CDK_zero=Any[]

    while i < N 
        
        if current_gap < gap_tol
            println("RELAXATION IS GOOD ENOUGH.")
            println()
            break
        end
    
        println()
        println(" >>>>>> ________________Iteration ", i+1, " ___________________ <<<<<<")
        println()

        temp = construct_CDK(data.moment[1])
        println()
        println("Number of positive eigenvalues of the moment matrix = ", temp[3], " and the smallest one is = ", temp[4])
        println("Kernel dimension = ", length(temp[1]))
        println()


        if temp[3] == size(data.moment[1], 1)       #For full-rank matrices, there are no kernel constraints

            K[temp[3]]=K[temp[3]]+1

            cdk = temp[2]
            println()
            coeffs = coefficients(cdk)
            moment_vector = compute_mom_vector(monomials(cdk),data.moment[1])
            println("y_star_previous = ", Vector{Float64}(moment_vector))
            println()
            L_ystar = coeffs'*moment_vector
            println()
            
            println("L_ystar = ",  L_ystar, " and k = ", k)

            popcdk = Vector{Polynomial{true, Float64}}(pop)

            push!(CDK_positive, (1-eps)*L_ystar - cdk)

            for i=1:length(CDK_positive)
                push!(popcdk, CDK_positive[i])    
            end
            
            println()
            println("Length of popcdk at iteration ",i+1, " is ",  length(popcdk))
            println()
            for i=1:length(popcdk)
                popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))    #Normalization
            end
            rng = Random.default_rng() 
            Random.seed!(rng, nothing)
            optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=false)

            if f(solcdk)< ubRef
                ubRef=f(solcdk)
                println("Global upper bound updated to ", ubRef)
            end

            current_opt = optcdk*normalization_factor

            if current_opt > ubRef+1e-6                    #to avoid some innacuracies
                println()
                println(" Overrestriction !!! ")
                Certain_overrestriction=true
                break
            end

            data=datacdk
            push!(Moment_matrices,data.moment[1])


            push!(OPT,round(current_opt, digits=6))
            println()
            
            current_gap= (abs((ubRef-current_opt))/abs(ubRef))*100
            println("Current gap = ", round(current_gap,digits=4),"%")
            
            push!(gap_history,round(current_gap,digits=4))
            println("Gap history = ", gap_history ,"%")
            println("History of lower bounds = ", OPT)
            




        else

            K[temp[3]]=K[temp[3]]+1
                                    
            cdk = temp[2]            
            coeffs = coefficients(cdk)
            
            moment_vector = compute_mom_vector(monomials(cdk),data.moment[1])
            println("y_star_previous: ", Vector{Float64}(moment_vector[1:n+1]))
            println()
            L_ystar = coeffs'*moment_vector
            
            println("L_ystar = ", L_ystar)
            println("Chosen level set, namely (1-eps)*L_ystar = ", (1-eps)*L_ystar)

            Gammas=Float64[]

            for j=1:temp[5]
                push!(Gammas, coefficients(temp[1][j])'* compute_mom_vector(temp[1][j],data.moment[1]))
            end
            println()
            println("Values of L_{y_star}(p_alpha^2) for polynomials p in the kernel: ", Gammas')
            println()
            
            popcdk = Vector{Polynomial{true, Float64}}(pop)
                    
            for i=1:length(popcdk)
                popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))    #Normalization
            end
            
            push!(CDK_positive, (1-eps)*L_ystar - cdk)

            for i=1:length(CDK_positive)
                push!(popcdk, CDK_positive[i]/maximum(abs.(coefficients(CDK_positive[i]))))    #Normalization
            end
                
            thresholds_kernel=Float64[]

            sum_kernel=sum([kernel_cutoff-temp[1][j] for j=1:temp[5]])

            push!(CDK_zero, sum_kernel)

            for i=1:length(CDK_zero)
                 push!(popcdk, CDK_zero[i]/maximum(abs.(coefficients(CDK_zero[i]))))         #Normalization
            end
            

            println()
            println("_______________ Solving the MODIFIED relaxation: ")
            println()
            println("Total number of constraints at iteration $(i+1) = ", length(popcdk)-1)
            println()
            Random.seed!(seed)

            log_file2 = "solver_log_$(i+1).txt"

    
            if running_time
        
                open(log_file2, "w") do f
                    redirect_stdout(f) do
                        redirect_stderr(f) do
                            rng = Random.default_rng() 
                            Random.seed!(rng, nothing)
                            optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
                        end
                    end
                end
                
                log_content2 = read(log_file2, String)
                time_match2 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content2)
                sdp_solving_time2 = parse(Float64, time_match2.captures[1])
                push!(Times, sdp_solving_time2)

            else
                rng = Random.default_rng() 
                Random.seed!(rng, nothing)
                optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
            end

            if f(solcdk) < ubRef
                ubRef=f(solcdk)
                println("Global upper bound updated to ", ubRef)
            end

            current_opt = optcdk*normalization_factor
            
            println()
            println("Current opt is ", current_opt)
            println("Current ub is ", ubRef)


            if current_opt > ubRef+1e-6                    #to avoid some innacuracies
                println()
                println(" Overrestriction !!! ")
                Certain_overrestriction=true
                break
            end


            data=datacdk
            push!(Moment_matrices,data.moment[1])
  
            push!(OPT, round(current_opt, digits=6))
            println()
            
            current_gap= (abs((ubRef-current_opt))/abs(ubRef))*100
            println("Current gap = ", round(current_gap,digits=4),"%")
            
            push!(gap_history,round(current_gap,digits=4))
            println("Gap history = ", gap_history ,"%")
            println("History of lower bounds = ", OPT)
            
            println(" ________________________________________________________________________________________________ ")

            


            
                
        end
        i=i+1
        
    end
                

    return OPT,gap_history, Moment_matrices, K, Certain_overrestriction,ubRef, Times

            
            
end 




function iterate_Ly_rsparse(x,n,percentage_dense,N,eps,seed,running_time=false,gap_tol=0.5,kernel_cutoff=0.001)


    """

    Performs iterative bound strengthening (i.e, implements H1) for  a particular randomly generated instance of QCQP 
    whose matrix Q has percentage_dense% dense entries . 
    
    # Arguments
    - `x::Vector`: Initial vector of decision variables.
    - `n::Int`: Dimension or size of the problem.
    - `percentage_dense::Float64`: percentage of dense entires within the matrix Q that later generates the QCQP
    - `N::Int`: Maximum number of iterations to perform.
    - `eps::Float64`: Perturbation factor deciding the volume of CDK sublevel sets.
    - `seed::Int`: Seed value that generates a particular POP instance
    - `running_time::Bool` (optional): If `true`, measures and returns the total runtime of H1. Default is `false`.
    - `gap_tol::Float64` (optional): Threshold for optimality gap tolerance. Default is `0.5`.
    - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controlling the strength of the Tikhonov regularization of CDK. 
       Default is `0.001`.
    
    # Returns
    A list with the following elements:
    1. `OPT`: Relaxation bound from each iteration.
    2. `gap_history`: A history of the optimality gaps at each iteration.
    3. `Moment_matrices`: Moment matrices computed during each iterations used to construct CDK sublevel constraints.
    4. `K`: Parameter to keep track of the rank of the moment matrices for each iteration.
    5. `Certain_overrestriction`: A measure or flag for over-restriction (when H1 is sure that further feasible set reductions would yield an invalid bound).
    6. `ubRef`: Upper bounds with respect to which the optimality gap is measured
    7. `Times`: SDP solving time of each iteration, if `running_time` is `true`.

    """

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS: ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q_rsparse(n,seed,percentage_dense)

    println("Dimension n = ",n)
    println("Threshold reduction coefficient  = ", 1-eps)
    println("Eigvals of Q = ", eigvals(Matrix(Q))')
    println()
    
    f=x'*Q*x+x'q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    Times = []
    Moment_matrices=[]
    Certain_overrestriction=false
    normalization_factor=maximum(abs.(coefficients(f)))

    println()
    println()
    println("_______________ SOLVING THE INITIAL RELAXATION: ")
    println()

    log_file1 = "solver_log1.txt"

    if running_time
        open(log_file1, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    Random.seed!(seed)
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
        push!(Times, sdp_solving_time1)
    else
        Random.seed!(seed)
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false,CS=false, solution=true,QUIET=!running_time)
    end
    println()
    println(" ________________________________________________________________________________________________ ")

    push!(Moment_matrices,data.moment[1])

    ubRef=f(sol)

    initial_gap=(abs((ubRef-opt))/abs(ubRef))*100

    current_gap=initial_gap
    current_opt=opt

    println()
    println("Value of the initial relaxation = ", opt)
    println("Global upper bound = ", f(sol))
    println("Initial optimality gap = ", round(initial_gap,digits=5), "%")
    println()

    
    OPT=[round(opt,digits=6)]
    gap_history=[round(current_gap,digits=4)]

    i=0

    k=1                                         #To keep track of the rank reduction of the moment matrix
    K=Int.(ones(size(data.moment[1], 1)))



    CDK_positive=Any[]
    CDK_zero=Any[]

    while i < N 
        
        if current_gap < gap_tol
            println("RELAXATION IS GOOD ENOUGH.")
            println()
            break
        end
    
        println()
        println(" >>>>>> ________________Iteration ", i+1, " ___________________ <<<<<<")
        println()

        temp = construct_CDK(data.moment[1])
        println()
        println("Number of positive eigenvalues of the moment matrix = ", temp[3], " and the smallest one is = ", temp[4])
        println("Kernel dimension = ", length(temp[1]))
        println()


        if temp[3] == size(data.moment[1], 1)       #For full-rank matrices, there are no kernel constraints

            K[temp[3]]=K[temp[3]]+1

            cdk = temp[2]
            println()
            coeffs = coefficients(cdk)
            moment_vector = compute_mom_vector(monomials(cdk),data.moment[1])
            println("y_star_previous = ", Vector{Float64}(moment_vector))
            println()
            L_ystar = coeffs'*moment_vector
            println()
            
            println("L_ystar = ",  L_ystar, " and k = ", k)

            popcdk = Vector{Polynomial{true, Float64}}(pop)

            push!(CDK_positive, (1-eps)*L_ystar - cdk)

            for i=1:length(CDK_positive)
                push!(popcdk, CDK_positive[i])    
            end
            
            println()
            println("Length of popcdk at iteration ",i+1, " is ",  length(popcdk))
            println()
            for i=1:length(popcdk)
                popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))    #Normalization
            end
            rng = Random.default_rng() 
            Random.seed!(rng, nothing)
            optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=false)

            if f(solcdk)< ubRef
                ubRef=f(solcdk)
                println("Global upper bound updated to ", ubRef)
            end

            current_opt = optcdk*normalization_factor

            if current_opt > ubRef+1e-6                    #to avoid some innacuracies
                println()
                println(" Overrestriction !!! ")
                Certain_overrestriction=true
                break
            end

            data=datacdk
            push!(Moment_matrices,data.moment[1])


            push!(OPT,round(current_opt, digits=6))
            println()
            
            current_gap= (abs((ubRef-current_opt))/abs(ubRef))*100
            println("Current gap = ", round(current_gap,digits=4),"%")
            
            push!(gap_history,round(current_gap,digits=4))
            println("Gap history = ", gap_history ,"%")
            println("History of lower bounds = ", OPT)
            




        else

            K[temp[3]]=K[temp[3]]+1
                                    
            cdk = temp[2]            
            coeffs = coefficients(cdk)
            
            moment_vector = compute_mom_vector(monomials(cdk),data.moment[1])
            #println("y_star_previous: ", Vector{Float64}(moment_vector[1:n+1]))
            println()
            L_ystar = coeffs'*moment_vector
            
            println("L_ystar = ", L_ystar)
            println("Chosen level set, namely (1-eps)*L_ystar = ", (1-eps)*L_ystar)

            Gammas=Float64[]

            for j=1:temp[5]
                push!(Gammas, coefficients(temp[1][j])'* compute_mom_vector(temp[1][j],data.moment[1]))
            end
            println()
            println("Values of L_{y_star}(p_alpha^2) for polynomials p in the kernel: ", Gammas')
            println()
            
            popcdk = Vector{Polynomial{true, Float64}}(pop)
                    
            for i=1:length(popcdk)
                popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))    #Normalization
            end
            
            push!(CDK_positive, (1-eps)*L_ystar - cdk)

            for i=1:length(CDK_positive)
                push!(popcdk, CDK_positive[i]/maximum(abs.(coefficients(CDK_positive[i]))))    #Normalization
            end
                
            thresholds_kernel=Float64[]

            sum_kernel=sum([kernel_cutoff-temp[1][j] for j=1:temp[5]])

            push!(CDK_zero, sum_kernel)

            for i=1:length(CDK_zero)
                 push!(popcdk, CDK_zero[i]/maximum(abs.(coefficients(CDK_zero[i]))))         #Normalization
            end
            

            println()
            println("_______________ Solving the MODIFIED relaxation: ")
            println()
            println("Total number of constraints at iteration $(i+1) = ", length(popcdk)-1)
            println()
            Random.seed!(seed)

            log_file2 = "solver_log_$(i+1).txt"

    
            if running_time
        
                open(log_file2, "w") do f
                    redirect_stdout(f) do
                        redirect_stderr(f) do
                            rng = Random.default_rng()                 #changing the seed allows Ipopt to get some other local solution, allowing for dynamical upper bound updates
                            Random.seed!(rng, nothing)
                            optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
                        end
                    end
                end
                
                log_content2 = read(log_file2, String)
                time_match2 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content2)
                sdp_solving_time2 = parse(Float64, time_match2.captures[1])
                push!(Times, sdp_solving_time2)

            else
                rng = Random.default_rng() 
                Random.seed!(rng, nothing)
                optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,CS=false,QUIET=!running_time)
            end

            if f(solcdk) < ubRef
                ubRef=f(solcdk)
                println("Global upper bound updated to ", ubRef)
            end

            current_opt = optcdk*normalization_factor
            
            println()
            println("Current opt is ", current_opt)
            println("Current ub is ", ubRef)


            if current_opt > ubRef+1e-6                    #to avoid some innacuracies
                println()
                println(" Overrestriction !!! ")
                Certain_overrestriction=true
                break
            end


            data=datacdk
            push!(Moment_matrices,data.moment[1])
  
            push!(OPT, round(current_opt, digits=6))
            println()
            
            current_gap= (abs((ubRef-current_opt))/abs(ubRef))*100
            println("Current gap = ", round(current_gap,digits=4),"%")
            
            push!(gap_history,round(current_gap,digits=4))
            println("Gap history = ", gap_history ,"%")
            println("History of lower bounds = ", OPT)
            
            println(" ________________________________________________________________________________________________ ")

            
                
        end
        i=i+1
        
    end
                

    return OPT,gap_history, Moment_matrices, K, Certain_overrestriction,ubRef, Times

            
            
end 



















function Iterate_Ly_random_Multiple_Instances(x, n, N, eps, seed, filename, running_time=false,gap_tol=0.5, kernel_cutoff=0.001)
        """        
        Applies H1 to different randomly generated QCQP instances.
        
        # Arguments
        - `x::Vector`: Initial vector of decision variables.
        - `n::Int`: Dimension or size of the problem.
        - `N::Int`: Maximum number of iterations to perform for each QCQP instance.
        - `eps::Float64`: Perturbation factor deciding the volume of CDK sublevel sets.
        - `seed::Vector`: Vector of different seed values that generate different  POP instances.
        - `filename::String`: The name of the ".txt" file where H1 logs should be saved.
        - `running_time::Bool` (optional): If `true`, measures and returns the total runtime for each instance. Default is `false`.
        - `gap_tol::Float64` (optional): Threshold for optimality gap tolerance. Default is `0.5`.
        - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controlling the strength of the Tikhonov regularization of CDK. 
           Default is `0.001`. 
            
        # Returns
        A list with the following elements:
        1. `All_opts`: History of bound strengthening for each instance.
        2. `ALL_gaps`: History of optimality gaps for each instance.
        3. `All_MM`: History of moment matrices for each iteration of each instance.
        4. `All_MM_ranks`: History of the rank changes for the moment matrices for each iteration of each instance.
        5. `All_overrestrictions`: A flag for over-restriction (H1 is certain that further feasible set reductions are not necessary)
        6. `All_UBS`: Available upper bound for all instances.
        7. `All_Times`: SDP solving time measurements iterations and for all instances, if `running_time` is `true`.
        
        """
    
        All_opts = Any[]
        ALL_gaps = Any[]
        All_MM = Any[]
        All_MM_ranks = Any[]
        All_Times = Any[]
        All_UBS=[]
        All_overrestrictions=[]
    
    open(filename, "w") do io
        
        for i = 1:length(seed)
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()______________ \e[1mSOLVING THE INSTANCE $i\e[0m _______________()()()()()")
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io, "()()()()()______________ SOLVING THE INSTANCE $i _______________()()()()()")
            println(io)

            opts, gaps, mm, mm_ranks, overrestriction, ub, r_time = iterate_Ly_random(x,n,N,eps,seed[i],running_time,gap_tol,kernel_cutoff)
            
            push!(All_opts, opts)
            push!(ALL_gaps, gaps)
            push!(All_MM, mm)
            push!(All_MM_ranks, mm_ranks)
            push!(All_UBS, ub)
            push!(All_overrestrictions,overrestriction)

            if running_time
                push!(All_Times,sum(r_time))
            end
                
            
            println(io, "___Instance $i Results:___")
            println(io,"")
            println(io, "Computed bounds: ", opts)
            println(io,"")
            println(io, "Best found upper bound: ", ub)
            println(io,"")
            println(io,"Overrestriction certified: ", overrestriction)
            println(io, "Gaps: ", gaps)
            for i=1:length(mm)
                println(io, "Moment Matrix at iteration $i: ", mm[i])
                println(io,"")
            end
            println(io, "Rank of moment matrices: ", mm_ranks)
            if running_time
                println(io, "Total Running Time: ", sum(r_time))
            end
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, All_MM, All_MM_ranks, All_overrestrictions, All_UBS, All_Times
    
end



function Iterate_Ly_rsparse_Multiple_Instances(x, n, percentage_dense, N, eps, seed, filename, running_time=false,gap_tol=0.5, kernel_cutoff=0.001)

    """        
        Applies H1 to different randomly generated QCQP instances with percentage_dense% non-zero entries.
        
        # Arguments
        - `x::Vector`: Initial vector of decision variables.
        - `n::Int`: Dimension or size of the problem.
        - `N::Int`: Maximum number of iterations to perform for each QCQP instance.
        - `eps::Float64`: Perturbation factor deciding the volume of CDK sublevel sets.
        - `seed::Vector`: Vector of different seed values that generate different  POP instances.
        - `filename::String`: The name of the ".txt" file where H1 logs should be saved.
        - `running_time::Bool` (optional): If `true`, measures and returns the total runtime for each instance. Default is `false`.
        - `gap_tol::Float64` (optional): Threshold for optimality gap tolerance. Default is `0.5`.
        - `kernel_cutoff::Float64` (optional): Threshold $\beta$ controlling the strength of the Tikhonov regularization of CDK. 
           Default is `0.001`. 
            
        # Returns
        A list with the following elements:
        1. `All_opts`: History of bound strengthening for each instance.
        2. `ALL_gaps`: History of optimality gaps for each instance.
        3. `All_MM`: History of moment matrices for each iteration of each instance.
        4. `All_MM_ranks`: History of the rank changes for the moment matrices for each iteration of each instance.
        5. `All_overrestrictions`: A flag for over-restriction (H1 is certain that further feasible set reductions are not necessary)
        6. `All_UBS`: Available upper bound for all instances.
        7. `All_Times`: SDP solving time measurements iterations and for all instances, if `running_time` is `true`.
        
        """
    
        All_opts = Any[]
        ALL_gaps = Any[]
        All_MM = Any[]
        All_MM_ranks = Any[]
        All_Times = Any[]
        All_UBS=[]
        All_overrestrictions=[]
    
    open(filename, "w") do io
        
        for i = 1:length(seed)
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()______________ \e[1mSOLVING THE INSTANCE $i\e[0m _______________()()()()()")
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io, "()()()()()______________ SOLVING THE INSTANCE $i _______________()()()()()")
            println(io)

            opts, gaps, mm, mm_ranks, overrestriction, ub, r_time = iterate_Ly_rsparse(x,n,percentage_dense,N,eps,seed[i],running_time,gap_tol,kernel_cutoff)
            
            push!(All_opts, opts)
            push!(ALL_gaps, gaps)
            push!(All_MM, mm)
            push!(All_MM_ranks, mm_ranks)
            push!(All_UBS, ub)
            push!(All_overrestrictions,overrestriction)

            if running_time
                push!(All_Times,sum(r_time))
            end
                
            
            println(io, "___Instance $i Results:___")
            println(io,"")
            println(io, "Computed bounds: ", opts)
            println(io,"")
            println(io, "Best found upper bound: ", ub)
            println(io,"")
            println(io,"Overrestriction certified: ", overrestriction)
            println(io, "Gaps: ", gaps)
            for i=1:length(mm)
                println(io, "Moment Matrix at iteration $i: ", mm[i])
                println(io,"")
            end
            println(io, "Rank of moment matrices: ", mm_ranks)
            if running_time
                println(io, "Total Running Time: ", sum(r_time))
            end
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, All_MM, All_MM_ranks, All_overrestrictions, All_UBS, All_Times
    
end





function AnalyseResults_H1(results,data,delta=0.5)

    
    """`

    Analyzes the performance of H1

        # Arguments
        - `results`: A data structure containing the output of the H1 optimization process. It is expected to include bounds, gaps, and timing information 
           for each test case.
        - `data`: A data structure containing TRUE reference values and problem-specific details (in form of a list), such as:
            - `data[3]`: Upper bounds for each test case.
            - `data[4]`: True optimal values.
            - `data[6]`: Timing information for the second-order moment relaxation.
            - `data[7]`: Seeds or identifiers from which POP/QCQP instances were generated.
        - `delta::Float64`: Desired gap tolerance (in %)
    
        
        # Returns
        A tuple with the following elements:
        1. **Average gaps**: 
            - `[PREavg, POSTavg]`: Mean gaps (in percentage) before and after applying H1, for non-failed cases.
        2. **Maximum gaps**: 
            - `[PREmax, POSTmax]`: Maximum gaps (in percentage) before and after applying H1, for non-failed  cases.
        3. **Timing information**:
            - `[H2_time, order2_time]`: Mean runtime for H1 and second-order relaxation, for non-failed  cases.
        4. **Failure analysis**:
            - `[length(failed_seeds), count_gaps_below_threshold, identified_overrestrictions]`: Number of failed cases (failed = H1 produced an upper 
               bound), count of Solved cases (Solved = post-H2 gaps are below a delta% threshold), and number of cases where H1 is certain that no further 
               feasible set reductions are possible.
        5. **Details of failed cases**:
            - `[failed, failed_seeds]`: Indices of failed cases and their corresponding seeds.
        6. **Accuracy of IPOPT results**:
            - `[Ipopt_not_exact, Ipopt_not_exact_seeds, length(Ipopt_not_exact)]`: Indices of cases where IPOPT bounds are not exact, corresponding seeds, 
               and the total count of such cases.
        7. **ITERATIONSavg**: Mean runtime for H1 over for non-failed  cases.
            - 

        
        # Notes
        - A case is considered "failed" if the lower bound exceeds the upper bound by more than `1e-6`.
        - The function computes gaps as percentages of the true values.
        - Seeds provide a means of identifying and analyzing specific cases.
    
        """
    
    tot_number_cases=length(data[7])

    failed=[i for i=1:tot_number_cases if results[1][i][end]>data[4][i]+1e-6]  # failed cases - lb > ub

    POSTvsTRUE=[[results[1][i][end],data[4][i]] for i=1:tot_number_cases] #bounds
    PREvsTRUE=[[results[1][i][1],data[4][i]] for i=1:tot_number_cases]  #bounds

    POSTGaps =[abs(POSTvsTRUE[i][2] - POSTvsTRUE[i][1]) / abs(POSTvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100
    PREGaps = [abs(PREvsTRUE[i][2] - PREvsTRUE[i][1]) / abs(PREvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100


    POSTavg = mean(POSTGaps)  # post H2 mean gap
    PREavg = mean(PREGaps)  # pre H2 mean gap

    POSTmax = maximum(POSTGaps)   # post H2 mmax  gap
    PREmax = maximum(PREGaps)   # pre H2 max gap
    
    count_gaps_below_threshold = count(x -> x < delta, POSTGaps)  #solved cases

    order2_time=mean([data[6][i] for i in 1:tot_number_cases if !(i in failed)])
    H1_time=mean([results[end][i] for i in 1:tot_number_cases if !(i in failed)])

 
    #Ipopt_not_exact = findall(i -> abs(data[3][i] - data[4][i]) > 1e-5, 1:length(data[4]))
    UB_not_exact=findall(i -> abs(results[end-1][i] - data[4][i]) > 1e-5, 1:length(data[4]))

    failed_seeds = data[7][failed]
    
    non_failed_indices = [i for i in 1:tot_number_cases if !(i in failed)]
    identified_overrestrictions = sum(results[end-2][i] for i in non_failed_indices)

    
    ITERATIONS = [length(results[1][i]) - 1 for i in non_failed_indices]

    for (idx, i) in enumerate(non_failed_indices)
        if results[end-2][i] 
            ITERATIONS[idx] += 1
        end
    end

            
    ITERATIONSavg=mean(ITERATIONS)

    return [[PREavg,POSTavg],[PREmax,POSTmax], [H1_time,order2_time],[length(failed_seeds),count_gaps_below_threshold,identified_overrestrictions],   [failed,failed_seeds], [UB_not_exact,data[7][UB_not_exact],length(UB_not_exact)], [ITERATIONSavg]]

end






#_____________________________________________________________________________________________________

################################# Some helper functions ##############################################

#_____________________________________________________________________________________________________


function get_monomial_elements(monomial)
    terms = []
    for term in unique(variables(monomial))
        exp = degree(monomial, term)
        append!(terms, repeat([term], exp))
    end
    return terms
end

function compute_mom_vector(monomials,MomMat)

    ind=[]
    for mon in monomials
        push!(ind,[findfirst(item -> item == get_monomial_elements(mon)[i], x)+1 for i=1:length(get_monomial_elements(mon))])
    
    end
    
    for vec in ind
        if length(vec)<2
            push!(vec,1)
        end
    end
    
    ystar=[]
    for i=1:length(ind)-1
        push!(ystar,MomMat[ind[i][1],ind[i][2]])
    end
    push!(ystar,1)

    return ystar

end



function compute_mom_vector_cl(monomials, MomMat, cliques_no)
    ind = []

    for mon in monomials
        elements = get_monomial_elements(mon)
        clique_str = map(string, cliques_no)  # Normalize clique elements as strings
        mon_str = map(string, elements)  # Normalize monomial elements as strings

        push!(ind, [
            let pos = findfirst(item -> item == mon_str[i], clique_str)
                pos === nothing ? throw(ErrorException("Element $(elements[i]) not found in cliques_no")) : pos + 1
            end
            for i in 1:length(elements)
        ])
    end

    # Ensure every vector in `ind` has at least 2 elements
    for vec in ind
        if length(vec) < 2
            push!(vec, 1)
        end
    end

    # Compute ystar using indices
    ystar = []
    for i in 1:length(ind) - 1
        push!(ystar, MomMat[ind[i][1], ind[i][2]])
    end
    push!(ystar, 1)

    return ystar
end











