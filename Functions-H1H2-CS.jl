function generate_Q_and_q_bd(n,p,qq,seed)
    Random.seed!(seed)
    
    blocks = [randn(qq, qq) for _ in 1:p] 
    symmetric_blocks = [0.5 * (B + B') for B in blocks]
    Q = zeros(n, n)
    for i in 1:p
        start_idx = (i-1)*qq + 1
        end_idx = i*qq
        Q[start_idx:end_idx, start_idx:end_idx] .= symmetric_blocks[i]
    end
        
    q = randn(n)
    
    return Q, q
end


##############################################################################################################################################
                                         # H2-related FUNCTIONS
##############################################################################################################################################


function find_clique_index(k::Int, p::Int, qq::Int)::Union{Int, Nothing}
    n = p * qq
    if k < 1 || k > n
        println("Index $k is out of bounds.")
        return nothing
    end
    clique_index = ceil(Int, k / qq)
    return clique_index
end

function construct_CDKmarg_SVD_CS(Mm, loc_idx, gl_idx, threshold=0.001)


    eigen_result = eigen(Mm[[1,loc_idx+1],[1,loc_idx+1]])

    Qvec = eigen_result.vectors
    Qval = eigen_result.values

    p_alpha_squared=[]   #Vector of orhonormal polynomials 
    
    for j=1:length(Qval)
        push!(p_alpha_squared, (vcat(1,x[gl_idx])'*Qvec[:,j])^2)
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





function Run_H2_CS(x,n,p,qq,seed,solIpopt,tau,running_time=false,kernel_cutoff=0.0001)

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS_____________________ ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q_bd(n,p,qq,seed)


    println("Eigvals of Q = ", eigvals(Q)')
    println()
    
    f = x' * Q * x + x' * q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    println()
    println()
    println("_______________ SOLVING THE INITIAL RELAXATION _____________________ ")
    println()
    println()

    cliques_bd = [collect(i:i+qq-1) for i in 1:qq:(p*qq)]


    Random.seed!(seed)
    log_file1 = "solver_log1.txt"

    if running_time

        open(log_file1, "w") do f
            redirect_stdout(f) do
                redirect_stderr(f) do
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false, solution=true,QUIET=!running_time,cliques=cliques_bd)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
    else
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false, solution=true,QUIET=!running_time, cliques=cliques_bd)
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
        clique_index = find_clique_index(i, p, qq)
        local_i = i - (clique_index - 1) * qq
        clique_matrix = data.moment[clique_index]
        #println(clique_matrix[[1,local_i],[1,local_i]])
        push!(CDK_marg_SVD,construct_CDKmarg_SVD_CS(clique_matrix,local_i,i))
    end
    
    popcdk = Vector{Polynomial{true, Float64}}(pop)
    Gammas=[]
    Gammas_kernel=[]

    for i=1:n
        temp=CDK_marg_SVD[i]
        gamma_i=temp[1](solIpopt[i])+1e-2 # to avoid numerical issues
        push!(Gammas,gamma_i)
        

        if gamma_i < tau
            if temp[4] > 0
                push!(Gammas_kernel, sum(temp[2])(solIpopt[i])+kernel_cutoff)
                #println()
                #println("I add these TWO: ", i)
                #println([sum(temp[2])(solIpopt[i])+kernel_cutoff-sum(temp[2]), gamma_i-temp[1]])
                push!(popcdk,sum(temp[2])(solIpopt[i])+kernel_cutoff-sum(temp[2]), gamma_i-temp[1])
            else
                #println()
                #println("I add this ONE: ", i)
                #println(gamma_i-temp[1])
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
                    optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,QUIET=!running_time)
                end
            end
        end
        
    # Read the log file to extract the required information
        log_content2 = read(log_file2, String)
        time_match2 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content2)
        sdp_solving_time2 = parse(Float64, time_match2.captures[1])
    else
        optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,QUIET=!running_time)
    end

        
    println()
    println("Summary: ____________________________________________________________________ ")
    
    optcdk=optcdk*maximum(abs.(coefficients(pop[1])))
    gapcdk= abs((ubRef-optcdk)/ubRef)*100
    println("Gammas = ",Gammas)
    println()
    println("Gammas_kernel = ", Gammas_kernel)
    println("\e[1mNEW BOUND\e[0m = ", optcdk)
    println()

    if running_time
        total_time=sdp_solving_time1+sdp_solving_time2
    else
        total_time=""
    end
    
    return [opt,optcdk],[gap,gapcdk],Gammas,Gammas_kernel,[ubRef,f(solcdk)],[data.moment,datacdk.moment],total_time       
            
end 




function Run_H2_Multiple_Instances_CS(x, n, p, qq, seed, solIpopt, tau, filename, running_time=false, kernel_cutoff=0.0001)
    
        All_opts = Any[]
        ALL_gaps = Any[]
        ALL_Gammas = Any[]
        ALL_UBS = Any[]
        All_MM = Any[]
        All_times = Any[]
    open(filename, "w") do io
        
        for i = 1:length(seed)
            println("()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println("()()()()()______________ \e[1mSOLVING THE INSTANCE $i\e[0m _______________()()()()()")
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io)

            opts, gaps, gammas, gammas_kernel, UBS, mm, r_time = Run_H2_CS(x, n, p, qq, seed[i], solIpopt[i], tau, running_time, kernel_cutoff)
            
            push!(All_opts, opts)
            push!(ALL_Gammas, gammas)
            push!(ALL_UBS, UBS)
            push!(ALL_gaps, gaps)
            push!(All_MM, mm)
            if running_time
                push!(All_times,r_time)
            end
                
            
            println(io, "Instance $i Results:")
            println(io, "Computed bounds: ", opts)
            println(io, "Gaps: ", gaps)
            println(io, "Gammas: ", gammas)
            println(io, "Known upper bounds: ", UBS)
            for i=1:p
                println(io, "Moment Matrix for clique $i: ", [mm[1][i],mm[2][i]])
            end
            if running_time
                println(io, "Total Running Time: ", r_time)
            end
            println(io, "()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()()")
            println(io, "()()()()()________________________________________________________________()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, ALL_Gammas, ALL_UBS, All_MM, All_times
end



function AnalyseResults_H2(results,data)

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
    
    count_gaps_below_threshold = count(x -> x < 0.5, POSTGaps)  #solved cases

    order2_time=mean([data[6][i] for i in 1:tot_number_cases if !(i in failed)])
    H2_time=mean([results[end][i] for i in 1:tot_number_cases if !(i in failed)])

 
    Ipopt_not_exact = findall(i -> abs(data[3][i] - data[4][i]) > 1e-5, 1:tot_number_cases)

    failed_seeds = data[7][failed]
    Ipopt_not_exact_seeds=data[7][Ipopt_not_exact]

    return [[PREavg,POSTavg],[PREmax,POSTmax], [H2_time,order2_time],[length(failed_seeds),count_gaps_below_threshold],[failed,failed_seeds],[Ipopt_not_exact,Ipopt_not_exact_seeds,length(Ipopt_not_exact)]]

end

##############################################################################################################################################
                                         # H1-related FUNCTIONS
##############################################################################################################################################




function get_monomial_elements(monomial)
    terms = []
    for term in unique(variables(monomial))
        exp = degree(monomial, term)
        append!(terms, repeat([term], exp))
    end
    return terms
end

function compute_mom_vector_CS(monomials, MomMat, cliques_no)
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

function construct_CDK_CS(Mm, threshold=0.001)
    Cdk = []
    Kernel = []
    posEV = []
    negEV = [] 
    minEV = []

    cliques_no1 = [x[(i-1)*qq+1:i*qq] for i in 1:p]  
    local_basis = []
    for i in 1:length(cliques_no1)
        push!(local_basis, vcat(1, cliques_no1[i]))
    end

    for i = 1:p
        eigen_result = eigen(Mm[i])

        Qvec = eigen_result.vectors
        Qval = eigen_result.values

        p_alpha_squared = []  # Vector of orthonormal polynomials 
    
        for j = 1:length(Qval)
            push!(p_alpha_squared, (local_basis[i]' * Qvec[:, j])^2)
        end

        positiveEV = count(x -> abs(x) > threshold, Qval)
        negativeEV = count(x -> abs(x) <= threshold, Qval)

        if positiveEV == length(Qval)
            cdk = sum(p_alpha_squared[j] / Qval[j] for j in 1:length(Qval))
        else
            cdk = sum(p_alpha_squared[j] / Qval[j] for j in (length(Qval) - positiveEV + 1):length(Qval))
        end

        push!(Cdk, cdk)
        push!(Kernel, p_alpha_squared[1:negativeEV])
        push!(posEV, positiveEV)
        push!(negEV, negativeEV)
        push!(minEV, minimum(Qval))
    end

    return Kernel, Cdk, posEV, minEV, negEV
end


function iterate_Ly_block_diag_CS(x,n,p,qq,N,eps,seed,running_time=false,gap_tol=0.5,kernel_cutoff=0.0001)

    println()
    println("_______________ INITIAL PROBLEM CHARACTERISTICS: ")
    println()

    Random.seed!(seed)
    Q, q = generate_Q_and_q_bd(n,p,qq,seed)

    println("Dimension n = ",n)
    println("Threshold reduction coefficient  = ", 1-eps)
    println("Eigvals of Q = ", eigvals(Q)')
    println()
    
    f=x'*Q*x+x'q
    d=1
    pop=[f]
    push!(pop,vcat(x.*(1 .-x))...)

    cliques_bd = [collect(i:i+qq-1) for i in 1:qq:(p*qq)]  #variable indices
    cliques_mon = [x[(i-1)*qq+1:i*qq] for i in 1:p]   #monomials in the clique


    Times = []
    UBH=[]
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
                    opt, sol, data = cs_tssos_first(pop, x, d, TS=false, solution=true,QUIET=!running_time,cliques=cliques_bd)
                end
            end
        end
        
        log_content1 = read(log_file1, String)
        time_match1 = match(r"SDP solving time:\s*([\d\.]+)\s*seconds", log_content1)
        sdp_solving_time1 = parse(Float64, time_match1.captures[1])
        push!(Times, sdp_solving_time1)
    else
        Random.seed!(seed)
        opt, sol, data = cs_tssos_first(pop, x, d, TS=false,solution=true,QUIET=!running_time,cliques=cliques_bd)
    end
    println()
    println(" ________________________________________________________________________________________________ ")

    push!(Moment_matrices,data.moment)
    push!(UBH,f(sol))

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
    K=[Int.(ones(size(data.moment[i], 1))) for i=1:p]



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
        

        temp = construct_CDK_CS(data.moment)
        

        for k=1:p                                   #Update the CDK constraints for each clique
            println()
            println("Information about the clique $k:")
            println()
            println("Number of positive eigenvalues of the moment matrix is ", temp[3][k]', " and the smallest one is  ", temp[4][k]')
            println("Kernel dimension is ", length(temp[1][k]))
            
            K[k][temp[3][k]]=K[k][temp[3][k]]+1
                                    
            cdk = temp[2][k]           
            coeffs = coefficients(cdk)
            moment_vector = compute_mom_vector_CS(monomials(cdk),data.moment[k],cliques_mon[k])
            println("y_star_previous: ", Vector{Float64}(moment_vector[end-qq:end-1]))
            println()
            L_ystar = coeffs'*moment_vector
    
            println("L_ystar = ", L_ystar)
            println("Chosen level set for clique $k, namely (1-eps)*L_ystar = ", (1-eps)*L_ystar)
    
            Gammas=Float64[]
    
            for j=1:temp[5][k]
                push!(Gammas, coefficients(temp[1][k][j])'* compute_mom_vector_CS(monomials(temp[1][k][j]),data.moment[k],cliques_mon[k]))
            end
                
            println()
            println("Values of L_{y_star}(p_alpha^2) for polynomials p in the kernel: ", Gammas')
            println()
            
            push!(CDK_positive, (1-eps)*L_ystar - cdk)

            thresholds_kernel=Float64[]
    
            sum_kernel=sum([kernel_cutoff-temp[1][k][j] for j=1:temp[5][k]])
    
            push!(CDK_zero, sum_kernel)
        end
            

        popcdk = Vector{Polynomial{true, Float64}}(pop)     #Cnstruct the updated POP
                
        for i=1:length(popcdk)
            popcdk[i]=popcdk[i]/maximum(abs.(coefficients(popcdk[i])))    #Normalization
        end
        for i=1:length(CDK_positive)
            push!(popcdk, CDK_positive[i]/maximum(abs.(coefficients(CDK_positive[i]))))    #Normalization
        end

        for i=1:length(CDK_zero)
             push!(popcdk, CDK_zero[i]/maximum(abs.(coefficients(CDK_zero[i]))))         #Normalization
        end
                
    
        println()
        println("_______________ Solving the MODIFIED relaxation: ")
        println()
        println("Total number of constraints at iteration $(i+1) = ", length(popcdk)-1)
        println()

        log_file2 = "solver_log_$(i+1).txt"
    
        
        if running_time
            open(log_file2, "w") do f
                redirect_stdout(f) do
                    redirect_stderr(f) do
                        rng = Random.default_rng() 
                        Random.seed!(rng, nothing)
                        optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,QUIET=!running_time,cliques=cliques_bd)
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
            optcdk,solcdk,datacdk=cs_tssos_first(popcdk,x,d,solution=true,TS=false,QUIET=!running_time,cliques=cliques_bd)
        end

        push!(UBH,f(solcdk))
    
        if f(solcdk) < ubRef
            ubRef=f(solcdk)
            println(" --- !!! ---- GLOBAL UB UPDATED TO: ", ubRef)
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
        push!(Moment_matrices,data.moment)

        push!(OPT, round(current_opt, digits=6))
        println()
        
        current_gap= (abs((ubRef-current_opt))/abs(ubRef))*100
        println("Current gap = ", round(current_gap,digits=4),"%")
        
        push!(gap_history,round(current_gap,digits=4))
        println("Gap history = ", gap_history ,"%")
        println("History of lower bounds = ", OPT)
        
        println(" ________________________________________________________________________________________________ ")
    
                
                    
        i=i+1
            
        
    end
                

    return OPT, gap_history, Moment_matrices, K, Certain_overrestriction, UBH, Times

            
end 


function Run_H1_Multiple_Instances_CS(x, n, p, qq, N, eps, seed, filename, running_time=false,gap_tol=0.5, kernel_cutoff=0.0001)
    
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
            println(io)

            opts, gaps, mm, mm_ranks, overrestriction, ub, r_time = iterate_Ly_block_diag_CS(x, n, p, qq, N, eps, seed[i], running_time, gap_tol, kernel_cutoff)
            
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
            println(io, "()()()()()________________________________________________________________()()()()()")

        end

    end
    
    return All_opts, ALL_gaps, All_MM, All_MM_ranks, All_overrestrictions, All_UBS, All_Times
end








function AnalyseResults_H1(results,data,gap_tol=0.5)

    tot_number_cases=length(data[7])

    failed=[i for i=1:tot_number_cases if results[1][i][end]>data[4][i]+1e-6]  # failed cases - lb > ub

    POSTvsTRUE=[[results[1][i][end],data[4][i]] for i=1:tot_number_cases] #bounds
    PREvsTRUE=[[results[1][i][1],data[4][i]] for i=1:tot_number_cases]  #bounds

    POSTGaps =[abs(POSTvsTRUE[i][2] - POSTvsTRUE[i][1]) / abs(POSTvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100
    PREGaps = [abs(PREvsTRUE[i][2] - PREvsTRUE[i][1]) / abs(PREvsTRUE[i][2]) for i in 1:tot_number_cases if !(i in failed)]*100


    POSTavg = mean(POSTGaps)  # post H1 mean gap
    PREavg = mean(PREGaps)  # pre H1 mean gap

    POSTmax = maximum(POSTGaps)   # post H1 mmax  gap
    PREmax = maximum(PREGaps)   # pre H1 max gap
    
    count_gaps_below_threshold = count(x -> x < gap_tol, POSTGaps)  #solved cases

    order2_time=mean([data[6][i] for i in 1:tot_number_cases if !(i in failed)])
    H1_time=mean([results[end][i] for i in 1:tot_number_cases if !(i in failed)])

 
    UB_not_exact=findall(i -> abs(minimum(results[end-1][i]) - data[4][i]) > 1e-5, 1:length(data[4]))

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





















