#Generic template for running model selection comparison simulations for my dissertation

using Distributed
#Add additional processes
addprocs(length(Sys.cpu_info())-1)

@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using DataFrames
@everywhere using Distributed
@everywhere using Distributions
@everywhere using GLM
@everywhere using StatsModels
@everywhere using StatsBase
@everywhere using Combinatorics
@everywhere using CSV

#Set random seed
@everywhere Random.seed!(myid()+2172023)

#Set our initial parameters
intercept = 2.0
b1 = 1.0
b2 = 1.0
true_sig = 2

#Set up vector of ns to loop through
ns = 100:100:1000

#Number of iterations
num_iters = 1000

#Loop through all n values to generate predictive ability
#Should be safe to do this as a threaded loop since each iteration is completely independent

plot_data = @distributed (append!) for kk in eachindex(ns)
    #N for this iteration
    n = ns[kk]
    #Get storage for this iteration - each worker will have its own
    out = DataFrame(iteration = 1:num_iters,

    Min_AIC_PSSRR = zeros(Float64,num_iters),
    AIC_Rule_Of_2_PSSRR = zeros(Float64,num_iters),
    Stand_AIC_Method_PSSRR = zeros(Float64,num_iters),
    Min_AIC_Spec = zeros(Float64,num_iters),
    AIC_Rule_Of_2_Spec = zeros(Float64,num_iters),
    Stand_AIC_Method_Spec = zeros(Float64,num_iters),

    Min_BIC_PSSRR = zeros(Float64,num_iters),
    BIC_Rule_Of_2_PSSRR = zeros(Float64,num_iters),
    Stand_BIC_Method_PSSRR = zeros(Float64,num_iters),
    Min_BIC_Spec = zeros(Float64,num_iters),
    BIC_Rule_Of_2_Spec = zeros(Float64,num_iters),
    Stand_BIC_Method_Spec = zeros(Float64,num_iters),

    Min_AICc_PSSRR = zeros(Float64,num_iters),
    AICc_Rule_Of_2_PSSRR = zeros(Float64,num_iters),
    Stand_AICc_Method_PSSRR = zeros(Float64,num_iters),
    Min_AICc_Spec = zeros(Float64,num_iters),
    AICc_Rule_Of_2_Spec = zeros(Float64,num_iters),
    Stand_AICc_Method_Spec = zeros(Float64,num_iters)
    )
    for jj = 1:num_iters
        #Generate covariates
        dataset = DataFrame(x = 5*rand(n), z = 5*rand(n), w = 5*rand(n),
        u = 5*rand(n), v = 5*rand(n), q = 5*rand(n))
        #Generate outcomes
        dataset.y = intercept*ones(Float64,n) + b1*dataset.x + b2*dataset.z + rand(Normal(0,true_sig),n)
        #Get test data Set
        pred_data = DataFrame(x = 5*rand(1000), z = 5*rand(1000), w = 5*rand(1000),
        u = 5*rand(1000), v = 5*rand(1000), q = 5*rand(1000))
        pred_data.y = intercept*ones(Float64,1000) + b1*pred_data.x + b2*pred_data.z + rand(Normal(0,true_sig),1000)
        #Get predicted mean
        pred_mu = intercept*ones(Float64,1000) + b1*pred_data.x + b2*pred_data.z
        #Generate every single regression model in Julia
        var_names = names(dataset)[1:(length(names(dataset))-1)]
        this_index = 1
        model_dict = Dict{Integer, Any}()
        poss_combs = combinations(1:(ncol(dataset)-1)) |> collect
        #Allocate storage for statistics about models 
        temp = DataFrame(Index = 1:length(poss_combs),
        AIC = zeros(Float64,length(poss_combs)),
        BIC = zeros(Float64,length(poss_combs)),
        AICc = zeros(Float64,length(poss_combs)),
        p = zeros(Float64,length(poss_combs)),
        Correctly_Spec = zeros(Float64,length(poss_combs)))
        for ii in eachindex(poss_combs)
            this_comb = poss_combs[ii]
            these_varnames = var_names[this_comb]
            model_dict[ii] = lm(@eval(@formula($(Meta.parse(join(["y ~ 1",join(these_varnames," + ")]," + "))))),dataset)
            #Save all of our data
            temp.p[ii] = length(these_varnames)+1
            temp.AIC[ii] = aic(model_dict[ii])
            temp.BIC[ii] = bic(model_dict[ii])
            temp.AICc[ii] = aicc(model_dict[ii])
            if ii == 7
                temp.Correctly_Spec[ii] = 1
            end
        end
        cand_AIC = temp.AIC[nrow(temp)]
        cand_BIC = temp.BIC[nrow(temp)]
        cand_AICc = temp.AICc[nrow(temp)]
        max_params = temp.p[nrow(temp)]
        #Calc further things
        temp.k = max_params*ones(Float64,nrow(temp)) - temp.p

        temp.AIC_Diff = temp.AIC - cand_AIC*ones(Float64,nrow(temp))
        temp.Stand_AIC_Diff = replace!((temp.AIC_Diff./sqrt.(2*temp.k)), NaN=>0)

        temp.BIC_Diff = temp.BIC - cand_BIC*ones(Float64,nrow(temp))
        temp.Stand_BIC_Diff = replace!((temp.BIC_Diff./sqrt.(2*temp.k)), NaN=> sqrt(1/2)*(1-log(n)))

        temp.AICc_Diff = temp.AICc - cand_AICc*ones(Float64,nrow(temp))
        ones_help = ones(Float64,nrow(temp))
        temp.Model_Cutoff_AICc = sqrt.(temp.k/2).*(ones_help-((ones_help./(n*ones_help-temp.p-ones_help))).*((2*ones_help*n*(n+1))./(n*ones_help-ones_help*max_params-ones_help)))
        temp.Stand_AICc_Diff = replace!((temp.AICc_Diff./sqrt.(2*temp.k)), NaN=>maximum(temp.Model_Cutoff_AICc))
        #Calculate denominator for MSE ratio
        mse_denom = mean((pred_mu - pred_data.y).^2)

        #Now do various selection procedures and calculate MSE
        ###########
        ### AIC ###
        ###########
        temp_filt = filter(:AIC => n -> n <= minimum(temp.AIC)+2, temp)
        temp_filt = sort(temp_filt, [:p, :AIC])
        out.AIC_Rule_Of_2_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.AIC_Rule_Of_2_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = filter(:Stand_AIC_Diff => n -> n <= -sqrt(1/2)+2, temp)
        temp_filt = sort(temp_filt, [:p, :Stand_AIC_Diff])
        out.Stand_AIC_Method_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Stand_AIC_Method_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = sort(temp, [:AIC])
        out.Min_AIC_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Min_AIC_Spec[jj] = temp_filt.Correctly_Spec[1]

        ###########
        ### BIC ###
        ###########
        temp_filt = filter(:BIC => n -> n <= minimum(temp.BIC)+2, temp)
        temp_filt = sort(temp_filt, [:p, :BIC])
        out.BIC_Rule_Of_2_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.BIC_Rule_Of_2_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = filter(:Stand_BIC_Diff => x -> x <= sqrt(1/2)*(1-log(n))+2, temp)
        temp_filt = sort(temp_filt, [:p, :Stand_BIC_Diff])
        out.Stand_BIC_Method_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Stand_BIC_Method_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = sort(temp, [:BIC])
        out.Min_BIC_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Min_BIC_Spec[jj] = temp_filt.Correctly_Spec[1]

        ############
        ### AICc ###
        ############
        temp_filt = filter(:AICc => n -> n <= minimum(temp.AICc)+2, temp)
        temp_filt = sort(temp_filt, [:p, :AICc])
        out.AICc_Rule_Of_2_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.AICc_Rule_Of_2_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = filter(:Stand_AICc_Diff => x -> x <= maximum(temp.Model_Cutoff_AICc)+2, temp)
        temp_filt = sort(temp_filt, [:p, :Stand_AICc_Diff])
        out.Stand_AICc_Method_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Stand_AICc_Method_Spec[jj] = temp_filt.Correctly_Spec[1]

        temp_filt = sort(temp, [:AICc])
        out.Min_AICc_PSSRR[jj] = mean((predict(model_dict[temp_filt.Index[1]], pred_data) - pred_data.y).^2)/mse_denom
        out.Min_AICc_Spec[jj] = temp_filt.Correctly_Spec[1]

    end
    #Output this row of plotting data
    DataFrame(n = n,
        Min_AIC_PSSRR = mean(out.Min_AIC_PSSRR),
        AIC_Rule_Of_2_PSSRR = mean(out.AIC_Rule_Of_2_PSSRR),
        Stand_AIC_Method_PSSRR = mean(out.Stand_AIC_Method_PSSRR),
        Min_AIC_Spec = mean(out.Min_AIC_Spec),
        AIC_Rule_Of_2_Spec = mean(out.AIC_Rule_Of_2_Spec),
        Stand_AIC_Method_Spec = mean(out.Stand_AIC_Method_Spec),

        Min_BIC_PSSRR = mean(out.Min_BIC_PSSRR),
        BIC_Rule_Of_2_PSSRR = mean(out.BIC_Rule_Of_2_PSSRR),
        Stand_BIC_Method_PSSRR = mean(out.Stand_BIC_Method_PSSRR),
        Min_BIC_Spec = mean(out.Min_BIC_Spec),
        BIC_Rule_Of_2_Spec = mean(out.BIC_Rule_Of_2_Spec),
        Stand_BIC_Method_Spec = mean(out.Stand_BIC_Method_Spec),

        Min_AICc_PSSRR = mean(out.Min_AICc_PSSRR),
        AICc_Rule_Of_2_PSSRR = mean(out.AICc_Rule_Of_2_PSSRR),
        Stand_AICc_Method_PSSRR = mean(out.Stand_AICc_Method_PSSRR),
        Min_AICc_Spec = mean(out.Min_AICc_Spec),
        AICc_Rule_Of_2_Spec = mean(out.AICc_Rule_Of_2_Spec),
        Stand_AICc_Method_Spec = mean(out.Stand_AICc_Method_Spec)
    )

end

#Save this plotting data to plot in R
CSV.write("H:/Dissertation/writing_thesis/thesis_tex_files/data/plot_data_sim1.csv",plot_data)