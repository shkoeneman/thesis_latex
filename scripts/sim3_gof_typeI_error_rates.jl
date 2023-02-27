#Generic template for bootstrap model selection test for my thesis

using Distributed
#Add additional processes
addprocs(length(Sys.cpu_info())-1)

@everywhere using LinearAlgebra
@everywhere using Random
@everywhere using DataFrames
@everywhere using Distributed
@everywhere using Distributions
@everywhere using HypothesisTests
@everywhere using GLM
@everywhere using StatsModels
@everywhere using StatsBase
@everywhere using Combinatorics
@everywhere using CSV

#Set random seed
@everywhere Random.seed!(myid()+10000)

#Set our initial parameters
intercept = 2.0
b1 = 2.0
b2 = 2.0
true_sig = 2

#True model will be linear, but with heteroskedasticity

#Set up vector of ns to loop through
ns = 100:100:1000

#Number of iterations
num_iters = 1000
boot_iters = 1000
level = .05

#Define functions we will need
#Function to get sandwich estimate
@everywhere function asymp_var_sandwich(model_lm)

    #Extract things needed from model
    n = size(model_lm.mm.m)[1];
    y = model_lm.mf.data.y;
    X = model_lm.mm.m;

    #Get fitted parameter values
    Beta = coef(model_lm);
    sighat2 = (transpose(y - (X*Beta)) * (y - (X*Beta))*(1/n));

    #Calculate observed information matrix and central terms based on score
    Info = [transpose(X)*X/sighat2 transpose(transpose(y - X*Beta)*X/(sighat2^2));
    (transpose(y - X*Beta)*X/(sighat2^2)) -n/(2*sighat2^2) + ((transpose(y - X*Beta)*(y - X*Beta)))/(sighat2^3)];
    Info_inv = inv(Info);

    #Calculate central term
    cent_term = zeros(Float64,size(Info_inv)[1],size(Info_inv)[2]);

    for cc = 1:n
        score = [(y[cc] - dot(X[cc,:],Beta))*X[cc,:]/(sighat2);
        (y[cc] - dot(X[cc,:],Beta))^2/(2*sighat2^2) - (1/(2*sighat2));
        ]
        cent_term = cent_term + (score*transpose(score));
    end
    #Return results
    dict = Dict{String, Any}()
    dict["asymp_var"] = (Info_inv*cent_term*Info_inv)[size(Info_inv)[1],size(Info_inv)[2]];
    dict["sighat2"] = sighat2;
    return dict

end

#Loop through all n values to generate power/type I error of test
#Should be safe to do this as a distributed loop since each iteration is completely independent

plot_data = @distributed (append!) for kk = 1:length(ns)
    #N for this iteration
    n = ns[kk]
    #Get storage for this iteration - each worker will have its own
    out = zeros(Float64, num_iters)
    #Try test with accelerated bootstrap CIs 
    #out_acc = zeros(Float64, num_iters)
    #Allocate storage for other tests for heteroskedasticity
    out_white = zeros(Float64, num_iters)
    out_bp = zeros(Float64, num_iters)

    for jj = 1:num_iters
        #Generate covariates
        dataset = DataFrame(x = 5*rand(n),
        z = 5*rand(n))

        #Generate outcomes
        dataset.y = zeros(Float64,nrow(dataset))

        for ii = 1:length(dataset.y)
            
            dataset.y[ii] = intercept + b1*dataset.x[ii] + b2*dataset.z[ii] + rand(Normal(0,true_sig),1)[1]
            #dataset.y[ii] = intercept + b1*dataset.x[ii] + b2*dataset.z[ii] + rand(Normal(0,5+dataset.x[ii]+dataset.z[ii]),1)[1]

        end
        
        #Get bootstrap Distribution
        boot_dist = zeros(Float64, boot_iters)
        @views for ii = 1:boot_iters
            #Bootstrap sample
            boot_dataset = dataset[sample(1:n,n,replace=true),:]
            #Fit model
            model_lm = lm(@formula(y ~ x + z),boot_dataset)
            #Get statistics
            results = asymp_var_sandwich(model_lm)
            #Calculate from delta method
            boot_dist[ii] = n^2*(1/(results["sighat2"]^2))*results["asymp_var"]
        end
        #Store result of this quantile bootstrap test
        out[jj] = ( (2*n < quantile(boot_dist,0+(level/2))) | (2*n > quantile(boot_dist,1-(level/2))) ) ? 1 : 0

        #Perform other heteroskedasticity tests
        model_lm = lm(@formula(y ~ x + z),dataset)
        out_white[jj] = (pvalue(WhiteTest(model_lm.mm.m, (dataset.y-predict(model_lm)))) < level) ? 1 : 0
        out_bp[jj] = (pvalue(BreuschPaganTest(model_lm.mm.m, (dataset.y-predict(model_lm)))) < level) ? 1 : 0

        #Do what we need to get the accelerated bootstrap CI
        #results = asymp_var_sandwich(model_lm)
        #pt_est = n^2*(1/(results["sighat2"]^2))*results["asymp_var"]
        #alpha = [0+(level/2),1-(level/2)]
        #z0 = quantile(Normal(), mean(boot_dist .< pt_est))
        #zu = quantile(Normal(), alpha)
        ##Calculate a from finite-sample Jackknife
        #I = zeros(Float64, n)
        #@views for ii = 1:n
        #    #Fit model
        #    model_lm = lm(@formula(y ~ x + z),dataset[filter((x) -> x != ii,1:n),:])
        #    #Get statistics
        #    results = asymp_var_sandwich(model_lm)
        #    #Calculate from delta method
        #    jack_est = n^2*(1/(results["sighat2"]^2))*results["asymp_var"]
        #    I[ii] = (n-1)*(pt_est - jack_est)
        #end
        #a = (sum(I.^3)/sum(I.^2)^1.5)/6
        ##Adjusted quantiles
        #u_unadjusted = cdf(Normal(), z0 .+ (z0 .+ zu)./(1 .- a*(z0.+zu)))
        ##Get accelerated interval
        #acc_CI = quantile(boot_dist, u_unadjusted)
        ##Store result
        #out_acc[jj] = ( (2*n < acc_CI[1]) | (2*n > acc_CI[2] ) ) ? 1 : 0


    end
    #Output this row of data
    DataFrame(N = n,
        Power_Boot_Quant = mean(out),
        Power_White = mean(out_white),
        Power_Breusch_Pagan = mean(out_bp)
    )

end
print(plot_data)
#Save this plotting data to plot in R
CSV.write("H:/Dissertation/writing_thesis/thesis_tex_files/data/table_data_sim3.csv",plot_data)

