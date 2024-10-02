include("../src/polyatree.jl")
include("../src/partitions.jl")
include("simulation_utils.jl")

using Distributions
using DataFrames
using Serialization
using ProgressBars


NREP = 100
NDATAS = [50, 100, 250, 500, 750, 1000]
XGRID = collect(LinRange(1e-8, 1-1e-8, 1000))

DG1 = Uniform(0, 0.5)
DG2 = MixtureModel([
    Uniform(0, 0.25),
    Uniform(0.125, 0.25),
    Uniform(0.5, 1.0)], [1/6, 1/2, 1/3])
DG3 = truncated(Normal(0.5, 0.1), 0.0, 1.0)
DG4 = Uniform(0, 0.2)
DG4 = MixtureModel([
    Uniform(0, 0.2),
    Uniform(0.7, 0.9)], [2/3, 1/3])
DG6 = MixtureModel([
    Beta(2, 15),
    Beta(15, 2)], [1/2, 1/2])
DG = [DG1, DG2, DG3, DG4, DG4, DG5, DG6]

TRUE_DENS = [pdf.(d, XGRID) for d in DG]

# Tuples of "model-id" and "alpha0". For all models max_depth is 10
# and beta0 = alpha0
MAX_DEPTH = 10
PRIORS = [
    ("GFPT1", 0.05, -1, false),
    ("GFPT1", 0.1, -1, false),
    ("GFPT2", 0.1, 0.5, true),
    ("GFPT2", 0.1, 1.0, true),
    ("GFPT2", 0.1, 2.0, true),
    ("GFPT2", 0.1, 5.0, true),
    ("GFPT2", 0.1, 0.5, false),
    ("GFPT2", 0.1, 1.0, false),
    ("GFPT2", 0.1, 2.0, false),
    ("GFPT2", 0.1, 5.0, false)
]

function build_model(model_id, alpha0, beta0, increasing_beta)
    if model_id == "GFPT1"
        return GFPT1(Poisson(5.0), MAX_DEPTH, alpha0)
    elseif model_id == "GFPT2"
        return GFPT2(Poisson(5.0), MAX_DEPTH, alpha0, beta0, increasing_beta)
    else 
        throw("Non-reckognized model_id: '$model_id' ")
    end
end

function get_posterior_summaries(pt::GFPT2, data)
    mcmc_chain = run_mcmc(data, pt, 10000, 5000, 1);
    pred_dens = predictive_density(XGRID, mcmc_chain)
    post_n_proba = zeros(MAX_DEPTH)
    for tree in mcmc_chain
        post_n_proba[tree.curr_n] += 1
    end
    post_n_proba ./= length(mcmc_chain)

    post_n_mean = sum(collect(1:MAX_DEPTH) .* post_n_proba)
    return post_n_mean, pred_dens
end


function get_posterior_summaries(pt::GFPT1, data)
    model = update(data, pt)
    pred_dens = predictive_density(XGRID, model)
    post_n = sum(collect(1:MAX_DEPTH) .* model.prob_n)
    return post_n, pred_dens
end


function run_one_iter(iternum)
    out = []

    for prior in PRIORS
        for (i, data_distribution) in enumerate(DG)
            for ndata in NDATAS
                model = build_model(prior...)
                data = rand(data_distribution, ndata)

                post_n, pred_dens = get_posterior_summaries(model, data)

                # Compute summaries
                l1 = l1_dist(TRUE_DENS[i], pred_dens, XGRID)
                length = arclength(TRUE_DENS[i] - pred_dens, XGRID)

                out = push!(
                    out,
                    (prior..., i, ndata, l1, length, post_n, iternum)
                )
            end
        end
    end
    return DataFrame(out, ["Model", "alpha0", "beta0", "increasing_beta", "DataGen", 
                           "Ndata", "L1", "LENGTH", "POST_N", "ITER"])
end

function main()
    tmp =  Array{DataFrame}(undef, NREP)
    @Threads.threads for i in ProgressBar(1:NREP)
        tmp[i] = run_one_iter(i)
    end
    out = reduce(vcat, tmp)
    Serialization.serialize("simulation2_results.dta", out)
end

main()