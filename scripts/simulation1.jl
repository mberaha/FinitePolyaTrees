include("../src/polyatree.jl")
include("../src/partitions.jl")
include("simulation_utils.jl")

using AddPackage
@add using Distributions
@add using DataFrames
@add using Serialization
@add using ProgressBars


NREP = 50
NDATAS = [50, 100, 1000, 5000, 10000]
XGRID = collect(LinRange(1e-8, 1-1e-8, 1000))

DG1 = Uniform(0, 0.5)
DG2 = MixtureModel([
    Uniform(0, 0.25),
    Uniform(0.125, 0.25),
    Uniform(0.5, 1.0)], [1/6, 1/2, 1/3])
DG3 = truncated(Normal(0.5, 0.1), 0.0, 1.0)
DG = [DG1, DG2, DG3]

TRUE_DENS = [pdf.(d, XGRID) for d in DG]

# Tuples of "model-id" and "alpha0". For all models max_depth is 10
# and beta0 = alpha0
MAX_DEPTH = 10
PRIORS = [
    ("PT", 0.05),
    ("PT", 0.1),
    ("PT", 2.0),
    ("PT", 10.0),
    ("GFPT1", 0.05),
    ("GFPT1", 0.1),
    ("GFPT1", 2.0),
    ("GFPT1", 10.0),
]

function build_model(model_id, alpha0)
    if model_id == "PT"
        return PolyaTree(MAX_DEPTH, alpha0)
    elseif model_id == "GFPT1"
        return GFPT1(Poisson(5.0), MAX_DEPTH, alpha0)
    else 
        throw("Non-reckognized model_id: '$model_id' ")
    end
end


function run_one_iter(iternum)
    out = []

    for prior in PRIORS
        for (i, data_distribution) in enumerate(DG)
            for ndata in NDATAS
                model = build_model(prior[1], prior[2])
                data = rand(data_distribution, ndata)
                model = update(data, model)
                pred_dens = predictive_density(XGRID, model)

                # Compute summaries
                l1 = l1_dist(TRUE_DENS[i], pred_dens, XGRID)
                hell = hell_dist(TRUE_DENS[i], pred_dens, XGRID)
                length = arclength(TRUE_DENS[i] - pred_dens, XGRID)

                post_n = -1
                mode_n = -1
                if prior[1] != "PT"
                    post_n = sum(collect(1:MAX_DEPTH) .* model.prob_n)
                    mode_n = argmax(model.prob_n)
                end

                out = push!(
                    out,
                    (prior[1], prior[2], i, ndata, l1, hell, length, post_n, mode_n, iternum)
                )
            end
        end
    end
    return DataFrame(out, ["Model", "alpha0", "DataGen", "Ndata", "L1", "HELL", "LENGTH", "POST_N", "MODE_N", "ITER"])
end

function main()
    tmp =  Array{DataFrame}(undef, NREP)
    @Threads.threads for i in ProgressBar(1:NREP)
        tmp[i] = run_one_iter(i)
    end
    out = reduce(vcat, tmp)
    Serialization.serialize("simulation1_results.dta", out)
end

main()