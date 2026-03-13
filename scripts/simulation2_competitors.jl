# competitors_ptt_sim.jl
# Run OPT / APT predictive densities (PTT) on your synthetic datasets.

# --- paths / includes ---------------------------------------------------------
# assumes this file lives in, e.g., scripts/ and wrappers in ../src/ptt_wrappers.jl
include("../src/polyatree.jl")
include("../src/partitions.jl")
include("simulation_utils.jl")

using AddPackage
using Random
@add using Distributions
@add using DataFrames
@add using Serialization
@add using RCall

using ProgressBars: ProgressBar

# --- ensure R-side PTT is available ------------------------------------------
R"""
if (!requireNamespace("remotes", quietly=TRUE))
  install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("PTT", quietly=TRUE)) {
  remotes::install_github("MaStatLab/PTT")
}
library(PTT)
"""

# --- import your two wrapper functions ----------------------------------------
# They must be defined in src/ptt_wrappers.jl exactly as you posted:
#   apt_pred_density(X, grid; max_resol=8, rho0=0.2)
#   opt_pred_density(X, grid; max_resol=8, rho0=0.2)
# If they live inside a module, `using .ThatModule` here.

# --- experiment settings ------------------------------------------------------
const NREP    = 96
const NDATAS  = [100, 500, 1000]                       # can add more, e.g. [200, 500, 1000]
const XGRID   = collect(LinRange(1e-8, 1 - 1e-8, 1000))
const Δx      = XGRID[2] - XGRID[1]

Random.seed!(20241004)
const SEEDS = rand(1:10_000_000, NREP)

# data generators (on [0,1])
DG1 = Uniform(0, 0.5)
DG2 = MixtureModel([Uniform(0,0.25), Uniform(0.125,0.25), Uniform(0.5,1.0)], [1/6, 1/2, 1/3])
DG3 = truncated(Normal(0.5, 0.1), 0.0, 1.0)
DG4 = Uniform(0, 0.2)
DG5 = MixtureModel([Uniform(0,0.2), Uniform(0.7,0.9)], [2/3, 1/3])
DG6 = MixtureModel([Beta(2,15), Beta(15,2)], [1/2, 1/2])
const DG  = [DG1, DG2, DG3, DG4, DG5, DG6]
const TRUE_DENS = [pdf.(d, XGRID) for d in DG]

# Which competitor settings to run (edit/extend as you wish)
# Keep it backbone-simple: one setting per method to start.
const SETTINGS = [
    ("OPT", 12, 0.2),
    ("APT", 12, 0.2),
]

# --- small API to call wrappers -----------------------------------------------
function run_competitor(method, X::Vector{Float64}, grid::Vector{Float64};
                        max_resol::Int, rho0::Float64)
    m = String(method)
    if m == "OPT"
        return opt_pred_density(X, grid; max_resol=max_resol, rho0=rho0)
    elseif m == "APT"
        return apt_pred_density(X, grid; max_resol=max_resol, rho0=rho0)
    else
        error("Unknown competitor method: $method")
    end
end

# --- one iteration ------------------------------------------------------------
function run_one_iter(iternum::Int)
    println("Run iter #", iternum); flush(stdout)

    # collect NAMED tuples to avoid type-assert errors
    out = NamedTuple[]

    # make R’s RNG deterministic per-iteration (optional)
    R"set.seed($(SEEDS[iternum]))"

    for (method, max_resol, rho0) in SETTINGS
        for (i, dgen) in enumerate(DG)
            for ndata in NDATAS
                try
                    # seed & sample
                    Random.seed!(SEEDS[iternum] + i + ndata)
                    X = rand(dgen, ndata)

                    pred = run_competitor(method, X, XGRID; max_resol=max_resol, rho0=rho0)

                    # sanity check: integral
                    integral = sum(pred) * Δx
                    if !isfinite(integral) || abs(integral - 1.0) > 0.1
                        @warn "Skipping (bad integral) method=$(method) DG=$i n=$ndata ∫≈$integral"
                        continue
                    end

                    # metrics
                    l1          = l1_dist(TRUE_DENS[i], pred, XGRID)
                    curve_len   = arclength(TRUE_DENS[i] - pred, XGRID)

                    push!(out, (
                        Competitor = String(method),
                        max_resol  = max_resol,
                        rho0       = rho0,
                        DataGen    = i,
                        Ndata      = ndata,
                        L1         = l1,
                        LENGTH     = curve_len,
                        ITER       = iternum
                    ))
                catch e
                    @warn "Iter $iternum failed: method=$(method) DG=$i n=$ndata — $(e)"
                    continue
                end
            end
        end
    end

    # Safe DataFrame even if out is empty
    if isempty(out)
        return DataFrame(Competitor = String[],
                         max_resol  = Int[],
                         rho0       = Float64[],
                         DataGen    = Int[],
                         Ndata      = Int[],
                         L1         = Float64[],
                         LENGTH     = Float64[],
                         ITER       = Int[])
    else
        return DataFrame(out)
    end
end

# --- main (sequential; RCall is not thread-safe) ------------------------------
function main()
    tmp = Vector{DataFrame}(undef, NREP)
    for i in ProgressBar(1:NREP)
        tmp[i] = run_one_iter(i)
    end
    out = reduce(vcat, tmp)
    Serialization.serialize("simulation2_competitors_results.dta", out)
end

main()
