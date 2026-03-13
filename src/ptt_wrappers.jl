using RCall

R"""
if (!requireNamespace("remotes", quietly=TRUE))
  install.packages("remotes", repos="https://cloud.r-project.org")
if (!requireNamespace("PTT", quietly=TRUE)) {
  remotes::install_github("MaStatLab/PTT")
}
library(PTT)
"""

# APT predictive density: apt(X=..., Xpred=..., max.resol=..., rho0=0.2)$pred
apt_pred_density(X::AbstractVector{<:Real}, grid::AbstractVector{<:Real};
                 max_resol::Int=8, rho0::Real=0.2)::Vector{Float64} = begin
    R"""
    res  <- apt(X = $X, Xpred = $grid, max.resol = $max_resol, rho0 = $rho0)
    pred <- as.numeric(res$pred)
    """
    rcopy(Vector{Float64}, R"pred")
end

# OPT predictive density (same calling shape)
opt_pred_density(X::AbstractVector{<:Real}, grid::AbstractVector{<:Real};
                 max_resol::Int=8, rho0::Real=0.2)::Vector{Float64} = begin
    R"""
    res  <- opt(X = $X, Xpred = $grid, max.resol = $max_resol, rho0 = $rho0)
    pred <- as.numeric(res$pred)
    """
    rcopy(Vector{Float64}, R"pred")
end