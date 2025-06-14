include("utils.jl")
include("gfpt1.jl")


mutable struct MultiscaleBPT
    tree::GFPT1
    etas::Vector{Float64}
    min_order::Int64
    max_order::Int64
    base::Int64
end


function MultiscaleBPT(
        prior_n::Distribution, max_n::Int64, alpha0::Float64, 
        etas::Vector{Float64}, min_order=0, base=10)

    alphas = Vector{Float64}()
    for l in 1:max_n
        alphas = push!(alphas, alpha0 * l^2)
    end

    return MultiscaleBPT(
        BenfordGFPT1(prior_n, max_n, alphas, base), 
        etas, 
        min_order, 
        min_order + length(etas) - 1,
        base)
end


function update(data::Vector{Float64}, pt::MultiscaleBPT)
    scaled_data, magnitudes = scale_and_magnitudes(data, pt.base)

    eta_post = copy(pt.etas)
    for order in pt.min_order:pt.max_order
        cnt = sum(magnitudes .== order)
        eta_post[order - pt.min_order + 1] += cnt
    end

    pt_post = update(scaled_data, pt.tree)

    return MultiscaleBPT(pt_post, eta_post, pt.min_order, pt.max_order, pt.base)
end


function sample_log_lik(data::Array{Float64}, bpt::MultiscaleBPT)
    scaled_data, magnitudes = scale_and_magnitudes(data, bpt.base)

    order_probas = rand(Dirichlet(bpt.etas))
    order_log_liks = logpdf.(
        Categorical(order_probas), 
        magnitudes .- bpt.min_order .+ 1);

    perm = sortperm(scaled_data)
    inverse_perm = invperm(perm)

    scaled_sorted_data = scaled_data[perm]
    pt_log_liks_sorted = log.(
        sample_pt_density(scaled_sorted_data, bpt.tree))

    out = order_log_liks .+ pt_log_liks_sorted[inverse_perm] .- (magnitudes .+ 1) .* log(bpt.base)
    return out
end

function log_lik_chain(data::Vector{Float64}, pt::MultiscaleBPT, N_MC=1000)
    out = zeros(N_MC, length(data))
    for (i, pt) in enumerate(pt_chain)
        out[i, :] .=  sample_log_lik(data, pt)
    end

    return out
end