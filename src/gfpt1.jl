include("finite_pt.jl")
include("utils.jl")

"""
    GFPT1

Gaussian finite Polya tree where the depth is random.

# Fields
- `pt::PolyaTree` : underlying finite Polya tree.
- `prob_n::Vector{Float64}` : prior probabilities for the number of levels.
"""
mutable struct GFPT1
    pt::PolyaTree
    prob_n::Vector{Float64}
end


# COSTRUCTORS

"""
    GFPT1(p::NestedPartitions, prior_n::Vector{Float64}; alpha0=1, base=2)

Create a `GFPT1` model with a fixed partition `p` and prior probabilities
`prior_n` on the depth.
"""
function GFPT1(p::NestedPartitions, prior_n::Vector{Float64},
              alpha0=1, base=2)
    @assert(length(p) == length(prior_n))
    pt = PolyaTree(p, alpha0, base)
    return GFPT1(pt, prior_n)
end

"""
    GFPT1(prior_n::Distribution, max_n::Int64; alpha0=1, base=2)

Construct a `GFPT1` where the partition has at most `max_n` levels and the
prior on the depth is given by `prior_n`.
"""
function GFPT1(prior_n::Distribution, max_n=Int64,
               alpha0=1, base=2)
    depth = max_n
    partitions = Vector{SimplePartition}()
    prob_n = zeros(max_n)
    for d in 1:depth
        endpoints = collect(LinRange(0, 1, base^d + 1))
        partitions = push!(partitions, SimplePartition(endpoints))
        prob_n[d] = pdf(prior_n, d)
    end
    prob_n ./= sum(prob_n)
    partitions = NestedPartitions(partitions)
    pt = PolyaTree(partitions, alpha0, base)
    return GFPT1(pt, prob_n)
end


"""
    BenfordGFPT1(prior_n::Distribution, max_n::Int64,
                 alphas::Vector{Float64}, base::Int64)

Construct a `GFPT1` where the underlying Polya tree is initialised to follow
Benford's law.
"""
function BenfordGFPT1(prior_n::Distribution, max_n::Int64, alphas::Vector{Float64},
                      base::Int64)

    pt = BenfordPT(max_n, alphas, base)

    prob_n = zeros(max_n)
    for d in 1:max_n
        prob_n[d] = pdf(prior_n, d)
    end
    prob_n ./= sum(prob_n)

    return GFPT1(pt, prob_n)
end

# POSTERIOR

"""
    update(data::Vector{Float64}, pt::GFPT1)

Update the `GFPT1` object with the observations in `data`.
"""
function update(data::Vector{Float64}, pt::GFPT1)
    post_tree = update(data, pt.pt)
    base = pt.pt.base
    m = post_tree.partition.depth
    log_probas_n = zeros(m)
    log_probas_num = zeros(m)
    log_probas_den = zeros(m)


    log_numerator = 0
    for i in 1:m
        log_denominator = 0
        curr_alphas = post_tree.alphas[i]
        for j in 1:base:length(curr_alphas)
            tmp =  curr_alphas[j:(j+base-1)]
            if all(tmp .== 0)
                continue
            else 
                tmp = tmp[tmp .> 0]
            end
            # println("alphas: ", tmp, ", contrib: ", log_mv_beta(tmp))
            log_numerator += log_mv_beta(tmp) 
        end
        log_denominator = sum(log.(post_tree.partition.lengths[i]) .* post_tree.counts[i]) 
        log_probas_n[i] = log(pt.prob_n[i]) + log_numerator - log_denominator 
        log_probas_num[i] = log_numerator
        log_probas_den[i] = log_denominator
    end
    return GFPT1(post_tree, softmax(log_probas_n))
end

# PREDICTIVE

"""
    get_nested_lengths(pt::PolyaTree, binseq::Vector{Int})

Return the lengths of all sub-intervals corresponding to prefixes of
`binseq`.
"""
function get_nested_lengths(pt::PolyaTree, binseq::Array{Int})
    return get_length.(Ref(pt), get_subarrays(binseq))
end


"""
    predictive_density(xgrid::Vector{Float64}, gfpt::GFPT1; proba_threshold=1e-6)

Predictive density of the GFPT1 model evaluated on `xgrid`.
`xgrid` must be sorted.
"""
function predictive_density(xgrid::Array{Float64}, gfpt::GFPT1, proba_threshold=1e-6)
    pt = gfpt.pt
    prob_n = gfpt.prob_n

    # TODO @Mario: reduce compute time by truncating N with proba_threshold

    idxs = find_intervals.(xgrid, Ref(pt.partition))
    out = zeros(length(xgrid))
    m = pt.partition.depth

    dens_cumprod = zeros(m)
    curr_binseq = digits(idxs[1][end] - 1, base=pt.base, pad=m) |> reverse
    
    dens_cumprod[1] = get_alpha(pt, curr_binseq[1:1]) / sum(
        get_next_alphas(pt, Int[]) )
    for j in 2:m
        dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / sum(
            get_next_alphas(pt, curr_binseq[1:j-1]))
    end
    # out[1] = dens_cumprod[end] / get_length(pt, curr_binseq)
    out[1] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))

    old_binseq = curr_binseq
    for i in 2:length(xgrid)
        curr_binseq = digits(idxs[i][end] - 1, base=pt.base, pad=m) |> reverse
        if (curr_binseq == old_binseq)
            # out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
            out[i] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))
            continue
        end

        first_diff = argmin(curr_binseq .== old_binseq)
        if first_diff == 1
            dens_cumprod[1] = get_alpha(pt, curr_binseq[1:1]) / sum(
                get_next_alphas(pt, Int[]))
            for j in 2:m
                dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / sum(
                    get_next_alphas(pt, curr_binseq[1:j-1]))
            end
        else 
            for j in first_diff:m
                dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / sum(
                    get_next_alphas(pt, curr_binseq[1:j-1]))
            end
        end
        # out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
        out[i] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))
        old_binseq = curr_binseq
    end
    return out
    # return out / sum(out * (xgrid[2] - xgrid[1]))
end


"""
    sample_pt_density(xgrid::Vector{Float64}, gfpt::GFPT1)

Sample a random density from the posterior `gfpt` and evaluate it on `xgrid`.
"""
function sample_pt_density(xgrid::Array{Float64}, gfpt::GFPT1)
    depth = rand(Categorical(gfpt.prob_n))


    trunc_partition = NestedPartitions(
        depth, 
        gfpt.pt.partition.base, 
        gfpt.pt.partition.levels[1:depth],
        gfpt.pt.partition.lengths[1:depth],
    )

    pt = PolyaTree(
        trunc_partition,
        gfpt.pt.base,
        gfpt.pt.alphas[1:depth],
        gfpt.pt.counts[1:depth],
        )

    return sample_pt_density(xgrid, pt)
end


"""
    sample_log_lik(data::Vector{Float64}, gfpt::GFPT1)

Draw one density from `gfpt` and return the log-likelihood of `data`.
"""
function sample_log_lik(data::Vector{Float64}, gfpt::GFPT1)
    perm = sortperm(data)
    inverse_perm = invperm(perm)
    sorted_data = data[perm]

    log_liks = log.(sample_pt_density(sorted_data, gfpt))

    return log_liks[inverse_perm]
end


"""
    log_lik_chain(data::Vector{Float64}, gfpt::GFPT1, N_MC=1000)

Monte Carlo estimate of the log-likelihood over `N_MC` posterior draws.
"""
function log_lik_chain(data::Vector{Float64}, gfpt::GFPT1, N_MC=1000)
    out = zeros(N_MC, length(data))
    for (i, pt) in enumerate(pt_chain)
        out[i, :] .=  sample_log_lik(data, gfpt)
    end

    return out
end

