include("finite_pt.jl")
include("utils.jl")

mutable struct GFPT1
    pt::PolyaTree
    prob_n::Vector{Float64}
end


# COSTRUCTORS

function GFPT1(p::NestedPartitions, prior_n::Vector{Float64}, 
              alpha0=1)
    @assert(length(p) == length(prior_n))
    pt = PolyaTree(p, alpha0)
    return GFPT1(pt, prior_n)
end

function GFPT1(prior_n::Distribution, max_n=Int64,
               alpha0=1)
    depth = max_n
    partitions = Vector{SimplePartition}()
    prob_n = zeros(max_n)
    for d in 1:depth
        endpoints = collect(LinRange(0, 1, 2^d + 1))
        partitions = push!(partitions, SimplePartition(endpoints))
        prob_n[d] = pdf(prior_n, d)
    end
    prob_n ./= sum(prob_n)
    partitions = NestedPartitions(partitions)
    pt = PolyaTree(partitions, alpha0)
    return GFPT1(pt, prob_n)
end

# POSTERIOR

function update(data::Vector{Float64}, pt::GFPT1)
    post_tree = update(data, pt.pt)

    m = post_tree.partition.depth
    log_probas_n = zeros(m)
    log_numerator = 0

    for i in 1:m
        log_denominator = 0
        curr_alphas = post_tree.alphas[i]
        for j in 1:2:length(curr_alphas)
            log_numerator +=  logbeta(curr_alphas[j], curr_alphas[j+1]) 
        end
        log_denominator = sum(log.(post_tree.partition.lengths[i]) .* post_tree.counts[i]) 
        log_probas_n[i] = log(pt.prob_n[i]) + log_numerator - log_denominator 
    end

    return GFPT1(post_tree, softmax(log_probas_n))
end

# PREDICTIVE

function get_nested_lengths(pt::PolyaTree, binseq::Array{Int})
    return get_length.(Ref(pt), get_subarrays(binseq))
end


function predictive_density(xgrid::Array{Float64}, gfpt::GFPT1, proba_threshold=1e-6)
    """
    We assume xgrid is already sorted and equispaced
    """
    pt = gfpt.pt
    prob_n = gfpt.prob_n

    # TODO @Mario: reduce compute time by truncating N with proba_threshold

    idxs = find_intervals.(xgrid, Ref(pt.partition))
    out = zeros(length(xgrid))
    m = pt.partition.depth

    dens_cumprod = zeros(m)
    curr_binseq = digits(idxs[1][end] - 1, base=2, pad=m) |> reverse
    
    dens_cumprod[1] = get_alpha(pt, curr_binseq[1:1]) / (
        get_alpha(pt, [0]) + get_alpha(pt, [1]) )
    for j in 2:m
        dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / (
            get_alpha(pt, push!(curr_binseq[1:j-1], 0)) + get_alpha(pt, push!(curr_binseq[1:j-1], 1))
        )
    end
    # out[1] = dens_cumprod[end] / get_length(pt, curr_binseq)
    out[1] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))

    old_binseq = curr_binseq
    for i in 2:length(xgrid)
        curr_binseq = digits(idxs[i][end] - 1, base=2, pad=m) |> reverse
        if (curr_binseq == old_binseq)
            # out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
            out[i] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))
            continue
        end

        first_diff = argmin(curr_binseq .== old_binseq)
        if first_diff == 1
            dens_cumprod[1] = get_alpha(pt, curr_binseq[1:1]) / (
                get_alpha(pt, [0]) + get_alpha(pt, [1]) )
            for j in 2:m
                dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / (
                    get_alpha(pt, push!(curr_binseq[1:j-1], 0)) + get_alpha(pt, push!(curr_binseq[1:j-1], 1))
                )
            end
        else 
            for j in first_diff:m
                dens_cumprod[j] = dens_cumprod[j - 1] * get_alpha(pt, curr_binseq[1:j]) / (
                get_alpha(pt, push!(curr_binseq[1:j-1], 0)) + get_alpha(pt, push!(curr_binseq[1:j-1], 1))
            )
            end
        end
        # out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
        out[i] = sum(prob_n .* dens_cumprod ./ get_nested_lengths(pt, curr_binseq))
        old_binseq = curr_binseq
    end
    return out
    # return out / sum(out * (xgrid[2] - xgrid[1]))
end
