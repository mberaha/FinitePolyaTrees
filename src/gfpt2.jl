using AddPackage
@add using SpecialFunctions

include("finite_pt.jl")
include("utils.jl")



mutable struct GFPT2
    partition_relative_lengths::Vector{Vector{Float64}}
    partition_endpoints::Vector{Vector{Float64}}
    partition_lengths::Vector{Vector{Float64}}
    counts::Array{Array{Float64,1},1}

    prior_prob_n::Vector{Float64}
    prior_alphas::Vector{Vector{Float64}}
    prior_betas::Vector{Vector{Float64}}

    curr_alphas::Vector{Vector{Float64}}
    curr_n::Int64
end

function GFPT2(partition_relative_lengths, partition_endpoints, partition_lengths, counts,
              prob_n, prior_alphas, prior_betas)

    alphas = Vector{Vector{Float64}}()
    curr_n = argmax(prob_n)
    return GFPT2(partition_relative_lengths, partition_endpoints, partition_lengths, counts,
                 prob_n, prior_alphas, prior_betas, alphas, curr_n)

end


# COSTRUCTORS

function GFPT2(prior_n::Distribution, max_n=Int64,
               alpha0=1, beta0=1, increasing_beta=false)

    depth = max_n
    
    a = Array{Array{Float64,1},1}()
    b = Array{Array{Float64,1},1}() 
    c = Array{Array{Float64,1},1}()

    prob_n = zeros(max_n)
    prob_n[1] = pdf(prior_n, 1)
    a = push!(a, ones(2) * alpha0)
    b = push!(b, ones(2) * beta0)
    c = push!(c, ones(2) * 0)


    for d in 2:depth
        prob_n[d] = pdf(prior_n, d)
        curr = ones(2^d) # * d^2
        a = push!(a, curr * d^2 * alpha0)
        if increasing_beta
            b = push!(b, curr * d^2  * beta0)
        else
            b = push!(b, curr * beta0)
        end
        c = push!(c, curr * 0.0)
    end
    prob_n ./= sum(prob_n)

    partition_rel_lengths = sample_beta_sequence(b)
    partition_lengths = lengths_from_relative_lengths(partition_rel_lengths)
    partition_endpoints = endpoints_from_lengths(partition_lengths)

    return GFPT2(partition_rel_lengths, partition_endpoints, partition_lengths, c,
                 prob_n, a, b)
end

# POSTERIOR

function update_counts_and_alphas(data, endpoints, pt)
    function intervals_from_endpoints(x, endpoints)
        depth = length(endpoints)
        idx = find_interval(x, endpoints[end])
        d = digits(idx - 1, base=2, pad=depth) |> reverse
        out = Vector{Int}(undef, depth)
        for l in 1:depth
            out[l] = bit_to_int(d[1:l]) + 1
        end
        return out
    end

    pt_to_interval = mapreduce(
        permutedims, vcat, 
        intervals_from_endpoints.(data, Ref(endpoints)))

    depth = length(endpoints)
    new_alphas = deepcopy(pt.prior_alphas[1:depth])
    new_counts = pt.counts[1:depth] .* 0
    for l in 1:depth
        counts = countmap(pt_to_interval[:, l])
        for (idx, c) in counts
            new_alphas[l][idx] += c
            new_counts[l][idx] += c
        end
    end
    return new_counts, new_alphas

end

function sample_n!(data, pt::GFPT2)
    depth = length(pt.prior_prob_n)
    
    log_probas_n = zeros(depth)
    log_numerator = 0

    for i in 1:depth
        log_denominator = 0
        curr_alphas = pt.curr_alphas[i]
        for j in 1:2:length(curr_alphas)
            log_numerator +=  logbeta(curr_alphas[j], curr_alphas[j+1]) 
        end
        log_denominator = sum(log.(pt.partition_lengths[i]) .* pt.counts[i]) 
        log_probas_n[i] = log(pt.prior_prob_n[i]) + log_numerator - log_denominator 
    end

    pt.curr_n = rand(Categorical(softmax(log_probas_n)))
    return pt
end

function marg_log_lik(alphas, lengths, counts)
    log_num = 0
    depth = length(alphas)

    for d in 1:depth
        curr_alphas = alphas[d]
        for j in 1:2:length(curr_alphas)
            log_num +=  logbeta(curr_alphas[j], curr_alphas[j+1]) 
        end
    end
    log_den = sum(log.(lengths[end]) .* counts[end]) 
    return log_num - log_den
end

function sample_r!(data, pt::GFPT2)
    max_n = length(pt.prior_prob_n)
    curr_n = pt.curr_n

    prop_rel_lenghts = sample_beta_sequence(pt.prior_betas[1:curr_n])
    prop_lenghts = lengths_from_relative_lengths(prop_rel_lenghts)
    prop_endpoints = endpoints_from_lengths(prop_lenghts)
    prop_counts, prop_alphas = update_counts_and_alphas(data, prop_endpoints, pt)

    log_a_rate = (marg_log_lik(prop_alphas, prop_lenghts, prop_counts) -
        marg_log_lik(pt.curr_alphas[1:curr_n], pt.partition_lengths[1:curr_n], 
                        pt.counts[1:curr_n]))

    if log(rand()) < log_a_rate
        pt.counts[1:curr_n] = prop_counts
        pt.curr_alphas[1:curr_n] = prop_alphas
        pt.partition_lengths[1:curr_n] = prop_lenghts
        pt.partition_relative_lengths[1:curr_n] = prop_rel_lenghts
        pt.partition_endpoints[1:curr_n] = prop_endpoints
    end

    for d in (curr_n+1):max_n
        r = [rand(Beta(
                pt.prior_betas[d][i], pt.prior_betas[d][i+1])) for i in 1:2:2^d]
        pt.partition_relative_lengths[d] .= r
    end

    pt.partition_lengths = lengths_from_relative_lengths(pt.partition_relative_lengths)
    pt.partition_endpoints = endpoints_from_lengths(pt.partition_lengths)
    pt.counts, pt.curr_alphas = update_counts_and_alphas(data, pt.partition_endpoints, pt)

    return pt
end

function run_mcmc(
        data, pt, burnin, iter, thin=1, xgrid=collect(LinRange(1e-8, 1-1e-8, 1000)))

    pt.counts, pt.curr_alphas = update_counts_and_alphas(data, pt.partition_endpoints, pt)

    for _ in 1:burnin
        pt = sample_r!(data, pt);
        pt = sample_n!(data, pt);
    end


    out = []

    for iternum in 1:iter
        pt = sample_r!(data, pt);
        pt = sample_n!(data, pt);
        if iternum % thin == 0
            out = push!(out, deepcopy(pt))
        end
    end
    return out
end


# PREDICTIVE
function predictive_density(xgrid::Array{Float64}, gfpt::GFPT2)
    """
    We assume xgrid is already sorted and equispaced
    """

    n = gfpt.curr_n
    partitions = NestedPartitions(
        [SimplePartition(gfpt.partition_endpoints[depth]) for depth in 1:n])
    simple_pt = PolyaTree(partitions, gfpt.curr_alphas[1:n], gfpt.counts[1:n])

    return predictive_density(xgrid, simple_pt)
end


function predictive_density(xgrid::Array{Float64}, pt_chain=Vector{GFPT2})
    pred_dens = zeros(length(xgrid))

    for pt in pt_chain
        pred_dens .+= predictive_density(xgrid, pt)
    end
    pred_dens ./= length(pt_chain)
end
