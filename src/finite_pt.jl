include("partitions.jl")
include("binary_tree.jl")
include("utils.jl")

using AddPackage
@add using Integrals
@add using StatsBase
@add using Distributions
@add using SpecialFunctions
@add using NNlib
@add using IterTools

"""
    PolyaTree

Finite Polya tree prior used for density estimation.

# Fields
- `partition::NestedPartitions`: partition structure on `[0,1]`.
- `base::Int`: branching factor of the tree.
- `alphas::Array{Array{Float64,1},1}`: Dirichlet parameters for each level.
- `counts::Array{Array{Float64,1},1}`: observed counts per partition cell.
"""
mutable struct PolyaTree
    # state
    partition::NestedPartitions
    base::Int
    # params
    alphas::Array{Array{Float64,1},1}
    counts::Array{Array{Float64,1},1}
end


# Constructors

"""
    PolyaTree(p::NestedPartitions, alpha0=1, base=2)

Create a `PolyaTree` on the provided `NestedPartitions` object.  The array of
Dirichlet parameters at level `l` is initialised to `alpha0 * l^2` for each
cell and the count arrays are set to zero.
"""
function PolyaTree(p::NestedPartitions, alpha0=1, base=2)
    a = Array{Array{Float64,1},1}()
    c = Array{Array{Float64,1},1}()
    for l in 1:p.depth
        curr = ones(Int(base^l)) .* l^2
        a = push!(a, curr * alpha0)
        c = push!(c, curr * 0.0)
    end
    return PolyaTree(p, base, a, c)
end

"""
    PolyaTree(depth::Int64, alpha0=1, base=2)

Construct a `PolyaTree` with `depth` levels and uniform partitions of the
unit interval.  This is a convenience wrapper around
`PolyaTree(NestedPartitions(...))`.
"""
function PolyaTree(depth::Int64, alpha0=1, base=2)

    partitions = Vector{SimplePartition}()
    for d in 1:depth
        endpoints = collect(LinRange(0.0, 1.0, Int(base^d + 1)))
        partitions = push!(partitions, SimplePartition(endpoints))
    end
    partitions = NestedPartitions(partitions)
    return PolyaTree(partitions, alpha0, base)
end


"""
    BenfordPT(depth::Int64, alphas::Vector{Float64}, base::Int64=10)

Construct a `PolyaTree` whose prior probabilities follow Benford's law for the
first digit.  The vector `alphas` controls the strength of the prior at each
level and must have length equal to `depth`.
"""
function BenfordPT(depth::Int64, alphas::Vector{Float64}, base::Int64=10)
    if (length(alphas) != depth)
        throw("Length of alphas must be equal to depth")
    end


    partitions = Vector{SimplePartition}()
    for d in 1:depth
        endpoints = collect(LinRange(0.0, 1, base^d + 1))
        partitions = push!(partitions, SimplePartition(endpoints))
    end
    partitions = NestedPartitions(partitions)

    bendford_probas = joint_benford_probas(depth, base)
    a = Array{Array{Float64,1},1}()
    a = push!(a, bendford_probas[1] * alphas[1])

    c = Array{Array{Float64,1},1}()
    c = push!(c, zeros(base))

    # for l in 2:depth
    #     cond_probas = zeros(length(bendford_probas[l]))
    #     for i in 1:length(cond_probas)
    #         den_proba = bendford_probas[l-1][Int(ceil(i / base))]
    #         num_proba = bendford_probas[l][i]
    #         if (num_proba > 0) & (den_proba > 0)
    #             cond_probas[i] = num_proba / den_proba
    #         end
    #     end
    #     a = push!(a, cond_probas .* alphas[l])
    #     c = push!(c, zeros(base^l))
    # end

    for l in 2:depth
        cond_probas = zeros(length(bendford_probas[l]))
        for b in 1:base:base^(l)
            batch = bendford_probas[l][b:(b+base-1)]
            den_proba = sum(batch)
            if den_proba > 0
                cond_probas[b:(b+base-1)] .= batch ./ den_proba
            end
        end
        a = push!(a, cond_probas .* alphas[l])
        c = push!(c, zeros(base^l))
    end
    return PolyaTree(partitions, base, a, c)
end


"""
    joint_benford_probas(num_digits, base::Int64=10)

Return a vector of arrays containing the joint Benford probabilities for
`num_digits` digits in the given `base`.
"""
function joint_benford_probas(num_digits, base::Int64=10)
    out = Array{Array{Float64,1},1}()
    for l in 1:num_digits
        out = push!(out, zeros(base^l))
    end

    base10_digits = collect(0:base-1)
    for depth in 1:num_digits
        powers = (1.0 * base) .^ collect((depth-1):-1:0)
        for digits in product(repeat([base10_digits], depth)...)
            if digits[1] == 0
                continue 
            end
            base10_num = sum(digits .* powers)
            joint_proba = log(base, 1 + 1.0 / base10_num)
            idx = Int(sum(digits .* powers)) + 1
            out[depth][idx] = joint_proba
        end 
    end
    return out
end


# POSTERIOR

"""
    update(data::Vector{Float64}, pt::PolyaTree)

Return the posterior `PolyaTree` after observing the vector of data points
`data`.
"""
function update(data::Vector{Float64}, pt::PolyaTree)
    pt_to_interval = mapreduce(
        permutedims, vcat, find_intervals.(data, Ref(pt.partition)))
     
    new_alphas = pt.alphas
    new_counts = pt.counts
    for l in 1:pt.partition.depth
        counts = countmap(pt_to_interval[:, l])
        for (idx, c) in counts
            new_alphas[l][idx] += c
            new_counts[l][idx] += c
        end
    end
    return PolyaTree(pt.partition, pt.base, new_alphas, new_counts)
end

# PREDICTIVE

"""
    get_alpha(pt::PolyaTree, binseq::Vector{Int})

Retrieve the Dirichlet parameter corresponding to a path `binseq` in the tree.
"""
function get_alpha(pt::PolyaTree, binseq::Vector{Int})
    return get_from_binseq(pt.alphas, binseq, pt.base)
end

"""
    get_next_alphas(pt::PolyaTree, binseq::Vector{Int})

Return the Dirichlet parameters of the children of the node specified by
`binseq`.
"""
function get_next_alphas(pt::PolyaTree, binseq::Vector{Int})
    out = [get_from_binseq(pt.alphas, [binseq; i], pt.base) for i in 0:(pt.base-1)]
    return out
end

"""
    get_length(pt::PolyaTree, binseq::Vector{Int})

Length of the interval associated with `binseq` in the partition.
"""
function get_length(pt::PolyaTree, binseq::Vector{Int})
    l =  get_from_binseq(pt.partition.lengths, binseq, pt.base)
    return l
end


"""
    predictive_density(xgrid::Vector{Float64}, pt::PolyaTree)

Evaluate the predictive density of `pt` on the grid `xgrid`.  The grid must be
sorted and equispaced.
"""
function predictive_density(xgrid::Array{Float64}, pt::PolyaTree)
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
    out[1] = dens_cumprod[end] / get_length(pt, curr_binseq)

    old_binseq = curr_binseq
    for i in 2:length(xgrid)
        curr_binseq = digits(idxs[i][end] - 1, base=pt.base, pad=m) |> reverse
        if (curr_binseq == old_binseq)
            out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
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
        out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
        old_binseq = curr_binseq
    end
    return out
end


"""
    sample_pt_density(xgrid, pt)

Draw a random density from the Polya tree prior and evaluate it on `xgrid`.
"""
function sample_pt_density(xgrid, pt)
    probas = sample_dirichlet_sequence(pt.alphas)

    idxs = find_intervals.(xgrid, Ref(pt.partition))
    out = zeros(length(xgrid))
    m = pt.partition.depth

    dens_cumprod = zeros(m)
    curr_binseq = digits(idxs[1][end] - 1, base=pt.base, pad=m) |> reverse
    
    dens_cumprod[1] = get_from_binseq(probas, curr_binseq[1:1], pt.base)
    
    for j in 2:m
        dens_cumprod[j] = dens_cumprod[j - 1] * get_from_binseq(probas, curr_binseq[1:j], pt.base)
    end
    out[1] = dens_cumprod[end] / get_length(pt, curr_binseq)

    old_binseq = curr_binseq
    for i in 2:length(xgrid)
        curr_binseq = digits(idxs[i][end] - 1, base=pt.base, pad=m) |> reverse
        if (curr_binseq == old_binseq)
            out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
            continue
        end

        first_diff = argmin(curr_binseq .== old_binseq)
        if first_diff == 1
            dens_cumprod[1] = get_from_binseq(probas, curr_binseq[1:1], pt.base)
            first_diff = 2
        end

        for j in first_diff:m
            dens_cumprod[j] = dens_cumprod[j - 1] * get_from_binseq(probas, curr_binseq[1:j], pt.base)
        end
        out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
        old_binseq = curr_binseq
    end
    return out
end
