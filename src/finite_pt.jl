include("partitions.jl")
include("binary_tree.jl")
include("utils.jl")

using AddPackage
@add using Integrals
@add using StatsBase
@add using Distributions
@add using SpecialFunctions
@add using NNlib

mutable struct PolyaTree
    # state
    partition::NestedPartitions

    # params
    alphas::Array{Array{Float64,1},1}
    counts::Array{Array{Float64,1},1}
end

# Constructors

function PolyaTree(p::NestedPartitions, alpha0=1)
    a = Array{Array{Float64,1},1}()
    b = Array{Array{Float64,1},1}()
    c = Array{Array{Float64,1},1}()
    for l in 1:p.depth
        curr = ones(2^l) * l^2
        a = push!(a, curr * alpha0)
        c = push!(c, curr * 0.0)
    end
    return PolyaTree(p, a, c)
end


function PolyaTree(depth::Int64, alpha0=1)
    partitions = Vector{SimplePartition}()
    for d in 1:depth
        endpoints = collect(LinRange(0, 1, 2^d + 1))
        partitions = push!(partitions, SimplePartition(endpoints))
    end
    partitions = NestedPartitions(partitions)
    return PolyaTree(partitions, alpha0)
end


# POSTERIOR

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
    return PolyaTree(pt.partition, new_alphas, new_counts)
end

# PREDICTIVE


function get_alpha(pt::PolyaTree, binseq::Array{Int})
    return get_from_binseq(pt.alphas, binseq)
end

function get_length(pt::PolyaTree, binseq::Array{Int})
    l =  get_from_binseq(pt.partition.lengths, binseq)
    return l
end


function predictive_density(xgrid::Array{Float64}, pt::PolyaTree)
    """
    We assume xgrid is already sorted and equispaced
    """
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
    out[1] = dens_cumprod[end] / get_length(pt, curr_binseq)
    old_binseq = curr_binseq
    for i in 2:length(xgrid)
        curr_binseq = digits(idxs[i][end] - 1, base=2, pad=m) |> reverse
        if (curr_binseq == old_binseq)
            out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
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
        out[i] = dens_cumprod[end] / get_length(pt, curr_binseq)
        old_binseq = curr_binseq
    end
    return out
end
