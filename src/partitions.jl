include("utils.jl")

struct SimplePartition
    endpoints::Array{Float64}
    lengths::Array{Float64}
end

function SimplePartition(endpoints::Array{Float64})
    lengths = zeros(length(endpoints) - 1)
    for i in 1:(length(endpoints) - 1)
        lengths[i] = endpoints[i+1] - endpoints[i]
    end
    return SimplePartition(endpoints, lengths)
end

function find_interval(x::Float64, endpoints::Vector{Float64})
    return last(searchsorted(endpoints, x))
end

function find_interval(x::Float64, p::SimplePartition)
    return find_interval(x, p.endpoints)
end

struct NestedPartitions
    depth::Int
    base::Int
    levels::Array{SimplePartition}
    lengths::Array{Array{Float64}}
end

function NestedPartitions(partitions::Array{SimplePartition})
    depth = length(partitions)
    base = length(partitions[1].lengths)
    lengths = []
    for i in 1:depth
        lengths = push!(lengths, partitions[i].lengths)
    end
    return NestedPartitions(depth, base, partitions, lengths)
end

function check_patition(p::NestedPartitions)
    base = p.base
    for d in 1:(p.depth - 1)
        for (idx, e) in enumerate(p.levels[d].endpoints)
            @assert p.levels[d+1][base * idx - 1] == e
            @assert e <= p.levels[d+1][base * idx]
            @assert p.levels[d][idx + 1] <= p.levels[d+1][base * idx]
        end
    end
end

function find_intervals(x, p::NestedPartitions)
    base = p.base
    idx = find_interval(x, p.levels[p.depth])
    d = digits(idx - 1, base=base, pad=p.depth) |> reverse
    out = Vector{Int}(undef, p.depth)
    for l in 1:p.depth
        out[l] = bit_to_int(d[1:l], base) + 1
    end
    return out
end