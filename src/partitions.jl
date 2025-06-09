include("utils.jl")

"""
    SimplePartition

Represents a partition of the unit interval with fixed endpoints.

# Fields
- `endpoints` : vector of break points.
- `lengths` : length of each interval.
"""
struct SimplePartition
    endpoints::Array{Float64}
    lengths::Array{Float64}
end

"""
    SimplePartition(endpoints::Vector{Float64})

Create a `SimplePartition` from its endpoints.
"""
function SimplePartition(endpoints::Array{Float64})
    lengths = zeros(length(endpoints) - 1)
    for i in 1:(length(endpoints) - 1)
        lengths[i] = endpoints[i+1] - endpoints[i]
    end
    return SimplePartition(endpoints, lengths)
end

"""
    find_interval(x::Float64, endpoints::Vector{Float64})

Return the index of the interval containing `x`.
"""
function find_interval(x::Float64, endpoints::Vector{Float64})
    return last(searchsorted(endpoints, x))
end

"""
    find_interval(x::Float64, p::SimplePartition)

Shortcut for `find_interval(x, p.endpoints)`.
"""
function find_interval(x::Float64, p::SimplePartition)
    return find_interval(x, p.endpoints)
end

"""
    NestedPartitions

Hierarchy of `SimplePartition`s defining a tree partition of the unit
interval.

# Fields
- `depth` : number of levels.
- `base` : branching factor.
- `levels` : vector of `SimplePartition`s for each level.
- `lengths` : cached lengths for fast access.
"""
struct NestedPartitions
    depth::Int
    base::Int
    levels::Array{SimplePartition}
    lengths::Array{Array{Float64}}
end

"""
    NestedPartitions(partitions::Vector{SimplePartition})

Construct a `NestedPartitions` object from an array of `SimplePartition`s.
"""
function NestedPartitions(partitions::Array{SimplePartition})
    depth = length(partitions)
    base = length(partitions[1].lengths)
    lengths = []
    for i in 1:depth
        lengths = push!(lengths, partitions[i].lengths)
    end
    return NestedPartitions(depth, base, partitions, lengths)
end

"""
    check_patition(p::NestedPartitions)

Assert basic consistency of the nested partition structure.
"""
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

"""
    find_intervals(x, p::NestedPartitions)

Return the sequence of interval indices for value `x` across all levels.
"""
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