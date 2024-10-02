function bit_to_int(arr)
    return Int(sum(arr .* (2 .^ collect(length(arr)-1:-1:0))))
end

function get_from_binseq(x, binseq::Array{Int})
    level = length(binseq)
    return x[level][bit_to_int(binseq) + 1]
end

function lengths_from_relative_lengths(rel_lengths::Vector{Vector{Float64}})
    lengths = Vector{Vector{Float64}}()
    depth = length(rel_lengths)
    lengths = push!(lengths, [rel_lengths[1][1], 1.0 - rel_lengths[1][1]])

    for d in 2:depth
        curr_lengths = reduce(
            vcat, [[lengths[d-1][i] * rel_lengths[d][i],  
                    lengths[d-1][i] * (1 - rel_lengths[d][i])] 
                    for i in 1:length(lengths[d-1])])
        lengths = push!(lengths, curr_lengths)
    end

    return lengths
end

function endpoints_from_lengths(lengths::Vector{Vector{Float64}})
    endpoints = Vector{Vector{Float64}}()
    depth = length(lengths)
    endpoints = push!(endpoints, [0.0, lengths[1][1], 1.0])
    for d in 2:depth
        curr_endpoints = [0; cumsum(lengths[d])]
        @assert abs(curr_endpoints[end] - 1.0) < 1e-4
        endpoints = push!(endpoints, curr_endpoints)
    end
    return endpoints
end

function sample_beta_sequence(params::Vector{Vector{Float64}})
    depth = length(params)
    out = Vector{Vector{Float64}}()
    for d in 1:depth
        curr = [rand(Beta(params[d][i], params[d][i+1])) for i in 1:2:length(params[d])]
        out = push!(out, curr)
    end
    return out
end

function get_subarrays(x)
    out = []
    for i in 1:length(x)
        out = push!(out, x[1:i])
    end
    return out
end

