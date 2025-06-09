using SpecialFunctions


function bit_to_int(arr, base=2)
    return Int(sum(arr .* (base .^ collect(length(arr)-1:-1:0))))
end


function get_from_binseq(x, binseq::Array{Int}, base=2)
    level = length(binseq)
    return x[level][bit_to_int(binseq, base) + 1]
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

function rand_ext_dirichlet(params)
    out = zeros(length(params))
    if sum(params) > 0
        keep_mask = params .> 0
        out[keep_mask] = rand(Dirichlet(params[keep_mask]))
    end
    return out
end


function sample_dirichlet_sequence(params::Vector{Vector{Float64}})
    depth = length(params)
    base = length(params[1])
    out = Vector{Vector{Float64}}()
    for d in 1:depth
        sample = [rand_ext_dirichlet(params[d][i:i+(base-1)])
                  for i in 1:base:length(params[d])]
        curr = reduce(vcat, sample)
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


function log_mv_beta(x)
    return sum(loggamma.(x)) - loggamma(sum(x))
end


function order_of_magnitude(x::Real, base=10)
    if x <= 0
        throw(ArgumentError("x must be positive"))
    end
    return floor(Int, log(base, x))
end


function scale_and_magnitudes(data::Vector{Float64}, base=10)
    # Compute orders of magnitude (integer part of log10)
    magnitudes = floor.(Int, log.(base, abs.(data)))

    # Scale data into (0.1, 1.0] using the order of magnitude
    scaled_data = data ./ ((1.0 * base) .^ (magnitudes .+ 1))

    return scaled_data, magnitudes
end


function compute_waic(L::AbstractMatrix)
    M, n = size(L)  # M = number of iterations, n = number of datapoints
    lppd = 0.0      # log pointwise predictive density
    p_waic = 0.0    # effective number of parameters

    for j in 1:n
        m = maximum(L[:, j])
        lppd_j = m + log(mean(exp.(L[:, j] .- m)))
        lppd += lppd_j

        p_waic += var(L[:, j])
    end

    return -2 * (lppd - p_waic)
end
