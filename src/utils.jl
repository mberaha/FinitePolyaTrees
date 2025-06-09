using SpecialFunctions


"""
    bit_to_int(arr; base=2)

Convert an array of digits `arr` in given `base` to an integer.
"""
function bit_to_int(arr, base=2)
    return Int(sum(arr .* (base .^ collect(length(arr)-1:-1:0))))
end


"""
    get_from_binseq(x, binseq; base=2)

Return `x` indexed by the binary sequence `binseq`.
"""
function get_from_binseq(x, binseq::Array{Int}, base=2)
    level = length(binseq)
    return x[level][bit_to_int(binseq, base) + 1]
end


"""
    lengths_from_relative_lengths(rel_lengths)

Compute absolute lengths from relative split proportions.
"""
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


"""
    endpoints_from_lengths(lengths)

Convert lengths at each level to cumulative endpoints.
"""
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


"""
    sample_beta_sequence(params)

Draw independent beta random variables for each pair of parameters.
"""
function sample_beta_sequence(params::Vector{Vector{Float64}})
    depth = length(params)
    out = Vector{Vector{Float64}}()
    for d in 1:depth
        curr = [rand(Beta(params[d][i], params[d][i+1])) for i in 1:2:length(params[d])]
        out = push!(out, curr)
    end
    return out
end

"""
    rand_ext_dirichlet(params)

Dirichlet sampler that gracefully handles zero parameters.
"""
function rand_ext_dirichlet(params)
    out = zeros(length(params))
    if sum(params) > 0
        keep_mask = params .> 0
        out[keep_mask] = rand(Dirichlet(params[keep_mask]))
    end
    return out
end


"""
    sample_dirichlet_sequence(params)

Sample Dirichlet vectors level by level using `params`.
"""
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


"""
    get_subarrays(x)

Return all leading subarrays of `x`.
"""
function get_subarrays(x)
    out = []
    for i in 1:length(x)
        out = push!(out, x[1:i])
    end
    return out
end


"""
    log_mv_beta(x)

Multivariate beta function on the log scale.
"""
function log_mv_beta(x)
    return sum(loggamma.(x)) - loggamma(sum(x))
end


"""
    order_of_magnitude(x; base=10)

Return the integer order of magnitude of `x` in the given base.
"""
function order_of_magnitude(x::Real, base=10)
    if x <= 0
        throw(ArgumentError("x must be positive"))
    end
    return floor(Int, log(base, x))
end


"""
    scale_and_magnitudes(data; base=10)

Scale `data` to the interval `(0.1, 1]` and return their magnitudes.
"""
function scale_and_magnitudes(data::Vector{Float64}, base=10)
    # Compute orders of magnitude (integer part of log10)
    magnitudes = floor.(Int, log.(base, abs.(data)))

    # Scale data into (0.1, 1.0] using the order of magnitude
    scaled_data = data ./ ((1.0 * base) .^ (magnitudes .+ 1))

    return scaled_data, magnitudes
end


"""
    compute_waic(L)

Compute the WAIC given a matrix of log-likelihood values `L`.
"""
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
