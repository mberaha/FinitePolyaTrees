function kl_div(p_dens, q_dens, xgrid)
    delta = (xgrid[2]-xgrid[1])
    return sum(p_dens .* (log.(p_dens) .- log.(q_dens))) * delta
end

function hell_dist(p_dens, q_dens, xgrid)
    delta = (xgrid[2]-xgrid[1])
    integrand = (sqrt.(p_dens) .- sqrt.(q_dens)).^2
    integral = sum(integrand) * delta
    return sqrt(0.5 * integral)
end

function l1_dist(p_dens, q_dens, xgrid)
    return 0.5 * sum(abs.(p_dens .- q_dens)) * (xgrid[2] - xgrid[1])
end

function l2_dist(p_dens, q_dens, xgrid)
    delta = (xgrid[2]-xgrid[1])
    integrand = (p_dens .- q_dens).^2
    integral = sum(integrand) * delta
    return integral
end

function arclength(curve, xgrid, normalize=true)
    diffs = curve[2:end] .- curve[1:end-1]
    deltas = xgrid[2:end] .- xgrid[1:end-1]
    out = sum(
        (diffs.^2 .+ deltas.^2).^(1/2)
    )
    if normalize
        out -= sum(deltas)
    end
    return out
end

function sobolev_dist(p_dens, q_dens, xgrid)
    delta = (xgrid[2]-xgrid[1])
    integrand1 = (p_dens .- q_dens).^2
    tmp1 = sum(integrand1) * delta

    p_der = (p_dens[2:end] - p_dens[1:(end-1)]) / delta
    q_der = (q_dens[2:end] - q_dens[1:(end-1)]) / delta
    integrand2 = (p_der .- q_der).^2
    tmp2 = sum(integrand2) * delta

    return sqrt(tmp1 + tmp2)
end