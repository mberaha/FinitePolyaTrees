using DataFrames
using CSV
using Base.Threads
using Profile
using Serialization


include("../src/polyatree.jl")
include("../src/utils.jl")

function to_matrix(x)
    return mapreduce(permutedims, vcat, x)
end

function read_data(filename)
    return float.(DataFrame(CSV.File(filename, header=false))[:, 1])
end

function parallel_log_lik(data, model)
    results = Vector{Any}(undef, 1000)

    @threads for i in 1:1000
        results[i] = sample_log_lik(data, model)
    end

    matrix_result = to_matrix(results)
    return matrix_result
end

function get_posterior_summaries(pt2_chain)
    post_n_proba = zeros(20)
    for tree in pt2_chain
        post_n_proba[tree.curr_n] += 1
    end
    post_n_proba ./= length(pt2_chain)

    post_n_mean = sum(2 .^ collect(1:20) .* post_n_proba)
    return post_n_mean
end


function run_one(data, name)
    digits10, magnitudes10 = scale_and_magnitudes(data, 10)
    min_order = minimum(magnitudes10)
    max_order = maximum(magnitudes10)
    etas = ones(max_order - min_order + 1)
    # -2 to subtract '0.'
    Nmax10 = maximum([length(string(x)) for x in digits10]) - 2
    Nmax10 = min(Nmax10, 7)

    bpt10 = MultiscaleBPT(
        Poisson(Nmax10 / 2), 
        Nmax10, 
        1.0, 
        etas, 
        minimum(magnitudes10), 
        10)
        
    bpt10 = update(data, bpt10)
    bpt_ll = parallel_log_lik(data, bpt10)
    waic_bpt10 = compute_waic(bpt_ll)

    _, magnitudes2 = scale_and_magnitudes(data, 2)
    min_order = minimum(magnitudes2)
    max_order = maximum(magnitudes2)
    etas = ones(max_order - min_order + 1)

    Nmax2 = Int(ceil(log(2, 10^Nmax10)))
    Nmax2 = min(Nmax2, 24)


    bpt2 = MultiscaleBPT(Poisson(Nmax2/2), Nmax2, 0.1, etas, minimum(magnitudes2), 2)
    bpt2 = update(data, bpt2)
    bpt_ll = parallel_log_lik(data, bpt2)
    waic_bpt2 = compute_waic(bpt_ll)

    V = maximum(data) * 1.01
    normalized_data = data ./ V


    gfpt1 = GFPT1(Poisson(Nmax2/2), Nmax2, 0.1)
    gfpt1 = update(normalized_data, gfpt1)
    gfpt1_ll = parallel_log_lik(normalized_data, gfpt1)    
    gfpt1_ll .-= log(V)
    waic1 = compute_waic(gfpt1_ll)

    # println("waic_bpt10: ", waic_bpt10, ", waic_bpt2: ", waic_bpt2, ", waic_gfpt1: ", waic1)

    gfpt2 = GFPT2(Poisson(5), 10, 0.1, 2.0)
    gfpt2_chain = run_mcmc(normalized_data, gfpt2, 5000, 1000)
    # gfpt2_ll = log_lik_chain(normalized_data, gfpt2_chain) .- log(V)
    # waic2 = compute_waic(gfpt2_ll)
    waic2 = 0

    println("Posterior Mean of q^N")
    post_n = bpt10.tree.prob_n
    n_bpt10 = sum(10 .^ collect(1:length(post_n)) .*  post_n)

    post_n = bpt2.tree.prob_n
    n_bpt2 = sum(2 .^ collect(1:length(post_n)) .*  post_n)

    post_n = gfpt1.prob_n
    n_gfpt1 = sum(2 .^ collect(1:length(post_n)) .*  post_n)

    n_gfpt2 = get_posterior_summaries(gfpt2_chain)

    println("gfpt1: ", n_gfpt1, ", gfpt2: ", n_gfpt2, ", bpt2: ", n_bpt2, ", bpt10: ", n_bpt10)

    # println("bpt10: ", n_bpt10, ", bpt2: ", n_bpt2, ", gfpt1: ", n_gfpt1, ", gfpt2: ", n_gfpt2)

    return waic1, waic2, waic_bpt2, waic_bpt10
end


function main()
    # EURODIST

    rows = []

    println("EURODIST")
    data = read_data("data/eurodist.csv")
    waic1, waic2, waic_bpt2, waic_bpt10 = run_one(data, "eurodist")
    rows = push!(rows, ("eurodist",  waic1, waic2, waic_bpt2, waic_bpt10 ))

    println("GDP")
    data = read_data("data/gdp.csv")
    waic1, waic2, waic_bpt2, waic_bpt10  = run_one(data, "gdp")
    rows = push!(rows, ("gdp",  waic1, waic2, waic_bpt2, waic_bpt10 ))

    println("CENSUS")
    data = read_data("data/census.csv")
    waic1, waic2, waic_bpt2, waic_bpt10 = run_one(data, "census")
    rows = push!(rows, ("census",  waic1, waic2, waic_bpt2, waic_bpt10 ))

    println("TWITTER")
    data = read_data("data/twitter_friends.csv")
    waic1, waic2, waic_bpt2, waic_bpt10  = run_one(data, "twitter")
    rows = push!(rows, ("twitter",  waic1, waic2, waic_bpt2, waic_bpt10 ))

    println("INCOME")
    data = read_data("data/income.csv")
    waic1, waic2, waic_bpt2, waic_bpt10  = run_one(data, "income")
    rows = push!(rows, ("income",  waic1, waic2, waic_bpt2, waic_bpt10 ))

    # df = DataFrame(rows)
    # CSV.write("data/real_data_no_gfpt2_out.csv", df)
end

main()