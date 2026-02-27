using Test
using Random
using DrWatson
@quickactivate "BayesContinuation"

include(srcdir("bayes_entropy_est.jl"))

import Attractors: extract_attractors
extract_attractors(::Function) = Dict{Int, Any}()

function mock_get_mapper(a, atts)
    return (u0) -> (u0[1] < a ? 1 : 2)
end

function tile_mean_entropy(a, observers)
    entropies = Float64[]
    for obs in observers
        (xmin, xmax) = obs.physical_bounds[1]
        dx = xmax - xmin
        p1 = clamp((a - xmin) / dx, 0.0, 1.0)
        if p1 <= 0.0 || p1 >= 1.0
            push!(entropies, 0.0)
        else
            push!(entropies, -p1 * log(p1) - (1.0 - p1) * log(1.0 - p1))
        end
    end
    return mean(entropies)
end

@testset "mock deterministic system error bounds" begin
    Random.seed!(1234)

    SPARSE_N = 50
    DENSE_N = 500
    BAYES_FACTOR = 10.0
    N_TILES = 10
    GLOBAL_BOUNDS = ((-1.0, 1.0), (-1.0, 1.0))
    LAMBDA = 1.0
    BETA = 0.5
    params = @strdict SPARSE_N DENSE_N BAYES_FACTOR N_TILES GLOBAL_BOUNDS LAMBDA BETA

    a_range = range(-0.8, 0.8, length = 6)

    history_mean_S, history_var_S, _, _, _ = estimate_entropy(params, a_range, mock_get_mapper)

    observers = generate_tiling(GLOBAL_BOUNDS, N_TILES, BETA)

    for (i, a) in enumerate(a_range)
        ref = tile_mean_entropy(a, observers)
        err = abs(history_mean_S[i] - ref)
        bound = 5.0 * sqrt(history_var_S[i]) + 0.1
        @test err <= bound
    end
end
