"""
rossler_K_sweep_helper.jl
=========================
Shared layer for the Rössler figures 4a–4c: the network, the binary basin map, and the
two sweeps over the coupling `K` at fixed rewiring probability `p`.

    ẋᵢ = −yᵢ − zᵢ − K (L x)ᵢ       ← diffusive coupling through x
    ẏᵢ =  xᵢ + a yᵢ
    żᵢ =  b  + zᵢ(xᵢ − c)


## The basin map is binary

There are no attractors to find here, and no reason to find them: the question is
whether a trajectory synchronizes, which is one bit per initial condition. So
`RosslerSyncMap` integrates for a fixed time, measures the Golomb–Rinzel coherence, and
"""

using DrWatson
@quickactivate "BayesianBasinTracking"
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Graphs
using OrdinaryDiffEq: Vern9
using SpecialFunctions
using Attractors
using CairoMakie
using LaTeXStrings
using ProgressMeter

include(srcdir("inference_stuff.jl"))

# --- MSF constants for Rössler (a=0.2, b=0.2, c=9.0), Boccaletti et al. 2002 ---------
const MSF_α1 = 0.1232
const MSF_α2 = 4.663
const MSF_Rmax = MSF_α2 / MSF_α1    # ≈ 37.85

# ============================================================================
# Rössler network ODE
# ============================================================================

mutable struct RosslerParams
    N::Int
    a::Float64
    b::Float64
    c::Float64
    K::Float64
    L::SparseMatrixCSC{Float64, Int}
    Lx::Vector{Float64}     # pre-allocated cache for L*x
end

RosslerParams(N, a, b, c, K, L) = RosslerParams(N, a, b, c, K, L, zeros(N))

function rossler_network!(du, u, p, t)
    (; N, a, b, c, K, L, Lx) = p
    x  = view(u,     1:N)
    y  = view(u,   N+1:2N)
    z  = view(u,  2N+1:3N)
    dx = view(du,    1:N)
    dy = view(du,  N+1:2N)
    dz = view(du, 2N+1:3N)
    mul!(Lx, L, x)
    @. dx = -y - z - K * Lx
    @. dy =  x + a * y
    @. dz =  b + z * (x - c)
    return nothing
end

# Golomb–Rinzel coherence (Golomb & Rinzel 1994) of a T×N matrix of x-components:
#   R = Var_t(x̄(t)) / mean_i(Var_t(xᵢ(t)))
# R = 1 → perfect synchrony;  R ≈ 1/N → incoherent.
function golomb_rinzel_coherence(X)
    mean_var_i = mean(var(X; dims = 1))
    mean_var_i < 1e-12 && return 1.0
    return var(vec(mean(X; dims = 2))) / mean_var_i
end


struct RosslerSyncMap{DS <: DynamicalSystem} <: BasinMap
    ds::DS
    N::Int
    r_thresh::Float64
    Ttr::Float64            # transient, discarded
    T::Float64              # time the coherence is measured over
end

function (bmap::RosslerSyncMap)(u0)
    # only the x-components are needed, and `container = Vector` keeps the record out of
    # `SVector{N}`, which for a network of this size is not worth compiling
    X, = trajectory(bmap.ds, bmap.T, u0;
        Ttr = bmap.Ttr, Δt = 1.0, save_idxs = 1:bmap.N, container = Vector,
    )
    # an integration that blew up says nothing about synchrony; `trajectory` pads the rest
    # of the record with the last state, which could well look coherent
    successful_step(bmap.ds) || return -1
    R = golomb_rinzel_coherence(Matrix(X))
    return (isnan(R) || R ≤ bmap.r_thresh) ? -1 : 1
end

Attractors._extract_attractors(::RosslerSyncMap) = Dict{Int, StateSpaceSet}()
Attractors.reset_mapper!(::RosslerSyncMap) = nothing

"One basin map, carrying its own system — the continuation re-parameterises it in place."
function sync_map(N, a, b, c, K, L, r_thresh, T_transient, T_measure)
    par = RosslerParams(N, a, b, c, K, L)
    diffeq = (alg = Vern9(), adaptive = false, dt = 0.1, maxiters = Int(1e8))
    ds = CoupledODEs(rossler_network!, zeros(3 * N), par; diffeq)
    return RosslerSyncMap(ds, N, r_thresh, T_transient, T_measure)
end

# The region the initial conditions are drawn from. A `Tuple` of pairs, not a `Vector`:
# the region's dimension has to be in its type.
global_region(N) = Tuple(vcat(
    [(-12.0, 12.0) for _ in 1:N],
    [(-12.0, 12.0) for _ in 1:N],
    [( -8.0, 35.0) for _ in 1:N],
))

"""
Laplacian of the WS graph at rewiring probability `p_val`, and the `K` range spanning its
MSF interval `Iₛ`. Returns `nothing` if the network cannot synchronize at any `K`.
"""
function get_network_and_coupling(N, k_deg, p_val, graph_seed, n_K_steps)
    g = watts_strogatz(N, k_deg, p_val; rng = Xoshiro(graph_seed))
    L = Float64.(laplacian_matrix(g))
    λs = filter(>(1e-10), sort(real.(eigvals(Matrix(L)))))
    isempty(λs) && return nothing
    λ_min, λ_max = first(λs), last(λs)
    K_lo, K_hi = MSF_α1 / λ_min, MSF_α2 / λ_max
    (λ_max / λ_min >= MSF_Rmax || K_lo >= K_hi) && return nothing
    return L, range(K_lo, K_hi; length = n_K_steps)
end


function ksweep_once(L, K_range, N, a, b, c, r_thresh, T_transient, T_measure,
                     sparse_n, dense_n, n_tiles, λ, β)
    bmap = sync_map(N, a, b, c, first(K_range), L, r_thresh, T_transient, T_measure)

    # `history = true` is what makes the estimators recoverable afterwards: without it
    # the sampler overwrites `alphas` and `etas` at every parameter.
    sampler = BayesianUpdateSampler(global_region(N), n_tiles;
        sparse_n, dense_n, λ, β, seed = 20260802, history = true,
    )

    pcurve = [Dict(:K => K) for K in K_range]

    # The matcher is never called (no attractors), so its configuration is irrelevant.
    fractions, attractors = global_continuation(
        AttractorSeedContinueMatch(bmap), pcurve, sampler,
    )

    est = bayes_estimates(sampler)
    return (; fractions, K = collect(K_range),
        est.mean_S, est.var_S, est.min_eta, est.n_panics,
        est.volumes, est.vol_var, est.full_S, est.full_eta,
    )
end

"Average, per parameter, a series of `Dict`s over realisations; a missing label counts 0."
function average_dict_series(series)
    n = length(series)
    return map(eachindex(first(series))) do i
        labels = reduce(union, (keys(s[i]) for s in series))
        Dict(k => sum(get(s[i], k, 0.0) for s in series) / n for k in labels)
    end
end

# Computation functions (wrapped for produce_or_load)

function rossler_Ksweep(d)
    @unpack N_osc, k_degree, graph_seed, n_avg, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            sparse_n, dense_n, n_tiles, λ, β = d

    runs = []
    for seed in graph_seed .+ (0:(n_avg - 1))
        net = get_network_and_coupling(N_osc, k_degree, p_val, seed, n_K_steps)
        # a network with no MSF interval has nothing to sweep over
        isnothing(net) && continue
        L, K_range = net
        println("p=$p_val  seed=$seed  Iₛ=[$(round(first(K_range), digits = 3)), " *
                "$(round(last(K_range), digits = 3))]")
        push!(runs, ksweep_once(L, K_range, N_osc, a_ros, b_ros, c_ros,
            r_thresh, T_transient, T_measure, sparse_n, dense_n, n_tiles, λ, β))
    end
    isempty(runs) && error("p=$p_val → all $n_avg network realisations linearly unstable")

    # Every realisation has its own `Iₛ`, so the `K` axis is averaged along with the rest.
    mean_over_runs(f) = sum(f(r) for r in runs) ./ length(runs)
    K_vec = mean_over_runs(r -> r.K)
    K_range = range(first(K_vec), last(K_vec); length = n_K_steps)

    return @strdict(
        K_range, n_valid = length(runs),
        fractions = average_dict_series([r.fractions for r in runs]),
        volumes = average_dict_series([r.volumes for r in runs]),
        vol_var = average_dict_series([r.vol_var for r in runs]),
        mean_S = mean_over_runs(r -> r.mean_S),
        var_S = mean_over_runs(r -> r.var_S),
        min_eta = mean_over_runs(r -> r.min_eta),
        n_panics = mean_over_runs(r -> float.(r.n_panics)),
        full_S = mean_over_runs(r -> r.full_S),
        full_eta = mean_over_runs(r -> r.full_eta),
    )
end

function rossler_Ksweep_montecarlo(d)
    @unpack N_osc, k_degree, graph_seed, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure, n_mc = d

    net = get_network_and_coupling(N_osc, k_degree, p_val, graph_seed, n_K_steps)
    isnothing(net) && error("p=$p_val → linearly unstable, cannot run montecarlo")
    L, K_range = net
    println("p=$p_val  Iₛ=[$(round(first(K_range), digits = 3)), " *
            "$(round(last(K_range), digits = 3))]")

    region = global_region(N_osc)
    sync_fracs = zeros(length(K_range))
    for (i, K) in enumerate(K_range)
        # one map per thread: each carries its own integrator, which it mutates
        bmaps = [sync_map(N_osc, a_ros, b_ros, c_ros, K, L,
                          r_thresh, T_transient, T_measure) for _ in 1:Threads.nthreads()]
        labels = zeros(Int, n_mc)
        @showprogress @Threads.threads for j in 1:n_mc
            u0 = [lo + rand() * (hi - lo) for (lo, hi) in region]
            labels[j] = bmaps[Threads.threadid()](u0)
        end
        sync_fracs[i] = count(==(1), labels) / n_mc
        println("  K=$(round(K, digits = 4))  sync_frac=$(round(sync_fracs[i], digits = 3))")
    end

    return @strdict(sync_fracs, K_range)
end

# ============================================================================
# Shared parameters
# ============================================================================

N_osc      = 100
k_degree   = 8
graph_seed = 12345

a_ros, b_ros, c_ros = 0.2, 0.2, 9.0

r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0
n_K_steps   = 50

# Bayesian monitoring. `n_tiles` splits *every* dimension of the 3N-dimensional region,
# so anything above 1 here means `n_tiles^(3N)` boxes.
λ        = 0.7
β        = 0.5
sparse_n = 20
dense_n  = 200
n_tiles  = 1

n_avg = 10      # network realisations averaged over
n_mc  = 500     # initial conditions per `K` in the Monte Carlo reference

p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

lab_args = (; yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)
