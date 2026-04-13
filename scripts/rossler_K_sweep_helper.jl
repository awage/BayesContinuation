using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Graphs
using OrdinaryDiffEq
using Attractors
using CairoMakie
using LaTeXStrings
using ProgressMeter

include(srcdir("BayesContinuation.jl"))
using .BayesContinuation

# ============================================================================
# Rössler oscillator network — Menck & Kurths (2013) setup
#
# ODE for node i:
#   ẋᵢ = −yᵢ − zᵢ − K (L x)ᵢ       ← diffusive coupling through x
#   ẏᵢ =  xᵢ + a yᵢ
#   żᵢ =  b  + zᵢ(xᵢ − c)
#
# Synchronous state: all nodes on the same Rössler trajectory.
# MSF theory (Pecora & Carroll) gives the stability interval:
#   K ∈ Iₛ = (α₁/λ_min, α₂/λ_max)
# with α₁ = 0.1232, α₂ = 4.663, a=0.2, b=0.2, c=9.0 (Boccaletti et al. 2002)
#
# Synchronizability: R = λ_max/λ_min < α₂/α₁ ≈ 37.85  (Iₛ non-empty)
# ============================================================================

MSF_α1   = 0.1232
MSF_α2   = 4.663
MSF_Rmax = MSF_α2 / MSF_α1    # ≈ 37.85

# ============================================================================
# Rössler network ODE
# ============================================================================

struct RosslerParams
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

function rossler_network_jac!(J, u, p, t)
    (; N, a, b, c, K, L) = p
    z = view(u, 2N+1:3N)

    fill!(J, 0.0)
    @inbounds for i in 1:N
        J[i, N + i]  = -1.0
        J[i, 2N + i] = -1.0
        J[N + i, i]      = 1.0
        J[N + i, N + i]  = a
        J[2N + i, i]      = z[i]
        J[2N + i, 2N + i] = u[i] - c
    end

    rows = rowvals(L)
    vals = nonzeros(L)
    @inbounds for col in 1:N
        for idx in nzrange(L, col)
            row = rows[idx]
            J[row, col] += -K * vals[idx]
        end
    end
    return nothing
end

# Golomb–Rinzel coherence measure (Golomb & Rinzel 1994):
#   R = Var_t(x̄(t)) / mean_i(Var_t(xᵢ(t)))
# R = 1 → perfect synchrony;  R ≈ 1/N → incoherent.
function golomb_rinzel_coherence(X)
    x_bar       = vec(mean(X; dims=2))
    var_pbar    = var(x_bar)
    mean_var_pi = mean(var(X; dims=1))
    mean_var_pi < 1e-12 && return 1.0
    return var_pbar / mean_var_pi
end

# ============================================================================
# Mapper: fixed network (L) and coupling (K) — returns 1 (sync) or 0
# ============================================================================

struct RosslerSyncMapper
    N::Int
    r_thresh::Float64
    T_transient::Float64
    T_measure::Float64
    p::RosslerParams
    f::ODEFunction
    diverge_thresh::Float64
end

function (m::RosslerSyncMapper)(u0)
    tspan  = (0.0, m.T_transient + m.T_measure)
    saveat = range(m.T_transient, m.T_transient + m.T_measure; step = 1.0)
    cb     = DiscreteCallback(
        (u, t, integrator) -> any(abs.(u) .> m.diverge_thresh),
        terminate!
    )
    prob = ODEProblem(m.f, collect(float(u0)), tspan, m.p)
    sol  = solve(prob, AutoVern9(Rodas5P());
                 reltol = 1e-9, maxiters = Int(1e8), callback = cb, saveat = saveat)
    sol.retcode == ReturnCode.Terminated && return 0
    X = reduce(hcat, sol.u)[1:m.N, :]'  # T × N  (x-components only)
    R = golomb_rinzel_coherence(X)
    isnan(R) && return 0
    return R > m.r_thresh ? 1 : 0
end

# Trivial mapper — for linearly-unstable networks; always returns 0.
struct TrivialMapper end
(::TrivialMapper)(u0) = 0

# ============================================================================
# Helpers: network construction and mapper factory
# ============================================================================

function get_network_and_coupling(N, k_deg, p_val, graph_seed, n_K_steps)
    g     = watts_strogatz(N, k_deg, p_val; rng = Random.Xoshiro(graph_seed))
    L     = laplacian_matrix(g)
    λs    = sort(real.(eigvals(Matrix(L))))
    λs_nz = filter(>(1e-10), λs)
    isempty(λs_nz) && return nothing
    λ_min = first(λs_nz)
    λ_max = last(λs_nz)
    R     = λ_max / λ_min
    K_lo  = MSF_α1 / λ_min
    K_hi  = MSF_α2 / λ_max
    (R >= MSF_Rmax || K_lo >= K_hi) && return nothing
    K_range = range(K_lo, K_hi; length = n_K_steps)
    return L, K_range
end

function rossler_mapper_factory(N, a, b, c, L,
                                r_thresh, T_transient, T_measure;
                                diverge_thresh = 1e4)
    f = ODEFunction(rossler_network!; jac = rossler_network_jac!)
    return K_val -> RosslerSyncMapper(N, r_thresh, T_transient, T_measure,
                                      RosslerParams(N, a, b, c, K_val, L), f, diverge_thresh)
end

# ============================================================================
# Computation functions (wrapped for produce_or_load)
# ============================================================================

function rossler_Ksweep(d)
    @unpack N_osc, k_degree, graph_seed, n_avg, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            sparse_n, dense_n, n_tiles, λ = d

    global_bounds = vcat(
        [(-12.0, 12.0) for _ in 1:N_osc],
        [(-12.0, 12.0) for _ in 1:N_osc],
        [( -8.0, 35.0) for _ in 1:N_osc],
    )
    params = @strdict sparse_n dense_n n_tiles global_bounds λ

    acc_mean_S   = nothing
    acc_var_S    = nothing
    acc_max_llr  = nothing
    acc_n_panics = nothing
    acc_full_S   = nothing
    acc_full_llr = nothing
    acc_volumes  = nothing
    acc_vol_var  = nothing
    acc_K        = nothing
    n_valid      = 0

    for seed_offset in 0:(n_avg - 1)
        seed   = graph_seed + seed_offset
        result = get_network_and_coupling(N_osc, k_degree, p_val, seed, n_K_steps)
        isnothing(result) && continue

        L, K_range = result
        println("p=$p_val  seed=$seed  Iₛ=[$(round(first(K_range), digits=3)), $(round(last(K_range), digits=3))]")

        get_map = rossler_mapper_factory(
            N_osc, a_ros, b_ros, c_ros, L,
            r_thresh, T_transient, T_measure
        )

        res        = estimate_entropy(params, K_range, GenericFactory(get_map))
        h_mean_S   = res.history_mean_S
        h_var_S    = res.history_var_S
        h_max_llr  = res.history_max_llr
        h_n_panics = res.history_n_panics
        fh_S       = res.full_history_S
        fh_llr     = res.full_history_llr
        h_volumes  = res.history_volumes
        h_vol_var  = res.history_vol_var
        K_vec      = collect(K_range)

        if n_valid == 0
            acc_mean_S   = h_mean_S
            acc_var_S    = h_var_S
            acc_max_llr  = h_max_llr
            acc_n_panics = float.(h_n_panics)
            acc_full_S   = fh_S
            acc_full_llr = fh_llr
            acc_volumes  = [Dict(kv for kv in hv) for hv in h_volumes]
            acc_vol_var  = [Dict(kv for kv in vv) for vv in h_vol_var]
            acc_K        = K_vec
        else
            acc_mean_S   .+= h_mean_S
            acc_var_S    .+= h_var_S
            acc_max_llr  .+= h_max_llr
            acc_n_panics .+= h_n_panics
            acc_full_S   .+= fh_S
            acc_full_llr .+= fh_llr
            acc_K        .+= K_vec
            for (i, hv) in enumerate(h_volumes)
                for (k, v) in hv
                    acc_volumes[i][k] = get(acc_volumes[i], k, 0.0) + v
                end
            end
            for (i, vv) in enumerate(h_vol_var)
                for (k, v) in vv
                    acc_vol_var[i][k] = get(acc_vol_var[i], k, 0.0) + v
                end
            end
        end
        n_valid += 1
    end

    n_valid == 0 && error("p=$p_val → all $n_avg network realisations linearly unstable")

    acc_mean_S   ./= n_valid
    acc_var_S    ./= n_valid
    acc_max_llr  ./= n_valid
    acc_n_panics ./= n_valid
    acc_full_S   ./= n_valid
    acc_full_llr ./= n_valid
    acc_K        ./= n_valid
    for hv in acc_volumes; for k in keys(hv); hv[k] /= n_valid; end; end
    for vv in acc_vol_var; for k in keys(vv); vv[k] /= n_valid; end; end

    K_range          = range(first(acc_K), last(acc_K); length = n_K_steps)
    history_mean_S   = acc_mean_S
    history_var_S    = acc_var_S
    history_max_llr  = acc_max_llr
    history_n_panics = acc_n_panics
    full_history_S   = acc_full_S
    full_history_llr = acc_full_llr
    history_volumes  = acc_volumes
    history_vol_var  = acc_vol_var

    return @strdict(history_mean_S, history_var_S, history_max_llr,
                    history_n_panics, full_history_S, full_history_llr,
                    history_volumes, history_vol_var, K_range)
end

function rossler_Ksweep_montecarlo(d)
    @unpack N_osc, k_degree, graph_seed, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            n_mc = d

    global_bounds = vcat(
        [(-12.0, 12.0) for _ in 1:N_osc],
        [(-12.0, 12.0) for _ in 1:N_osc],
        [( -8.0, 35.0) for _ in 1:N_osc],
    )

    result = get_network_and_coupling(N_osc, k_degree, p_val, graph_seed, n_K_steps)
    isnothing(result) && error("p=$p_val → linearly unstable, cannot run montecarlo")
    L, K_range = result
    println("p=$p_val  Iₛ=[$(round(first(K_range), digits=3)), $(round(last(K_range), digits=3))]")

    get_map    = rossler_mapper_factory(N_osc, a_ros, b_ros, c_ros, L, r_thresh, T_transient, T_measure)
    sync_fracs = zeros(length(K_range))

    for (k_idx, K_val) in enumerate(K_range)
        mappers = [get_map(K_val) for _ in 1:Threads.nthreads()]
        hits    = zeros(Int, n_mc)
        @showprogress @Threads.threads for i in 1:n_mc
            tid     = Threads.threadid()
            u0      = [lo + rand() * (hi - lo) for (lo, hi) in global_bounds]
            hits[i] = mappers[tid](u0)
        end
        sync_fracs[k_idx] = mean(hits)
        println("  K=$(round(K_val, digits=4))  sync_frac=$(round(sync_fracs[k_idx], digits=3))")
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

λ        = 0.7
sparse_n = 20
dense_n  = 200
n_tiles  = 1
n_avg    = 10
n_mc     = 500

p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))
