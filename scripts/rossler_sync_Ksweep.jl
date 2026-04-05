using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Graphs
using JLD2
using OrdinaryDiffEq: Vern9, Rodas5P, AutoVern9, ODEProblem, solve
using Attractors
using CairoMakie

include(srcdir("bayes_entropy_est.jl"))

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
#
# Continuation parameter: rewiring probability p of the WS graph.
# For each p, basin stability is averaged over n_K couplings sampled from Iₛ.
# ============================================================================

# --- MSF constants for Rössler (a=0.2, b=0.2, c=9.0) ----------------------
const MSF_α1   = 0.1232
const MSF_α2   = 4.663
const MSF_Rmax = MSF_α2 / MSF_α1    # ≈ 37.85

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

# Phase order parameter r = |N⁻¹ Σ exp(i φⱼ)|, φⱼ = atan(yⱼ, xⱼ)
# NOTE: assumes spiral attractor wrapping around origin in x-y plane.
function phase_order_parameter(u, N)
    re = 0.0; im = 0.0
    @inbounds for i in 1:N
        φ = atan(u[N + i], u[i])
        re += cos(φ);  im += sin(φ)
    end
    return sqrt(re^2 + im^2) / N
end

# Golomb–Rinzel coherence measure (Golomb & Rinzel 1994):
#
#        Var_t( p̄(t) )
#   R = ─────────────────────────────
#        mean_i( Var_t( p_i(t) ) )
#
# R = 1  → perfect synchrony;  R ≈ 1/N  → incoherent (independent nodes).
# Uses x-components as the observable (coupling variable).
# Takes the full trajectory (iterable of state vectors of length 3N).
function golomb_rinzel_coherence(traj, N)
    T       = length(traj)
    sum_pi  = zeros(N)    # Σ_t xᵢ(t)
    sum_pi2 = zeros(N)    # Σ_t xᵢ(t)²
    sum_pb  = 0.0         # Σ_t x̄(t)
    sum_pb2 = 0.0         # Σ_t x̄(t)²

    for row in traj
        pb = 0.0
        @inbounds for i in 1:N
            xi      = row[i]
            sum_pi[i]  += xi
            sum_pi2[i] += xi * xi
            pb         += xi
        end
        pb        /= N
        sum_pb    += pb
        sum_pb2   += pb * pb
    end

    var_pbar    = sum_pb2 / T - (sum_pb / T)^2          # Var_t(x̄)
    mean_var_pi = mean(@inbounds sum_pi2[i] / T - (sum_pi[i] / T)^2 for i in 1:N)

    mean_var_pi < 1e-12 && return 1.0   # nodes are stationary → trivially coherent
    return var_pbar / mean_var_pi
end

# ============================================================================
# Mapper for a fixed network (L) and a set of K values
# Returns 1 (sync) if majority of K trials synchronise, else 0
# ============================================================================

struct RosslerSyncMapper
    N::Int
    r_thresh::Float64
    T_transient::Float64
    T_measure::Float64
    ds::CoupledODEs
end

function (m::RosslerSyncMapper)(u0)
    y, t = trajectory(m.ds, m.T_measure, u0; Ttr = m.T_transient)
    R = golomb_rinzel_coherence(y, m.N)
    isnan(R) && return 0
    return R > m.r_thresh ? 1 : 0
end

Attractors.extract_attractors(::RosslerSyncMapper) = Dict{Int, Nothing}()

# ============================================================================
# Trivial mapper — returned when the network is linearly unstable (Iₛ empty)
# Always returns 0: no synchrony possible.
# ============================================================================

struct TrivialMapper end

(::TrivialMapper)(u0) = 0
Attractors.extract_attractors(::TrivialMapper) = Dict{Int, Nothing}()

# ============================================================================
# Factory: build mapper for a given rewiring probability p_val
# ============================================================================

function get_network_and_coupling(N, k_deg, p_val, graph_seed, n_K_steps)
    # Build WS graph and its Laplacian
    g     = watts_strogatz(N, k_deg, p_val; rng = Random.Xoshiro(graph_seed))
    L     = laplacian_matrix(g)

    # Laplacian spectrum (real, non-negative)
    λs    = sort(real.(eigvals(Matrix(L))))
    λs_nz = filter(>(1e-10), λs)
    isempty(λs_nz) && return nothing

    λ_min = first(λs_nz)
    λ_max = last(λs_nz)
    R     = λ_max / λ_min
    K_lo  = MSF_α1 / λ_min
    K_hi  = MSF_α2 / λ_max

    # Linearly unstable: synchronizability condition fails
    (R >= MSF_Rmax || K_lo >= K_hi) && return nothing

    K_range = range(K_lo, K_hi; length = n_K_steps)
    return L, K_range
end

# Factory for a fixed network (L): returns _get_mapper(K_val, atts) -> RosslerSyncMapper
function rossler_mapper_factory(N, a, b, c, L,
                                r_thresh, T_transient, T_measure)
    function _get_mapper(K_val, _atts)
        p     = RosslerParams(N, a, b, c, K_val, L)
        diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = Int(1e8), adaptive = false, dt = 0.1)
        ds    = CoupledODEs(rossler_network!, zeros(N * 3), p; diffeq)
        return RosslerSyncMapper(N, r_thresh, T_transient, T_measure, ds)
    end
    return _get_mapper
end

# ============================================================================
# Computation function (wrapped for produce_or_load)
# ============================================================================

function rossler_Ksweep(d)
    @unpack N_osc, k_degree, graph_seed, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            sparse_n, dense_n, n_tiles, λ = d

    global_bounds = vcat(
        [(-12.0, 12.0) for _ in 1:N_osc],
        [(-12.0, 12.0) for _ in 1:N_osc],
        [( -8.0, 35.0) for _ in 1:N_osc],
    )
    params = @strdict sparse_n dense_n n_tiles global_bounds λ

    result = get_network_and_coupling(N_osc, k_degree, p_val, graph_seed, n_K_steps)
    isnothing(result) && error("p=$p_val → linearly unstable, cannot run continuation")
    L, K_range = result
    println("p=$p_val  Iₛ=[$(round(first(K_range), digits=3)), $(round(last(K_range), digits=3))]")

    get_map = rossler_mapper_factory(
        N_osc, a_ros, b_ros, c_ros, L,
        r_thresh, T_transient, T_measure
    )

    history_mean_S, history_var_S, history_max_llr, history_n_panics,
        history_att, full_history_S, history_volumes =
            estimate_entropy(params, K_range, get_map)

    return @strdict(history_mean_S, history_var_S, history_max_llr,
                    history_n_panics, history_att, full_history_S,
                    history_volumes, K_range)
end

# ============================================================================
# Setup
# ============================================================================

# Network parameters — Menck & Kurths (2013): N=100, ⟨k⟩=8
N_osc      = 100
k_degree   = 8
graph_seed = 12345
p_val      = 0.8

# Rössler parameters
a_ros, b_ros, c_ros = 0.2, 0.2, 9.0

# Integration settings
r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0
n_K_steps   = 50

# Bayesian continuation (over K)
λ        = 0.7
sparse_n = 60
dense_n  = 1000
n_tiles  = 1

# Dense sampling in (0, 0.15) where synchronization transitions sharply,
# coarser beyond.
p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

for p_val in p_vals
    params = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ

    try 
        data, file = produce_or_load(
            datadir("data"), params, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = true,
            filename = hash
        )

        @unpack history_mean_S, history_var_S, history_max_llr, history_n_panics,
                history_att, full_history_S, history_volumes, K_range = data
        # ============================================================================
        # Plot — sync fraction and LLR detector across K sweep
        # ============================================================================

        sync_fracs = [get(hv, 1, 0.0) for hv in history_volumes]
        K_vec      = collect(K_range)
        println("Loaded from: $file")

        fig = Figure(size = (750, 650))

        ax1 = Axis(fig[1, 1],
            title  = "Rössler network — WS(N=$N_osc, ⟨k⟩=$k_degree, p=$p_val)",
            ylabel = "Sync fraction  S_B",
        )
        lines!(ax1, K_vec, sync_fracs, color = :black, linewidth = 2)
        scatter!(ax1, K_vec, sync_fracs, color = :black, markersize = 5)
        ylims!(ax1, 0, 1)

        ax2 = Axis(fig[2, 1],
            ylabel = "η  (log Bayes factor)",
            xlabel = "Coupling  K",
        )
        lines!(ax2, K_vec, history_max_llr, color = :black, linewidth = 2)
        hlines!(ax2, [0.0], color = :red, linestyle = :dash, linewidth = 1)

        panic_idx = findall(>(0), history_n_panics)
        if !isempty(panic_idx)
            scatter!(ax2, K_vec[panic_idx], history_max_llr[panic_idx],
                     color = :red, markersize = 8, label = "panic")
        end

        save(plotsdir(savename("rossler_sync_Ksweep",(;p = p_val),"png")), fig)
        println("Saved → scripts/rossler_sync_Ksweep.png")
        println("Sync fraction range : ", extrema(sync_fracs))
        println("Panics triggered    : ", sum(history_n_panics))
    catch
    end
end
