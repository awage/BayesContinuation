using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Graphs
using OrdinaryDiffEq: Vern9, Rodas5P, AutoVern9, ODEProblem, solve
using Attractors
using JLD2
using CairoMakie

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
#
# Continuation parameter: rewiring probability p of the WS graph.
# For each p a single K is drawn uniformly at random from Iₛ.
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
function phase_order_parameter(u, N)
    re = 0.0; im = 0.0
    @inbounds for i in 1:N
        φ = atan(u[N + i], u[i])
        re += cos(φ);  im += sin(φ)
    end
    return sqrt(re^2 + im^2) / N
end

# ============================================================================
# Mapper: fixed network (L) and fixed coupling (K)
# Returns 1 (sync) or 0
# ============================================================================

struct RosslerSyncMapper
    N::Int
    r_thresh::Float64
    T_transient::Float64
    T_measure::Float64
    T_total::Float64
    ds::CoupledODEs
end

function (m::RosslerSyncMapper)(u0)
    y, t = trajectory(m.ds, m.T_total, u0; Ttr = m.T_transient)
    r = mean(phase_order_parameter(row, m.N) for row in y)
    isnan(r) && return 0
    return r > m.r_thresh ? 1 : 0
end

# ============================================================================
# Trivial mapper — returned when the network is linearly unstable (Iₛ empty)
# Always returns 0: no synchrony possible.
# ============================================================================

struct TrivialMapper end

(::TrivialMapper)(u0) = 0

# ============================================================================
# Helper: build WS graph and return (L, K_lo, K_hi), or nothing if unstable
# ============================================================================

function get_network_and_coupling(N, k_deg, p_val, graph_seed)
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

    return L, K_lo, K_hi
end

# ============================================================================
# Factory: continuation over p.
# At each p step: build a new WS network, draw one random K ∈ Iₛ.
# ============================================================================

function rossler_mapper_factory(N, k_deg, a, b, c,
                                r_thresh, T_transient, T_measure,
                                graph_seed, K)
    T_total = T_transient + T_measure
    function _get_mapper(p_val)
        result = get_network_and_coupling(N, k_deg, p_val, graph_seed)
        isnothing(result) && return TrivialMapper()
        L, K_lo, K_hi = result
        p      = RosslerParams(N, a, b, c, K, L)
        diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = Int(1e8), adaptive = false, dt = 0.1)
        ds     = CoupledODEs(rossler_network!, zeros(N * 3), p; diffeq)
        return RosslerSyncMapper(N, r_thresh, T_transient, T_measure, T_total, ds)
    end
    return _get_mapper
end

# ============================================================================
# Computation function (wrapped for produce_or_load)
# ============================================================================

function rossler_psweep(d)
    @unpack N_osc, k_degree, graph_seed, K, n_p, p_start, p_end,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            sparse_n, dense_n, n_tiles, λ = d

    global_bounds = vcat(
        [(-12.0, 12.0) for _ in 1:N_osc],
        [(-12.0, 12.0) for _ in 1:N_osc],
        [( -8.0, 35.0) for _ in 1:N_osc],
    )
    params = @strdict sparse_n dense_n n_tiles global_bounds λ

    p_range = exp10.(range(log10(p_start), log10(p_end); length = n_p))
    # K_rng   = Random.Xoshiro(K_seed)

    get_map = rossler_mapper_factory(
        N_osc, k_degree, a_ros, b_ros, c_ros,
        r_thresh, T_transient, T_measure,
        graph_seed, K
    )

    history_mean_S, history_var_S, history_max_llr, history_n_panics,
        full_history_S, history_volumes =
            estimate_entropy(params, p_range, GenericFactory(get_map))

    return @strdict(history_mean_S, history_var_S, history_max_llr,
                    history_n_panics, full_history_S,
                    history_volumes, p_range)
end

# ============================================================================
# Setup
# ============================================================================

# Network parameters — Menck & Kurths (2013): N=100, ⟨k⟩=8
N_osc      = 100
k_degree   = 8
graph_seed = 12345
K = 0.2

# Rössler parameters
a_ros, b_ros, c_ros = 0.2, 0.2, 9.0

# Integration settings
r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0

# Bayesian continuation (over p)
λ        = 0.7
sparse_n = 50
dense_n  = 100
n_tiles  = 1

# Rewiring probability sweep — log-spaced, p=0 excluded (degenerate spectrum)
p_start = 0.01
p_end   = 1.0
n_p     = 50

params = @strdict N_osc k_degree graph_seed K n_p p_start p_end a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ

data, file = produce_or_load(
    datadir("data"), params, rossler_psweep;
    prefix = "rossler_psweep", storepatch = false, suffix = "jld2", force = false
)

@unpack history_mean_S, history_var_S, history_max_llr, history_n_panics,
        full_history_S, history_volumes, p_range = data

# ============================================================================
# Plot — S_B(p) and LLR detector
# ============================================================================

sync_vol = [get(hv, 1, 0.0) for hv in history_volumes]
p_vec    = collect(p_range)
println("Loaded from: $file")

fig = Figure(size = (750, 650))

ax1 = Axis(fig[1, 1],
    title  = "Rössler network — WS(N=$N_osc, ⟨k⟩=$k_degree)",
    ylabel = "Basin stability  S_B",
    xscale = log10,
)
lines!(ax1, p_vec, sync_vol, color = :black, linewidth = 2)
scatter!(ax1, p_vec, sync_vol, color = :black, markersize = 6)
xlims!(ax1, p_start, p_end);  ylims!(ax1, 0, 1)

ax2 = Axis(fig[2, 1],
    ylabel = "η  (log Bayes factor)",
    xlabel = "Rewiring probability  p",
    xscale = log10,
)
lines!(ax2, p_vec, history_max_llr, color = :black, linewidth = 2)
hlines!(ax2, [0.0], color = :red, linestyle = :dash, linewidth = 1)
xlims!(ax2, p_start, p_end)

panic_idx = findall(>(0), history_n_panics)
if !isempty(panic_idx)
    scatter!(ax2, p_vec[panic_idx], history_max_llr[panic_idx],
             color = :red, markersize = 8, label = "panic")
end

save(plotsdir("rossler_sync_psweep.png"), fig)
println("Saved → scripts/rossler_sync_psweep.png")
println("S_B range     : ", extrema(sync_vol))
println("Panics total  : ", sum(history_n_panics))
