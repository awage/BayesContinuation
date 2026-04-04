using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Random
using SparseArrays
using Graphs
using OrdinaryDiffEq: Vern9, ODEProblem, solve
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
function phase_order_parameter(u, N)
    re = 0.0; im = 0.0
    @inbounds for i in 1:N
        φ = atan(u[N + i], u[i])
        re += cos(φ);  im += sin(φ)
    end
    return sqrt(re^2 + im^2) / N
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
    T_total::Float64
    ds::CoupledODEs
end

function (m::RosslerSyncMapper)(u0)
    y, t = trajectory(m.ds, m.T_total, u0; Ttr = m.T_transient)
    r = mean(phase_order_parameter(row, m.N) for row in y)
    return r > m.r_thresh ? 1 : 0
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
    T_total = T_transient + T_measure
    function _get_mapper(K_val, _atts)
        p     = RosslerParams(N, a, b, c, K_val, L)
        diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = Int(1e8))
        ds    = CoupledODEs(rossler_network!, zeros(N * 3), p; diffeq)
        return RosslerSyncMapper(N, r_thresh, T_transient, T_measure, T_total,  ds)
    end
    return _get_mapper
end

# ============================================================================
# Setup
# ============================================================================

# Network parameters — Menck & Kurths (2013): N=100, ⟨k⟩=8
N_osc      = 100
k_degree   = 8
graph_seed = 12345

# Rössler parameters
a_ros, b_ros, c_ros = 0.2, 0.2, 9.0

# Integration settings
r_thresh    = 0.90      # sync if mean order parameter > r_thresh
T_transient = 100.0     
T_measure   = 200.0     
n_K_steps   = 40        # K values in Iₛ for the K-continuation

# Bayesian continuation (over K, for each fixed network)
λ        = 0.7
sparse_n = 40
dense_n  = 500
n_tiles  = 1            # single box — state space is 3N-dimensional

# IC box in 3N-dimensional space: x,y ∈ (-12,12), z ∈ (0,25)
global_bounds = vcat(
    [(-12.0, 12.0) for _ in 1:N_osc],
    [(-12.0, 12.0) for _ in 1:N_osc],
    [(  -8, 35.0) for _ in 1:N_osc],
)

# Rewiring probability sweep (outer loop — no continuation across p)
p_start = 0.01
p_end   = 1.0
n_p     = 30
p_range = exp10.(range(log10(p_start), log10(p_end); length = n_p))

params = @strdict N_osc k_degree sparse_n dense_n n_tiles global_bounds λ

# ============================================================================
# Outer loop over p: run K-continuation independently for each network
# ============================================================================

S_B_vec         = Float64[]          # basin stability averaged over K, per p
total_panics    = Int[]
p_valid         = Float64[]          # p values for which Iₛ is non-empty

for p_val in p_range
    result = get_network_and_coupling(N_osc, k_degree, p_val, graph_seed, n_K_steps)
    if isnothing(result)
        println("p=$(round(p_val, digits=3))  → linearly unstable, skipping")
        continue
    end
    L, K_range = result
    println("p=$(round(p_val, digits=3))  Iₛ=[$(round(first(K_range), digits=3)), $(round(last(K_range), digits=3))]")

    get_map = rossler_mapper_factory(
        N_osc, a_ros, b_ros, c_ros, L,
        r_thresh, T_transient, T_measure
    )

    _, _, _, history_n_panics, _, _, history_volumes =
        estimate_entropy(params, K_range, get_map)

    sync_fracs = [get(hv, 1, 0.0) for hv in history_volumes]
    push!(S_B_vec, mean(sync_fracs))
    push!(total_panics, sum(history_n_panics))
    push!(p_valid, p_val)
end

# ============================================================================
# Plot — mirrors Fig. 2 of Menck & Kurths (2013)
# ============================================================================

fig = Figure(size = (750, 500))

ax1 = Axis(fig[1, 1],
    title  = "Rössler network — WS(N=$N_osc, ⟨k⟩=$k_degree)",
    ylabel = "Basin stability  S_B",
    xlabel = "Rewiring probability  p",
    xscale = log10,
)
lines!(ax1, p_valid, S_B_vec, color = :black, linewidth = 2)
scatter!(ax1, p_valid, S_B_vec, color = :black, markersize = 6)
xlims!(ax1, p_start, p_end);  ylims!(ax1, 0, 1)

save(scriptsdir("rossler_sync_estimation.png"), fig)
println("Saved → scripts/rossler_sync_estimation.png")
println("S_B range : ", extrema(S_B_vec))
println("Total panics per p : ", total_panics)
