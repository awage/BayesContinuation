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

function rossler_network_jac!(J, u, p, t)
    (; N, a, b, c, K, L) = p
    z = view(u, 2N+1:3N)

    fill!(J, 0.0)
    # Blocks indexed as: x = 1:N, y = N+1:2N, z = 2N+1:3N
    @inbounds for i in 1:N
        # ∂ẋᵢ/∂xⱼ = −K Lᵢⱼ  (set via Laplacian below)
        # ∂ẋᵢ/∂yᵢ = −1
        J[i, N + i]  = -1.0
        # ∂ẋᵢ/∂zᵢ = −1
        J[i, 2N + i] = -1.0

        # ∂ẏᵢ/∂xᵢ = 1
        J[N + i, i]      = 1.0
        # ∂ẏᵢ/∂yᵢ = a
        J[N + i, N + i]  = a

        # ∂żᵢ/∂xᵢ = zᵢ
        J[2N + i, i]      = z[i]
        # ∂żᵢ/∂zᵢ = xᵢ − c
        J[2N + i, 2N + i] = u[i] - c
    end

    # ∂ẋᵢ/∂xⱼ = −K Lᵢⱼ  (top-left N×N block)
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
#
#        Var_t( p̄(t) )
#   R = ─────────────────────────────
#        mean_i( Var_t( p_i(t) ) )
#
# R = 1  → perfect synchrony;  R ≈ 1/N  → incoherent (independent nodes).
# Uses x-components as the observable (coupling variable).
# Takes the full trajectory (iterable of state vectors of length 3N).
function golomb_rinzel_coherence(X)
    # X is T × N, x-components only
    x_bar       = vec(mean(X; dims=2))  # mean over nodes at each time step → length T
    var_pbar    = var(x_bar)            # Var_t(x̄)
    mean_var_pi = mean(var(X; dims=1))  # mean_i( Var_t(xᵢ) )
    mean_var_pi < 1e-12 && return 1.0
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

# ============================================================================
# Trivial mapper — returned when the network is linearly unstable (Iₛ empty)
# Always returns 0: no synchrony possible.
# ============================================================================

struct TrivialMapper end

(::TrivialMapper)(u0) = 0

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
                                r_thresh, T_transient, T_measure;
                                diverge_thresh = 1e4)
    f = ODEFunction(rossler_network!; jac = rossler_network_jac!)
    return K_val -> RosslerSyncMapper(N, r_thresh, T_transient, T_measure,
                                      RosslerParams(N, a, b, c, K_val, L), f, diverge_thresh)
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
        full_history_S, full_history_llr, history_volumes =
            estimate_entropy(params, K_range, GenericFactory(get_map))

    return @strdict(history_mean_S, history_var_S, history_max_llr,
                    history_n_panics, full_history_S, full_history_llr,
                    history_volumes, K_range)
end

# ============================================================================
# Pure Monte Carlo sweep — reference computation (no Bayesian continuation)
#
# For each K in K_range, samples n_mc initial conditions uniformly from
# global_bounds and classifies each as sync (1) or not (0) via the mapper.
# Simulations for a given K are parallelised with @Threads.@threads.
# One mapper per thread is created to avoid races on the shared Lx buffer.
# ============================================================================

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

    get_map = rossler_mapper_factory(
        N_osc, a_ros, b_ros, c_ros, L,
        r_thresh, T_transient, T_measure
    )

    sync_fracs = zeros(length(K_range))

    for (k_idx, K_val) in enumerate(K_range)
        # One mapper per thread — each has its own Lx buffer, avoiding races.
        mappers = [get_map(K_val) for _ in 1:Threads.nthreads()]
        hits    = zeros(Int, n_mc)

        @showprogress @Threads.threads for i in 1:n_mc
            tid = Threads.threadid()
            u0  = [lo + rand() * (hi - lo) for (lo, hi) in global_bounds]
            hits[i] = mappers[tid](u0)
        end

        sync_fracs[k_idx] = mean(hits)
        println("  K=$(round(K_val, digits=4))  sync_frac=$(round(sync_fracs[k_idx], digits=3))")
    end

    return @strdict(sync_fracs, K_range)
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
r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0
n_K_steps   = 50

# Bayesian continuation (over K)
λ        = 0.7
sparse_n = 50
dense_n  = 500
n_tiles  = 1

p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

for p_val in p_vals

    params = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ

    try 
        data, file = produce_or_load(
            datadir("data"), params, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        @unpack history_mean_S, history_var_S, history_max_llr, history_n_panics,
                full_history_S, history_volumes, K_range = data
        # ============================================================================
        # Plot — sync fraction and LLR detector across K sweep
        # ============================================================================

        sync_fracs = [get(hv, 1, 0.0) for hv in history_volumes]
        K_vec      = collect(K_range)
        println("Loaded from: $file")

        fig = Figure(size = (750, 650))

        ax1 = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsvisible = false, ylabelsize = 20,
            # title  = "Rössler network — WS(N=$N_osc, ⟨k⟩=$k_degree, p=$p_val)",
            ylabel = L"S_B",
        )
        lines!(ax1, K_vec, sync_fracs, color = :black, linewidth = 2)
        scatter!(ax1, K_vec, sync_fracs, color = :black, markersize = 5)
        ylims!(ax1, 0, 1)

        ax2 = Axis(fig[2, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"\eta  \text{(log Bayes factor)}",
            xlabel = L"K",
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

# ============================================================================
# Monte Carlo reference sweep
# ============================================================================

n_mc = 5   # IC samples per K step

for p_val in p_vals

    params_mc = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure n_mc

    try
        data, file = produce_or_load(
            datadir("data"), params_mc, rossler_Ksweep_montecarlo;
            prefix = "rossler_Ksweep_mc", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        @unpack sync_fracs, K_range = data
        K_vec = collect(K_range)
        println("MC loaded from: $file")

        fig = Figure(size = (750, 400))
        ax  = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"S_B \text{ (MC)}",
            xlabel = L"K",
        )
        lines!(ax, K_vec, sync_fracs, color = :black, linewidth = 2)
        scatter!(ax, K_vec, sync_fracs, color = :black, markersize = 5)
        ylims!(ax, 0, 1)

        save(plotsdir(savename("rossler_sync_Ksweep_mc", (; p = p_val), "png")), fig)
        println("Saved MC plot  →  rossler_sync_Ksweep_mc_p=$(p_val).png")
        println("Sync fraction range : ", extrema(sync_fracs))
    catch e
        @warn "MC sweep failed for p=$p_val" exception=e
    end
end
