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
    @unpack N_osc, k_degree, graph_seed, n_avg, p_val, n_K_steps,
            a_ros, b_ros, c_ros, r_thresh, T_transient, T_measure,
            sparse_n, dense_n, n_tiles, λ = d

    global_bounds = vcat(
        [(-12.0, 12.0) for _ in 1:N_osc],
        [(-12.0, 12.0) for _ in 1:N_osc],
        [( -8.0, 35.0) for _ in 1:N_osc],
    )
    params = @strdict sparse_n dense_n n_tiles global_bounds λ

    # Accumulate results over n_avg network realisations
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
        isnothing(result) && continue   # skip linearly-unstable realisations

        L, K_range = result
        println("p=$p_val  seed=$seed  Iₛ=[$(round(first(K_range), digits=3)), $(round(last(K_range), digits=3))]")

        get_map = rossler_mapper_factory(
            N_osc, a_ros, b_ros, c_ros, L,
            r_thresh, T_transient, T_measure
        )

        res = estimate_entropy(params, K_range, GenericFactory(get_map))
        h_mean_S   = res.history_mean_S
        h_var_S    = res.history_var_S
        h_max_llr  = res.history_max_llr
        h_n_panics = res.history_n_panics
        fh_S       = res.full_history_S
        fh_llr     = res.full_history_llr
        h_volumes  = res.history_volumes
        h_vol_var  = res.history_vol_var

        K_vec = collect(K_range)

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

    # Average
    acc_mean_S   ./= n_valid
    acc_var_S    ./= n_valid
    acc_max_llr  ./= n_valid
    acc_n_panics ./= n_valid
    acc_full_S   ./= n_valid
    acc_full_llr ./= n_valid
    acc_K        ./= n_valid
    for hv in acc_volumes
        for k in keys(hv); hv[k] /= n_valid; end
    end
    for vv in acc_vol_var
        for k in keys(vv); vv[k] /= n_valid; end
    end

    K_range = range(first(acc_K), last(acc_K); length = n_K_steps)

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
sparse_n = 20
dense_n  = 200
n_tiles  = 1
n_avg    = 10   # number of network realisations to average over

p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

for p_val in p_vals

    params = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ

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

n_mc = 500   # IC samples per K step

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
        
        n_avg = 1 # Single run
        params = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
        # Bayesian Rossler refference: 
        data, file = produce_or_load(
            datadir("data"), params, rossler_Ksweep;
            prefix = "rossler_Ksweep_sing", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        @unpack history_n_panics, history_max_llr, history_volumes, K_range, history_vol_var = data
        bayes_sync_fracs = [get(hv, 1, 0.0) for hv in history_volumes]
        bayes_vol_std    = sqrt.([ get(vv, 1, 0.0) for vv in history_vol_var ])
        # MC binomial std: √(p̂(1−p̂)/n_mc)
        mc_std = sqrt.(sync_fracs .* (1 .- sync_fracs) ./ n_mc)

        fig = Figure(size = (750, 400))
        ax  = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"S_B",
            xlabel = L"K",
        )
        band!(ax, K_vec, sync_fracs .- mc_std, sync_fracs .+ mc_std, color = (:black, 0.2))
        lines!(ax, K_vec, sync_fracs, color = :black, linewidth = 2, label = "MC")
        band!(ax, K_vec, bayes_sync_fracs .- bayes_vol_std, bayes_sync_fracs .+ bayes_vol_std,
              color = (:red, 0.2))
        lines!(ax, K_vec, bayes_sync_fracs, color = :red, linewidth = 2, label = "Bayes")
        axislegend(ax; position = :lt)
        ylims!(ax, 0, 1)

        save(plotsdir(savename("rossler_sync_Ksweep_mc", (; p = p_val), "png")), fig)
        println("Saved MC plot  →  rossler_sync_Ksweep_mc_p=$(p_val).png")
        println("Sync fraction range : ", extrema(sync_fracs))


        fig = Figure(size = (750, 400))
        ax = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"\eta  \text{(log Bayes factor)}",
            xlabel = L"K",
        )
        lines!(ax, K_vec, history_max_llr, color = :black, linewidth = 2)
        hlines!(ax, [0.0], color = :red, linestyle = :dash, linewidth = 1)

        panic_idx = findall(>(0), history_n_panics)
        if !isempty(panic_idx)
            scatter!(ax, K_vec[panic_idx], history_max_llr[panic_idx],
                     color = :red, markersize = 8, label = "panic")
        end

        save(plotsdir(savename("rossler_sync_Ksweep_alarms",(;p = p_val),"png")), fig)


    catch e
        @warn "MC sweep failed for p=$p_val" exception=e
    end
end

# ============================================================================
# Distance (RMSE over K) between MC and Bayes sync fractions, as a function of p
# ============================================================================

p_distance = Float64[]
dist_rmse  = Float64[]
dist_err   = Float64[]   # SE[RMSE] from MC binomial variance (delta method)

n_avg = 1  # single-run Bayes used as reference

for p_val in p_vals
    try
        params_mc = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure n_mc
        data_mc, _ = produce_or_load(
            datadir("data"), params_mc, rossler_Ksweep_montecarlo;
            prefix = "rossler_Ksweep_mc", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        params_bayes = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
        data_bayes, _ = produce_or_load(
            datadir("data"), params_bayes, rossler_Ksweep;
            prefix = "rossler_Ksweep_sing", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        mc_fracs    = data_mc["sync_fracs"]
        bayes_fracs = [get(hv, 1, 0.0) for hv in data_bayes["history_volumes"]]

        diff_sq  = (mc_fracs .- bayes_fracs).^2
        rmse     = sqrt(mean(diff_sq))
        # Delta-method: Var[RMSE] ≈ mean(diff_sq * var_mc) / RMSE²
        var_mc   = mc_fracs .* (1 .- mc_fracs) ./ n_mc
        se_rmse  = rmse > 1e-12 ? sqrt(mean(diff_sq .* var_mc)) / rmse : 0.0

        push!(p_distance, p_val)
        push!(dist_rmse,  rmse)
        push!(dist_err,   se_rmse)
    catch e
        @warn "Distance computation failed for p=$p_val" exception=e
    end
end

if !isempty(p_distance)
    fig_dist = Figure(size = (750, 400))
    ax_dist  = Axis(fig_dist[1, 1],
        yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
        ylabel = L"\mathrm{RMSE}(S_B^{\mathrm{MC}},\, S_B^{\mathrm{Bayes}})",
        xlabel = L"p",
    )
    lines!(ax_dist,   p_distance, dist_rmse, color = :black, linewidth = 2)
    scatter!(ax_dist, p_distance, dist_rmse, color = :black, markersize = 6)
    ylims!(ax_dist, 0, nothing)
    save(plotsdir("rossler_sync_distance_vs_p.png"), fig_dist)
    println("Saved distance plot → rossler_sync_distance_vs_p.png")
end

# ============================================================================
# Alarm probability vs p
#
# For each p: load the averaged (n_avg=10) Ksweep data and compute
#   P_alarm(p) = sum(history_n_panics) / n_K_steps
# ============================================================================

n_avg      = 10
p_alarm    = Float64[]
prob_alarm = Float64[]

for p_val in p_vals
    params_alarm = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    try
        data, _ = produce_or_load(
            datadir("data"), params_alarm, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        h_panics = data["history_n_panics"]
        push!(p_alarm,    p_val)
        push!(prob_alarm, sum(h_panics) / n_K_steps)
    catch e
        @warn "Alarm probability failed for p=$p_val" exception=e
    end
end

if !isempty(p_alarm)
    fig_alarm = Figure(size = (650, 400))
    ax_alarm  = Axis(fig_alarm[1, 1],
        yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
        ylabel = L"P_{\mathrm{alarm}}",
        xlabel = L"p",
    )
    lines!(ax_alarm,   p_alarm, prob_alarm, color = :steelblue, linewidth = 2)
    scatter!(ax_alarm, p_alarm, prob_alarm, color = :steelblue, markersize = 6)
    ylims!(ax_alarm, 0, nothing)
    save(plotsdir("rossler_alarm_prob_vs_p.png"), fig_alarm)
    println("Saved alarm probability plot → rossler_alarm_prob_vs_p.png")
end
