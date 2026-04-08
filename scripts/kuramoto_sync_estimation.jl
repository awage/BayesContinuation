using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Random
using OrdinaryDiffEq: Tsit5, ODEProblem, solve
using CairoMakie
using ProgressMeter

include(srcdir("BayesContinuation.jl"))
using .BayesContinuation

# ============================================================================
# First-order Kuramoto model (all-to-all coupling)
#   dθᵢ/dt = ωᵢ + (K/N) Σⱼ sin(θⱼ − θᵢ)
# ============================================================================

"""
Mean-field form: z = (1/N) Σ exp(iθⱼ), then dθᵢ/dt = ωᵢ + K·Im[z·exp(-iθᵢ)].
O(N) per evaluation instead of O(N²).
"""
function kuramoto!(dθ, θ, p, t)
    ω, K, N = p
    re_z = 0.0; im_z = 0.0
    @inbounds for j in 1:N
        re_z += cos(θ[j])
        im_z += sin(θ[j])
    end
    re_z /= N; im_z /= N
    @inbounds for i in 1:N
        dθ[i] = ω[i] + K * (im_z * cos(θ[i]) - re_z * sin(θ[i]))
    end
    return nothing
end

function kuramoto_order_parameter(θ)
    N = length(θ)
    re = 0.0; im = 0.0
    @inbounds for j in 1:N
        re += cos(θ[j])
        im += sin(θ[j])
    end
    return sqrt(re^2 + im^2) / N
end

# ============================================================================
# Mapper: callable struct wrapping the Kuramoto integrator
# ============================================================================

struct KuramotoSyncMapper
    ω::Vector{Float64}
    K_val::Float64
    N::Int
    r_thresh::Float64
    T_transient::Float64
    T_measure::Float64
end

function (m::KuramotoSyncMapper)(u0)
    p = (m.ω, m.K_val, m.N)
    T_total = m.T_transient + m.T_measure
    prob = ODEProblem(kuramoto!, u0, (0.0, T_total), p)
    sol = solve(prob, Tsit5(); save_everystep=false,
                saveat=m.T_transient:1.0:T_total,
                abstol=1e-6, reltol=1e-6)
    r_mean = mean(kuramoto_order_parameter(u) for u in sol.u)
    return r_mean > m.r_thresh ? 1 : 0
end

function get_mapper_kuramoto_sync(ω, N_osc, r_thresh, T_transient, T_measure)
    return K_val -> KuramotoSyncMapper(ω, K_val, N_osc, r_thresh, T_transient, T_measure)
end

# ============================================================================
# Run
# ============================================================================

# Kuramoto parameters
N_osc = 10
rng = Random.Xoshiro(10002)
ω = randn(rng, N_osc)       # natural frequencies (standard normal)

# Bayesian continuation parameters
λ = 0.7
sparse_n = 100
dense_n = sparse_n^2
n_tiles = 1                  # single box (no tiling in high-D)
global_bounds = [(-Float64(π), Float64(π)) for _ in 1:N_osc]

# Coupling range (log-spaced)
Ki = 1; Kf = 10; Kl = 50
K_range = 10 .^ range(0, 1; length=Kl)

# Sync threshold and integration times
r_thresh = 0.8
T_transient = 200.0
T_measure = 100.0

params = @strdict K_range sparse_n dense_n n_tiles global_bounds λ

factory = GenericFactory(get_mapper_kuramoto_sync(ω, N_osc, r_thresh, T_transient, T_measure))

history_mean_S, history_var_S, history_max_llr, history_n_panics,
    full_history_S, history_volumes =
        estimate_entropy(params, K_range, factory)

# ============================================================================
# Plot
# ============================================================================

sync_vol = [get(hv, 1, 0.0) for hv in history_volumes]

fig = Figure(size = (700, 600))

# Sync basin volume
ax1 = Axis(fig[1, 1], ylabel = "Sync basin volume", xscale = log10)
lines!(ax1, collect(K_range), sync_vol, color = :black)
xlims!(ax1, Ki, Kf)
ylims!(ax1, 0, 1)

# Log Bayes Factor
ax2 = Axis(fig[2, 1], ylabel = "η (log BF)", xlabel = "K (coupling)", xscale = log10)
lines!(ax2, collect(K_range), history_max_llr, color = :black)
hlines!(ax2, [0.0], color = :red, linestyle = :dash)
xlims!(ax2, Ki, Kf)

# Mark panics
panic_idx = findall(>(0), history_n_panics)
if !isempty(panic_idx)
    scatter!(ax2, collect(K_range)[panic_idx], history_max_llr[panic_idx],
             color = :red, markersize = 8)
end

save(scriptsdir("kuramoto_sync_estimation.png"), fig)
println("Plot saved to scripts/kuramoto_sync_estimation.png")
println("Sync volume range: ", extrema(sync_vol))
println("Panics triggered: ", sum(history_n_panics))
