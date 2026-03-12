using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Attractors
using Random
using Graphs
using OrdinaryDiffEq: Tsit5
using ProgressMeter

include(srcdir("bayes_entropy_est.jl"))

function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, incidence, P) = p
    ωs = view(u, N+1:2N)
    du[1:N] .= ωs
    sine_term = K .* (incidence * sin.(incidence' * u[1:N]))
    @. du[N+1:end] .= P - α*ωs - sine_term
    return nothing
end

mutable struct KuramotoParameters{M}
    N::Int
    α::Float64
    incidence::M
    P::Vector{Float64}
    K::Float64
end

function KuramotoParameters(; N = 10, α = 0.1, K = 6.0, seed = 53867481290)
    rng = Random.Xoshiro(seed)
    g = random_regular_graph(N, 3; rng)
    incidence = incidence_matrix(g, oriented=true)
    P = [isodd(i) ? +1.0 : -1.0 for i = 1:N]
    return KuramotoParameters(N, α, incidence, P, K)
end

# Project to (θ₁, ω₁) — the phase and velocity of oscillator 1.
# This 2D cross-section is compatible with the Bayesian tiling infrastructure.
function get_mapper_kuramoto(K_val, N, grid_rec, atts = nothing)
    p = KuramotoParameters(; N, K = K_val)
    diffeq = (alg = Tsit5(), reltol = 1e-9, maxiters = 1e6)
    ds = CoupledODEs(second_order_kuramoto!, zeros(2*N), p; diffeq)

    # Map full 2N state → 2D: (θ₁, ω₁)
    _proj_state(y) = [y[1], y[N+1]]
    # Map 2D back to full 2N: set all other angles and velocities to 0
    _complete(y) = length(y) == 2 ? [y[1]; zeros(N-1); y[2]; zeros(N-1)] : y
    psys = ProjectedDynamicalSystem(ds, _proj_state, _complete)

    mapper = AttractorsViaRecurrences(psys, grid_rec; sparse = true, Δt = 1.,
        show_progress = false, mx_chk_fnd_att = 100,
        mx_chk_safety = Int(1e7),
        force_non_adaptive = true,
        Ttr = 400.)

    if !isnothing(atts) && !isempty(atts)
        seed_mapper!(mapper, atts)
        mtch = MatchBySSSetDistance(; distance = Hausdorff(), threshold = Inf, use_vanished = false)
        matching_map!(mapper.bsn_nfo.BoA.attractors, atts, mtch)
    end

    return mapper
end

function kuramoto_bayes_continuation(params)
    @unpack K_range, N, SPARSE_N, DENSE_N, BAYES_FACTOR, N_TILES, GLOBAL_BOUNDS, LAMBDA = params

    (xmin, xmax), (ymin, ymax) = GLOBAL_BOUNDS
    xg_rec = range(xmin, xmax; length = 201)
    yg_rec = range(ymin, ymax; length = 201)
    grid_rec = (xg_rec, yg_rec)

    get_map(K, atts) = get_mapper_kuramoto(K, N, grid_rec, atts)

    history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes =
        estimate_entropy(params, K_range, get_map)

    return @strdict(history_mean_S, history_var_S, history_max_llr, history_n_panics,
                    history_att, full_history_S, history_volumes)
end


# Bayesian entropy monitoring params
BETA = 0.5
LAMBDA = 0.7
SPARSE_N = 20
DENSE_N = SPARSE_N^2
BAYES_FACTOR = 5.0

N_TILES = 8
# Bounds for the (θ₁, ω₁) cross-section
GLOBAL_BOUNDS = ((-pi, pi), (-15.0, 15.0))

# Kuramoto parameters
N = 10   # Number of oscillators (system dimension = 2N)
Ki = 0.0
Kf = 10.0
Kl = 50
K_range = range(Ki, Kf; length = Kl)

params = @strdict K_range N SPARSE_N DENSE_N BAYES_FACTOR N_TILES GLOBAL_BOUNDS LAMBDA

data, file = produce_or_load(
    datadir("data"),
    params,
    kuramoto_bayes_continuation;
    prefix = "kuramoto_bayes", storepatch = false,
    suffix = "jld2", force = true
)

@unpack history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes = data

println("Done. Mean entropy range: ", extrema(history_mean_S))
println("Max LLR range: ", extrema(history_max_llr))


using CairoMakie

all_labels = sort(collect(reduce(union, keys.(history_volumes))))
vol_series = Dict(k => [get(hv, k, 0.0) for hv in history_volumes] for k in all_labels)

fig = Figure(resolution = (800, 1000))

upper_band = history_mean_S .+ (3.0 .* sqrt.(history_var_S))
lower_band = history_mean_S .- (3.0 .* sqrt.(history_var_S))

# Global Entropy
ax1 = Axis(fig[1, 1], title = "Mean Basin Entropy", ylabel = "Sb")
lines!(ax1, K_range, history_mean_S, color = :black)
xlims!(ax1, Ki, Kf)
band!(ax1, K_range, lower_band, upper_band,
      color = (:black, 0.2), label = "Confidence (±3σ)")

# Panic mode count (The Detector)
ax2 = Axis(fig[2, 1], title = "Panic Tiles per Step", ylabel = "# panics")
stairs!(ax2, K_range, history_n_panics, color = :red)
xlims!(ax2, Ki, Kf)

# Basin volumes — stacked band chart
colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1], title = "Relative Basin Volumes", ylabel = "Volume fraction", xlabel = "K (coupling)")
let lower = zeros(length(K_range))
    for (i, k) in enumerate(all_labels)
        upper = lower .+ vol_series[k]
        band!(ax3, K_range, lower, upper,
              color = (colors[mod1(i, length(colors))], 0.8),
              label = "Basin $k")
        lines!(ax3, K_range, upper, color = colors[mod1(i, length(colors))], linewidth = 0.8)
        lower = copy(upper)
    end
end
axislegend(ax3, position = :rt)
xlims!(ax3, Ki, Kf)
ylims!(ax3, 0, 1)

# Entropy heatmap per box
ax4 = Axis(fig[4, 1], title = "Entropy per Box", xlabel = "K (coupling)", ylabel = "Box ID")
heatmap!(ax4, K_range, 1:(N_TILES^2), full_history_S, colormap = :viridis)

save("tiling_entropy_monitor_kuramoto.png", fig)
