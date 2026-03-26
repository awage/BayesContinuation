using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using Statistics
using Attractors
using Random
using SparseArrays
using Graphs
using OrdinaryDiffEq:Vern9
using ProgressMeter

include(srcdir("bayes_entropy_est.jl"))
mutable struct KuramotoParameters{M}
    N::Int
    α::Float64
    Δ::M
    ΔT::M
    P::Vector{Float64}
    K::Float64
    # Both of these are dummies
    x::Vector{Float64}
    y::Vector{Float64}
end
function KuramotoParameters(; N, K, α = 0.1, seed = 53867481290)
    rng = Random.Xoshiro(seed)
    g = random_regular_graph(N, 3; rng)
    Δ = incidence_matrix(g, oriented=true)
    P = [isodd(i) ? +1.0 : -1.0 for i = 1:N]
    x = Δ' * zeros(N)
    y = zeros(N)
    ΔT = sparse(Matrix(Δ'))
    return KuramotoParameters(N, α, Δ, ΔT, P, K, x, y)
end
using LinearAlgebra: mul!
function second_order_kuramoto!(du, u, p, t)
    (; N, α, K, Δ, ΔT, P, x, y) = p
    φs = view(u, 1:N)
    ωs = view(u, N+1:2N)
    dφs = view(du, 1:N)
    dωs = view(du, N+1:2N)
    dφs .= ωs
    mul!(x, ΔT, φs)
    x .= sin.(x)
    mul!(y, Δ, x)
    y .*= K
    # the full sine term is y now.
    @. dωs = P - α*ωs - y
    return nothing
end


# Project to (θ₁, ω₁) — the phase and velocity of oscillator 1.
# This 2D cross-section is compatible with the Bayesian tiling infrastructure.
function get_mapper_kuramoto(K_val, N, grid_rec, atts = nothing)
    p = KuramotoParameters(; N, K = K_val)
    diffeq = (alg = Vern9(), reltol = 1e-9, maxiters = 1e8)
    ds = CoupledODEs(second_order_kuramoto!, zeros(2*N), p; diffeq)

    _complete(y) = (length(y) == N) ? zeros(2*N) : y;
    _proj_state(y) = y[N+1:2*N]
    psys = ProjectedDynamicalSystem(ds, _proj_state, _complete)
    yg = range(-15, 15; length = 51)
    grid = ntuple(x -> yg, dimension(psys))
    mapper = AttractorsViaRecurrences(psys, grid;  Δt = 0.1,
        consecutive_recurrences = 800, consecutive_attractor_steps = 10)

    if !isnothing(atts) && !isempty(atts)
        seed_mapper!(mapper, atts)
        mtch = MatchBySSSetDistance(; distance = Hausdorff(), threshold = Inf, use_vanished = false)
        matching_map!(mapper.bsn_nfo.BoA.attractors, atts, mtch)
    end

    return mapper
end

function kuramoto_bayes_continuation(params)
    @unpack K_range, N, sparse_n, dense_n, n_tiles, global_bounds, λ = params

    get_map(K, atts) = get_mapper_kuramoto(K, N, nothing, atts)

    history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes =
        estimate_entropy(params, K_range, get_map)

    return @strdict(history_mean_S, history_var_S, history_max_llr, history_n_panics,
                    history_att, full_history_S, history_volumes)
end


# Bayesian entropy monitoring params
λ = 0.7
sparse_n = 40
dense_n = sparse_n^2

n_tiles = 1

# Kuramoto parameters
N = 10   # Number of oscillators (system dimension = 2N)
# Initial conditions span the full 2N-dimensional state space:
# first N dims = angles ∈ (-π, π), next N dims = velocities ∈ (-pi, pi)
global_bounds = vcat([(-pi, pi) for _ in 1:N], [(-pi, pi) for _ in 1:N])
Ki = 0.0
Kf = 10.0
Kl = 100
K_range = range(Ki, Kf; length = Kl)

params = @strdict K_range N sparse_n dense_n n_tiles global_bounds λ

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
heatmap!(ax4, K_range, 1:size(full_history_S, 2), full_history_S, colormap = :viridis)

save("tiling_entropy_monitor_kuramoto.png", fig)
