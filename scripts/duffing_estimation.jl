using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using OrdinaryDiffEq:Vern9
using Statistics
using StaticArrays
using Attractors
using ProgressMeter

include(srcdir("bayes_entropy_est.jl"))

@inline @inbounds function duffing(u, p, t)
    d = p[1]; F = p[2]; omega = p[3]
    du1 = u[2]
    du2 = -d*u[2] + u[1] - u[1]^3 + F*sin(omega*t)
    return SVector{2}(du1, du2)
end

function get_mapper_duffing(d, F, ω, grid_rec, atts = nothing; consecutive_recurrences = 200)
    diffeq = (reltol = 1e-9, abstol = 1e-9,  alg = Vern9(), maxiters = 1e9)
    ds = CoupledODEs(duffing, rand(2), [d, F, ω]; diffeq)
    smap = StroboscopicMap(ds, 2*pi/ω)
    mapper = AttractorsViaRecurrences(smap, grid_rec; consecutive_recurrences)

    if !isnothing(atts) && !isempty(atts)
        seed_mapper!(mapper, atts)
        mtch = MatchBySSSetDistance(; distance = Hausdorff(), threshold = Inf, use_vanished = false)
        rmap = matching_map!(mapper.bsn_nfo.BoA.attractors, atts, mtch)
    end
    return mapper
end

function duffing_bayes_continuation(params)

    @unpack ω_range, F, d, SPARSE_N, DENSE_N, BAYES_FACTOR, N_TILES, GLOBAL_BOUNDS, LAMBDA = params
    
    # For recurrence finding
    xg_rec = range(-5, 5, length = 30001)
    yg_rec = range(-5, 5, length = 30001)
    grid_rec = (xg_rec, yg_rec)

    get_map(ω, atts) = get_mapper_duffing(d, F, ω, grid_rec, atts)

    history_mean_S, history_var_S, history_max_score, history_att, full_history_S = estimate_entropy(params, ω_range, get_map)

    return @strdict(history_mean_S, history_var_S, history_max_score, history_att, full_history_S)
end


# Bayesian entropy monitoring params
BETA = 0.5
LAMBDA = 0.7
SPARSE_N = 20
DENSE_N = SPARSE_N^2
BAYES_FACTOR = 5.0

N_TILES = 5
GLOBAL_BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))

# Duffing parameters
d = 0.2; F=0.2; ω=1.;  # smooth boundary


# Integrator resolution for the stroboscopic map

# Sweep forcing amplitude (gamma)
ωi = 0.1
ωf = 2.50
len = 150
ω_range = range(ωi, ωf, length = len)


params = @strdict ω_range d F SPARSE_N DENSE_N BAYES_FACTOR N_TILES GLOBAL_BOUNDS LAMBDA

data, file = produce_or_load(
    datadir("data"), 
    params, 
    duffing_bayes_continuation;
    prefix = "duffing_bayes", storepatch = false,
    suffix = "jld2", force = false
)

@unpack history_mean_S, history_var_S, history_max_score, history_att, full_history_S = data




println("Done. Mean entropy range: ", extrema(history_mean_S))
println("Max log Bayes factor range: ", extrema(history_max_score))


# PLOTTING
fig = Figure(resolution = (800, 800))

upper_band = history_mean_S .+ (3.0 .* sqrt.(history_var_S))
lower_band = history_mean_S .- (3.0 .* sqrt.(history_var_S))

    #  Global Entropy
ax1 = Axis(fig[1, 1], title = "Mean Basin Entropy", ylabel = "Sb")
lines!(ax1, ω_range, history_mean_S, color = :black)
xlims!(ax1, ωi, ωf)
band!(ax1, ω_range, lower_band, upper_band, 
        color = (:black, 0.2), # Transparent gray
        label = "Confidence (±3σ)"
    )

# Max KL Divergence (The Detector)
ax2 = Axis(fig[2, 1], title = "Max Spatial Bayes score", ylabel = "B_10")
lines!(ax2, ω_range, history_max_score, color = :red)
hlines!(ax2, [BAYES_FACTOR], color = :gray, linestyle = :dash, label="Panic Threshold")
xlims!(ax2, ωi, ωf)

# Visualizing how entropy evolves in the boxes over time (flattened)
ax3 = Axis(fig[3, 1], title = "Entropy per Box", xlabel=L"\omega", ylabel="Box ID")
heatmap!(ax3, ω_range, 1:(N_TILES^2), full_history_S, colormap=:viridis)

save("tiling_entropy_monitor_duffing.png", fig)
