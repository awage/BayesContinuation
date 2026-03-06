using DrWatson
@quickactivate
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using Attractors
using ProgressMeter

include(srcdir("bayes_entropy_est.jl"))

function henon_bayes_continuation(d)

    @unpack a_range, b, SPARSE_N, DENSE_N, BAYES_FACTOR, N_TILES, GLOBAL_BOUNDS, LAMBDA = d

    a = 1.0
    # For recurrence finding
    xg_rec = range(-4, 4, length = 1000)
    yg_rec = range(-4, 4, length = 1000)
    grid_rec = (xg_rec, yg_rec)

    get_map(a, atts)  = get_mapper(a, b, grid_rec, atts)

    # Do the estimation 
    history_mean_S, history_var_S, history_max_llr, history_att, full_history_S = estimate_entropy(params, a_range, get_map) 

    return @strdict(history_mean_S, history_var_S, history_max_llr, history_att, full_history_S)
end

BETA = 0.5
LAMBDA = 0.7       # Forgetting factor
SPARSE_N = 10      # Routine monitoring samples
DENSE_N = SPARSE_N^2     # Panic mode samples (re-learning)
BAYES_FACTOR = 5 # Threshold to trigger Panic Mode

# Tiling Configuration
N_TILES = 8 
GLOBAL_BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))

# Parameters
ai = 1.0; af = 2.0; 
b = -0.3;  
al = 150 # Steps
a_range = range(ai, af, length = al)


params = @strdict a_range b SPARSE_N DENSE_N BAYES_FACTOR N_TILES GLOBAL_BOUNDS LAMBDA

data, file = produce_or_load(
    datadir("data"), 
    params, 
    henon_bayes_continuation;
    prefix = "henon_bayes", storepatch = false,
    suffix = "jld2", force = true
)

@unpack history_mean_S, history_var_S, history_max_llr, history_att, full_history_S = data

# PLOTTING
fig = Figure(resolution = (800, 800))

upper_band = history_mean_S .+ (3.0 .* sqrt.(history_var_S))
lower_band = history_mean_S .- (3.0 .* sqrt.(history_var_S))

    #  Global Entropy
ax1 = Axis(fig[1, 1], title = "Mean Basin Entropy", ylabel = "Sb")
lines!(ax1, a_range, history_mean_S, color = :black)
xlims!(ax1, ai, af)
band!(ax1, a_range, lower_band, upper_band, 
        color = (:black, 0.2), # Transparent gray
        label = "Confidence (±3σ)"
    )

# Max KL Divergence (The Detector)
ax2 = Axis(fig[2, 1], title = "Max Divergence", ylabel = "D_W")
lines!(ax2, a_range, history_max_llr, color = :red)
# hlines!(ax2, [BAYES_FACTOR], color = :gray, linestyle = :dash, label="Panic Threshold")
xlims!(ax2, ai, af)

# Visualizing how entropy evolves in the boxes over time (flattened)
ax3 = Axis(fig[3, 1], title = "Entropy per Box", xlabel="a", ylabel="Box ID")
heatmap!(ax3, a_range, 1:(N_TILES^2), full_history_S, colormap=:viridis)

save("tiling_entropy_monitor.png", fig)
