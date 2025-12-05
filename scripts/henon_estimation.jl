using DrWatson
@quickactivate
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using Attractors
using ProgressMeter

include(srcdir("bayes_entropy_est.jl"))

BETA = 0.5
LAMBDA = 0.7       # Forgetting factor
SPARSE_N = 20      # Routine monitoring samples
DENSE_N = SPARSE_N^2     # Panic mode samples (re-learning)
KL_THRESHOLD = 0.7 # Threshold to trigger Panic Mode

# Tiling Configuration
N_TILES = 20 
GLOBAL_BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))

# Parameters
ai = 1.0; af = 2.0; 
b = -0.3;  
al = 200 # Steps
a_range = range(ai, af, length = al)

# For recurrence finding
xg_rec = range(-4, 4, length = 1000)
yg_rec = range(-4, 4, length = 1000)
grid_rec = (xg_rec, yg_rec)


params = @strdict SPARSE_N DENSE_N KL_THRESHOLD N_TILES GLOBAL_BOUNDS LAMBDA
get_map(a, atts)  = get_mapper(a, b, grid_rec, atts)

# Do the estimation 
history_mean_S, history_var_S, history_max_KL, history_att, full_history_S = estimate_entropy(params, a_range, get_map) 

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
ax2 = Axis(fig[2, 1], title = "Max Spatial Surprise (KL)", ylabel = "KL")
lines!(ax2, a_range, history_max_KL, color = :red)
hlines!(ax2, [KL_THRESHOLD], color = :gray, linestyle = :dash, label="Panic Threshold")
xlims!(ax2, ai, af)

# Visualizing how entropy evolves in the boxes over time (flattened)
ax3 = Axis(fig[3, 1], title = "Entropy per Box", xlabel="a", ylabel="Box ID")
heatmap!(ax3, a_range, 1:(N_TILES^2), full_history_S, colormap=:viridis)

save("tiling_entropy_monitor.png", fig)
