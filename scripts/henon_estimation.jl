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

    @unpack a_range, b, sparse_n, dense_n,  n_tiles, global_bounds, λ = d

    a = 1.0
    # For recurrence finding
    xg_rec = range(-4, 4, length = 1000)
    yg_rec = range(-4, 4, length = 1000)
    grid_rec = (xg_rec, yg_rec)

    get_map(a, atts)  = get_mapper(a, b, grid_rec, atts)

    # Do the estimation 
    history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes = estimate_entropy(params, a_range, get_map)

    return @strdict(history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes)
end

λ = 0.7       # Forgetting factor
sparse_n = 15      # routine monitoring samples
dense_n = sparse_n^2     # panic mode samples (re-learning)

# Tiling Configuration
n_tiles = 20 
global_bounds = ((-2.0, 2.0), (-2.0, 2.0))

# Parameters
ai = 1.0; af = 2.0; 
b = -0.3;  
al = 1000 # Steps
a_range = range(ai, af, length = al)


params = @strdict a_range b sparse_n dense_n n_tiles global_bounds λ

data, file = produce_or_load(
    datadir("data"), 
    params, 
    henon_bayes_continuation;
    prefix = "henon_bayes", storepatch = false,
    suffix = "jld2", force = true
)

@unpack history_mean_S, history_var_S, history_max_llr, history_n_panics, history_att, full_history_S, history_volumes = data

# Collect all basin labels that appear across all steps
all_labels = sort(collect(reduce(union, keys.(history_volumes))))
# Build per-label volume time series (missing → 0)
vol_series = Dict(k => [get(hv, k, 0.0) for hv in history_volumes] for k in all_labels)

# PLOTTING
fig = Figure(resolution = (800, 1000))

upper_band = history_mean_S .+ (3.0 .* sqrt.(history_var_S))
lower_band = history_mean_S .- (3.0 .* sqrt.(history_var_S))

# Global Entropy
ax1 = Axis(fig[1, 1], title = "Mean Basin Entropy", ylabel = "Sb")
lines!(ax1, a_range, history_mean_S, color = :black)
xlims!(ax1, ai, af)
band!(ax1, a_range, lower_band, upper_band,
        color = (:black, 0.2),
        label = "Confidence (±3σ)")

# Panic mode count (The Detector)
ax2 = Axis(fig[2, 1], title = "Panic Tiles per Step", ylabel = "# panics")
stairs!(ax2, a_range, history_n_panics, color = :red)
xlims!(ax2, ai, af)

# Basin volumes — stacked band chart
colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1], title = "Relative Basin Volumes", ylabel = "Volume fraction", xlabel = "a")
let lower = zeros(length(a_range))
    for (i, k) in enumerate(all_labels)
        upper = lower .+ vol_series[k]
        band!(ax3, a_range, lower, upper,
              color = (colors[mod1(i, length(colors))], 0.8),
              label = "Basin $k")
        lines!(ax3, a_range, upper, color = colors[mod1(i, length(colors))], linewidth = 0.8)
        lower = copy(upper)
    end
end
axislegend(ax3, position = :rt)
xlims!(ax3, ai, af)
ylims!(ax3, 0, 1)

# Entropy heatmap per box
ax4 = Axis(fig[4, 1], title = "Entropy per Box", xlabel = "a", ylabel = "Box ID")
heatmap!(ax4, a_range, 1:(n_tiles^2), full_history_S, colormap = :viridis)

save("tiling_entropy_monitor_henon.png", fig)
