using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using OrdinaryDiffEq:Vern9
using Statistics
using StaticArrays
using Attractors
# using ProgressMeter
#
include(srcdir("BayesContinuation.jl"))
using .BayesContinuation
include(srcdir("compute.jl"))

@inline @inbounds function duffing(u, p, t)
    d = p[1]; F = p[2]; omega = p[3]
    du1 = u[2]
    du2 = -d*u[2] + u[1] - u[1]^3 + F*sin(omega*t)
    return SVector{2}(du1, du2)
end

function get_mapper_duffing(d, F, ω, grid_rec, atts = nothing; consecutive_recurrences = 800)
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

    @unpack ω_range, F, d, sparse_n, dense_n,  n_tiles, global_bounds = params
    
    # For recurrence finding
    xg_rec = range(-7, 7, length = 3001)
    yg_rec = range(-7, 7, length = 3001)
    grid_rec = (xg_rec, yg_rec)

    factory = AttractorMapperFactory((ω, atts) -> get_mapper_duffing(d, F, ω, grid_rec, atts))

    history_mean_S, history_var_S, history_max_llr, history_n_panics, full_history_S, full_history_llr, history_volumes = estimate_entropy(params, ω_range, factory)

    return @strdict(history_mean_S, history_var_S, history_max_llr, history_n_panics, full_history_S, full_history_llr, history_volumes)
end


# Bayesian entropy monitoring params
λ = 0.7
sparse_n = 20
dense_n = sparse_n^2

n_tiles = 15
global_bounds = ((-2.0, 2.0), (-2.0, 2.0))

# Duffing parameters
d = 0.2; F=0.2; ω=1.;  # smooth boundary


# Integrator resolution for the stroboscopic map

# Sweep forcing amplitude (gamma)
ωi = 0.2
ωf = 1.5
len = 200
ω_range = range(ωi, ωf, length = len)


params = @strdict ω_range d F sparse_n dense_n n_tiles global_bounds λ 

data, file = produce_or_load(
    datadir("data"), 
    params, 
    duffing_bayes_continuation;
    prefix = "duffing_bayes", storepatch = false,
    suffix = "jld2", force = false, 
    filename = hash
)

@unpack history_mean_S, history_var_S, history_max_llr, history_n_panics, full_history_S, history_volumes = data


println("Done. Mean entropy range: ", extrema(history_mean_S))
println("Max LLR range: ", extrema(history_max_llr))


using CairoMakie

lab_args = (;yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)
# Collect all basin labels that appear across all steps
all_labels = sort(collect(reduce(union, keys.(history_volumes))))
# Build per-label volume time series (missing → 0)
vol_series = Dict(k => [get(hv, k, 0.0) for hv in history_volumes] for k in all_labels)

# PLOTTING
fig = Figure(resolution = (800, 1000))

upper_band = history_mean_S .+ (3.0 .* sqrt.(history_var_S))
lower_band = history_mean_S .- (3.0 .* sqrt.(history_var_S))

# Global Entropy
ax1 = Axis(fig[1, 1];  ylabel = L"S_b", lab_args...)
lines!(ax1, ω_range, history_mean_S, color = :black)
xlims!(ax1, ωi, ωf)
band!(ax1, ω_range, lower_band, upper_band,
        color = (:black, 0.2),
        label = "Confidence (±3σ)")
Label(fig[1, 1, TopLeft()], "(a)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Panic mode count (The Detector)
ax2 = Axis(fig[2, 1];  ylabel = "# alarms", lab_args...)
stairs!(ax2, ω_range, history_n_panics, color = :red)
xlims!(ax2, ωi, ωf)
Label(fig[2, 1, TopLeft()], "(b)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Basin volumes — stacked band chart
colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1];  ylabel = "Volume fraction", xlabel = L"\omega", xlabelsize = 20, lab_args...)
let lower = zeros(length(ω_range))
    for (i, k) in enumerate(all_labels)
        upper = lower .+ vol_series[k]
        band!(ax3, ω_range, lower, upper,
              color = (colors[mod1(i, length(colors))], 0.8),
              label = "Basin $k")
        lines!(ax3, ω_range, upper, color = colors[mod1(i, length(colors))], linewidth = 0.8)
        lower = copy(upper)
    end
end
# let 
#     for (i, k) in enumerate(all_labels)
#         lines!(ax3, ω_range, vol_series[k], color = colors[mod1(i, length(colors))], linewidth = 0.8, 
#               label = "Basin $k")
#     end
# end
axislegend(ax3, position = :rt)
xlims!(ax3, ωi, ωf)
ylims!(ax3, 0, 1)
Label(fig[3, 1, TopLeft()], "(c)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)


# Entropy heatmap per box
# ax4 = Axis(fig[4, 1], title = "Entropy per Box", xlabel = L"\omega", ylabel = "Box ID")
# heatmap!(ax4, ω_range, 1:(n_tiles^2), full_history_S, colormap = :viridis)

save(plotsdir("fig3.png"), fig)
