"""
duffing_estimation.jl
=====================
Bayesian basin monitoring of the driven Duffing oscillator

    ẋ = y
    ẏ = -d y + x - x³ + F sin(ω t)

sweeping the driving frequency `ω` over [0.2, 1.5] at fixed `d = 0.2`, `F = 0.2`,
on the stroboscopic section at the forcing period.

Same shape as `henon_estimation.jl`: one basin map, one sampler, one
`global_continuation`, then `bayes_estimates(sampler)`. See that script for what the
old `estimate_entropy` interface used to do by hand and no longer has to.

## Why the rule is written in rescaled time

The stroboscopic section is at the forcing period `2π/ω`, so sweeping `ω` changes the
period of the map itself, not just a parameter of the vector field. A global
continuation advances by calling `set_parameters!` on a single long-lived system, and
the period of a `StroboscopicMap` is not one of its parameters.

Substituting `s = ω t` fixes this exactly. The equations become

    dx/ds = y / ω
    dy/ds = (-d y + x - x³ + F sin s) / ω

whose forcing period is `2π` for every `ω`, while `s = 2πn` is `t = 2πn/ω` — the same
instants the original section samples. So the map, its attractors and its basins are
unchanged, and `ω` is now an ordinary parameter that `set_parameters!` can move.
"""

using DrWatson
@quickactivate "BayesContinuation"
using LinearAlgebra
using OrdinaryDiffEq: Vern9
using Statistics
using StaticArrays
using Attractors

include(srcdir("BayesContinuation.jl"))
using .BayesContinuation

# Duffing in rescaled time `s = ω t`; see the docstring. p = [d, F, ω].
@inline @inbounds function duffing_rescaled(u, p, s)
    d = p[1]; F = p[2]; ω = p[3]
    du1 = u[2] / ω
    du2 = (-d * u[2] + u[1] - u[1]^3 + F * sin(s)) / ω
    return SVector{2}(du1, du2)
end

function duffing_bayes_continuation(params)
    @unpack ω_range, F, d, sparse_n, dense_n, n_tiles, global_bounds, λ, β = params

    # For recurrence finding
    grid_rec = (range(-7, 7; length = 3001), range(-7, 7; length = 3001))

    diffeq = (reltol = 1e-9, abstol = 1e-9, alg = Vern9(), maxiters = 1e9)
    ds = CoupledODEs(duffing_rescaled, [0.1, 0.1], [d, F, first(ω_range)]; diffeq)
    smap = StroboscopicMap(ds, 2π)          # fixed period, by the rescaling
    bmap = BasinMapRecurrences(smap, grid_rec;
        consecutive_recurrences = 1000, show_progress = false)

    # `history = true` is what makes the estimators recoverable afterwards: without it
    # the sampler overwrites `alphas` and `etas` at every parameter.
    sampler = BayesianUpdateSampler(global_bounds, n_tiles;
        sparse_n, dense_n, λ, β, seed = 20260802, history = true,
    )

    # Index 3 of `[d, F, ω]` is the swept parameter.
    pcurve = [Dict(3 => ω) for ω in ω_range]

    # Hausdorff with an infinite threshold: match every attractor to its nearest
    # predecessor however far it moved, as the original script did.
    algo = RecurrencesFindAndMatch(bmap; distance = Hausdorff(), threshold = Inf)
    fractions, attractors = global_continuation(algo, pcurve, sampler)

    est = bayes_estimates(sampler)
    # `@strdict` only takes bare names or `key = value`; a field access like
    # `est.mean_S` is rejected (at run time, after the whole sweep).
    return @strdict(
        fractions,
        mean_S = est.mean_S, var_S = est.var_S,
        min_eta = est.min_eta, n_panics = est.n_panics,
        volumes = est.volumes, vol_var = est.vol_var,
        full_S = est.full_S, full_eta = est.full_eta,
    )
end

# Bayesian entropy monitoring params
λ = 0.7
β = 0.5
sparse_n = 20
dense_n = sparse_n^2

n_tiles = 15
global_bounds = ((-2.0, 2.0), (-2.0, 2.0))

# Duffing parameters
d = 0.2; F = 0.2    # smooth boundary

# Sweep the driving frequency
ωi = 0.2
ωf = 1.5
len = 20
ω_range = range(ωi, ωf, length = len)

params = @strdict ω_range d F sparse_n dense_n n_tiles global_bounds λ β

data, file = produce_or_load(
    datadir("data"),
    params,
    duffing_bayes_continuation;
    prefix = "duffing_bayes", storepatch = false,
    suffix = "jld2", force = true,
    filename = hash
)

@unpack mean_S, var_S, min_eta, n_panics, full_S, volumes, fractions = data

println("Done. Mean entropy range: ", extrema(mean_S))
println("Minimum η range: ", extrema(min_eta))

using CairoMakie

lab_args = (; yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)
# Panel (c) shows the basin fractions measured by the continuation itself, not the
# posterior volumes `est.volumes` (which are still saved, for comparison).
all_labels = sort!(Int.(reduce(union, keys.(fractions))))
vol_series = volume_series(fractions, all_labels)

# PLOTTING
fig = Figure(size = (800, 1000))

upper_band = mean_S .+ (3.0 .* sqrt.(var_S))
lower_band = mean_S .- (3.0 .* sqrt.(var_S))

# Global entropy
ax1 = Axis(fig[1, 1]; ylabel = L"S_b", lab_args...)
lines!(ax1, ω_range, mean_S, color = :black)
xlims!(ax1, ωi, ωf)
band!(ax1, ω_range, lower_band, upper_band,
        color = (:black, 0.2),
        label = "Confidence (±3σ)")
Label(fig[1, 1, TopLeft()], "(a)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Panic mode count (the detector)
ax2 = Axis(fig[2, 1]; ylabel = "# alarms", lab_args...)
stairs!(ax2, ω_range, n_panics, color = :red)
xlims!(ax2, ωi, ωf)
Label(fig[2, 1, TopLeft()], "(b)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Basin fractions — stacked band chart
colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1]; ylabel = "Basin fraction", xlabel = L"\omega", lab_args...)
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
axislegend(ax3, position = :rt)
xlims!(ax3, ωi, ωf)
ylims!(ax3, 0, 1)
Label(fig[3, 1, TopLeft()], "(c)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Entropy heatmap per box
# ax4 = Axis(fig[4, 1], title = "Entropy per Box", xlabel = L"\omega", ylabel = "Box ID")
# heatmap!(ax4, ω_range, 1:(n_tiles^2), full_S, colormap = :viridis)

save(plotsdir("fig3.png"), fig)
