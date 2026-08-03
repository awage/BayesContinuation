"""
henon_estimation.jl
===================
Reference example for the Bayesian basin monitoring, on the Hénon map

    x' = a - x² + b y
    y' = x

sweeping `a` over [1.0, 2.0] at fixed `b = -0.3`.

This is the template the other sweep scripts should follow, and it is deliberately
short: the continuation is a plain `Attractors.global_continuation`, and the whole of
this package's contribution is the `bayes_estimates(sampler)` line that reads the
figures off the history the sampler kept along the way.

Compared to the old `estimate_entropy` interface, four things are gone and nothing
replaces them:

  * the `MapperFactory` closure that rebuilt a mapper at every parameter — the
    continuation sets the parameters on a single `BasinMapRecurrences` and calls
    `reset_mapper!` itself;
  * the seeding of the new mapper from the previous attractors — step 1 of
    `AttractorSeedContinueMatch`;
  * the `matching_map!` call that kept the attractor IDs stable — the continuation
    loop matches each parameter against the previous one before the sampler ever
    sees a label, so the priors are automatically comparable across the sweep;
  * the sweep loop itself.

What the script has to provide is the basin map, the sampler, and the parameter
curve. Note `pcurve` is a vector of `index => value` dictionaries, which is how a
global continuation names parameters; here index 1 is `a` and index 2 is `b`.
"""

using DrWatson
@quickactivate
using CairoMakie
using Statistics
using Attractors
# include(srcdir("compute.jl"))
include(srcdir("BayesContinuation.jl"))
using .BayesContinuation

function henon_rule(u, p, n) # here `n` is "time", but we don't use it.
    x, y = u # system state
    a, b = p # system parameters
    xn = a - x^2 + b*y
    yn = x
    return SVector(xn, yn)
end

# One basin map for the whole sweep; the continuation re-parameterises and resets it.
# Both continuations below build it exactly the same way, so that the only difference
# between them is the initial condition sampler.
function henon_basin_map(a0, b)
    ds = DeterministicIteratedMap(henon_rule, [0.0, 0.0], [a0, b])
    grid_rec = (range(-4, 4; length = 1000), range(-4, 4; length = 1000))
    return BasinMapRecurrences(ds, grid_rec;
        consecutive_recurrences = 50000, Ttr = 5000, show_progress = false,
    )
end

henon_pcurve(a_range, b) = [Dict(1 => a, 2 => b) for a in a_range]

function henon_bayes_continuation(d)
    @unpack a_range, b, sparse_n, dense_n, n_tiles, global_bounds, λ, β = d

    bmap = henon_basin_map(first(a_range), b)

    # `history = true` is what makes the estimators recoverable afterwards: without it
    # the sampler overwrites `alphas` and `etas` at every parameter.
    sampler = BayesianUpdateSampler(global_bounds, n_tiles;
        sparse_n, dense_n, λ, β, seed = 20260802, history = true,
    )

    pcurve = henon_pcurve(a_range, b)

    fractions, attractors = global_continuation(
        RecurrencesFindAndMatch(bmap; distance = Hausdorff(), threshold = Inf), pcurve, sampler,
    )

    est = bayes_estimates(sampler)
    return @strdict(
        fractions,
        mean_S = est.mean_S, var_S = est.var_S,
        min_eta = est.min_eta, n_panics = est.n_panics,
        volumes = est.volumes, vol_var = est.vol_var,
        full_S = est.full_S, full_eta = est.full_eta,
    )
end

# The same continuation with the *vanilla* sampler: `n_ics` initial conditions drawn
# uniformly over the whole region at every parameter, no boxes, no priors, no
# re-sampling. This is what `global_continuation` does on its own, and it is the
# reference the Bayesian run has to be compared against: it estimates the same basin
# fractions, it just cannot say where in the region they changed, and it spends its
# whole budget at every parameter whether anything moved or not.
function henon_vanilla_continuation(d)
    @unpack a_range, b, n_ics, global_bounds = d

    bmap = henon_basin_map(first(a_range), b)

    # Same region the Bayesian sampler tiles, sampled uniformly instead.
    region = HRectangle(SVector(minimum.(global_bounds)), SVector(maximum.(global_bounds)))
    sampler = RandomICSampler(n_ics, region, 20260802)

    fractions, attractors = global_continuation(
        RecurrencesFindAndMatch(bmap; distance = Hausdorff(), threshold = Inf), henon_pcurve(a_range, b), sampler,
    )

    return @strdict fractions
end

λ = 0.7                  # forgetting factor
β = 0.5                  # Dirichlet base pseudo-count
sparse_n = 15            # routine monitoring samples per box
dense_n = sparse_n^2     # panic mode samples per box (re-learning)

# Tiling configuration
n_tiles = 20
global_bounds = ((-2.0, 2.0), (-2.0, 2.0))

# Parameters
ai = 1.0; af = 2.0
b = -0.3
al = 200                 # steps
a_range = range(ai, af, length = al)

params = @strdict a_range b sparse_n dense_n n_tiles global_bounds λ β

data, file = produce_or_load(
    datadir("data"),
    params,
    henon_bayes_continuation;
    prefix = "henon_bayes", storepatch = false,
    suffix = "jld2", force = false,
    filename = hash
)

@unpack mean_S, var_S, n_panics, volumes, fractions = data

all_labels = sort(collect(reduce(union, keys.(fractions))))
vol_series = volume_series(fractions, all_labels)

# PLOTTING
fig = Figure(size = (800, 1000))

lab_args = (; yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)

upper_band = mean_S .+ (3.0 .* sqrt.(var_S))
lower_band = mean_S .- (3.0 .* sqrt.(var_S))

# Global entropy
ax1 = Axis(fig[1, 1]; ylabel = L"S_b", lab_args...)
lines!(ax1, a_range, mean_S, color = :black)
xlims!(ax1, ai, af)
band!(ax1, a_range, lower_band, upper_band,
        color = (:black, 0.2),
        label = "Confidence (±3σ)")
Label(fig[1, 1, TopLeft()], "(a)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Panic mode count (the detector)
ax2 = Axis(fig[2, 1]; ylabel = "# alarms", lab_args...)
stairs!(ax2, a_range, n_panics, color = :red)
xlims!(ax2, ai, af)
Label(fig[2, 1, TopLeft()], "(b)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

# Basin volumes — stacked band chart
colors = Makie.wong_colors()
ax3 = Axis(fig[3, 1]; ylabel = "Volume fraction", xlabel = "a", lab_args...)
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
Label(fig[3, 1, TopLeft()], "(c)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

save(plotsdir("fig2.png"), fig)

# ----------------------------------------------------------------------------------------
# Comparison with the vanilla continuation
# ----------------------------------------------------------------------------------------
# Equal budget per parameter: one sparse round of the Bayesian sampler is `sparse_n`
# points in each of the `n_tiles^2` boxes. The Bayesian run spends more than this at the
# parameters where a box panics (`dense_n` extra points per flagged box, panel (b) above),
# and the same everywhere else.
n_ics = n_tiles^2 * sparse_n

params_vanilla = @strdict a_range b n_ics global_bounds

data_vanilla, file_vanilla = produce_or_load(
    datadir("data"),
    params_vanilla,
    henon_vanilla_continuation;
    prefix = "henon_vanilla", storepatch = false,
    suffix = "jld2", force = true,
    filename = hash
)

fractions_vanilla = data_vanilla["fractions"]

# One series per basin for each run, over the union of the labels the two runs found.
cmp_labels = sort(collect(union(reduce(union, keys.(fractions)),
                               reduce(union, keys.(fractions_vanilla)))))
vol_bayes = volume_series(fractions, cmp_labels)
vol_vanilla = volume_series(fractions_vanilla, cmp_labels)

println("Vanilla run: $n_ics initial conditions per parameter, ",
        length(cmp_labels), " basins over the sweep.")
for k in cmp_labels
    d = vol_bayes[k] .- vol_vanilla[k]
    println("  basin $k: mean |Δfraction| = ", round(mean(abs, d), digits = 4),
            ", max = ", round(maximum(abs, d), digits = 4))
end

figc = Figure(size = (800, 1000))

# (a) Bayesian fractions, (b) vanilla fractions: same stacked bands, same colours
for (row, (title, series)) in enumerate((
        ("Bayesian sampler", vol_bayes), ("Vanilla sampler", vol_vanilla)))
    ax = Axis(figc[row, 1]; ylabel = "Basin fraction", title, lab_args...)
    let lower = zeros(length(a_range))
        for (i, k) in enumerate(cmp_labels)
            upper = lower .+ series[k]
            band!(ax, a_range, lower, upper,
                  color = (colors[mod1(i, length(colors))], 0.8), label = "Basin $k")
            lines!(ax, a_range, upper, color = colors[mod1(i, length(colors))],
                   linewidth = 0.8)
            lower = copy(upper)
        end
    end
    xlims!(ax, ai, af); ylims!(ax, 0, 1)
    row == 1 && axislegend(ax, position = :rt)
end

# (c) the difference, basin by basin
ax_diff = Axis(figc[3, 1];
    ylabel = "Bayes − vanilla", xlabel = "a", lab_args...)
for (i, k) in enumerate(cmp_labels)
    lines!(ax_diff, a_range, vol_bayes[k] .- vol_vanilla[k],
           color = colors[mod1(i, length(colors))], label = "Basin $k")
end
hlines!(ax_diff, [0.0], color = :black, linestyle = :dash, linewidth = 0.8)
xlims!(ax_diff, ai, af)

save(plotsdir("fig2_vanilla_comparison.png"), figc)
