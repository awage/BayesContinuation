"""
henon_global_continuation_mwe.jl
================================
Minimal working example: driving `Attractors.global_continuation` with the
`BayesianUpdateSampler` of `Attractors/src/continuation/sampler_api.jl`, on the
Hénon map

    x' = a - x² + b y
    y' = x

sweeping `a` over [2.0, 2.25] at fixed `b = -0.3`. In that range the map has one
bounded chaotic attractor whose basin shrinks as `a` grows, until a boundary
crisis somewhere around a ≈ 2.15 destroys it and every initial condition escapes
to infinity (label -1 from the basin map). That crisis is exactly the kind of
event the sampler is meant to catch.

The sampler replaces the usual `RandomICSampler`: instead of drawing a fixed
number of uniform initial conditions at every parameter, it tiles the region into
`N_TILES^2` boxes, samples each box with `sparse_n` points, tests the resulting
label counts against that box's Dirichlet prior with a log Bayes factor η, and
re-samples with `dense_n` points only the boxes whose η went negative.
`global_continuation` drives all of it through `generate_ics`, `update_sampler!`
and `resampling_required`.

Two things worth noting about this particular setup:

  * The sampler's priors are keyed by attractor label and `global_continuation`
    calls `reset_mapper!` at every parameter, so the labels must mean the same
    thing at every step. A recurrence mapper re-issues its IDs in discovery
    order, which is only safe here because this range holds a single bounded
    attractor: it is always ID 1, and divergence is always -1. With several
    coexisting attractors, use a mapper with fixed IDs such as
    `BasinMapProximity`.
  * The sampler keeps no history — `alphas` and `etas` are overwritten in place
    at every parameter. Anything worth keeping has to be recorded from inside
    the loop, which is why the sweep below is run one parameter at a time.

Run with:
    julia --project=. scripts/henon_global_continuation_mwe.jl
"""

using Attractors

# ---------------------------------------------------------------------------
# 1. The dynamical system and its basin map
# ---------------------------------------------------------------------------

henon_rule(u, p, n) = SVector(p[1] - u[1]^2 + p[2] * u[2], u[1])

const B = -0.3
const A_RANGE = range(1.8, 2.; length = 200)

ds = DeterministicIteratedMap(henon_rule, [0.0, 0.0], [first(A_RANGE), B])

# The grid has to be wide enough that escaping trajectories leave it, which is how
# they end up labelled -1.
grid = (range(-4.0, 4.0; length = 1000), range(-4.0, 4.0; length = 1000))
bmap = BasinMapRecurrences(ds, grid;
    sparse = true, consecutive_recurrences = 2000, Ttr = 100, show_progress = false,
)

# ---------------------------------------------------------------------------
# 2. The Bayesian sampler
# ---------------------------------------------------------------------------

const REGION = ((-2.0, 2.0), (-2.0, 2.0))   # region that gets tiled
const N_TILES = 10                           # => 16 boxes

sampler = BayesianUpdateSampler(REGION, N_TILES;
    sparse_n = 20,   # ics per box during routine monitoring
    dense_n  = 20^2,   # ics per box when a box asks for a re-sample
    λ = 0.7,         # forgetting factor of the prior
    β = 0.5,         # Dirichlet base pseudo-count
    seed = 20250730,
)

# ---------------------------------------------------------------------------
# 3. The continuation, one parameter at a time so the diagnostics can be kept
# ---------------------------------------------------------------------------

algo = RecurrencesFindAndMatch(bmap; distance = Hausdorff(), threshold = Inf)
# algo = AttractorSeedContinueMatch(bmap)

println("parameter sweep: a ∈ [$(first(A_RANGE)), $(last(A_RANGE))], b = $B")
println("boxes: $(N_TILES^2), sparse_n: $(sampler.sparse_n), dense_n: $(sampler.dense_n)\n")
println(rpad("a", 8), rpad("ics", 6), rpad("min η", 9), rpad("re-sampled boxes", 20),
        "basin fractions")

fractions_cont = Dict{Int, Float64}[]
for a in A_RANGE
    n_ics = length(sampler)  # read *before* the step: what it is about to draw
    fs_cont, _ = global_continuation(
        algo, [Dict(1 => a, 2 => B)], sampler; show_progress = false,
    )
    fs = only(fs_cont)
    push!(fractions_cont, fs)

    # `alphas` and `etas` hold only the current state, so read them now.
    resampled = findall(η -> η < 0, sampler.etas)
    println(
        rpad(round(a; digits = 4), 8),
        rpad(n_ics, 6),
        rpad(round(minimum(sampler.etas); digits = 2), 9),
        rpad(isempty(resampled) ? "-" : string(resampled), 20),
        Dict(k => round(v; digits = 3) for (k, v) in fs),
    )
end

# ---------------------------------------------------------------------------
# 4. What the sampler ended up believing
# ---------------------------------------------------------------------------

# `α_k / Σα` is box `i`'s estimate of the fraction of itself belonging to basin `k`;
# the boxes all have the same volume, so averaging gives the global basin fractions.
function basin_fractions_from_priors(s)
    v = Dict{Int, Float64}()
    for α in s.alphas
        α₀ = sum(values(α))
        for (k, a) in α
            v[k] = get(v, k, 0.0) + a / α₀ / length(s.alphas)
        end
    end
    return v
end

println("\nat a = $(round(last(A_RANGE); digits = 4)):")
println("  fractions from the priors:  ",
        Dict(k => round(v; digits = 3) for (k, v) in basin_fractions_from_priors(sampler)))
println("  fractions from the mapper:  ",
        Dict(k => round(v; digits = 3) for (k, v) in fractions_cont[end]))
