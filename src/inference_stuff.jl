"""
inference_stuff.jl
==================
Estimators over the Dirichlet posteriors that `Attractors.BayesianUpdateSampler`
maintains, one per box of its tiling.

`include(srcdir("inference_stuff.jl"))` this file into a script; it defines plain
functions in the script's own namespace and needs `SpecialFunctions` (digamma,
trigamma) and `Attractors` to be loaded first. The typical use is

    sampler = BayesianUpdateSampler(region, n_tiles; sparse_n, history = true)
    fractions, attractors = global_continuation(RecurrencesFindAndMatch(bmap), pcurve, sampler)
    est = bayes_estimates(sampler)   # mean_S, var_S, min_eta, n_panics, volumes, ...

The sampler keeps those posteriors in `sampler.alphas :: Vector{Dict{Int, Float64}}`:
`alphas[i][k]` is the pseudo-count box `i` assigns to attractor `k`, so
`alphas[i][k] / Σ alphas[i]` is that box's estimate of the fraction of itself that
belongs to basin `k`. Every function below is a functional of those numbers alone.

Nothing here samples, tiles, tests or continues anything — the tiling, the point
generation, the log Bayes factor η, the panic/re-sample logic and the per-parameter
record of `alphas`/`etas` all live in `Attractors/src/continuation/sampler_api.jl`.
What is *not* upstream, and is the reason this file exists, are the basin-entropy
estimators and their variances.

The file has two halves. The first takes one *slice* — the state of the boxes at a
single parameter — and is what the estimators are actually defined on. The second
maps those over the sampler's history to give the series a figure needs; see
[`bayes_estimates`](@ref).
"""

# ===========================================================================
# Part 1 — one slice: the boxes at a single parameter
# ===========================================================================

# ---------------------------------------------------------------------------
# Entropy of a single box
# ---------------------------------------------------------------------------

"""
    bayes_entropy(α) → Float64

Posterior mean of the Shannon entropy of a Dirichlet(α) distribution,

    E[S] = ψ(α₀ + 1) − Σₖ (αₖ/α₀) ψ(αₖ + 1),    α₀ = Σₖ αₖ

`α` is a `label => pseudo-count` dictionary such as an entry of `sampler.alphas`.
Labels absent from `α` carry no mass and do not contribute.
"""
function bayes_entropy(alpha::AbstractDict{Int, Float64})
    a0 = sum(values(alpha))
    a0 > 0 || return 0.0
    term2 = 0.0
    for val in values(alpha)
        term2 += (val / a0) * digamma(val + 1)
    end
    return digamma(a0 + 1) - term2
end

"""
    bayes_entropy_variance(α) → Float64

Exact posterior variance of the Shannon entropy under Dirichlet(α), following
Wolpert & Wolf (1995), Theorem 16 (Eqs. 16.1–16.2). The `i ≠ j` cross terms are
summed with the `(Σ)² − Σ(·²)` trick, so the cost is O(K) rather than O(K²).

`α` may be a `label => pseudo-count` dictionary or a plain vector of pseudo-counts.
"""
function bayes_entropy_variance(alpha::Union{AbstractVector{Float64}, AbstractDict{Int, Float64}})
    vals = isa(alpha, AbstractDict) ? collect(values(alpha)) : alpha
    a0 = sum(vals)
    a0 > 0 || return 0.0

    # Polygamma terms of the total, needed by every summand
    psi_a0_1 = digamma(a0 + 1)
    psi_a0_2 = digamma(a0 + 2)
    tri_a0_2 = trigamma(a0 + 2)

    E_S = 0.0      # E[S], Eq. 16.1
    sum_A = 0.0    # Σ Aᵢ, for the cross-term trick
    sum_A2 = 0.0   # Σ Aᵢ²
    sum_a_sq = 0.0 # Σ αᵢ²
    D = 0.0        # diagonal (i == j) terms of Eq. 16.2

    for a_i in vals
        a_i > 0 || continue
        E_S -= (a_i / a0) * (digamma(a_i + 1) - psi_a0_1)

        A_i = a_i * (digamma(a_i + 1) - psi_a0_2)
        sum_A += A_i
        sum_A2 += A_i^2
        sum_a_sq += a_i^2

        D += (a_i * (a_i + 1)) / (a0 * (a0 + 1)) *
             ((digamma(a_i + 2) - psi_a0_2)^2 + trigamma(a_i + 2) - tri_a0_2)
    end

    C = ((sum_A^2 - sum_A2) - tri_a0_2 * (a0^2 - sum_a_sq)) / (a0 * (a0 + 1))
    # Var = E[S²] − E[S]²; clamp away float noise around 0
    return max(0.0, (C + D) - E_S^2)
end

# ---------------------------------------------------------------------------
# Aggregation over the boxes of a tiling
# ---------------------------------------------------------------------------

"""
    box_entropies(alphas) → Vector{Float64}

`bayes_entropy` of every box, in the order of `sampler.alphas`.
"""
box_entropies(alphas::AbstractVector{<:AbstractDict{Int, Float64}}) =
    [bayes_entropy(a) for a in alphas]

"""
    mean_entropy(alphas) → Float64

Basin entropy of the whole region: the mean over boxes of their posterior mean
entropy. All boxes of a `BayesianUpdateSampler` have the same volume, so an
unweighted mean is the right aggregation.
"""
mean_entropy(alphas::AbstractVector{<:AbstractDict{Int, Float64}}) =
    isempty(alphas) ? 0.0 : sum(bayes_entropy, alphas) / length(alphas)

"""
    mean_entropy_variance(alphas) → Float64

Variance of [`mean_entropy`](@ref). The boxes are treated as independent, so the
variance of their mean is `(1/N²) Σᵢ Var[Sᵢ]`.
"""
function mean_entropy_variance(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    n = length(alphas)
    n > 0 || return 0.0
    return sum(bayes_entropy_variance, alphas) / n^2
end

# ---------------------------------------------------------------------------
# Basin volumes
# ---------------------------------------------------------------------------

"""
    basin_volumes(alphas) → Dict{Int, Float64}

Relative volume of each basin, as believed by the priors:

    V_k ≈ (1/N) Σᵢ αᵢₖ / αᵢ₀

Values sum to 1. This is the *prior-based* estimate, which carries the memory of
previous parameters through the sampler's forgetting factor λ. It is not the same
quantity as the `fractions_cont` returned by a global continuation, which
`Attractors.weighted_fractions` computes from the labels of the current parameter
only; comparing the two is a useful check that the priors have kept up.
"""
function basin_volumes(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    vol = Dict{Int, Float64}()
    n = length(alphas)
    n > 0 || return vol
    for alpha in alphas
        a0 = sum(values(alpha))
        a0 > 0 || continue
        for (k, a) in alpha
            vol[k] = get(vol, k, 0.0) + a / (a0 * n)
        end
    end
    return vol
end

"""
    basin_volume_variance(alphas) → Dict{Int, Float64}

Posterior variance of each entry of [`basin_volumes`](@ref). Within one box,
Dirichlet(α) gives `Var[pₖ] = αₖ(α₀ − αₖ) / (α₀²(α₀ + 1))`; the boxes are averaged
as independent contributions, so `Var[V_k] = (1/N²) Σᵢ Var[pᵢₖ]`.
"""
function basin_volume_variance(alphas::AbstractVector{<:AbstractDict{Int, Float64}})
    var_vol = Dict{Int, Float64}()
    n = length(alphas)
    n > 0 || return var_vol
    for alpha in alphas
        a0 = sum(values(alpha))
        a0 > 0 || continue
        for (k, ak) in alpha
            var_vol[k] = get(var_vol, k, 0.0) + ak * (a0 - ak) / (a0^2 * (a0 + 1) * n^2)
        end
    end
    return var_vol
end

"""
    panic_boxes(etas) → Vector{Int}

The boxes that asked for a dense re-sample at a parameter, given their log Bayes
factors. An alarm is a *negative* η: the box's history explains its data worse than
no history at all.
"""
panic_boxes(etas::AbstractVector{<:Real}) = findall(<(0), etas)

# ===========================================================================
# Part 2 — the whole sweep: series over a sampler's history
# ===========================================================================

"""
    bayes_estimates(sampler::BayesianUpdateSampler) → NamedTuple
    bayes_estimates(history) → NamedTuple

Every quantity the figures need, computed from the per-parameter record the sampler
kept during a [`global_continuation`](@ref). The sampler must have been built with
`history = true`; `history` may also be given directly as the
`(; alphas, etas)` of `Attractors.sampler_history`.

This is the whole reason the continuation itself does not have to live here: run
`global_continuation` unmodified, then map the Part 1 estimators over the history.

Returns, with one entry per parameter of the `pcurve` that was swept:

- `mean_S`, `var_S`: basin entropy of the region and its variance, from
  [`mean_entropy`](@ref) and [`mean_entropy_variance`](@ref).
- `min_eta`: the smallest η over the boxes — the alarm signal. A *maximum* would
  track the quietest box and never signal anything.
- `n_panics`, `panic_boxes`: how many boxes, and which, raised the alarm.
- `volumes`, `vol_var`: [`basin_volumes`](@ref) and [`basin_volume_variance`](@ref)
  of the priors.
- `full_S`, `full_eta`: the `(n_parameters, n_boxes)` matrices behind the aggregates,
  for when the spatial distribution of the alarm matters.
"""
function bayes_estimates(history::NamedTuple)
    alphas, etas = history.alphas, history.etas
    isempty(alphas) && throw(ArgumentError(
        "the sampler kept no history; build it with `BayesianUpdateSampler(...; history = true)`"
    ))
    n_p, n_b = length(alphas), length(first(alphas))
    return (;
        mean_S = [mean_entropy(a) for a in alphas],
        var_S = [mean_entropy_variance(a) for a in alphas],
        min_eta = [minimum(e) for e in etas],
        n_panics = [count(<(0), e) for e in etas],
        panic_boxes = [panic_boxes(e) for e in etas],
        volumes = [basin_volumes(a) for a in alphas],
        vol_var = [basin_volume_variance(a) for a in alphas],
        full_S = [bayes_entropy(alphas[i][j]) for i in 1:n_p, j in 1:n_b],
        full_eta = [etas[i][j] for i in 1:n_p, j in 1:n_b],
    )
end

bayes_estimates(sampler::BayesianUpdateSampler) = bayes_estimates(sampler_history(sampler))

"""
    volume_series(volumes, labels = ...) → Dict{Int, Vector{Float64}}

Turn the per-parameter `volumes` dictionaries of [`bayes_estimates`](@ref) into one
time series per basin, filling with `0.0` the parameters at which a basin does not
exist. This is the shape a stacked band plot wants. `Attractors.continuation_series`
does the same for the `fractions_cont` of a continuation, but fills with `NaN`.

The `fractions_cont` of a continuation have the same shape and work here too. They may
arrive as a `Vector{Dict}` rather than a `Vector{Dict{Int, Float64}}` (a JLD2 round trip
loses the element type), hence the loose signature.
"""
function volume_series(volumes::AbstractVector{<:AbstractDict},
                       labels = sort!(collect(reduce(union, keys.(volumes)))))
    return Dict{Int, Vector{Float64}}(
        Int(k) => [Float64(get(v, k, 0.0)) for v in volumes] for k in labels
    )
end
