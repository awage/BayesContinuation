"""
BayesContinuation
=================
Bayesian monitoring of basin structure along a global continuation.

The method itself — tiling a region of state space, holding a Dirichlet prior over
attractor labels in each box, testing each new sparse sample against that prior with
a log Bayes factor η, re-sampling densely the boxes where η goes negative, and
recording `alphas`/`etas` at every parameter — lives entirely upstream in
`Attractors.BayesianUpdateSampler`. So does the continuation loop that drives it.

What is left here is one file of estimators. Run an ordinary `global_continuation`
with a history-keeping sampler, then read the figures off the history:

    sampler = BayesianUpdateSampler(region, n_tiles; sparse_n, history = true)
    fractions, attractors = global_continuation(RecurrencesFindAndMatch(bmap), pcurve, sampler)
    est = bayes_estimates(sampler)   # mean_S, var_S, min_eta, n_panics, volumes, ...
"""
module BayesContinuation

using SpecialFunctions   # digamma, trigamma
using Attractors
using Attractors: BayesianUpdateSampler, sampler_history

include("inference_stuff.jl")

# Part 1 — estimators over the boxes at a single parameter
export bayes_entropy, bayes_entropy_variance
export box_entropies, mean_entropy, mean_entropy_variance
export basin_volumes, basin_volume_variance, panic_boxes

# Part 2 — series over a sampler's history
export bayes_estimates, volume_series

end
