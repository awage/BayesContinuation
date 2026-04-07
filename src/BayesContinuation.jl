module BayesContinuation

using DrWatson
using Distributions
using SpecialFunctions
using StaticArrays
using Statistics
using Attractors
using ProgressMeter

include("compute.jl")
include("inference_stuff.jl")
include("bayes_entropy_est.jl")

# Mapper factory
export MapperFactory, TrackedFactory, PlainFactory
# Backward-compatible aliases
export AttractorOracle, GenericOracle

# From inference_stuff.jl
export LocalBoxObserver
export create_observer
export initialize_prior_from_data!
export bayes_entropy
export bayes_entropy_variance
export generate_tiling
export pick_random_point
export basin_volumes
export log_marginal_likelihood
export compute_log_bayes_factor
export test_continuity

# From bayes_entropy_est.jl
export estimate_entropy

# From compute.jl
export henon_rule
export seed_mapper!
export seeding
export get_mapper

end
