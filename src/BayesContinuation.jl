module BayesContinuation

using DrWatson
using SpecialFunctions
using StaticArrays
using Statistics
using Attractors
using ProgressMeter

# ============================================================================
# Mapper factory types
# ============================================================================

"""
    MapperFactory

Abstract type for mapper factories used by `estimate_entropy`.
Subtypes define how to build a mapper for a given parameter value and how to
maintain any internal state (e.g. attractor matching) across steps.
"""
abstract type MapperFactory end

"""
    GenericFactory(f)

Wraps a stateless `param → mapper` closure.
Parallel sampling is allowed.
"""
struct GenericFactory{F} <: MapperFactory
    build::F
end

"""
    AttractorMapperFactory(f)

Wraps a `(param, prev_attractors) → mapper` closure that uses Attractors.jl
for attractor tracking and matching between parameter steps.
Stores `prev_attractors` internally and updates it after each step.
Parallel sampling is disabled because the underlying mapper holds mutable state.
"""
mutable struct AttractorMapperFactory{F} <: MapperFactory
    build::F
    prev_attractors::Any
end
AttractorMapperFactory(f) = AttractorMapperFactory(f, nothing)

# Build a mapper for the given parameter value.
build_mapper(mf::GenericFactory,         param) = mf.build(param)
build_mapper(mf::AttractorMapperFactory, param) = mf.build(param, mf.prev_attractors)

# Update stored attractor state after a step. No-op for GenericFactory.
update!(::GenericFactory,            _)      = nothing
update!(mf::AttractorMapperFactory,  mapper) = (mf.prev_attractors = extract_attractors(mapper))

# Thread safety: AttractorMapperFactory mappers hold mutable state.
_parallel_allowed(::GenericFactory)         = true
_parallel_allowed(::AttractorMapperFactory) = false

include("inference_stuff.jl")
include("bayes_entropy_est.jl")

# Mapper factory types and interface
export MapperFactory, GenericFactory, AttractorMapperFactory
export build_mapper, update!

# From inference_stuff.jl
# export LocalBoxObserver
# export create_observer
# export initialize_prior_from_data!
# export bayes_entropy
# export bayes_entropy_variance
# export generate_tiling
# export pick_random_point
# export basin_volumes
# export log_marginal_likelihood
# export compute_log_bayes_factor
# export test_continuity

# From bayes_entropy_est.jl
export estimate_entropy

end
