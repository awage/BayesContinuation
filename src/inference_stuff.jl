using Distributions
using SpecialFunctions

"""
Computes Dirichlet Entropy where alpha is a Dictionary of {Label => Count}.
Missing keys are assumed to have 0 mass (or handled via β in construction).
"""
function bayes_entropy(alpha::Dict{Int, Float64})
    # alpha_0 is the sum of all pseudo-counts in the dictionary
    a0 = sum(values(alpha))
    
    # Term 1: digamma(sum + 1)
    term1 = digamma(a0 + 1)
    
    # Term 2: sum (alpha_i / alpha_0) * digamma(alpha_i + 1)
    term2 = 0.0
    for val in values(alpha)
        term2 += (val / a0) * digamma(val + 1)
    end
    
    return term1 - term2
end


"""
    bayes_entropy_variance_exact(alpha)

Computes the EXACT variance of the Shannon entropy for a Dirichlet distribution.
Ref: Wolpert & Wolf (1995), Theorem 16 (Eq 16.1 and 16.2).
Uses an O(K) algebraic reduction to avoid K^2 cross-term summations.
"""
function bayes_entropy_variance(alpha::Union{Vector{Float64}, Dict{Int, Float64}})
    vals = isa(alpha, Dict) ? collect(values(alpha)) : alpha
    a0 = sum(vals)
    
    if a0 <= 0
        return 0.0
    end
    
    # Precompute common polygamma terms for the total sum a0
    psi_a0_1 = digamma(a0 + 1)
    psi_a0_2 = digamma(a0 + 2)
    tri_a0_2 = trigamma(a0 + 2)
    
    E_S = 0.0      # Expected Entropy: E[S]
    sum_A = 0.0    # For the O(K) cross-term trick
    sum_A2 = 0.0   # For the O(K) cross-term trick
    sum_a_sq = 0.0 # Sum of alpha_i squared
    D = 0.0        # Diagonal terms
    
    for a_i in vals
        if a_i > 0
            # 1. Expectation Term (Eq 16.1)
            E_S -= (a_i / a0) * (digamma(a_i + 1) - psi_a0_1)
            
            # 2. Cross Term Components (for Eq 16.2 i != j)
            A_i = a_i * (digamma(a_i + 1) - psi_a0_2)
            sum_A += A_i
            sum_A2 += A_i^2
            sum_a_sq += a_i^2
            
            # 3. Diagonal Terms (for Eq 16.2 i == j)
            psi_ai_2 = digamma(a_i + 2)
            tri_ai_2 = trigamma(a_i + 2)
            
            term_D = (a_i * (a_i + 1)) / (a0 * (a0 + 1)) * 
                     ( (psi_ai_2 - psi_a0_2)^2 + tri_ai_2 - tri_a0_2 )
            D += term_D
        end
    end
    
    # Combine the Cross Terms using the (Sum)^2 - Sum(Squares) trick
    cross_part1 = sum_A^2 - sum_A2
    cross_part2 = -tri_a0_2 * (a0^2 - sum_a_sq)
    C = (cross_part1 + cross_part2) / (a0 * (a0 + 1))
    
    # Exact Second Moment E[S^2]
    E_S2 = C + D
    
    # Variance = E[S^2] - (E[S])^2
    # Ensure it doesn't drop trivially below 0 due to float imprecision
    return max(0.0, E_S2 - E_S^2) 
end

mutable struct LocalBoxObserver
    # Physical boundaries per dimension: [(x_min, x_max), (y_min, y_max), ...]
    physical_bounds::Vector{Tuple{Float64, Float64}}
    # Dictionary of Dirichlet parameters: Label => Weight
    alpha::Dict{Int, Float64}
    last_entropy::Float64
    last_llr::Float64
end

function create_observer(phys_bounds, β::Float64)
    # Initialize with empty dictionary (conceptual mass is β everywhere)
    return LocalBoxObserver(
        phys_bounds,
        Dict{Int, Float64}(),
        0.0,
        0.0
    )
end

"""
Updates the priors of an observer based on a dense initialization (the initial basins).
"""
function initialize_prior_from_data!(obs::LocalBoxObserver, data_view::AbstractArray, β::Float64)
    # Clear current
    empty!(obs.alpha)
    
    # Count occurrences in the dense data view
    counts = Dict{Int, Int}()
    for val in data_view
        counts[val] = get(counts, val, 0) + 1
    end
    
    # Convert to alpha = count + β
    for (k, c) in counts
        obs.alpha[k] = c + β
    end
    
    # Initialize stats
    obs.last_entropy = bayes_entropy(obs.alpha)
end

function initialize_prior_from_data!(obs::LocalBoxObserver, mapper, β::Float64, N::Int64)
    # Clear current
    empty!(obs.alpha)
    
    new_counts = Dict{Int, Int}()
    for _ in 1:N
        u0 = pick_random_point(obs)
        label = mapper(u0) 
        new_counts[label] = get(new_counts, label, 0) + 1
    end
    
    # Convert to alpha = count + β
    for (k, c) in new_counts
        obs.alpha[k] = c + β
    end
    
end

"""
    basin_volumes(observers)

Estimates the relative volume of each basin as the mean expected probability
of each attractor label across all observer tiles. Each tile has equal physical
area, so the global fraction of phase space belonging to basin k is:

    V_k ≈ (1/N_tiles) Σ_i  α_{i,k} / α_{i,0}

Returns a `Dict{Int,Float64}` mapping label → relative volume (values sum to 1).
"""
function basin_volumes(observers::Vector{LocalBoxObserver})
    all_labels = Set{Int}()
    for obs in observers
        union!(all_labels, keys(obs.alpha))
    end

    vol = Dict{Int, Float64}()
    n = length(observers)
    for k in all_labels
        total = 0.0
        for obs in observers
            a0 = sum(values(obs.alpha))
            if a0 > 0
                total += get(obs.alpha, k, 0.0) / a0
            end
        end
        vol[k] = total / n
    end
    return vol
end

"""
Pick a random point inside the observer's N-dimensional box.
"""
function pick_random_point(obs::LocalBoxObserver)
    return [xmin + rand() * (xmax - xmin) for (xmin, xmax) in obs.physical_bounds]
end

"""
    generate_tiling(global_bounds, n_tiles, β)

Creates an N-dimensional tiling of `n_tiles` per dimension, yielding `n_tiles^D` observers
where `D = length(global_bounds)`. Each element of `global_bounds` is a `(min, max)` pair.
"""
function generate_tiling(global_bounds, n_tiles, β)
    D = length(global_bounds)
    edges = [collect(range(Float64(lo), Float64(hi); length = n_tiles + 1))
             for (lo, hi) in global_bounds]

    observers = Vector{LocalBoxObserver}()
    for idx in CartesianIndices(ntuple(_ -> n_tiles, D))
        bounds = [(edges[d][idx[d]], edges[d][idx[d]+1]) for d in 1:D]
        push!(observers, create_observer(bounds, β))
    end
    return observers
end


"""
    log_marginal_likelihood(counts, alpha, β)

Computes the log-marginal likelihood (log-evidence) of observed counts `c`
under a Dirichlet-Multinomial model with prior `alpha`.

    L(α) = lnΓ(α₀) − lnΓ(N_s + α₀) + Σᵢ [lnΓ(cᵢ + αᵢ) − lnΓ(αᵢ)]

where α₀ = Σ αᵢ and N_s = Σ cᵢ. Categories not present in `alpha` get
the base prior `β`.
"""
function log_marginal_likelihood(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, β::Float64)
    all_keys = union(keys(new_counts), keys(alpha))

    alpha_0 = 0.0
    N_s = sum(values(new_counts))
    log_lik = 0.0

    for k in all_keys
        a_k = get(alpha, k, β)
        c_k = get(new_counts, k, 0)
        alpha_0 += a_k
        log_lik += loggamma(c_k + a_k) - loggamma(a_k)
    end

    log_lik += loggamma(alpha_0) - loggamma(N_s + alpha_0)
    return log_lik
end

"""
    compute_log_bayes_factor(new_counts, alpha, β)

Computes the Log Bayes Factor η comparing two hypotheses:
  - H_hist (Stability): data generated by current prior `alpha`
  - H_reset (Crisis): data generated by uninformative prior αᵢ = β

    η = L(α_prior) − L(α_reset)

When η < 0, the uninformative model explains the data better than the
historical record, indicating a structural change.

Returns: η (Float64)
"""
function compute_log_bayes_factor(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, β::Float64)
    # Log-evidence under historical prior
    L_hist = log_marginal_likelihood(new_counts, alpha, β)

    # Log-evidence under reset (uninformative) prior: α_i = β for all categories
    all_keys = union(keys(new_counts), keys(alpha))
    alpha_reset = Dict{Int, Float64}(k => β for k in all_keys)
    L_reset = log_marginal_likelihood(new_counts, alpha_reset, β)

    return L_hist - L_reset
end

"""
    test_continuity(new_counts, alpha, β)

Tests whether the sparse sample is consistent with the prior (H₀: no structural change)
using the log-marginal Bayes Factor.

Returns: (reject::Bool, η::Float64, 0.0)
  - reject = true  → Panic Mode: η < 0, the uninformative model fits better
  - reject = false → Continue: the historical prior explains the data
  - Third element is a placeholder for backward compatibility (no p-value in this test)
"""
function test_continuity(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, β::Float64;
                         gamma::Float64=0.01)
    eta = compute_log_bayes_factor(new_counts, alpha, β)
    reject = eta < 0.0
    return reject, eta, 0.0
end

# ============================================================================
#  G-statistic detection (kept for comparison)
# ============================================================================

"""
    compute_g_statistic(new_counts, alpha, β)

Computes the Williams-corrected G-statistic for testing whether the observed
sparse sample `new_counts` is consistent with the Dirichlet prior `alpha`.

The G-statistic is:
    D = 2 Σ cᵢ ln(cᵢ/Ns / p̂ᵢ)
where p̂ᵢ = αᵢ / α₀ is the prior predictive mean.

Williams' correction improves the χ² approximation for small samples:
    D_W = D / (1 + (K+1)/(6Ns))

Returns: (D_W, K) where K is the number of categories.
"""
function compute_g_statistic(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, β::Float64)
    # Union of all categories seen in prior or new data
    all_keys = union(keys(new_counts), keys(alpha))
    K = length(all_keys)
    
    # Total prior mass
    alpha_0 = 0.0
    for k in all_keys
        alpha_0 += get(alpha, k, β)
    end
    
    # Total observed counts
    N_s = sum(values(new_counts))
    
    # Compute G-statistic: D = 2 Σ cᵢ ln(cᵢ/Ns / p̂ᵢ)
    # Skip categories with cᵢ = 0 (they contribute 0)
    D = 0.0
    for k in all_keys
        c_i = get(new_counts, k, 0)
        if c_i > 0
            p_prior = get(alpha, k, β) / alpha_0   # prior predictive mean
            p_obs = c_i / N_s                          # observed proportion
            D += c_i * log(p_obs / p_prior)
        end
    end
    D *= 2.0
    
    # Williams' correction factor
    q = 1.0 + (K + 1) / (6 * N_s)
    D_W = D / q
    
    return D_W, K
end

"""
    test_continuity_gstat(new_counts, alpha, β; gamma=0.01)

Tests whether the sparse sample is consistent with the prior (H₀: no structural change).

Uses the Williams-corrected G-statistic compared against the χ²(K-1) distribution
at significance level γ.

Returns: (reject::Bool, D_W::Float64, p_value::Float64)
    - reject = true  → Panic Mode: the basin structure has changed
    - reject = false → Continue: update the posterior normally
"""
function test_continuity_gstat(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, β::Float64;
                         gamma::Float64=0.01)
    D_W, K = compute_g_statistic(new_counts, alpha, β)
    
    # Degrees of freedom
    dof = K - 1
    
    if dof <= 0
        # Only one category: no test needed, trivially consistent
        return false, D_W, 1.0
    end
    
    # Critical value from χ² distribution
    chi2_critical = quantile(Chisq(dof), 1.0 - gamma)
    
    # p-value for diagnostics
    p_value = 1.0 - cdf(Chisq(dof), D_W)
    
    reject = D_W > chi2_critical
    return reject, D_W, p_value
end

