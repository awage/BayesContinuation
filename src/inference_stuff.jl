using Distributions
using SpecialFunctions

"""
Computes Dirichlet Entropy where alpha is a Dictionary of {Label => Count}.
Missing keys are assumed to have 0 mass (or handled via beta in construction).
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
    # Physical boundaries [x_min, x_max], [y_min, y_max]
    physical_bounds::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}
    # Dictionary of Dirichlet parameters: Label => Weight
    alpha::Dict{Int, Float64}
    last_entropy::Float64
    last_llr::Float64
end

function create_observer(phys_bounds, beta::Float64)
    # Initialize with empty dictionary (conceptual mass is beta everywhere)
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
function initialize_prior_from_data!(obs::LocalBoxObserver, data_view::AbstractArray, beta::Float64)
    # Clear current
    empty!(obs.alpha)
    
    # Count occurrences in the dense data view
    counts = Dict{Int, Int}()
    for val in data_view
        counts[val] = get(counts, val, 0) + 1
    end
    
    # Convert to alpha = count + beta
    for (k, c) in counts
        obs.alpha[k] = c + beta
    end
    
    # Initialize stats
    obs.last_entropy = bayes_entropy(obs.alpha)
end

function initialize_prior_from_data!(obs::LocalBoxObserver, mapper, beta::Float64, N::Int64)
    # Clear current
    empty!(obs.alpha)
    
    new_counts = Dict{Int, Int}()
    for _ in 1:N
        u0 = pick_random_point(obs)
        label = mapper(u0) 
        new_counts[label] = get(new_counts, label, 0) + 1
    end
    
    # Convert to alpha = count + beta
    for (k, c) in new_counts
        obs.alpha[k] = c + beta
    end
    
end

"""
Pick a random physical point (x,y) inside the observer's box.
"""
function pick_random_point(obs::LocalBoxObserver)
    (xmin, xmax) = obs.physical_bounds[1]
    (ymin, ymax) = obs.physical_bounds[2]
    
    x = xmin + rand() * (xmax - xmin)
    y = ymin + rand() * (ymax - ymin)
    return [x, y]
end

function generate_tiling(global_bounds, n_tiles, beta)
    (gx_min, gx_max), (gy_min, gy_max) = global_bounds
    dx = (gx_max - gx_min) / n_tiles
    dy = (gy_max - gy_min) / n_tiles
    
    observers = Vector{LocalBoxObserver}()
    
    for i in 1:n_tiles
        for j in 1:n_tiles
            # Calculate local bounds
            loc_xmin = gx_min + (i-1)*dx
            loc_xmax = gx_min + i*dx
            loc_ymin = gy_min + (j-1)*dy
            loc_ymax = gy_min + j*dy
            
            # Create observer for this tile
            obs = create_observer(((loc_xmin, loc_xmax), (loc_ymin, loc_ymax)), beta)
            push!(observers, obs)
        end
    end
    return observers
end

# ============================================================================
#  G-statistic detection (replaces Bayes Factor)
# ============================================================================

"""
    compute_g_statistic(new_counts, alpha, beta)

Computes the Williams-corrected G-statistic for testing whether the observed
sparse sample `new_counts` is consistent with the Dirichlet prior `alpha`.

The G-statistic is:
    D = 2 Σ cᵢ ln(cᵢ/Ns / p̂ᵢ)
where p̂ᵢ = αᵢ / α₀ is the prior predictive mean.

Williams' correction improves the χ² approximation for small samples:
    D_W = D / (1 + (K+1)/(6Ns))

Returns: (D_W, K) where K is the number of categories.
"""
function compute_g_statistic(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, beta::Float64)
    # Union of all categories seen in prior or new data
    all_keys = union(keys(new_counts), keys(alpha))
    K = length(all_keys)
    
    # Total prior mass
    alpha_0 = 0.0
    for k in all_keys
        alpha_0 += get(alpha, k, beta)
    end
    
    # Total observed counts
    N_s = sum(values(new_counts))
    
    # Compute G-statistic: D = 2 Σ cᵢ ln(cᵢ/Ns / p̂ᵢ)
    # Skip categories with cᵢ = 0 (they contribute 0)
    D = 0.0
    for k in all_keys
        c_i = get(new_counts, k, 0)
        if c_i > 0
            p_prior = get(alpha, k, beta) / alpha_0   # prior predictive mean
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
    test_continuity(new_counts, alpha, beta; gamma=0.01)

Tests whether the sparse sample is consistent with the prior (H₀: no structural change).

Uses the Williams-corrected G-statistic compared against the χ²(K-1) distribution
at significance level γ.

Returns: (reject::Bool, D_W::Float64, p_value::Float64)
    - reject = true  → Panic Mode: the basin structure has changed
    - reject = false → Continue: update the posterior normally
"""
function test_continuity(new_counts::Dict{Int, Int}, alpha::Dict{Int, Float64}, beta::Float64; 
                         gamma::Float64=0.01)
    D_W, K = compute_g_statistic(new_counts, alpha, beta)
    
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

