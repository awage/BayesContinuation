using DrWatson
@quickactivate
using CairoMakie
using JLD2
using LinearAlgebra
using Statistics
using Attractors
using ProgressMeter

include(srcdir("compute.jl"))
include(srcdir("inference_stuff.jl"))

function estimate_entropy(params, a_range, get_mapper::Function)

    @unpack SPARSE_N, DENSE_N, BAYES_FACTOR, N_TILES, GLOBAL_BOUNDS, LAMBDA = params
    beta = get(params, :BETA, 0.5)

    sparse_n = SPARSE_N
    dense_n = DENSE_N
    bayes_factor = BAYES_FACTOR
    n_tiles = N_TILES
    global_bounds = GLOBAL_BOUNDS
    lambda = LAMBDA

    println("Initializing $(n_tiles)x$(n_tiles) observer grid...")
    observers = generate_tiling(global_bounds, n_tiles, beta)

    history_mean_S = Float64[]
    history_var_S = Float64[]
    history_max_score = Float64[]
    n_steps = length(a_range)
    full_history_S = zeros(Float64, n_steps, length(observers))
    full_history_score = zeros(Float64, n_steps, length(observers))
    mapper = get_mapper(a_range[1], nothing)

    step_entropies = Float64[]
    step_variances = Float64[]
    # Initialize Priors for ALL boxes and initialize the first frame
    for (i, obs) in enumerate(observers)
        initialize_prior_from_data!(obs, mapper, beta, dense_n)
        obs.last_entropy = bayes_entropy(obs.alpha)
        push!(step_entropies, obs.last_entropy)
        full_history_S[1,i] = obs.last_entropy
        var_k = bayes_entropy_variance(obs.alpha)
        push!(step_variances, var_k)
    end
    global_entropy_var = sum(step_variances)/(length(observers)^2)
    push!(history_var_S, global_entropy_var)
    push!(history_mean_S, mean(step_entropies))
    push!(history_max_score, 0.0)
        
    # collect found attractors for the continuity match
    # (and bifurcation diagram if needed)
    atts = extract_attractors(mapper)
    history_att = Array{typeof(atts)}(undef, n_steps)
    history_att[1] = atts



    @showprogress for (t_idx, a_val) in enumerate(a_range)
        if t_idx == 1; continue; end
        
        # Update System Dynamics and do the attractor seed and match
        mapper = get_mapper(a_val, history_att[t_idx-1])
        step_entropies = Float64[]
        step_variances = Float64[]
        step_score = Float64[]

        # Iterate over all boxes
        for (obs_idx, obs) in enumerate(observers)
            
            # 1. Decay Prior
            prior_alpha = Dict{Int, Float64}()
            for (k, v) in obs.alpha
                decayed_val = lambda * (v - beta) + beta
                prior_alpha[k] = decayed_val
            end

            # 2. Sparse Sampling
            new_counts = Dict{Int, Int}()
            for _ in 1:sparse_n
                u0 = pick_random_point(obs)
                label = mapper(u0) 
                new_counts[label] = get(new_counts, label, 0) + 1
            end

            # 3. Posterior Update
            post_alpha = copy(prior_alpha)
            for (label, count) in new_counts
                current_val = get(post_alpha, label, beta)
                post_alpha[label] = current_val + count
            end

            # 4. Compute Metrics
            entropy_curr = bayes_entropy(post_alpha)
            # score_curr = score_div(post_alpha, prior_alpha, beta)
            log_bayes_factor = compute_log_bayes_factor(new_counts, prior_alpha, beta)

            # 5. Check for Phase Transition (Panic Mode)
            if log_bayes_factor > bayes_factor
                initialize_prior_from_data!(obs, mapper, beta, dense_n)
                obs.last_entropy = bayes_entropy(obs.alpha)
                obs.last_score = log_bayes_factor 
            else 
                # Normal update
                obs.alpha = post_alpha
                obs.last_entropy = entropy_curr
                obs.last_score = log_bayes_factor
            end
            
            var_k = bayes_entropy_variance(post_alpha)

            push!(step_variances, var_k)
            push!(step_entropies, obs.last_entropy)
            push!(step_score, obs.last_score)
            
            # Store for full history
            full_history_S[t_idx, obs_idx] = obs.last_entropy
            full_history_score[t_idx, obs_idx] = obs.last_score
        end
        
        # collect found attractors for the continuity match
        history_att[t_idx]  = extract_attractors(mapper)
        global_entropy_var = sum(step_variances)/(length(observers)^2)
        push!(history_mean_S, mean(step_entropies))
        push!(history_max_score, maximum(step_score))
        push!(history_var_S, global_entropy_var)
    end

    return history_mean_S, history_var_S, history_max_score, history_att, full_history_S

end 
