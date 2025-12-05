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

    @unpack SPARSE_N, DENSE_N, KL_THRESHOLD, N_TILES, GLOBAL_BOUNDS, LAMBDA = params

    println("Initializing $(N_TILES)x$(N_TILES) observer grid...")
    observers = generate_tiling(GLOBAL_BOUNDS, N_TILES, BETA)

    history_mean_S = Float64[]
    history_var_S = Float64[]
    history_max_KL = Float64[]
    full_history_S = zeros(Float64, al, length(observers))
    full_history_KL = zeros(Float64, al, length(observers))
    mapper = get_map(a_range[1], nothing) 

    step_entropies = Float64[]
    step_variances = Float64[]
    # Initialize Priors for ALL boxes and initialize the first frame
    for (i, obs) in enumerate(observers)
        initialize_prior_from_data!(obs, mapper, BETA, DENSE_N)
        obs.last_entropy = bayes_entropy(obs.alpha)
        push!(step_entropies, obs.last_entropy)
        full_history_S[1,i] = obs.last_entropy
        var_k = bayes_entropy_variance(obs.alpha)
        push!(step_variances, var_k)
    end
    global_entropy_var = sum(step_variances)/(length(observers)^2)
    push!(history_var_S, global_entropy_var)
    push!(history_mean_S, mean(step_entropies))
    push!(history_max_KL, 0.0)
        
    # collect found attractors for the continuity match
    # (and bifurcation diagram if needed)
    atts = extract_attractors(mapper)
    history_att = Array{typeof(atts)}(undef, al)
    history_att[1] = atts



    @showprogress for (t_idx, a_val) in enumerate(a_range)
        if t_idx == 1 ; continue; end
        
        # Update System Dynamics and do the attractor seed and match
        mapper = get_map(a_val, history_att[t_idx-1]) 

        step_entropies = Float64[]
        step_variances = Float64[]
        step_kls = Float64[]

        # Iterate over all boxes
        for (obs_idx, obs) in enumerate(observers)
            
            # 1. Decay Prior
            prior_alpha = Dict{Int, Float64}()
            for (k, v) in obs.alpha
                decayed_val = LAMBDA * (v - BETA) + BETA
                prior_alpha[k] = decayed_val
            end

            # 2. Sparse Sampling
            new_counts = Dict{Int, Int}()
            for _ in 1:SPARSE_N
                u0 = pick_random_point(obs)
                label = mapper(u0) 
                new_counts[label] = get(new_counts, label, 0) + 1
            end

            # 3. Posterior Update
            post_alpha = copy(prior_alpha)
            for (label, count) in new_counts
                current_val = get(post_alpha, label, BETA)
                post_alpha[label] = current_val + count
            end

            # 4. Compute Metrics
            S_curr = bayes_entropy(post_alpha)
            KL_curr = kl_div(post_alpha, prior_alpha, BETA)

            # 5. Check for Phase Transition (Panic Mode)
            if KL_curr > KL_THRESHOLD
                initialize_prior_from_data!(obs, mapper, BETA, DENSE_N)
                obs.last_entropy = bayes_entropy(obs.alpha)
                obs.last_kl = KL_curr 
            else 
                # Normal update
                obs.alpha = post_alpha
                obs.last_entropy = S_curr
                obs.last_kl = KL_curr
            end
            
            var_k = bayes_entropy_variance(post_alpha)

            push!(step_variances, var_k)
            push!(step_entropies, obs.last_entropy)
            push!(step_kls, obs.last_kl)
            
            # Store for full history
            full_history_S[t_idx, obs_idx] = obs.last_entropy
            full_history_KL[t_idx, obs_idx] = obs.last_kl
        end
        
        # collect found attractors for the continuity match
        history_att[t_idx]  = extract_attractors(mapper)
        global_entropy_var = sum(step_variances)/(length(observers)^2)
        push!(history_mean_S, mean(step_entropies))
        push!(history_max_KL, maximum(step_kls))
        push!(history_var_S, global_entropy_var)
    end

    return history_mean_S, history_var_S, history_max_KL, history_att, full_history_S

end 

