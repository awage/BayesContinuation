function estimate_entropy(params, a_range, factory::MapperFactory; parallel=false)

    @unpack sparse_n, dense_n, n_tiles, global_bounds, λ = params
    β = get(params, "β", 0.5)

    if parallel && !_parallel_allowed(factory)
        @warn "parallel=true ignored: AttractorMapperFactory mappers hold mutable state. Running sequentially."
        parallel = false
    end

    println("Initializing $(n_tiles)x$(n_tiles) observer grid...")
    observers = generate_tiling(global_bounds, n_tiles, β)

    history_mean_S    = Float64[]
    history_var_S     = Float64[]
    history_max_llr   = Float64[]
    history_n_panics  = Int[]
    history_volumes   = Dict{Int, Float64}[]
    n_steps           = length(a_range)
    full_history_S    = zeros(Float64, n_steps, length(observers))
    full_history_llr  = zeros(Float64, n_steps, length(observers))

    mapper = build_mapper(factory, a_range[1])
    thread_mappers = parallel ?
        [build_mapper(factory, a_range[1]) for _ in 1:Threads.nthreads()] :
        nothing

    step_entropies = Float64[]
    step_variances = Float64[]
    for (i, obs) in enumerate(observers)
        if parallel
            initialize_prior_from_data!(obs, thread_mappers, β, dense_n)
        else
            initialize_prior_from_data!(obs, mapper, β, dense_n)
        end
        obs.last_entropy = bayes_entropy(obs.alpha)
        push!(step_entropies, obs.last_entropy)
        full_history_S[1, i] = obs.last_entropy
        push!(step_variances, bayes_entropy_variance(obs.alpha))
    end
    push!(history_var_S,    sum(step_variances) / length(observers)^2)
    push!(history_mean_S,   mean(step_entropies))
    push!(history_max_llr,  0.0)
    push!(history_n_panics, 0)
    push!(history_volumes,  basin_volumes(observers))
    update!(factory, mapper)

    @showprogress for (t_idx, a_val) in enumerate(a_range)
        if t_idx == 1; continue; end

        mapper = build_mapper(factory, a_val)
        if parallel
            thread_mappers = [build_mapper(factory, a_val) for _ in 1:Threads.nthreads()]
        end
        step_entropies = Float64[]
        step_variances = Float64[]
        step_llr       = Float64[]
        step_panics    = 0

        for (obs_idx, obs) in enumerate(observers)

            # 1. Decay prior
            prior_alpha = Dict{Int, Float64}(k => λ * v for (k, v) in obs.alpha)

            # 2. Sparse sampling
            labels = Vector{Int}(undef, sparse_n)
            if parallel
                Threads.@threads for i in 1:sparse_n
                    labels[i] = thread_mappers[Threads.threadid()](pick_random_point(obs))
                end
            else
                for i in 1:sparse_n
                    labels[i] = mapper(pick_random_point(obs))
                end
            end
            new_counts = Dict{Int, Int}()
            for label in labels
                new_counts[label] = get(new_counts, label, 0) + 1
            end

            # 3. Posterior update
            post_alpha = copy(prior_alpha)
            for (label, count) in new_counts
                post_alpha[label] = get(post_alpha, label, β) + count
            end

            # 4. Metrics
            entropy_curr = bayes_entropy(post_alpha)
            reject, llr, _ = test_continuity(new_counts, prior_alpha, β)

            # 5. Panic mode on rejection
            if reject
                step_panics += 1
                if parallel
                    initialize_prior_from_data!(obs, thread_mappers, β, dense_n)
                else
                    initialize_prior_from_data!(obs, mapper, β, dense_n)
                end
                obs.last_entropy = bayes_entropy(obs.alpha)
                obs.last_llr     = llr
            else
                obs.alpha        = post_alpha
                obs.last_entropy = entropy_curr
                obs.last_llr     = llr
            end

            push!(step_variances, bayes_entropy_variance(obs.alpha))
            push!(step_entropies, obs.last_entropy)
            push!(step_llr,       obs.last_llr)

            full_history_S[t_idx, obs_idx]   = obs.last_entropy
            full_history_llr[t_idx, obs_idx] = obs.last_llr
        end

        update!(factory, mapper)
        push!(history_mean_S,   mean(step_entropies))
        push!(history_max_llr,  maximum(step_llr))
        push!(history_n_panics, step_panics)
        push!(history_var_S,    sum(step_variances) / length(observers)^2)
        push!(history_volumes,  basin_volumes(observers))
    end

    return (; history_mean_S, history_var_S, history_max_llr, history_n_panics,
              full_history_S, full_history_llr, history_volumes)

end
