using DrWatson
@quickactivate "BayesContinuation"
if !isdefined(Main, :rossler_Ksweep)
    include(scriptsdir("rossler_K_sweep_helper.jl"))
end


# ============================================================================
# Fig 3b — S_B(K): MC vs single-run Bayes, and η(K) alarms, for each p
# ============================================================================

for p_val in p_vals

    params_mc = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure n_mc

    try
        data_mc, file_mc = produce_or_load(
            datadir("data"), params_mc, rossler_Ksweep_montecarlo;
            prefix = "rossler_Ksweep_mc", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        @unpack sync_fracs, K_range = data_mc
        K_vec = collect(K_range)
        println("MC loaded from: $file_mc")

        n_avg_sing = 1
        params_sing = @strdict N_osc k_degree graph_seed n_avg=n_avg_sing p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
        data_sing, _ = produce_or_load(
            datadir("data"), params_sing, rossler_Ksweep;
            prefix = "rossler_Ksweep_sing", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        @unpack history_n_panics, history_max_llr, history_volumes, K_range, history_vol_var = data_sing
        bayes_sync_fracs = [get(hv, 1, 0.0) for hv in history_volumes]
        bayes_vol_std    = sqrt.([get(vv, 1, 0.0) for vv in history_vol_var])
        mc_std           = sqrt.(sync_fracs .* (1 .- sync_fracs) ./ n_mc)

        # S_B(K): MC vs Bayes
        fig = Figure(size = (750, 400))
        ax  = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"S_B",
            xlabel = L"K",
        )
        band!(ax, K_vec, sync_fracs .- mc_std, sync_fracs .+ mc_std, color = (:black, 0.2))
        lines!(ax, K_vec, sync_fracs, color = :black, linewidth = 2, label = "MC")
        band!(ax, K_vec, bayes_sync_fracs .- bayes_vol_std, bayes_sync_fracs .+ bayes_vol_std,
              color = (:red, 0.2))
        lines!(ax, K_vec, bayes_sync_fracs, color = :red, linewidth = 2, label = "Bayes")
        axislegend(ax; position = :lt)
        ylims!(ax, 0, 1)
        save(plotsdir(savename("rossler_sync_Ksweep_mc", (; p = p_val), "png")), fig)
        println("Saved MC plot → rossler_sync_Ksweep_mc_p=$(p_val).png")

        # η(K): alarms
        fig = Figure(size = (750, 400))
        ax  = Axis(fig[1, 1],
            yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
            ylabel = L"\eta  \text{(log Bayes factor)}",
            xlabel = L"K",
        )
        lines!(ax, K_vec, history_max_llr, color = :black, linewidth = 2)
        hlines!(ax, [0.0], color = :red, linestyle = :dash, linewidth = 1)
        panic_idx = findall(>(0), history_n_panics)
        if !isempty(panic_idx)
            scatter!(ax, K_vec[panic_idx], history_max_llr[panic_idx],
                     color = :red, markersize = 8, label = "panic")
        end
        save(plotsdir(savename("rossler_sync_Ksweep_alarms", (; p = p_val), "png")), fig)

    catch e
        @warn "MC sweep failed for p=$p_val" exception=e
    end
end
