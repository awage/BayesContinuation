using DrWatson
@quickactivate "BayesContinuation"
include(scriptsdir("rossler_K_sweep_helper.jl"))

# ============================================================================
# Fig 4a — RMSE(p): distance between MC and single-run Bayesian S_B(K)
# ============================================================================

p_wiring = Float64[]
dist_rmse  = Float64[]

n_avg_sing = 1   # single-run Bayes as reference

for p_val in p_vals
    try
        params_mc = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure n_mc
        data_mc, _ = produce_or_load(
            datadir("data"), params_mc, rossler_Ksweep_montecarlo;
            prefix = "rossler_Ksweep_mc", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        params_sing = @strdict N_osc k_degree graph_seed n_avg=n_avg_sing p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
        data_sing, _ = produce_or_load(
            datadir("data"), params_sing, rossler_Ksweep;
            prefix = "rossler_Ksweep_sing", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )

        mc_fracs    = data_mc["sync_fracs"]
        bayes_fracs = [get(hv, 1, 0.0) for hv in data_sing["history_volumes"]]

        diff_sq = (mc_fracs .- bayes_fracs).^2
        rmse    = sqrt(mean(diff_sq))

        push!(p_wiring, p_val)
        push!(dist_rmse,  rmse)
    catch e
        @warn "RMSE computation failed for p=$p_val" exception=e
    end
end

if !isempty(p_wiring)
    fig_dist = Figure(size = (750, 400))
    ax_dist  = Axis(fig_dist[1, 1],
        yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
        ylabel = L"\mathrm{RMSE}(S_B^{\mathrm{MC}},\, S_B^{\mathrm{Bayes}})",
xlabel = L"p_{\text{wiring}}",
    )
    lines!(ax_dist,   p_wiring, dist_rmse, color = :black, linewidth = 2)
    scatter!(ax_dist, p_wiring, dist_rmse, color = :black, markersize = 6)
    ylims!(ax_dist, 0, nothing)
    save(plotsdir("fig4a.png"), fig_dist)
    println("Saved RMSE plot → rossler_sync_distance_vs_p.png")
end

# ============================================================================
# Fig 4b — p_wiring(p): fraction of K steps that triggered an alarm,
#           averaged over n_avg=10 network realisations.
#   p_wiring(p) = sum(history_n_panics) / n_K_steps
# ============================================================================

n_avg = 10

p_wiring    = Float64[]
prob_alarm = Float64[]

for p_val in p_vals
    params_alarm = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    try
        data, _ = produce_or_load(
            datadir("data"), params_alarm, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        @show h_panics = data["history_n_panics"]
        push!(p_wiring,    p_val)
        push!(prob_alarm, sum(h_panics) / n_K_steps)
    catch e
        @warn "Alarm probability failed for p=$p_val" exception=e
    end
end

if !isempty(p_wiring)
    fig_alarm = Figure(size = (650, 400))
    ax_alarm  = Axis(fig_alarm[1, 1],
        yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
        ylabel = L"p_{a}",
        xlabel = L"p_{\text{wiring}}",
        # yscale = log10
    )
    lines!(ax_alarm,   p_wiring, prob_alarm, color = :black, linewidth = 2)
    scatter!(ax_alarm, p_wiring, prob_alarm, color = :black, markersize = 6)
    # ylims!(ax_alarm, 0., nothing)
    save(plotsdir("fig4b.png"), fig_alarm)
    println("Saved alarm probability plot → rossler_alarm_prob_vs_p.png")
end

# ============================================================================
# Fig 4c — Gain G(p): expected sampling cost of Bayesian method vs full dense,
#   G = N_d / (N_s (1 - p_a) + N_d (1 - p_a))
# where p_a = prob_alarm, N_s = sparse_n, N_d = dense_n.
# ============================================================================

if !isempty(p_wiring)
    gain = dense_n ./ (sparse_n .* (1 .- prob_alarm) .+ dense_n .*prob_alarm)

    fig_gain = Figure(size = (650, 400))
    ax_gain  = Axis(fig_gain[1, 1],
        yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
        ylabel = L"G",
        xlabel = L"p_{\text{wiring}}",
    )
    lines!(ax_gain,   p_wiring, gain, color = :black, linewidth = 2)
    scatter!(ax_gain, p_wiring, gain, color = :black, markersize = 6)
    save(plotsdir("fig4c.png"), fig_gain)
    println("Saved gain plot → fig4c.png")
end
