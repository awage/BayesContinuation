"""
rossler_fig_4ab.jl
==================
Figure 4a — the basin of synchrony `B_S(K)` of the Rössler network at fixed `p`, measured
twice: by brute-force Monte Carlo, and by a single monitored `global_continuation`.
Figure 4b — the alarm `η(K)` that the monitored run raises along the way.

Both come out of `rossler_K_sweep_helper.jl`; see its docstring for the binary basin map
and the sweep. The point of (a) is that the Bayesian estimate follows the Monte Carlo
curve at a fraction of its cost, and the point of (b) is where it decided to spend it.
"""

using DrWatson
@quickactivate "BayesianBasinTracking"
if !isdefined(Main, :rossler_Ksweep)
    include(scriptsdir("rossler_K_sweep_helper.jl"))
end

p_val = 0.2

# ============================================================================
# The two sweeps
# ============================================================================

params_mc = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure n_mc

data_mc, file_mc = produce_or_load(
    datadir("data"), params_mc, rossler_Ksweep_montecarlo;
    prefix = "rossler_Ksweep_mc", storepatch = false, suffix = "jld2", force = false,
    filename = hash
)
@unpack sync_fracs, K_range = data_mc
K_vec = collect(K_range)
println("MC loaded from: $file_mc")

# One realisation only: this is the run whose cost is being compared to the Monte Carlo,
# and it uses the same `graph_seed`, hence the same network and the same `K` axis.
n_avg = 1
params_sing = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ β

data_sing, file_sing = produce_or_load(
    datadir("data"), params_sing, rossler_Ksweep;
    prefix = "rossler_Ksweep_sing", storepatch = false, suffix = "jld2", force = false,
    filename = hash
)
@unpack volumes, vol_var, min_eta, n_panics = data_sing
println("Bayes loaded from: $file_sing")

# The posterior volume of basin 1 (synchrony) and its standard deviation, against the
# binomial error of the Monte Carlo estimate.
bayes_sync_fracs = [get(v, 1, 0.0) for v in volumes]
bayes_vol_std    = sqrt.([get(v, 1, 0.0) for v in vol_var])
mc_std           = sqrt.(sync_fracs .* (1 .- sync_fracs) ./ n_mc)

# ============================================================================
# Fig 4a — B_S(K): Monte Carlo vs Bayes
# ============================================================================

fig = Figure(size = (750, 400))
ax = Axis(fig[1, 1]; ylabel = L"B_S", xlabel = L"K", lab_args...)
band!(ax, K_vec, sync_fracs .- mc_std, sync_fracs .+ mc_std, color = (:black, 0.2))
lines!(ax, K_vec, sync_fracs, color = :black, linewidth = 2, label = "MC")
band!(ax, K_vec, bayes_sync_fracs .- bayes_vol_std, bayes_sync_fracs .+ bayes_vol_std,
      color = (:red, 0.2))
lines!(ax, K_vec, bayes_sync_fracs, color = :red, linewidth = 2, label = "Bayes")
axislegend(ax; position = :lt)
ylims!(ax, 0.3, 1)
Label(fig[1, 1, TopLeft()], "(a)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

save(plotsdir("fig4a.png"), fig)
println("Saved → ", plotsdir("fig4a.png"))

# ============================================================================
# Fig 4b — η(K): the alarms
# ============================================================================

fig = Figure(size = (750, 400))
ax = Axis(fig[1, 1];
    ylabel = L"\eta  \text{(log Bayes factor)}", xlabel = L"K", lab_args...
)
lines!(ax, K_vec, min_eta, color = :black, linewidth = 2)
hlines!(ax, [0.0], color = :red, linestyle = :dash, linewidth = 1)
panic_idx = findall(>(0), n_panics)
if !isempty(panic_idx)
    scatter!(ax, K_vec[panic_idx], min_eta[panic_idx],
             color = :red, markersize = 8, label = "panic")
end
Label(fig[1, 1, TopLeft()], "(b)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

save(plotsdir("fig4b.png"), fig)
println("Saved → ", plotsdir("fig4b.png"))
