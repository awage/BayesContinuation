"""
rossler_fig4c.jl
================
Figure 4c — the basin of synchrony of the Rössler network averaged over its MSF interval,
`⟨B_S⟩_K`, as a function of the rewiring probability `p`, with an exponential fit.

One monitored `K` sweep per `p` (averaged over `n_avg` network realisations) comes from
`rossler_K_sweep_helper.jl`; see its docstring for the binary basin map and the sweep.
Every `p` reuses the `.jld2` that `rossler_fig_4ab.jl` produces for its own `p`, so run
this one second and it is mostly a load.
"""

using DrWatson
@quickactivate "BayesContinuation"
if !isdefined(Main, :rossler_Ksweep)
    include(scriptsdir("rossler_K_sweep_helper.jl"))
end
using LsqFit

n_avg = 10

p_values  = Float64[]
mean_sync = Float64[]
std_sync  = Float64[]

for p_val in p_vals
    params = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ β
    try
        data, _ = produce_or_load(
            datadir("data"), params, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        # posterior volume of basin 1 (synchrony) at each `K` of this network's `Iₛ`
        sync_fracs = [get(v, 1, 0.0) for v in data["volumes"]]
        push!(p_values,  p_val)
        push!(mean_sync, mean(sync_fracs))
        push!(std_sync,  std(sync_fracs))
    catch e
        # a `p` whose networks are all linearly unstable has no `Iₛ` to average over
        @warn "Failed to load p=$p_val" exception=e
    end
end

println("Loaded $(length(p_values)) p values")

# ============================================================================
# Exponential fit:  f(p) = exp(a + b·p) + c
# ============================================================================

const c_offset = 0.6    # the floor the decay settles on, fitted by eye
lin_model(x, θ) = θ[1] .+ θ[2] .* x
p0 = [0.5, -2.0]
# the first few `p` are the pre-transition plateau, and are left out of the fit
fit = curve_fit(lin_model, p_values[5:end], log.(mean_sync[5:end] .- c_offset), p0)
a_fit, b_fit = coef(fit)
println("Fit: a=$(round(a_fit, digits = 4))  b=$(round(b_fit, digits = 4))")

p_fine    = range(0.0, 1.0; length = 100)
fit_curve = exp.(lin_model(collect(p_fine), coef(fit))) .+ c_offset

# ============================================================================
# Plot
# ============================================================================

fig = Figure(size = (750, 400))
ax = Axis(fig[1, 1]; xlabel = L"p", ylabel = L"\langle B_S\rangle_K", lab_args...)
lines!(ax,   p_values, mean_sync, color = :steelblue, linewidth = 2)
scatter!(ax, p_values, mean_sync, color = :steelblue, markersize = 7)
lines!(ax, collect(p_fine), fit_curve, color = :orange, linewidth = 2,
       linestyle = :dash, label = L"a e^{b p} + c")
axislegend(ax; position = :rt, labelsize = 25)
Label(fig[1, 1, TopLeft()], "(c)",
        fontsize = 25,
        padding = (0, 50, -10, 0),
        halign = :right)

save(plotsdir("fig4c.png"), fig)
println("Saved → ", plotsdir("fig4c.png"))
