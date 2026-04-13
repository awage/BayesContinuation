using DrWatson
include(scriptsdir("rossler_K_sweep_helper.jl"))
using LsqFit

n_avg = 10

p_values  = Float64[]
mean_sync = Float64[]
std_sync  = Float64[]

for p_val in p_vals
    params = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    try
        data, _ = produce_or_load(
            datadir("data"), params, rossler_Ksweep;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        sync_fracs = [get(hv, 1, 0.0) for hv in data["history_volumes"]]
        push!(p_values,  p_val)
        push!(mean_sync, mean(sync_fracs))
        push!(std_sync,  std(sync_fracs))
    catch e
        @warn "Failed to load p=$p_val" exception=e
    end
end

println("Loaded $(length(p_values)) p values")

# ============================================================================
# Exponential fit:  f(p) = exp(a + b·p) + c
# ============================================================================

lin_model(x, θ) = θ[1] .+ θ[2] .* x
p0  = [0.5, -2.0]
fit = curve_fit(lin_model, p_values[5:end], log.(mean_sync[5:end] .- 0.6), p0)
a_fit, b_fit = coef(fit)
println("Fit: a=$(round(a_fit, digits=4))  b=$(round(b_fit, digits=4))")

p_fine    = range(0.0, 1.0; length = 100)
fit_curve = exp.(lin_model(collect(p_fine), coef(fit))) .+ 0.6

# ============================================================================
# Plot
# ============================================================================

fig = Figure(size = (650, 400))
ax  = Axis(fig[1, 1],
    yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
    xlabel = L"p",
    ylabel = L"\langle S_B\rangle_K",
)
lines!(ax,   p_values, mean_sync, color = :steelblue, linewidth = 2)
scatter!(ax, p_values, mean_sync, color = :steelblue, markersize = 7)
lines!(ax, collect(p_fine), fit_curve, color = :orange, linewidth = 2,
       linestyle = :dash, label = L"a e^{b p} + c")
axislegend(ax; position = :rt)

save(plotsdir("fig3c.png"), fig)
println("Saved → rossler_avg_vs_p.png")
