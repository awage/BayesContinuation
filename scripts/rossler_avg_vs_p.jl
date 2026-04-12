using DrWatson
@quickactivate "BayesContinuation"
using Statistics
using JLD2
using CairoMakie
using LsqFit

# Parameters that index into the saved files (must match rossler_sync_Ksweep.jl)
N_osc      = 100
k_degree   = 8
graph_seed = 12345

# Rössler parameters
a_ros, b_ros, c_ros = 0.2, 0.2, 9.0

# Integration settings
r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0
n_K_steps   = 50

# Bayesian continuation (over K)
λ        = 0.7
sparse_n = 20
dense_n  = 200
n_tiles  = 1
n_avg    = 10   # number of network realisations to average over

p_values  = Float64[]
mean_sync = Float64[]
std_sync  = Float64[]
p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

for p_val in p_vals
    # params = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    params = @strdict N_osc k_degree graph_seed n_avg p_val n_K_steps a_ros b_ros c_ros  r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    try 
        data, file = produce_or_load(
            datadir("data"), params, x -> nothing;
            prefix = "rossler_Ksweep", storepatch = false, suffix = "jld2", force = false,
            filename = hash
        )
        sync_fracs = [get(hv, 1, 0.0) for hv in data["history_volumes"]]
        push!(p_values,  p_val)
        push!(mean_sync, mean(sync_fracs))
        push!(std_sync,  std(sync_fracs))
    catch 
        sync_fracs = 0.
    end
end

println("Loaded $(length(p_values)) p values: ", p_values)

# ============================================================================
# Exponential fit:  f(p) = a·exp(b·p) + c
# ============================================================================

exp_model(x, θ) = θ[1] .+ θ[2] .* x
p0   = [0.5, -2.0]
fit  = curve_fit(exp_model, p_values[5:end], log.(mean_sync[5:end] .- 0.6), p0)
a_fit, b_fit = coef(fit)
println("Exponential fit: a=$(round(a_fit, digits=4))  b=$(round(b_fit, digits=4))  ")

p_fine    = range(0.0, 1.0; length = 100)
emodel(x, θ) = exp.(θ[1] .+ θ[2] .* x) .+ 0.6
fit_curve = emodel(collect(p_fine), coef(fit))
# fit_curve = exp_model(collect(p_fine), coef(fit))

# ============================================================================
# Plot: ⟨S_B⟩_K vs p
# ============================================================================

fig = Figure(size = (650, 400))
ax  = Axis(fig[1, 1],
    yticklabelsize = 15, xticklabelsize = 15, ylabelsize = 20, xlabelsize = 20,
    xlabel = L"p",
    ylabel = L"\langle S_B\rangle_K",
    # title  = "Rössler WS network — basin stability averaged over K sweep (N=$N_osc, ⟨k⟩=$k_degree)",
)

# band!(ax, p_values, mean_sync .- std_sync, mean_sync .+ std_sync;
#       color = (:steelblue, 0.2))
lines!(ax,   p_values, mean_sync, color = :steelblue, linewidth = 2)
scatter!(ax, p_values, mean_sync, color = :steelblue, markersize = 7)
lines!(ax, collect(p_fine), fit_curve, color = :orange, linewidth = 2, linestyle = :dash,
       label = L"a e^{b p} + c")
axislegend(ax; position = :rt)
# ylims!(ax, 0.2, 1)
# xlims!(ax, 0, 1)

outpath = plotsdir("rossler_avg_vs_p.png")
save(outpath, fig)
println("Saved → $outpath")
