using DrWatson
@quickactivate "BayesContinuation"
using Statistics
using JLD2
using CairoMakie

# ============================================================================
# Aggregate rossler_Ksweep results: for each p, compute mean sync fraction
# averaged over the K sweep.  Data was produced by rossler_sync_Ksweep.jl.
# ============================================================================

# Parameters that index into the saved files (must match rossler_sync_Ksweep.jl)
N_osc      = 100
k_degree   = 8
graph_seed = 12345
n_K_steps  = 100
a_ros, b_ros, c_ros = 0.2, 0.2, 9.0
r_thresh    = 0.90
T_transient = 100.0
T_measure   = 200.0
λ        = 0.7
# sparse_n = 20
# dense_n  = 100
sparse_n = 50
dense_n  = 1000
n_tiles  = 1

p_values  = Float64[]
mean_sync = Float64[]
std_sync  = Float64[]
p_vals = sort(unique(vcat(
    range(0.0,  0.15, step = 0.01),   # fine grid in transition region
    range(0.20, 1.00, step = 0.05),   # coarse grid elsewhere
)))

for p_val in p_vals
    params = @strdict N_osc k_degree graph_seed p_val n_K_steps a_ros b_ros c_ros r_thresh T_transient T_measure sparse_n dense_n n_tiles λ
    fpath  = datadir("data", savename("rossler_Ksweep", params, "jld2"))

    isfile(fpath) || continue

    data = load(fpath)
    try 
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
# Plot: ⟨S_B⟩_K vs p
# ============================================================================

fig = Figure(size = (650, 400))
ax  = Axis(fig[1, 1],
    xlabel = "Rewiring probability  p",
    ylabel = "Mean sync fraction  ⟨S_B⟩_K",
    title  = "Rössler WS network — basin stability averaged over K sweep (N=$N_osc, ⟨k⟩=$k_degree)",
)

band!(ax, p_values, mean_sync .- std_sync, mean_sync .+ std_sync;
      color = (:steelblue, 0.2))
lines!(ax,   p_values, mean_sync, color = :steelblue, linewidth = 2)
scatter!(ax, p_values, mean_sync, color = :steelblue, markersize = 7)
ylims!(ax, 0.2, 1)
xlims!(ax, 0, 1)

outpath = plotsdir("rossler_avg_vs_p.png")
save(outpath, fig)
println("Saved → $outpath")
