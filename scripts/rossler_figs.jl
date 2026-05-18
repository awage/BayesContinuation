using DrWatson
@quickactivate "BayesContinuation"

println("=== Loading helper ===")
include(scriptsdir("rossler_K_sweep_helper.jl"))

lab_args = (;yticklabelsize = 20, xticklabelsize = 20, ylabelsize = 25, xlabelsize = 25)

println("\n=== Fig 3ab: S_B(K) and η(K) — Bayesian and MC sweeps ===")
include(scriptsdir("rossler_K_sweep_fig_3ab.jl"))

println("\n=== Fig 3c: ⟨S_B⟩_K vs p ===")
include(scriptsdir("rossler_K_sweep_fig3c.jl"))

# println("\n=== Fig 4: RMSE(p) and P_alarm(p) ===")
# include(scriptsdir("rossler_K_sweep_fig4.jl"))

println("\nDone.")
