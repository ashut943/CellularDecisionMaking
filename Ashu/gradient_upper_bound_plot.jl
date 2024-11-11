using LsqFit, Plots

N_vals = [2, 3, 4, 5, 6, 7]
gradients = [0.31204645713527873, 0.19219334421294632, 0.1102455528686576,0.06530152969677465, 0.05256877989754358, 0.05132823584488033]

power_law_model(N, p) = p[1] * N .^ (-p[2])
exp_decay_model(N, p) = p[1] * exp.(-p[2] * N)

initial_params_power = [1.0, 1.0]
initial_params_exp = [1.0, 0.1]

fit_power = curve_fit(power_law_model, N_vals, gradients, initial_params_power)
fit_exp = curve_fit(exp_decay_model, N_vals, gradients, initial_params_exp)

a_power, b_power = fit_power.param
a_exp, b_exp = fit_exp.param

N_fit = range(2, stop=7, length=100)
gradients_power_fit = power_law_model(N_fit, fit_power.param)
gradients_exp_fit = exp_decay_model(N_fit, fit_exp.param)

scatter(N_vals, gradients, label="Observed Gradients", xlabel="N", ylabel="Gradient", title="Gradient as a Function of N", marker=:x, color=:black)
plot!(N_fit, gradients_power_fit, label="Power-law Fit", linestyle=:dash)
plot!(N_fit, gradients_exp_fit, label="Exponential Decay Fit", linestyle=:dot)

println("Power-law model: Gradient(N) ≈ $a_power * N^(-$b_power)")
println("Exponential decay model: Gradient(N) ≈ $a_exp * exp(-$b_exp * N)")

savefig("gradient_vs_N_power_exp_fits.png")
savefig("gradient_vs_N_power_exp_fits.svg")

plot!()