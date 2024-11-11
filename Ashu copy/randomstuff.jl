using Plots

# Define your functions
f(x) = sin(x)
g(x) = cos(x)

# Define the x range
x = range(0, stop=2π, length=100)

# Plot with a custom color scheme
plot(x, f.(x), label="f(x) = sin(x)", linewidth=2, color=:blue)
plot!(x, g.(x), label="g(x) = cos(x)", linewidth=2, color=:green)

# Fill the area between the two curves with a nice color blend
plot!(x, f.(x), fillrange=g.(x), fillcolor=:lightblue, fillalpha=0.5, legend=:topright)
