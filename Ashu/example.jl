using Plots

function plot_Q_with_colored_yticks(Q::Matrix, N::Int, special_ticks::Vector{Int}, filename)
    # Plot the heatmap with y-axis flipped
    p = heatmap(
        Q,
        c = :coolwarm,
        title = "Transition Rate Matrix Heatmap",
        xlabel = "States",
        ylabel = "States",
        size = (1600, 800),
        left_margin = 10Plots.mm,
        right_margin = 10Plots.mm,
        bottom_margin = 10Plots.mm,
        yflip = true  # Flip the y-axis to have indices increasing downwards
    )
    
    # Set custom y-ticks
    y_ticks = collect(1:size(Q, 1))
    yticks!(p, (y_ticks, string.(y_ticks)))
    
    # Annotate special ticks on the y-axis
    for i in special_ticks
        annotate!(p, 0.5, i, text("T", :red, 12, :left))  # Adjusted position and size
    end
    
    display(p)
    savefig(p, filename)
end

# Test the function with a sample matrix
function test_plot_Q_with_colored_yticks()
    N = 4
    Q = [0.1 0.2 0.3 0.4;
    0.5 0.6 0.7 0.8;
    0.9 1.0 1.1 1.2;
    1.3 1.4 1.5 1.6]
     # Generate a random NxN matrix
    special_ticks = [3, 7]  # Indices of special ticks
    filename = "Q_heatmap.png"
    
    plot_Q_with_colored_yticks(Q, N, special_ticks, filename)
end

# Run the test
test_plot_Q_with_colored_yticks()
