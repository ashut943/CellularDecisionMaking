using JuMP, Ipopt, Plots, Printf, LinearAlgebra, SCS, COSMO, Distributions, LightGraphs, FileIO, ImageIO, VideoIO,ProgressMeter, FFMPEG

function generate_filename(folder_name,base_name::String)
    return joinpath(folder_name, @sprintf("%s", base_name))
end

function is_irreducible(Q::Matrix{Float64})
    n = size(Q, 1)
    G = SimpleDiGraph(n) 
    for i in 1:n
        for j in 1:n
            if i != j && Q[i, j] > 0
                add_edge!(G, i, j)
            end
        end
    end
    return is_strongly_connected(G)
end

function hitting_time(Q,targetstates_good,targetstates_bad,startstates,λ)
    n=size(Q,1)
    A=copy(Q)
    b=-ones(n)
    targetstates=[targetstates_good;targetstates_bad]
    for target_state ∈ targetstates
        A[target_state, :] .= 0.0
        A[target_state, target_state] = 1.0
        if(target_state ∈ targetstates_good)
            b[target_state] = 0.0
        else
            b[target_state] = λ
        end
    end
    for i in 1:n
        if all(Q[i, :] .== 0.0) && i ∉ targetstates
            A[i, :] .= 0.0
            A[i, i] = 1.0
            b[i] = 0.0
        end
    end
    T = A \ b
    for i in 1:n
        if T[i]==0 && i ∉ targetstates
            T[i]=Inf 
        end
    end
    return [T[start_state] for start_state ∈ startstates]
end


function simulate_ctmc(Q::Array{Float64,2}, initial_state::Int, T::Float64)
    num_states = size(Q, 1)
    t = 0.0
    s = initial_state
    times = [t]
    states = [s]
    tolerance = 1e-9

    while t < T
        λ = -Q[s, s]
        if λ < -tolerance
            error("Rate out of current state is negative at state $s.")
        elseif λ == 0
            println("There is an absorbing state at state $s.")
            break
        end

        Δt = rand(Exponential(λ))
        t += Δt
        if t >= T
            break
        end

        rates = copy(Q[s, :])
        rates[s] = 0.0
        total_rate = sum(rates)
        
        if total_rate <= 0
            error("No transitions available from current state $s.")
        end

        probs = rates / total_rate
        dist = Categorical(probs)
        s = rand(dist)
        push!(times, t)
        push!(states, s)
    end

    if times[end] < T
        push!(times, T)
        push!(states, states[end])
    end

    return times, states
end

function simulate_mul_ctmc(Q::Array{Float64,2}, initial_state::Int, T::Float64, num_paths::Int, num_time_points::Int)
    #used chatgpt for this, was too lazy
    num_states = size(Q, 1)
    time_points = range(0, T, length=num_time_points)
    #here to make plotting feasible
    state_counts = zeros(Float64, num_states, num_time_points)
    
    for path in 1:num_paths
        times, states = simulate_ctmc(Q, initial_state, T)
        idx = 1
        for t_idx in 1:num_time_points
            t = time_points[t_idx]
            while idx < length(times) && times[idx+1] <= t
                #essentially check if the current checkpoint time is before or after the latest transition
                idx += 1
            end
            state = states[idx]
            state_counts[state, t_idx] += 1
        end
    end

    state_probs = state_counts / num_paths

    return time_points, state_probs
end

function plot_ctmc(times::Vector{Float64}, states::Vector{Int}, T::Float64)
    plot(
        xlabel="Time (t)", 
        ylabel="State (s)", 
        title="CTMC Simulation",
        yticks=collect(1:maximum(states)),
        xlim=(0, T), 
        ylim=(minimum(states)-0.5, maximum(states)+0.5),
        grid=:both,
        size=(1500, 1000),left_margin=10Plots.mm, right_margin=10Plots.mm,bottom_margin=10Plots.mm)
    
    for i in 1:(length(times)-1)
        plot!([times[i], times[i+1]], [states[i], states[i]], linewidth=2, label=false)
        if i < length(times)-1
            plot!([times[i+1], times[i+1]], [states[i], states[i+1]], linestyle=:dash, color=:gray, label=false)
        end
    end
    display(current())
end

function plot_Q_with_colored_yticks(Q::Matrix, N::Int, special_ticks::Vector{Int}, filename, λ::Float64=missing)
    # Plot the heatmap with y-axis flipped
    title_text = "Transition Rate Matrix Heatmap"
    title_text = @sprintf("Transition Rate Matrix Heatmap, N=%d", N)
    if λ !== missing
        title_text = @sprintf("Transition Rate Matrix Heatmap, N=%d, λ=%s", N, string(λ))
    end
    p = heatmap(
        Q,
        c = :matter,
        title = title_text,
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
        annotate!(p, 0.5, i, text("T", :red, 12, :right))  # Adjusted position and size
    end
    
    # display(p)
    savefig(p,filename * ".png")
    savefig(p,filename * ".svg")
end

function log_to_file(msg)
    println(msg)                 # Print to console
    println(log_file, msg)       # Write to file
end

function extract_lambda_value(filename::String)::Float64
    parts = split(filename, '_')
    idx = findfirst(x -> x == "lambda", parts)
    if idx !== nothing && idx < length(parts)
        try
            return parse(Float64, parts[idx + 1])
        catch
            return Inf
        end
    end
    return Inf
end

function create_movie(input_folder::String, output_file::String; frame_rate::Int = 10)
    frames = sort(filter(f -> endswith(f, ".png"), readdir(input_folder; join=true)), by = x -> extract_lambda_value(x))
    
    if isempty(frames)
        println("No frames found in the specified input folder.")
        return
    end
    list_file = joinpath(input_folder, "frames_list.txt")
    open(list_file, "w") do io
        for frame in frames
            println(io, "file '" * abspath(frame) * "'")
        end
    end
    ffmpeg_cmd = `"/opt/homebrew/bin/ffmpeg" -f concat -safe 0 -r $frame_rate -i $list_file -pix_fmt yuv420p -crf 20 -preset medium -y $output_file`
    run(ffmpeg_cmd)
    rm(list_file)
end