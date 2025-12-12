"""
Script for comparing multiscale to single scale solving on the following problem for a given
vector y and matrix A.

minₓ 0.5‖Ax - y‖₂² + λGL(x) s.t. ‖x‖₁ = 1, and x ≥ 0,

where GL is the Graph Laplacian regularizer 0.5x'Gx. This is the discrete version of the
following continuous problem,

min_f 0.5‖A(f) - y‖₂² + 0.5λ‖f′‖₂² ‖ s.t. ‖f‖₁ = 1, and f(t) ≥ 0,

where x[i] = f(t[i])Δt
"""

using Random
using LinearAlgebra
using Plots
using Statistics

########################
# Function Definitions #
########################

### Helpers ###

"""Converts the scale number s to the number of points to skip over in a coarsening of the problem"""
scale_to_skip(s) = 2^(s-1)

"""Converts the number of points in a discretization to the maximum scale number S"""
points_to_scale(n) = Int(log2(n-1))

"""Efficient calculation of ‖x‖₂²"""
norm2(x) = sum(x -> x^2, x)

"""Rectified Linear Unit"""
ReLU(x) = max(0, x)

"""Linearly interpolates a vector x.
If J=length(x), interpolate(x) will have length 2J-1."""
function interpolate(x)
    I = 2length(x)-1
    x̲ = zeros(I)
    for i in 1:I
        if iseven(i)
            x̲[i] = 0.5*(x[i÷2] + x[i÷2 + 1])
        else # i is odd
            x̲[i] = x[(i+1) ÷ 2]
        end
    end
    return x̲
end

### Regularizer ###

"""Graph Laplacian Matrix"""
laplacian_matrix(n) = SymTridiagonal([1;2*ones(n-2);1],-ones(n-1))

"""
    GL(x; Δt, scale=1)

Compute cx'Gx efficiently where
c = 0.5 / Δt^3 / scale_to_skip(scale)
G = laplacian_matrix(length(x))

Equivalent to c*norm(diff(x))^2.
"""
function GL(x; Δt, scale=1)
    n = length(x)
    total = (x[1] - x[2])^2 # type stable and saves a single call to initialize total
    for i in 2:(n-1)
        total += (x[i] - x[i+1])^2
    end
    return 0.5*total / Δt^3 / scale_to_skip(scale)
end

"""Gradient of GL(x)"""
∇GL(x; Δt, scale=1) =  laplacian_matrix(length(x)) * x / (Δt^3 * scale_to_skip(scale))

"""Efficient implementation of ∇GL(x) that stores the result in z"""
function ∇GL!(z, x; Δt, scale=1)
    n=length(x)
    z[1] = x[1] - x[2]
    for i in 2:(n-1)
        z[i] = -x[i-1] + 2x[i] - x[i+1]
    end
    z[n] = x[n] - x[n-1]
    Δt3_scaled = 1 / Δt^3 / scale_to_skip(scale)
    z .*= Δt3_scaled
    return z
end

### Problem Creation ###

"""Legendre Polynomial Measurement Basis Functions"""
g(t, m) = sum(binomial(m, k)*binomial(m+k, k)*((t - 1)/2)^k for k in 0:m) * sqrt((2m+1)/2)

"""Samples the Legendre polynomials on t"""
function make_measurement_matrix(t; n_measurements)
    m = 1:n_measurements
    A = g.(t', m) # equivalent to A[i, j] = g(t[j], m[i]) for all (i, j)
    # End points should be half as big to follow trapezoid rule
    # when multiplying with x
    A[:, begin] ./= 2
    A[:, end] ./= 2
    return A
end

"""
Makes A, x, and y where y is a (possibly noisy) version of A*x where

x[i] = f(t[i]) * Δt
"""
function make_problem(; t, f, σ=0, n_measurements)
    Δt = t.step |> Float64
    x = @. f(t) * Δt
    A = make_measurement_matrix(t; n_measurements)
    ϵ = randn(size(A, 1))
    y_clean = A*x
    y = y_clean + σ*ϵ/norm(ϵ)
    return A, x, y
end

### Problem Solving Functions ###

"""
    proj_scaled_simplex(y; S=1)

Projects (in Euclidian distance) the vector y into the scaled simplex:

    {y | y[i] ≥ 0 and sum(y) = sum_constraint}

[1] Yunmei Chen and Xiaojing Ye, "Projection Onto A Simplex", 2011
"""
function proj!(y; sum_constraint=1)
    n = length(y)

    y_sorted = sort(y) # Vectorize/extract input and sort all entries, will make a copy
    total = y_sorted[n]
    i = n - 1
    t = 0 # need to ensure t has scope outside the while loop
    while true
        t = (total - sum_constraint) / (n-i)
        y_i = y_sorted[i]
        if t >= y_i
            break
        else
            total += y_i
            i -= 1
        end

        if i >= 1
            continue
        else # i == 0
            t = (total - sum_constraint) / n
            break
        end
    end
    @. y = ReLU(y - t)
end

"""
Initializes x to a random point in the set
{x | ‖x‖₁ = sum_constraint, and x ≥ 0}.
Note this is not a *uniformly* random point on this set!
"""
function initialize_x(size; sum_constraint=1)
    x = rand(size) # uniform entries between [0, 1]. Ensures positive values unlike randn
    normalization = sum_constraint / sum(x)
    x .*= normalization
    return x
end

"""
    make_step_size(; A, Δt, λ, scale=1)

Calculates the inverse approximate smoothness of the loss function to use as the step size.

The exact smoothness is
opnorm(A'*A + (λ/Δt^3/scale_to_skip(scale))*laplacian_matrix(size(A, 2)))
which is upper bounded by
opnorm(A'*A) + (λ/Δt^3/scale_to_skip(scale)) * opnorm(laplacian_matrix(size(A, 2)))
with triangle inequality. We rewrite opnorm(A'*A) == opnorm(A*A'), because A*A' is a much smaller m×m matrix that is faster to compute and take its operator norm. And the Laplacian Matrix's opnorm is upper bounded by 4.
"""
make_step_size(; A, Δt, λ, scale=1) = 1 / (opnorm(Symmetric(A*A')) + 4λ/Δt^3/scale_to_skip(scale))

"""
    solve_problem(A, y;
        L, ∇L!, Δt, λ=λ, sum_constraint=1, scale=1, n=(size(A, 2) - 1) ÷ scale_to_skip(scale) + 1, x_init=initialize_x(n; sum_constraint), loss_tol=0.01, max_itr=8000, ignore_warnings=false)

Main algorithm for solving minₓ L(x) at a single scale.

# Plotting Keywords
- `show_plot`: Show any plotting of iterates
- `plot_first`: Plots the first `plot_first` number of iterates
- `plot_multiples`: Plots every `plot_multiple` number of iterates
- `plot_init`: Plot the initialization
- `plot_final`: Plot the final iterate
- `plot_number`: Start counting at `plot_number`. Note this variable is also used to set the alpha of each plot `alpha = plot_number/plot_final`.
- `p`: Handle to the plot
- `ground_truth`: If passed and plots are being shown, plot it
"""
function solve_problem(A, y; L, ∇L!, Δt, λ, sum_constraint=1, scale=1, n=(size(A, 2) - 1) ÷ scale_to_skip(scale) + 1, x_init=initialize_x(n; sum_constraint), loss_tol=0.01, max_itr=8000, ignore_warnings=false, show_plot=false, plot_first=7,ground_truth=nothing, plot_multiples=1,plot_init=true,plot_final=true, p=plot(;xlabel="\$t\$", ylabel="\$x\$",legend_columns=2,size=(450,250)), plot_number=1)#legendfontsize=12,

    @assert n == length(x_init)

    g = zeros(n) # gradient

    if scale != 1 # Avoid making a copy when scale == skip == 1
        skip = scale_to_skip(scale)
        A = A[:, begin:skip:end] * skip
    end

    α = make_step_size(; A, Δt, λ, scale)

    x = x_init # relabel
    i = 0 # iteration counter

    ∇L!(g, x; Δt, A, y, λ, scale) # obtain the first gradient, save it in g

    loss_per_itr = Float64[] # record the loss at each iteration
    push!(loss_per_itr, L(x; Δt, A, y, λ, scale))

    color = :blue
    if show_plot;
        alpha = plot_number / plot_first
        t = range(-1,1,length=n);
        # label="\$x_1^0\$"
        if !isnothing(ground_truth); plot!(p, t, ground_truth; label="\$x_1^*\$", color=:orange,linestyle=:dash,alpha=0.5,linewidth=5); end
        if plot_init;
        plot!(p, t, x_init; label="\$x_1^0\$", color, alpha);
        plot_number += 1
        end
    end

    plot_n_inner_iterations = plot_final ? plot_first - 1 : plot_first
    while loss_per_itr[i+1] > loss_tol

        # println(norm(x - proj!(x - α*g; sum_constraint)))

        @. x -= α * g
        proj!(x; sum_constraint)

        push!(loss_per_itr, L(x; Δt, A, y, λ, scale))

        i += 1

        if i == max_itr
            ignore_warnings || @warn "Reached maximum number of iterations $max_itr"
            break
        end

        if show_plot && plot_number <= plot_n_inner_iterations && i % plot_multiples == 0;
            alpha = plot_number / plot_first
            #label = plot_number == plot_first ? "\$x_1^k\$" : nothing
            plot!(p, t, x; label="\$x_1^{$(i)}\$", color, alpha);#label="\$x_1^{$(i)}\$"
            plot_number += 1
        end
        ∇L!(g, x; Δt, A, y, λ,scale) # next iteration's gradient (so it can be used in while loop condition)
    end

    if show_plot
        if plot_final;
            alpha = 1 # plot_number / plot_first
            #label = plot_number == plot_first ? "\$x_1^k\$" : nothing
            plot!(p, t, x; label="\$x_1^{$(i)}\$", color, alpha);#label="\$x_1^{$(i)}\$"
        end
        display(p);
    end

    return x, i, loss_per_itr
end

"""
    solve_problem_multiscale(A, y;
        L, ∇L!, Δt, λ, sum_constraint=1,  n_scales=points_to_scale(size(A, 2)),x_init=initialize_x(3; sum_constraint= sum_constraint / scale_to_skip(n_scales)), loss_tol=0.01, max_itr=8000, ignore_warnings=false, show_plot=false)

Main algorithm for solving minₓ L(x) at multiple scales.
"""
function solve_problem_multiscale(A, y; L, ∇L!, Δt, λ, sum_constraint=1,  n_scales=points_to_scale(size(A, 2)),x_init=initialize_x(3; sum_constraint= sum_constraint / scale_to_skip(n_scales)), loss_tol=0.01, max_itr=8000, ignore_warnings=false, show_plot=false, plot_first=7,ground_truth=nothing,plot_init=true,plot_final=true,p=plot(; xlabel="\$t\$", ylabel="\$x\$",legend_columns=2,size=(450,250)),plot_multiples=1,plot_coarse_scales=true)#legendfontsize=12,

    all_iterations = zeros(Int, n_scales)

    loss_per_itr_all_scales = Vector{Float64}[]

    time_per_scale = Float64[]
    time_per_interpolation = Float64[]

    plot_number = 1
    color = :blue
    if show_plot;
        if !isnothing(ground_truth);
            plot!(p, -1:Δt:1, ground_truth; label="\$x_1^*\$", color=:orange,linestyle=:dash,alpha=0.5,linewidth=5);
        end
        if plot_init;
            alpha = plot_number / plot_first
            t = range(-1,1,length=length(x_init));
            plot!(p, t, x_init; label="\$x_{$(n_scales)}^0\$", color, alpha);
            plot_number += 1
        end
    end
    # start_time = time()
    time = @elapsed x_S, i_S, loss = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(n_scales),
        x_init, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale=n_scales) # force one gradient step
    # end_time = time()
    push!(time_per_scale, time)

    push!(loss_per_itr_all_scales, loss)

    if show_plot && plot_coarse_scales;
        alpha = plot_number / plot_first
        t = range(-1,1;length=length(x_S))
        plot!(p, t, x_S; label="\$x_{$(n_scales)}^1\$", color, alpha);
        plot_number += 1
    end

    time = @elapsed x_s = interpolate(x_S)
    push!(time_per_interpolation,time)
    all_iterations[n_scales] = i_S

    # Middle scale solves
    for scale in (n_scales-1):-1:2 # Count down from larger to smaller scales
        # start_time = time()
        time = @elapsed x_s, i_s,loss = solve_problem(A, y; L, ∇L!, λ, sum_constraint = sum_constraint / scale_to_skip(scale),
            x_init=x_s, ignore_warnings=true, max_itr=1, loss_tol=0, Δt, scale) # force one gradient step

        # end_time = time()
        push!(time_per_scale, time)

        push!(loss_per_itr_all_scales, loss)

        if show_plot && plot_coarse_scales && plot_number <= plot_first;
            alpha = plot_number / plot_first
            t = range(-1,1,length=length(x_s));
            plot!(p, t, x_s; label="\$x_{$(scale)}^1\$", color, alpha);
            plot_number += 1
        end

        time = @elapsed x_s = interpolate(x_s)
        push!(time_per_interpolation,time)'

        all_iterations[scale] = i_s
    end

    # Finest scale solve
    # start_time = time()
    time = @elapsed x_1, i_1, loss = solve_problem(A, y; L, ∇L!, Δt, λ, sum_constraint, x_init=x_s, max_itr, loss_tol, scale=1, ignore_warnings, plot_init=false,plot_final,p,plot_first,plot_number,show_plot,plot_multiples)
    # end_time = time()
    push!(time_per_scale, time)

    push!(loss_per_itr_all_scales, loss)

    # if show_plot && plot_number <= plot_first;
    #     alpha = plot_number / plot_first
    #     t = range(-1,1,length=length(x_1));
    #     plot!(p, t, x_1; label="\$x_{1}^{$(i_1)}\$", color, alpha);
    #     plot_number += 1
    #     display(p)
    # end

    all_iterations[1] = i_1

    return x_1, all_iterations, loss_per_itr_all_scales, time_per_scale, time_per_interpolation
end

#######################
# Start of Benchmarks #
#######################
SEED = 3141592653
Random.seed!(SEED)

n_measurements = 5 # Number of Legendre polynomial measurements
n_scales = 10 # will run on about 2^10 points at the fine scale
λ = 1e-4 # Graph Laplacian regularization parameter
σ = 0.05 # Percent Gaussian noise in measurement y
percent_loss_tol = 0.05 # Iterate until the loss is within 5% of the ground truth

# Ground truth function
f(t) = -2.625t^4 - 1.35t^3 + 2.4t^2 + 1.35t + 0.225

# More problem setup
fine_scale_size = 2^n_scales + 1
t = range(-1, 1, length=fine_scale_size)
Δt = Float64(t.step)

A, x, y = make_problem(; t, f, σ, n_measurements)

"""Loss Function"""
L(x; Δt=Δt, λ=λ, A=A, y=y, scale=1) = 0.5 * norm2(A*x .- y) + λ .* GL(x; Δt, scale)

"""Gradient of Loss function L(x)"""
∇L(x; Δt, λ, A, y, scale=1) = A'*(A*x .- y) .+ λ .* ∇GL(x; Δt, scale)

"""Efficient in-place version of ∇L(x), storing the result in z."""
function ∇L!(z, x; Δt, λ, A, y, scale=1)
    ∇GL!(z, x; Δt, scale)
    mul!(z, A', A*x .- y, 1, λ) # mul!(C, A, B, α, β) == ABα+Cβ
    # This mul! function call is equivalent to
    # z .= A' * (A * x .- y) .* 1 .+ z .* λ
end

loss_tol = 0.00024 #L(x)*(1 + percent_loss_tol) # want xhat to be at least as good as our true x up to some percent

Random.seed!(SEED)
# Run it!!

# run once to compile
show_plot = false
time_single = @elapsed xhat, n_itr_single, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot, ground_truth=x)
time_multi = @elapsed xhat_multi, n_itr_multi, loss_per_itr_multi, time_per_scale, time_per_interpolation = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot, ground_truth=x,)

# Main timed run
show_plot = false
time_single = @elapsed xhat, n_itr_single, loss_per_itr_single = solve_problem(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot, ground_truth=x)
time_multi = @elapsed xhat_multi, n_itr_multi, loss_per_itr_multi, time_per_scale, time_per_interpolation = solve_problem_multiscale(A, y; L, ∇L!, loss_tol, Δt, λ, show_plot, ground_truth=x,)
#time_multi = sum(time_per_scale)
interpolation_time = sum(time_per_interpolation)
overhead_multi = time_multi - sum(time_per_scale) #- interpolation_time

function multi_time_and_loss(time_per_scale, overhead_multi, n_itr_multi)

    t = 0
    times  = Float64[]
    # losses = Float64[]
    overhead_per_scale = overhead_multi / (length(n_itr_multi)-1)

    push!(times, t)

    for (Δt_s, K_s) in zip(time_per_scale,reverse(n_itr_multi))
        Δt_s_k = Δt_s / K_s
        for k in 1:K_s
            t += Δt_s_k
            push!(times, t)
        end

        if K_s == n_itr_multi[1]
            break # skip the overhead on the final scale
        end

        t += overhead_per_scale
        push!(times, t)


    end

    return times
end
all_times_multi = multi_time_and_loss(time_per_scale, overhead_multi, n_itr_multi)

# p = plot(; xlabel="time (s)", ylabel="loss", yaxis=:log10)
# plot!(p, all_times_multi, vcat(loss_per_itr_multi...))
# display(p)
# all_times_single = range(0, time_single, length=(n_itr_single+1))
# plot!(p, all_times_single, loss_per_itr_single)

function save_and_render(p, file_name)
    svg_path = "../figure/"
    pdf_path = "../paper_files/mediabag/figure/"
    savefig(p, "$svg_path$file_name.svg")
    run(`inkscape $svg_path$file_name.svg --export-type=pdf --export-filename=$pdf_path$file_name.pdf`)
end

p = plot(; xlabel="time (ms)", ylabel="loss", yaxis=:log10,yticks=[exp10(n) for n in -3:0], size=(450,250),xlims=[0,1]) # xaxis=:log10,
all_times_single = range(0, time_single, length=(n_itr_single+1))
plot!(p, all_times_single*1e3, loss_per_itr_single; label="single", linewidth=2, linestyle=:dash)
plot!(p, all_times_multi*1e3, vcat(loss_per_itr_multi...); label="multi", linewidth=2, linestyle=:solid)

time_per_scale_overhead = [all_times_multi[i:i+1] for i in eachindex(all_times_multi)[2:2:(9*2)]]
loss_per_scale_overhead = [[loss_per_itr_multi[i][end], loss_per_itr_multi[i+1][begin]] for i in eachindex(loss_per_itr_multi)[begin:end-1]]

for (i,(t, l)) in enumerate(zip(time_per_scale_overhead, loss_per_scale_overhead))
    label = i==1 ? "overhead" : nothing
    plot!(p, t*1e3, l; label, alpha=0.25, color=:black, linewidth=8)
end
display(p)

save_and_render(p, "multi-vs-single-time-zoomed")

p = plot(; xlabel="time (ms)", ylabel="loss", yaxis=:log10,yticks=[exp10(n) for n in -3:0], size=(450,250), xlims=[0,20], ylims=[exp10(-3.5),ylims(p)[2]]) # ,
all_times_single = range(0, time_single, length=(n_itr_single+1))
plot!(p, all_times_single*1e3, loss_per_itr_single; label="single", linewidth=2, linestyle=:dash)
plot!(p, all_times_multi*1e3, vcat(loss_per_itr_multi...); label="multi", linewidth=2, linestyle=:solid)

display(p)

save_and_render(p, "multi-vs-single-time-full")
