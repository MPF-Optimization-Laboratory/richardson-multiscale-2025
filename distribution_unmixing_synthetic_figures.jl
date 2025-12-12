# generate synthetic data for KDE and tensor factorization

using Random
using Logging
using Plots

using BenchmarkTools
using KernelDensity
using Distributions

using BlockTensorFactorization

J = 65 # Number of samples in the x dimension
K = 65 # Number of samples in the y dimension
L = 65 # Number of samples in the z dimension

Random.seed!(314159)

# Three sources, product distributions
R = 3
source1a = Normal(4, 1)
source1b = Uniform(-7, 2)
source1c = Uniform(-1, 1)

source2a = Normal(0, 3)
source2b = Uniform(-2, 2)
source2c = Exponential(2)

source3a = Exponential(1)
source3b = Normal(0, 1)
source3c = Normal(0, 3)

source1 = product_distribution([source1a, source1b, source1c])
source2 = product_distribution([source2a, source2b, source2c])
source3 = product_distribution([source3a, source3b, source3c])

sources = (source1, source2, source3)

# x values to sample the densities at
x = range(-10,10,length=J)
Δx = x[2] - x[1]

# y values to sample the densities at
y = range(-10,10,length=K)
Δy = y[2] - y[1]

# z values to sample the densities at
z = range(-10,10,length=K)
Δz = z[2] - z[1]

# Collect sample points into a 3rd order array
xyz = [[x,y,z] for (j, x) in enumerate(x), (k, y) in enumerate(y), (l, z) in enumerate(z)]

source1_density = pdf.((source1,), xyz)
source2_density = pdf.((source2,), xyz)
source3_density = pdf.((source3,), xyz);

# Generate mixing matrix
p1 = [0, 0.4, 0.6]
p2 = [0.3, 0.3, 0.4]
p3 = [0.8, 0.2, 0]
p4 = [0.2, 0.7, 0.1]
p5 = [0.6, 0.1, 0.3]

A_true = hcat(p1,p2,p3,p4,p5)'

# Generate mixing distributions
distribution1 = MixtureModel([sources...], p1)
distribution2 = MixtureModel([sources...], p2)
distribution3 = MixtureModel([sources...], p3)
distribution4 = MixtureModel([sources...], p4)
distribution5 = MixtureModel([sources...], p5)

distributions = [distribution1, distribution2, distribution3, distribution4, distribution5]
I = length(distributions)

# Collect into a tensor that is size(Y) == (Sinks x Features x Samples)
sinks = [pdf.((d,), xyz) for d in distributions]
Y = cat(sinks...; dims=4)
Y = permutedims(Y, (4,1,2,3))
Y .*= Δx * Δy * Δz # Scale factors
Y_slices = eachslice(Y, dims=1)
correction = sum.(Y_slices) # should all be 1, but might be something like 0.9783... from truncating and discretizing pdfs
Y_slices ./= correction

# Main convergence criteria is Mean Relative Error
struct MeanRelError <: AbstractStat
    mean_rel_error_tol::Float64
end

MeanRelError(; mean_rel_error_tol, kwargs...) = MeanRelError(mean_rel_error_tol)

function (S::MeanRelError)(X, Y, _, _, _)
    non_zero_indexes = Y .> S.mean_rel_error_tol
    Y_non_zero = @view Y[non_zero_indexes]
    X_non_zero = @view X[non_zero_indexes]
    return mean(@. abs(X_non_zero - Y_non_zero) / Y_non_zero)
end

options = (
    rank=3,
    momentum=false,
    model=Tucker1,
    tolerance=(1e-6),
    converged=(ObjectiveValue),
    do_subblock_updates=true,
    constrain_init=true,
    constraints=[nonnegative!, simplex_rows!], # constraint on B first since it's the core
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, MeanRelError],
    mean_rel_error_tol = 1e-4,
    maxiter=50,
)

"""Force one iteration at the coarser scales"""
function multi_factorize_fixed_itr(Y, n_coarse_itr; kwargs...)
    continuous_dims = [2, 3, 4]
    decomposition, stats_data, _ = multiscale_factorize(Y; kwargs..., continuous_dims, scales=[2^(6-s) for s in 1:5], maxiter=n_coarse_itr, tolerance=0); #force n_coarse_itr
    decomposition = interpolate(decomposition, 2;dims=continuous_dims)
    @info "Factorizing at scale 1..."
    final_time = @elapsed decomposition, stats_data_final, kwargs = factorize(Y; kwargs..., decomposition)
    push!(stats_data, (stats_data_final, final_time))
    return decomposition, stats_data, kwargs
end

# Compile pass
decomposition, stats_data, kwargs = @time multi_factorize_fixed_itr(Y, 40; options...);

# Hide warnings
logger = ConsoleLogger(stdout, Logging.Error)
global_logger(logger)

number_of_coarse_iterations = [1:10...,12:2:20...,30,40]

# Manually execute the benchmark because we want to know how many fine scale iterations there were
n_trials = 20
all_fine_iterations = Vector{Int}[]
all_times_multi = Vector{Float64}[] # seconds

for K_S in number_of_coarse_iterations
    @show K_S
    n_fine_iterations_K_S = Int[]
    trial_times = Float64[]
    for trial_number in 1:n_trials
        @show trial_number
        trial_time = @elapsed _, stats_data, _ = multi_factorize_fixed_itr(Y, K_S; options...);
        n_fine_iterations = stats_data[end][1][end,:Iteration]
        push!(n_fine_iterations_K_S, n_fine_iterations)
        push!(trial_times, trial_time)
    end
    push!(all_fine_iterations, n_fine_iterations_K_S)
    push!(all_times_multi, trial_times)
end

n_iterations_single = Int[]
all_times_single = Float64[]  # seconds
for trial_number in 1:n_trials
    @show trial_number
    trial_time = @elapsed _, stats_data, _ = factorize(Y; options...)
    n_fine_iterations = stats_data[end,:Iteration]
    push!(n_iterations_single, n_fine_iterations)
    push!(times_single, trial_time)
end
# unhide warnings
logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger);

median_times_multi = map(median, all_times_multi)
top_quantile = 0.75
bot_quantile = 0.25

top_times_multi = map(x -> quantile(x, top_quantile), all_times_multi)
bot_times_multi = map(x -> quantile(x, bot_quantile), all_times_multi)

median_time_single = median(all_times_single)
top_time_single = quantile(all_times_single, top_quantile)
bot_time_single = quantile(all_times_single, bot_quantile)

p = plot(number_of_coarse_iterations, median_times_multi; xlabel="coarse scale iterations \$K_s\$", ylabel="time (s)",label="multi", ribbon=(median_times_multi - bot_times_multi, top_times_multi - median_times_multi), fillalpha=0.2, size=(450,250))
plot!([0,1], [median_time_single, median_times_multi[1]],
 ribbon=([median_time_single, median_times_multi[1]] - [bot_time_single, bot_times_multi[1]], [top_time_single,top_times_multi[1]] - [median_time_single, median_times_multi[1]]), fillalpha=0.2,label="single")


function save_and_render(p, file_name)
    svg_path = "../figure/"
    pdf_path = "../paper_files/mediabag/figure/"
    savefig(p, "$svg_path$file_name.svg")
    run(`inkscape $svg_path$file_name.svg --export-type=pdf --export-filename=$pdf_path$file_name.pdf`)
end

save_and_render(p, "time-vs-number-of-coarse-iterations")

# top_quantile = 0.90
# bot_quantile = 0.10

# n_iterations_multi = [all_fine_iterations[K_S] for K_S in number_of_coarse_iterations]

median_iterations_multi = map(median, n_iterations_multi)
top_iterations_multi = map(x -> quantile(x, top_quantile), n_iterations_multi)
bot_iterations_multi = map(x -> quantile(x, bot_quantile), n_iterations_multi)

median_iterations_single = median(n_iterations_single)
top_iterations_single = quantile(n_iterations_single, top_quantile)
bot_iterations_single = quantile(n_iterations_single, bot_quantile)

p = plot(number_of_coarse_iterations, median_iterations_multi; xlabel="coarse scale iterations \$K_s\$", ylabel="fine scale iterations \$K_1\$",label="multi", ribbon=(median_iterations_multi - bot_iterations_multi, top_iterations_multi - median_iterations_multi), fillalpha=0.2, size=(450,250))
plot!([0,1], [median_iterations_single, median_iterations_multi[1]],
 ribbon=([median_iterations_single, median_iterations_multi[1]] - [bot_iterations_single, bot_iterations_multi[1]], [top_iterations_single,top_iterations_multi[1]] - [median_iterations_single, median_iterations_multi[1]]), fillalpha=0.2,label="single")

save_and_render(p, "number-of-fine-iterations-vs-number-of-coarse-iterations")
