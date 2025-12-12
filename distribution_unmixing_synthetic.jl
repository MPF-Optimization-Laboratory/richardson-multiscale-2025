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
    tolerance=(0.05, 1e-6),
    converged=(MeanRelError, ObjectiveValue),
    do_subblock_updates=true,
    constrain_init=true,
    constraints=[nonnegative!, simplex_rows!], # constraint on B first since it's the core
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError, MeanRelError],
    mean_rel_error_tol = 1e-4,
    maxiter=50,
)

# First pass to compile
decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);

decomposition, stats_data, kwargs = factorize(Y; options...);

# Second pass to time
multi_time = @elapsed decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...);

iteration_time = sum(s[2] for s in stats_data)
overhead_time = multi_time - iteration_time
@show overhead_time
@show multi_time
percent_overhead = overhead_time / multi_time * 100
@show percent_overhead

single_time = @elapsed decomposition, stats_data, kwargs = factorize(Y; options...);

# Third pass to benchmark
# Hide warnings
logger = ConsoleLogger(stdout, Logging.Error)
global_logger(logger)

benchmark = BenchmarkGroup()

benchmark["multi"] = @benchmarkable multiscale_factorize(Y; continuous_dims=[2, 3, 4], options...) samples=20 seconds=20
benchmark["single"] = @benchmarkable factorize(Y; options...) samples=20 seconds=60

benchmark_result = run(benchmark)
display(benchmark_result["multi"])
display(benchmark_result["single"])

# unhide warnings
logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger);
