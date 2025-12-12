# Test of multiscale

using Random
using Logging

using HDF5
using BenchmarkTools

using BlockTensorFactorization

Random.seed!(314)

filename = "./geodata.h5"

myfile = h5open(filename, "r")

# Y = myfile["Y_257_points"] |> read
Y = myfile["Y_1025_points"] |> read

close(myfile)

size_Y = size(Y)
I,J,K=size(Y)

Y_backup = copy(Y)

# # Add an extra slice so the continuous dimension is one plus a power of 2
# Y = cat(Y, zeros(20, 7, 1) ;dims=3)

scaleB_rescaleA! = ConstraintUpdate(0, l1scale_average12slices! ∘ nonnegative!;
    whats_rescaled=(x -> eachcol(factor(x, 1)))
)

scaleA_rows! = ConstraintUpdate(1, l1scale_rows! ∘ nonnegative!;
    whats_rescaled=nothing
)

nonnegativeA! = ConstraintUpdate(1, nonnegative!)

nonnegativeB! = ConstraintUpdate(0, nonnegative!)
# [l1scale_average12slices! ∘ nonnegative!, nonnegative!]
# [scaleB_rescaleA!, scaleA_rows!]
# [nonnegativeB!, scaleA_rows!]

function rand_scaled_tucker1((I,J,K),R)
    # I, J, K = size_Y
    # R = 3

    A = abs.(randn(I, R))
    B = abs.(randn(R, J, K))

    l1scale_rows!(A) # ensures rows sum to 1 since A is already nonnegative
    l1scale_12slices!(B) # simplex! can make the factors too sparse

    return Tucker1((B, A))
end
 # This means Y = A*B

X_init = rand_scaled_tucker1(size_Y,3)

@assert check(simplex_12slices!, X_init)

# X_init = Tucker1(size_Y, 3; init=abs_randn)

options = (
    rank=3,
    momentum=false,
    # model=Tucker1,
    tolerance=(0.12),
    converged=(RelativeError),
    do_subblock_updates=true,
    constrain_init=false,
    constraints=[nonnegative!, simplex_rows!],#[l1scale_average12slices! ∘ nonnegative!, nonnegative!],
    stats=[Iteration, ObjectiveValue, GradientNNCone, RelativeError],
    maxiter=200
)

#[l1scale_average12slices! ∘ nonnegative!,nonnegative!]

# first pass to compile
decomposition, stats_data, kwargs = factorize(Y; options...);

# X_init_multi = rand_scaled_tucker1((I,J,2),3)

decomposition, stats_data, kwargs = multiscale_factorize(Y;
    continuous_dims=3, options...);

# Second pass to time
@time decomposition, stats_data, kwargs = factorize(Y; options...);

multi_time = @elapsed decomposition, stats_data, kwargs = multiscale_factorize(Y; continuous_dims=3, options...);

iteration_time = sum(s[2] for s in stats_data)
overhead_time = multi_time - iteration_time
@show overhead_time
percent_overhead = overhead_time / multi_time * 100
@show percent_overhead

# Third pass to benchmark

logger = ConsoleLogger(stdout, Logging.Error)
global_logger(logger)

benchmark = BenchmarkGroup()

benchmark["single"] = @benchmarkable factorize(Y; options...) seconds=15 samples=20
benchmark["multi"] = @benchmarkable multiscale_factorize(Y; continuous_dims=3,options...) seconds=5 samples=20

benchmark_results = run(benchmark)
display(benchmark_results["multi"])
display(benchmark_results["single"])

logger = ConsoleLogger(stdout, Logging.Info)
global_logger(logger);
