import Pkg
Pkg.activate(".")
#Pkg.add(["Plots", "Random"]) #, "Printf"
#using Printf
using Random
Random.seed!(123456789)
using Plots

# Plot settings
Plots.resetfontsizes(); #Plots.scalefontsizes(1.5)
plotfont="Computer Modern"
#plotfontsize=13
default(;
    #dpi=300,
    thickness_scaling = 1.5,
    size=(900,350),
    xlabel="\$t\$",
    ylabel="\$x\$",
    linewidth=2,
    markerstrokecolor=:white,
    legendfont=plotfont,
    fontfamily=plotfont,
    #legend=:outertopright,
    legend=true,
    legendfontsize=12,
    grid=true,
    xticks=true,
    yticks=-6:2:2,
    ylims=(-7,3.5),
    foreground_color_border=:black,
    foreground_color_axis=:black,
    tick_direction=:up,
    showaxis=true,
)

# Export settings
foldername = "./figure/"
extension = ".svg"
filepath(filename) = foldername * filename * extension
saveplot(filename) = savefig(filepath(filename))


function interpolate(Y::AbstractArray, scale; dims=1:ndims(Y), degree=0, kwargs...)
    # Quick exit if no interpolation is needed
    if scale == 1 || isempty(dims)
        return Y
    end

    Y = repeat(Y; inner=(d in dims ? scale : 1 for d in 1:ndims(Y)))

    # Chop the last slice of repeated dimensions since we only interpolate between
    # the values
    chop = (d in dims ? axis[begin:end-scale+1] : axis for (d, axis) in enumerate(axes(Y)))
    Y = Y[chop...]

    if degree == 0
        return Y
    elseif degree == 1 && scale == 2 # TODO generalize linear_smooth to other scales
        return linear_smooth(Y; dims, kwargs...)
    else
        error("interpolation of degree=$degree with scale=$scale not supported (YET!)")
    end
end

function linear_smooth(Y; dims=1:ndims(Y), kwargs...)
    return _linear_smooth!(1.0 * Y, dims)
    # makes a copy of Y and ensures the type can hold float like elements
end

function _linear_smooth!(Y, dims)
    all_dims = 1:ndims(Y)
    for d in dims
        axis = axes(Y, d)
        Y1 = @view Y[(i==d ? axis[begin+1:end-1] : (:) for i in all_dims)...]
        Y2 = @view Y[(i==d ? axis[begin+2:end] : (:) for i in all_dims)...]

        @. Y1 = 0.5 * (Y1 + Y2)
    end
    return Y
end

# Function definition #

#f(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
f(t) = 72t^3-80t^2+10t+1
I = 5
a, b = (0, 1)
t_fine = range(a, b, length=I)
t_coarse = t_fine[begin:2:end]
x_star_fine = f.(t_fine)
x_star_coarse = f.(t_coarse)

x_coarse = copy(x_star_coarse)
x_fine = interpolate(x_coarse, 2; degree=1)

t_cts = range(0,1,length=100)

## Exact Interpolation ##
blue = palette(:tab10)[1]
orange = palette(:tab10)[2]
green = :darkgreen #palette(:tab10)[3]

p = plot()

plot!(t_cts,f.(t_cts);
  label="\$f(t)\$",
  color=blue,
)


plot!(t_fine, x_fine;
  label="\$x_s\$",
  marker=:square,
  linestyle=:dot,
  color=green,
  markersize=8,
)

plot!(t_fine, x_star_fine;
  label="\$x^{\\!*}_s\$",
  marker=:circle,
  linestyle=:dash,
  color=orange,
  markersize=6,
)


display(p)

saveplot("interpolation-lemma-exact")

## Inexact Interpolation ##
δ = [-1, 1, -2]
x_coarse_with_error = x_coarse + δ

x_fine_with_error = interpolate(x_coarse_with_error, 2; degree=1)

p = plot()

plot!(t_cts,f.(t_cts);
  label="\$f(t)\$",
  color=blue
)


plot!(t_fine, x_fine_with_error;
  label="\$x_s\$",
  marker=:square,
  linestyle=:dot,
  color=green,
  markersize=8,
)

plot!(t_fine, x_star_fine;
  label="\$x_s^{\\!*}\$",
  marker=:circle,
  linestyle=:dash,
  color=orange,
  markersize=6,
)



display(p)

saveplot("interpolation-lemma-approx")

# Lipschitz Function Interpolation

f(t) = t - cos(3π*t)
f_prime(t) = 1 + 3π*sin(3π*t)
L = abs(f_prime(1/6)) # maximum on [0,1]
s = 0.5(a + b + (f(b) - f(a))/L)
u = 0.5(a + b - (f(b) - f(a))/L)
# lipschitz_line_a_top(t) = f(a) + L*(t-a)
# lipschitz_line_a_bot(t) = f(a) - L*(t-a)
# lipschitz_line_b_top(t) = f(b) - L*(t-b)
# lipschitz_line_b_bot(t) = f(b) + L*(t-b)

lipschitz_line_a_top = [f(a), f(a) + L*(s-a)]
lipschitz_line_a_bot = [f(a), f(a) - L*(u-a)]
lipschitz_line_b_top = [f(b), f(b) - L*(s-b)]
lipschitz_line_b_bot = [f(b), f(b) + L*(u-b)]

lipschitz_lines = [lipschitz_line_a_top, lipschitz_line_a_bot, lipschitz_line_b_top, lipschitz_line_b_bot]

lipschitz_domains = [[a,s],[a,u],[b,s],[b,u]]

tail(x) = x[begin+1:end]

p = plot(;
  legend=:outertopright,
  xlabel="\$\\lambda_1\$",
  legendfontsize=10,
)

lower, upper = (-6,6)

ylims!((lower, upper))
yticks!(lower:2:upper)

plot!(first(lipschitz_domains),first(lipschitz_lines);
  label="bounding\nparallelogram",
  color=:gray,
  linestyle=:dash,
)

plot!(tail(lipschitz_domains),tail(lipschitz_lines);
  label=false,
  color=:gray,
  linestyle=:dash,
)

plot!([a, b], [f(a), f(b)];
  label="\$ \\lambda_1  f(a) + \\lambda_2 f(b)\$",
  color=orange,
  linestyle=:dashdotdot,
  # marker=true
)

plot!(t_cts,f.(t_cts);
  label="\$f(\\lambda_1 a + \\lambda_2 b)\$",
  color=blue
)

scatter!([a, b], [f(a), f(b)];
  color=blue,
  markerstrokewidth=2,
  label=false
)

display(p)

saveplot("lipschitz-function-interpolation-typical")


# worst case

f(t)=t<0.5 ? t : 1-t
L = 1 # maximum on [0,1]
s = 0.5(a + b + (f(b) - f(a))/L)
u = 0.5(a + b - (f(b) - f(a))/L)
# lipschitz_line_a_top(t) = f(a) + L*(t-a)
# lipschitz_line_a_bot(t) = f(a) - L*(t-a)
# lipschitz_line_b_top(t) = f(b) - L*(t-b)
# lipschitz_line_b_bot(t) = f(b) + L*(t-b)

lipschitz_line_a_top = [f(a), f(a) + L*(s-a)]
lipschitz_line_a_bot = [f(a), f(a) - L*(u-a)]
lipschitz_line_b_top = [f(b), f(b) - L*(s-b)]
lipschitz_line_b_bot = [f(b), f(b) + L*(u-b)]

lipschitz_lines = [lipschitz_line_a_top, lipschitz_line_a_bot, lipschitz_line_b_top, lipschitz_line_b_bot]

lipschitz_domains = [[a,s],[a,u],[b,s],[b,u]]

tail(x) = x[begin+1:end]

p = plot(;
  legend=:outertopright,
  xlabel="\$\\lambda\$",
  legendfontsize=10,
)

lower, upper = (-0.6,0.6)

ylims!((lower, upper))
yticks!(lower:0.2:upper)

plot!(first(lipschitz_domains),first(lipschitz_lines);
  label="bounding\nparallelogram",
  color=:gray,
  linestyle=:dash,
)

plot!(tail(lipschitz_domains),tail(lipschitz_lines);
  label=false,
  color=:gray,
  linestyle=:dash,
)

plot!([a, b], [f(a), f(b)];
  label="\$\\lambda f(a)+ (1- \\lambda) f(b)\$",
  color=orange,
  marker=true
)

plot!(t_cts,f.(t_cts);
  label="\$f(\\lambda a+ (1- \\lambda) b)\$",
  color=blue
)

display(p)

saveplot("lipschitz-function-interpolation-worst-case")
