import Pkg
Pkg.activate(".")
#Pkg.add(["Plots", "Random"]) #, "Printf"
#using Printf
using Random
Random.seed!(123456789)
using Plots
import Measures: mm

# Plot settings
Plots.resetfontsizes(); #Plots.scalefontsizes(1.5)
plotfont="Computer Modern"
#plotfontsize=13
default(;
    #dpi=300,
    size=(600,300),
    thickness_scaling = 1.5,
    xlabel="\$t\$",
    ylabel="\$x\$",
    linewidth=2,
    left_margin = -4mm,
    bottom_margin = -2mm,
    legendfont=plotfont,
    fontfamily=plotfont,
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

f(t) = -84t^4 + 146.4t^3 - 74.4t^2 + 12t
I = 9
t = range(0, 1, length=I)
x = f.(t)

t_fine = range(0,1,length=100)

## Random initialization ##
begin
p = plot()

plot!(t_fine,f.(t_fine);
  label="\$x=f(t)\$",
)

x_rand = 2 .* rand(I)

plot!(t[begin:2:end], x_rand[begin:2:end];
  label="\$x_S^0\$ random",
  marker=:circle,
  linestyle=:dash,
)

display(p)
end
saveplot("multiscale-scheme-random-initialization")

## Coarse Solution ##

p = plot()

plot!(t_fine,f.(t_fine);
  label="\$x=f(t)\$",
)

plot!(t[begin:2:end], x[begin:2:end];
  label="\$x_S^{K_S}\$ solution",
  marker=:circle,
  linestyle=:dash,
)

display(p)

saveplot("multiscale-scheme-coarse-solution")

## Interpolated Solution ##

p = plot()

x_bar = x[begin:2:end]

xnterp = interpolate(x_bar, 2; degree=1)

#xnterp = xnterp[begin:end-1]

plot!(t_fine,f.(t_fine);
  label="\$x=f(t)\$",
)

plot!(t, xnterp;
  label="\$\\underbar{x}_S^{K_S}={x}_{S-1}^{0}\$ interpolated",
  marker=:circle,
  linestyle=:dash,
)

display(p)

saveplot("multiscale-scheme-interpolated-solution")

## Fine Solution ##

p=plot()

plot!(t_fine,f.(t_fine);
  label="\$x=f(t)\$",
)

plot!(t, x;
  label="\${x}_{S-1}^{K_{S-1}}\$ solution",
  marker=:circle,
  linestyle=:dash,
)

display(p)

saveplot("multiscale-scheme-fine-solution")
