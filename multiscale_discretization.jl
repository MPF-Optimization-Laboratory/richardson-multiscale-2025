using Plots

Plots.resetfontsizes(); #Plots.scalefontsizes(1.5)
plotfont="Computer Modern"
#plotfontsize=13
default(;
    #dpi=300,
    #default size=(600, 400),
    size=(900,350),
    thickness_scaling = 1.5,
    xlabel="domain \$t\$       ",
    ylabel="\$\\longleftarrow\$scale \$s\$",
    linewidth=2,
    legendfont=plotfont,
    fontfamily=plotfont,
    #legend=:outertopright,
    legend=nothing,
    grid=nothing,
    xticks=false,
    foreground_color_border=:white,
    foreground_color_axis=:white,
    tick_direction=:none,
    showaxis=true,
)

# Export settings
save = true
foldername = "./figure/"
extension = ".svg"
filepath(filename) = foldername * filename * extension
saveplot(filename) = savefig(filepath(filename))

# Setup
maximum_scale = 3
scales = 1:maximum_scale
n_points(s) = 2^(maximum_scale-s+1) + 1
t_values(s) = range(0, 1, length=n_points(s))
markers = [:circle, :pentagon, :rect, :utriangle]

# Plotting
p = plot()
for scale in scales
    t = t_values(scale)
    plot!(t, scale*ones(n_points(scale)); label="", color=:black)
    y=scale*ones(n_points(scale))
    if scale < maximum_scale
        even_t = t[begin:2:end]
        even_y = y[begin:2:end]
        odd_t = t[begin+1:2:end]
        odd_y = y[begin+1:2:end]
        scatter!(even_t, even_y; marker=:circle, markersize=7, label="s=$(scale)", markerstrokecolor=:white, color="darkcyan")
        scatter!(odd_t, odd_y; marker=:rect, markersize=7, label="s=$(scale)", markerstrokecolor=:white, color="darkmagenta")
    else
        scatter!(t, y; marker=:rect, markersize=7, label="s=$(scale)", markerstrokecolor=:white, color="darkmagenta")
    end
end

ylim_spacing = 0.4
ylims!((1-ylim_spacing,maximum_scale+ylim_spacing))
yticks!(1:maximum_scale)

xlims!((-0.02,1.1))

spacing = 0.15
above(x) = x + spacing
below(x) = x - spacing

variable = "x"

s = 3

annotate!(0.00, above(s), Plots.text("\$$(variable)_$(s)[1]\$", :bottom, 10))
annotate!(0.50, above(s), Plots.text("\$$(variable)_$(s)[2]\$", :bottom, 10))
annotate!(1.00, above(s), Plots.text("\$$(variable)_$(s)[3]\$", :bottom, 10))
annotate!(0.00, below(s), Plots.text("\$$(variable)_{($(s))}[1]\$", :top, 10))
annotate!(0.50, below(s), Plots.text("\$$(variable)_{($(s))}[2]\$", :top, 10))
annotate!(1.00, below(s), Plots.text("\$$(variable)_{($(s))}[3]\$", :top, 10))

s = 2

annotate!(0.00, above(s), Plots.text("\$$(variable)_$(s)[1]\$", :bottom, 10))
annotate!(0.25, above(s), Plots.text("\$$(variable)_$(s)[2]\$", :bottom, 10))
annotate!(0.50, above(s), Plots.text("\$$(variable)_$(s)[3]\$", :bottom, 10))
annotate!(0.75, above(s), Plots.text("\$$(variable)_$(s)[4]\$", :bottom, 10))
annotate!(1.00, above(s), Plots.text("\$$(variable)_$(s)[5]\$", :bottom, 10))
annotate!(0.25, below(s), Plots.text("\$$(variable)_{($(s))}[1]\$", :top, 10))
annotate!(0.75, below(s), Plots.text("\$$(variable)_{($(s))}[2]\$", :top, 10))

s = 1

annotate!(0.000, above(s), Plots.text("\$$(variable)_$(s)[1]\$", :bottom, 10))
annotate!(0.125, above(s), Plots.text("\$$(variable)_$(s)[2]\$", :bottom, 10))
annotate!(0.250, above(s), Plots.text("\$$(variable)_$(s)[3]\$", :bottom, 10))
annotate!(0.375, above(s), Plots.text("\$$(variable)_$(s)[4]\$", :bottom, 10))
annotate!(0.500, above(s), Plots.text("\$$(variable)_$(s)[5]\$", :bottom, 10))
annotate!(0.625, above(s), Plots.text("\$$(variable)_$(s)[6]\$", :bottom, 10))
annotate!(0.750, above(s), Plots.text("\$$(variable)_$(s)[7]\$", :bottom, 10))
annotate!(0.875, above(s), Plots.text("\$$(variable)_$(s)[8]\$", :bottom, 10))
annotate!(1.000, above(s), Plots.text("\$$(variable)_$(s)[9]\$", :bottom, 10))
annotate!(0.125, below(s), Plots.text("\$$(variable)_{($(s))}[1]\$", :top, 10))
annotate!(0.375, below(s), Plots.text("\$$(variable)_{($(s))}[2]\$", :top, 10))
annotate!(0.625, below(s), Plots.text("\$$(variable)_{($(s))}[3]\$", :top, 10))
annotate!(0.875, below(s), Plots.text("\$$(variable)_{($(s))}[4]\$", :top, 10))

# annotate!(1.05, 2.5, Plots.text("\$\\bar{$(variable)}\$", :bottom, 10))
annotate!(1.06, 2.6, Plots.text("\$\\bar{$(variable)} \\longrightarrow\$", :center, 10, rotation = 90))
annotate!(1.1, 2.6, Plots.text("coarsen", plotfont, :center, 10, rotation = 90))
annotate!(1.06, 1.5, Plots.text("\$\\longleftarrow \\underbar{$(variable)}\$", :center, 10, rotation = 90))
annotate!(1.1, 1.5, Plots.text("interpolate", plotfont, :center, 10, rotation = 90))

display(p)

# Saving
if save
    saveplot("discretization-scales")
end
