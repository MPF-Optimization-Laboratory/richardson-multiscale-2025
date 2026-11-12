using Plots

import Measures: mm

# Plot settings
Plots.resetfontsizes(); #Plots.scalefontsizes(1.5)
plotfont="Computer Modern"

default(;
    # dpi=300,
    size=(600,300),
    thickness_scaling = 1.5,
    linewidth=2,
    left_margin = -2mm,
    bottom_margin = 0mm,
    legendfont=plotfont,
    fontfamily=plotfont,
    right_margin=-7mm,
)

function save_and_render(p, file_name)
    svg_path = "../figure/"
    pdf_path = "../rendered/"
    savefig(p, "$svg_path$file_name.svg")
    run(`inkscape $svg_path$file_name.svg --export-type=pdf --export-filename=$pdf_path$file_name.pdf`)
end

cond_unreg = [
    3.063829787234043,
    4.38774761114803,
    1.1190101212676138e17,
    2.2839346438104026e18,
    1.167398435420243e18,
    4.783899661737982e18,
    5.471059802464222e19,
    1.1572874611989186e21,
    8.19335408222792e19,
    1.2214048317118949e20,
    1.1443651768747524e21,
    2.452976827145434e21
]

cond_reg = [
    3.0633518265495168,
    4.380991371725172,
    357.7366131846419,
    164.2106973471697,
    137.22989978479427,
    131.99194096071045,
    212.70548980422774,
    847.8259470356223,
    3386.5537553668883,
    13537.945157214821,
    54136.46594849752,
    216516.45704499327
]

scales = [
    12,
    11,
    10,
    9,
    8,
    7,
    6,
    5,
    4,
    3,
    2,
    1,
]
scale_to_points(s; S=maximum(scales)) = 2^(S - s + 1) + 1
number_of_points = scale_to_points.(scales)

condition_to_1_minus_rate(c) = 2 / (c + 1)


p = plot(;
    yaxis=:log10,
    xaxis=:log2,
    xticks=number_of_points .- 1,
    ylabel="condition number \$c\$",
    xlabel="problem size (# of points)",
    legend=:right,
    ylims=[1,4e21],
)
plot!(number_of_points, cond_unreg;
    label="\$\\lambda = 0\$ (unregularized)",
    color=4,
    linestyle=:dot,
    linewidth=2.5,
)

plot!(number_of_points, cond_reg;
    label="\$\\lambda = 10^{-4}\$ (regularized)",
    color=3,
    linestyle=:solid,
)

# more stable than (c - 1) / (c + 1) for c close to 1
condition_to_rate(c) = c / (c + 1) - 1 / (c + 1)

tick_nums = 0:5:20

# convert the 1 - 1e-i in log10-space with the range [1, 4e21]
# to the unit interval [0, 1] in plain space
power_to_unit_interval(i) = log10(2exp10(i)-1)/log10(4e21)

plot!(twinx();
 yaxis = ("rate \$q\$"),
 yticks = ([power_to_unit_interval(i) for i in tick_nums],["\$0\$", ("\$1 - 10^{-$i}\$" for i in tick_nums[begin+1:end])...]),
 ylims=[0,condition_to_rate(4e21)],
 )
display(p)


save_and_render(p, "conditioning_vs_problem_size")

###########################################################

show_extra_plots = false

if show_extra_plots

begin
p = plot(;
    yaxis=:log10,
    xaxis=:log2,
    xticks=number_of_points .- 1,
    ylabel="condition number \$c\$",
    xlabel="problem size (# of points)",
    legend=:right,
    xlims=[2^2+1,2^2+1],
    ylims=[1,10^7]
)
plot!(number_of_points, cond_unreg;
    label="\$\\lambda = 0\$ (unregularized)",
    color=4,
    linestyle=:dash,
)

plot!(number_of_points, cond_reg;
    label="\$\\lambda = 10^{-4}\$ (regularized)",
    color=3,
    linestyle=:solid
)
display(p)
end

condition_to_rate(c) = c / (c + 1) - 1 / (c + 1)

condition_to_1_minus_rate(c) = 2 / (c + 1)

rate_unreg = condition_to_rate.(cond_unreg)
rate_reg = condition_to_rate.(cond_reg)

one_minus_rate_unreg = condition_to_1_minus_rate.(cond_unreg)
one_minus_rate_reg = condition_to_1_minus_rate.(cond_reg)

begin
p = plot(;
    # yaxis=:log10,
    xaxis=:log2,
    xticks=number_of_points .- 1,
    ylabel="rate \$q\$",
    xlabel="problem size (# of points)",
    legend=:right
)
plot!(number_of_points, rate_unreg;
    label="\$\\lambda = 0\$ (unregularized)",
    color=4,
    linestyle=:dash,
)

plot!(number_of_points, rate_reg;
    label="\$\\lambda = 10^{-4}\$ (regularized)",
    color=3,
    linestyle=:solid
)
display(p)
end

begin
p = plot(;
    yaxis=:log10,
    xaxis=:log2,
    xticks=number_of_points .- 1,
    # yticks=([1e-5],["\$10^{-5}\$"]),
    ylabel="\$1 - q\$",
    xlabel="problem size (# of points)",
    legend=:right
)

plot!(number_of_points, one_minus_rate_reg;
    label="\$\\lambda = 10^{-4}\$ (regularized)",
    color=3,
    linestyle=:solid
)
plot!(number_of_points, one_minus_rate_unreg;
    label="\$\\lambda = 0\$ (unregularized)",
    color=4,
    linestyle=:dash,
)

display(p)
end

begin
tick_nums = 0:5:20
p = plot(;
    yaxis=:log10,
    xaxis=:log2,
    xticks=number_of_points .- 1,
    yticks=([exp10(-i) for i in tick_nums],["\$1 - 10^{-$i}\$" for i in tick_nums]),
    ylabel="\$q\$",
    xlabel="problem size (# of points)",
    legend=:right,
    yflip = true,
)

plot!(number_of_points, one_minus_rate_reg;
    label="\$\\lambda = 10^{-4}\$ (regularized)",
    color=3,
    linestyle=:solid
)
plot!(number_of_points, one_minus_rate_unreg;
    label="\$\\lambda = 0\$ (unregularized)",
    color=4,
    linestyle=:dash,
)

display(p)
end
end