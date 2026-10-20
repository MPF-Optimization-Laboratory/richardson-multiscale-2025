using Plots
using Contour

###########

import Measures: mm

# Plot settings
Plots.resetfontsizes(); #Plots.scalefontsizes(1.5)
plotfont="Computer Modern"

default(;
    #dpi=300,
    size=(600,300),
    thickness_scaling = 1.5,
    xlabel="rate \$q\$",
    ylabel="# of scales \$S\$",
    linewidth=2,
    left_margin = -2mm,
    bottom_margin = -2mm,
    legendfont=plotfont,
    fontfamily=plotfont,
)

function save_and_render(p, file_name)
    svg_path = "../figure/"
    pdf_path = "../rendered/"
    savefig(p, "$svg_path$file_name.svg")
    # run(`inkscape $svg_path$file_name.svg --export-type=pdf --export-filename=$pdf_path$file_name.pdf`)
end

###########
n = 100

q = range(0,  1, length=n)
S = range(1, 16, length=n)

σ(q, S) = q == 0.5 ? S - 1 : (2q - (2q)^S)/(1 - 2q)

f(q, S) = begin
    2^(S+2) * (q^3 * sqrt(0.5 + 2^(-S)) - q^S) / σ(q, S)
end

z = [f(q,S) for q in q, S in S]

linestyles = [:dash, :solid, :dot] #[:dashdotdot,  :dashdot, :solid, :dot, :dash]
Ls         = [ 10,     1,       .01]

begin
p = plot(legend=:outerright, legendtitle="\$L_{f^*}\$", 
        yticks=0:2:16, ylims=[0,16], xticks=0:0.2:1, xlims=[0,1])
for (cl, linestyle) in zip(levels(contours(q,S,z, Ls)), linestyles)
    lvl = level(cl) # the z-value of this contour level
    line = first(lines(cl))
    xs, ys = coordinates(line) # coordinates of this line segment
    plot!(xs, ys;
        linestyle,
        label="$(lvl)",
    )
end
annotate!([0.4, 0.8], [12, 2], ["\$S > S_{GM}\$", "\$S < S_{GM}\$"])
display(p)
end

save_and_render(p, "greedy_sufficient_conditions")


K_min(q) = log(sqrt(2) - 1)/log(q)

g(q,k,L=1) = begin
    log(q^(-(k+5)/4) / (1 + q^k) * (2 + 0.5L / (sqrt(2)*(1 + q^k) - 1))) / log(sqrt(2) / (1 + q^k))
end

n = 1000

q = range(0.01,  0.993, length=n)
k_max=100
K = range(1, k_max, length=n)

z = [g(q,K,1) for q in q, K in K]

linestyles = [:solid, :dot, :dash]
Ss         = [ 10,      15,    20]

begin
p = plot(legend=:outerright, legendtitle="\$S\$",
        yticks=([1,10,100],["1","10","100"]),
        # ylabe
         ylims=[1,k_max], xticks=0:0.2:1, xlims=[0,1],
         yaxis=:log10,
         ylabel="iterations \$K'\$",
         )
for (i, (cl, linestyle)) in enumerate(zip(levels(contours(q,K,z, Ss)), linestyles))
    i = i+3
    lvl = level(cl) # the z-value of this contour level
    haslabel = false
    for line in lines(cl)
        xs, ys = coordinates(line) # coordinates of this line segment
        
        # Skip contour lines under K_min
        any(((x,y),) -> y < K_min(x), zip(xs,ys)) ? continue : nothing

        if !haslabel
            plot!(xs, ys;
                linestyle,
                color=i,
                label="$(lvl)"
            )
            haslabel = true
        else
            plot!(xs, ys;
                linestyle,
                color=i,
                label=nothing
            )
        end
    end
end
annotate!([0.25, 0.875], [50, 2],  "\$S \\leq S_{LM}\$")
annotate!([0.7], [9.5], text("\$S \\geq S_{LM}\$", rotation = 20))
display(p)
end

save_and_render(p, "lazy_sufficient_conditions")