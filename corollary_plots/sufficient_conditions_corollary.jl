using Plots
using Contour

export_figs = true

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
Ls         = [ 10,     1,       .1]

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

export_figs && save_and_render(p, "greedy_sufficient_conditions")


#######################################

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

export_figs && save_and_render(p, "lazy_sufficient_conditions")

#################################

K_min(q) = log(sqrt(2) - 1)/log(q)

h(q,k,S) = begin
    (sqrt(2^S) / (q^(-(k+5)/4)*(1+q^k)^(S-1)) - 1 )*((1+q^k)*sqrt(2) - 1)
end

n = 1000

q = range(0.01,  0.993, length=n)
k_max=100
K = range(1, k_max, length=n)

z = [h(q,K,15) for q in q, K in K] # S = 10

linestyles = [:dash, :solid, :dot]
Ls         = [ 10,      1,    0.01]

begin
p = plot(legend=:outerright, legendtitle="\$L_{f^*}\$",
        yticks=([1,10,100],["1","10","100"]),
        # ylabe
         ylims=[1,k_max], xticks=0:0.2:1, xlims=[0,1],
         yaxis=:log10,
         ylabel="iterations \$K'\$",
         )
for (i, (cl, linestyle)) in enumerate(zip(levels(contours(q,K,z, Ls)), linestyles))
    i = i
    lvl = level(cl) # the z-value of this contour level
    haslabel = false
    for line in lines(cl)
        xs, ys = coordinates(line) # coordinates of this line segment
        
        # Skip contour lines under K_min
        # any(((x,y),) -> y < K_min(x), zip(xs,ys)) ? continue : nothing

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
annotate!([0.25, 0.875], [50, 2],  "\$10 \\leq S_{LM}\$")
annotate!([0.78], [16], text("\$10 \\geq S_{LM}\$", rotation = 38))
display(p)
end

export_figs && save_and_render(p, "lazy_sufficient_conditions_fixed_scale")


#################################
begin
K_min(q) = log(sqrt(2) - 1)/log(q)

h(q,k,S) = begin
    (sqrt(2^S) / (q^(-(k+5)/4)*(1+q^k)^(S-1)) - 1 )*((1+q^k)*sqrt(2) - 1)
end

n = 100

q = range(0.2,  0.98, length=n)
S_max = 32
S = range(1.6, S_max, length=n)

z = [h(q,25,S) for q in q, S in S] # K = 25

linestyles = [:dash, :solid, :dot]
Ls         = [ 10,      1,    0.1]

begin
p = plot(legend=:outerright, legendtitle="\$L_{f^*}\$",
        # yticks=([1,10,100],["1","10","100"]),
        # ylabe
         ylims=[0,S_max], xticks=0:0.2:1, xlims=[0,1],
        #  yaxis=:log10,
         ylabel="# of scales \$S\$",
         grid=true,
         )
for (i, (cl, linestyle)) in enumerate(zip(levels(contours(q,S,z,Ls)), linestyles))
    lvl = level(cl) # the z-value of this contour level
    haslabel = false
    for (j,line) in enumerate(lines(cl))
        xs, ys = coordinates(line) # coordinates of this line segment
        
        if j==2
            continue
        end

        # any(((x,y),) -> 25 < K_min(x), zip(xs,ys)) ? continue : nothing

        # if j == 2 && lvl == 0.01
        #     # need to correct this contour piece by flipping the x values at
        #     # the critical q value where K > K_min(q) for K=25
        #     # since computing this piece is poorly conditioned
        #     xs = minimum(xs) .- (xs .- minimum(xs))
        # end

        # if j == 1 && lvl == 0.01
        #     # glue the end of the first contour piece to the corrected second piece
        #     ys[end-4] = 2.2
        #     ys[end-3] = 2.1
        #     ys[end-2] = 2.05
        #     ys[end-1] = 2.1
        #     ys[end] = 2.3
        # end

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
annotate!([0.75, 0.3], [25, 10], ["\$S \\geq S_{LM}\$", "\$S \\leq S_{LM}\$"])
display(p)
end
end

export_figs && save_and_render(p, "lazy_sufficient_conditions_fixed_iteration")