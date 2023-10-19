using Serialization, PlotlyJS

export externalview, internalview, preplot

## Viewers

function externalview(args)
    path = tempname()
    serialize(path, args)
    jlcmd = "using CrystalEnergyGrids, PlotlyJS, Serialization; atexit(() -> close(stdin)); args = deserialize(\"$path\"); rm(\"$path\"); p = CrystalEnergyGrids.internalview(args); display(p); while true; read(stdin); sleep(10.0); end"
    cmd = `$(Base.julia_cmd()) --project=$(dirname(@__DIR__)) -e "$jlcmd"`
    p = Pipe()
    close(p.out)
    run(pipeline(Cmd(cmd; ignorestatus=true, detach=true); stdin=p.in); wait=false)
    @info "External Plotly window will appear momentarily"
end

function internalview(args)
    # download at the current resolution
    config = PlotConfig(toImageButtonOptions=attr(format="jpeg", height=nothing, width=nothing).fields)

    # do not show background axes
    shownothing = PlotlyJS.attr(; showbackground=false, showgrid=false, showticklabels=false, showspikes=false)
    # shownothing = PlotlyJS.attr()

    p = plot(args, PlotlyJS.Layout(; scene=PlotlyJS.attr(;
        camera_projection_type="orthographic", aspectmode="data",
        xaxis=shownothing, yaxis=shownothing, zaxis=shownothing,
        ),
        margin=PlotlyJS.attr(; t=0, b=0, l=0, r=0, autoexpand=false),
        showlegend=false,
        ); config);
    display(p)
end


## Bonding algorithm (mostly taken from CrystalNets.jl)

# Data from Blue Obelisk's data repository, corrected for H, C, O, N, S and F
# (see https://github.com/chemfiles/chemfiles/issues/301#issuecomment-574100048)
const vdwradii = Float32[1.0, 1.4, 2.2, 1.9, 1.8, 1.5, 1.4, 1.3, 1.2, 1.54, 2.4, 2.2, 2.1,
                         2.1, 1.95, 1.9, 1.8, 1.88, 2.8, 2.4, 2.3, 2.15, 2.05, 2.05, 2.05,
                         2.05, 2.0, 2.0, 2.0, 2.1, 2.1, 2.1, 2.05, 1.9, 1.9, 2.02, 2.9,
                         2.55, 2.4, 2.3, 2.15, 2.1, 2.05, 2.05, 2.0, 2.05, 2.1, 2.2, 2.2,
                         2.25, 2.2, 2.1, 2.1, 2.16, 3.0, 2.7, 2.5, 2.48, 2.47, 2.45, 2.43,
                         2.27, 2.25, 2.2, 2.1, 2.05, 2.0, 2.0, 2.05, 2.1, 2.05, 2.2, 2.3,
                         2.3, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.4, 2.0, 2.3, 2.0, 2.0, 2.0,
                         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0,
                         2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]

# Data from PeriodicTable.jl
const ismetal = Bool[0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1]

"""
    guess_bonds(system::AbstractSystem)

Return the bonds guessed from the system. The algorithm is mostly taken from CrystalNets.jl.

The returned list contains triplets of the form `(src, dst, ofs)` where `src` and `dst` are
the source and destination atoms, and `ofs` is the offset of the cell of the destination
compared to that of the source.
"""
function guess_bonds(system::AbstractSystem{3})
    n = length(system)
    bonds = Tuple{Int,Int,SVector{3,Int}}[]
    radii = Vector{Float32}(undef, n)
    wider_metallic_bonds = any(i -> atomic_number(system, i) ≥ 21, 1:n)
    mat = stack3(bounding_box(system))
    invmat = inv(mat)
    for i in 1:n
        t = atomic_number(system, i)
        if t isa Int && isassigned(vdwradii, t)
            radii[i] = vdwradii[t]*(1 + wider_metallic_bonds*ismetal[t]*0.5)
        else
            radii[i] = 0.0
        end
    end
    cutoff = (3*(0.75^3.1) * max(maximum(radii), 0.833))^2
    cutoff2 = 13*0.75/15
    buffer, ortho, safemin = prepare_periodic_distance_computations(mat)
    safemin2 = safemin^2
    buffer2 = MVector{3,Float64}(undef)
    for i in 1:n
        radius_i = radii[i]
        iszero(radius_i) && continue
        posi = NoUnits.(position(system, i)/u"Å")
        for j in (i+1):n
            radius_j = radii[j]
            iszero(radius_j) && continue
            posj = NoUnits.(position(system, j)/u"Å")
            buffer .= posi .- posj
            d2 = periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)
            maxdist = (cutoff2*(radius_i + radius_j))^2
            if d2 < cutoff && 0.25 < d2 < maxdist
                for ofsx in -1:1, ofsy in -1:1, ofsz in -1:1
                    ofs = SVector{3,Int}(ofsx, ofsy, ofsz)
                    if norm2(posi .- (posj .+ mat*ofs)) < maxdist
                        push!(bonds, (i, j, ofs))
                    end
                end
            end
        end
    end
    bonds
end


## Utils for tracing multiple segments in one Plotly line

function one_line(l; kwargs...)
    x = Union{Float64,Nothing}[]; y = Union{Float64,Nothing}[]; z = Union{Float64,Nothing}[]
    for (src, dst) in l
        push!(x, src[1], dst[1], nothing)
        push!(y, src[2], dst[2], nothing)
        push!(z, src[3], dst[3], nothing)
    end
    pop!(x); pop!(y); pop!(z)
    PlotlyJS.scatter(; x, y, z,
                       mode="lines", type="scatter3d", connectgaps=false, hoverinfo="skip",
                       kwargs...)
end

function colored_lines(l; kwargs...)
    isempty(l) && return GenericTrace{Dict{Symbol, Any}}[]
    percolor = Dict{String,Vector{NTuple{2,SVector{3,Float64}}}}()
    for (src, dst, color) in l
        push!(get!(percolor, color, NTuple{2,SVector{3,Float64}}[]), (src, dst))
    end
    [one_line(l; line_color=c, kwargs...) for (c, l) in percolor]
end


## Main functions

function preplot(system::AbstractSystem{3})
    n = length(system)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    color = Vector{String}(undef, n)
    size = Vector{Float64}(undef, n)
    hovertext = Vector{String}(undef, n)
    tokeep = Int[]
    for i in 1:n
        p = NoUnits.(position(system, i)/u"Å")
        x[i] = p[1]
        y[i] = p[2]
        z[i] = p[3]
        at = atomic_number(system, i)
        at != 0 && push!(tokeep, i)
        size[i] = 2 + at
        color[i] = get(atom_info, at, (:X, "#C2C2C2", "#C2C2C2"))[3]
        hovertext[i] = string(atomic_symbol(system, i), " (", i, ')')
    end

    ret = [PlotlyJS.scatter(; x=x[tokeep], y=y[tokeep], z=z[tokeep],
                              mode="markers", type="scatter3d", size_max=20,
                              hoverlabel_namelength=0, hovertext=hovertext[tokeep],
                              marker=PlotlyJS.attr(; color=color[tokeep], size=size[tokeep],
                                                     opacity=1,
                                                     line=PlotlyJS.attr(width=1, color="black")))]

    a1, a2, a3 = [NoUnits.(x/u"Å") for x in bounding_box(system)]

    # Plot the periodic box
    if all(==(Periodic()), boundary_conditions(system))
        line_color = "black"
        push!(ret, one_line([
            (zero(SVector{3,Float64}), a1),
            (zero(SVector{3,Float64}), a2),
            (zero(SVector{3,Float64}), a3),
            (a1, a1+a2),
            (a1, a1+a3),
            (a2, a1+a2),
            (a2, a2+a3),
            (a3, a1+a3),
            (a3, a2+a3),
            (a1+a2, a1+a2+a3),
            (a1+a3, a1+a2+a3),
            (a2+a3, a1+a2+a3),
        ]; line_color))
    end

    # Plot the chemical bonds
    bond_line = Tuple{SVector{3,Float64},SVector{3,Float64},String}[]
    for (i, j, ofs) in guess_bonds(system)
        thisofs = ofs[1]*a1 + ofs[2]*a2 + ofs[3]*a3
        refposi = NoUnits.(position(system, i)/u"Å")
        refposj = NoUnits.(position(system, j)/u"Å")
        push!(bond_line, (refposi, (refposi + refposj + thisofs)/2, color[i]))
        push!(bond_line, ((refposi + refposj - thisofs)/2, refposj, color[j]))
    end
    append!(ret, colored_lines(bond_line; line_width=3))

    ret
end

function preplot(field::AbstractArray{Float64,3}, reference::AbstractSystem{3}, block=nothing)
    Base.require_one_based_indexing(field)
    numA, numB, numC = size(field)
    axeA, axeB, axeC = [NoUnits.(x/u"Å") for x in bounding_box(reference)]
    # X, Y, Z = mgrid(range(0.0; step=(axeA[1]+axeB[1]+axeC[1])))
    stepA = axeA / numA
    stepB = axeB / numB
    stepC = axeC / numC
    indices = CartesianIndices((numA, numB, numC))
    n = length(indices)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    blockx = Float64[]
    blocky = Float64[]
    blockz = Float64[]
    for (i, ci) in enumerate(indices)
        iA, iB, iC = Tuple(ci)
        v = (iA-1)*stepA + (iB-1)*stepB + (iC-1)*stepC
        x[i] = v[1]
        y[i] = v[2]
        z[i] = v[3]
        if (block===nothing || block) && field[iA, iB, iC] == 1e100
            push!(blockx, v[1])
            push!(blocky, v[2])
            push!(blockz, v[3])
        end
    end
    # @show last(x), last(y), last(z)
    isomin, isomax = extrema(field)
    colorscale = if iszero(isomin) # density
        isomin = isomax/40
        colors.BuPu
    else # energy
        isomax = partialsort(vec(field), min(length(field), 10000)) # 10000th lowest value
        colors.ice
    end

    ret = [PlotlyJS.volume(; x, y, z, value=field[:],
                            suface_count=20, isomin, isomax,
                            opacity=0.6,
                            hoverinfo="skip",
                            colorscale
                            )]
    if (block===nothing || block) && !isempty(blockx)
        push!(ret,
            PlotlyJS.scatter(; x=blockx, y=blocky, z=blockz,
                mode="markers", type="scatter3d",
                hoverinfo="skip",
                marker=PlotlyJS.attr(; color="black", size=1, opacity=0.4,
                                        line=PlotlyJS.attr(width=0))
        ))
    end
    block===true && popfirst!(ret)
    ret
end


function preplot(points::Vector{CartesianIndex{3}}, field::AbstractArray{Float64,3}, reference::AbstractSystem{3})
    numA, numB, numC = size(field)
    axeA, axeB, axeC = [NoUnits.(x/u"Å") for x in bounding_box(reference)]
    # X, Y, Z = mgrid(range(0.0; step=(axeA[1]+axeB[1]+axeC[1])))
    stepA = axeA / numA
    stepB = axeB / numB
    stepC = axeC / numC
    n = length(points)
    x = Vector{Float64}(undef, n)
    y = Vector{Float64}(undef, n)
    z = Vector{Float64}(undef, n)
    color = Vector{String}(undef, n)
    hovertext = Vector{String}(undef, n)
    markersize = Vector{Float64}(undef, n)
    values = [field[p] for p in points]
    m, M = extrema(values)
    for i in 1:n
        a, b, c = Tuple(points[i])
        p = (a-1)*stepA + (b-1)*stepB + (c-1)*stepC
        x[i] = p[1]
        y[i] = p[2]
        z[i] = p[3]
        value = values[i]
        markersize[i] = round(Int, clamp(-log(value-m), 2, 20))
        hovertext[i] = string(value)
        color[i] = "blue"
    end

    [PlotlyJS.scatter(; x, y, z,
                        mode="markers", type="scatter3d",
                        hoverlabel_namelength=0, hovertext, size_max=20,
                        marker=PlotlyJS.attr(; color=color, size=markersize, opacity=1,
                                               line=PlotlyJS.attr(width=0)))]
end

"""
    preplot

Return a list of plotting recipes that can be fed to [`internalview`](@ref) or
[`externalview`](@ref)
"""
preplot
# preplots = preplot(setup.framework)
# append!(preplots, preplot(density))
# externalview(preplots) # or internalview(preplots)
