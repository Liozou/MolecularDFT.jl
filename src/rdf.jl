using Base.Threads
using LinearAlgebra: isdiag, det
using Statistics: mean
import Chemfiles
import BasicInterpolators: LinearInterpolator, CubicSplineInterpolator, NoBoundaries

function compute_angle(posA, posB)
    pos = posB - posA
    p = pos/norm(pos)
    θ = acos(p[3])
    1 - abs(p[3]) < 1e-13 && return (θ, 0.0)
    q = SVector{2,Float64}(pos[1], pos[2])
    q /= norm(q)
    ϕ = acos(q[1])
    if q[2] < 0
        ϕ = 2π - ϕ
    end
    return θ, ϕ
end

"""
    get_molecules(frame, mat; skipmolid)

Given a `frame` and the corresponding matrix `mat` of the unit cell, returns:
- moleculepos: the list of position of the geometric centre of each molecule (in fractional
  coordinates);
- moleculeangle: the list of orientations (θ, ϕ) of each molecule (in radians, assuming the
  molecules are linear and centrosymmetric);
- moleculesID: for each molecule, an integer that identifies its formula (e.g. 1 for CO2, 2
  for O2, 3 for N2). If `skipmolid` is set, only the number of atoms is used to identify a
  molecule (this is significantly faster currently);
- IDmap: for each molecule ID, the list of individual molecules with this composition;
- groups: the list of molecules: each sublist is the list of its atoms.
"""
function get_molecules(frame, mat; skipmolid)
    invmat = inv(mat)
    poss = [invmat*x for x in frame]
    buffer, ortho, safemin = prepare_periodic_distance_computations(mat)
    n = length(poss)
    groupmap = collect(1:n) # groupmap[i] is the id of the group to which atom i belongs
    groups = [[i] for i in 1:n] # groups[k] is the list of atoms l in group k, i.e. such that groupmap[l] == k
    offsets = zeros(SVector{3,Int}, n) # offsets[i] is the offset of atom i in molecule groupmap[i]
    ofs = zero(MVector{3,Int})
    for i in 1:n, j in (i+1):n
        buffer .= poss[i] .- poss[j]
        d = periodic_distance_with_ofs!(buffer, ofs, mat, ortho, safemin)
        @assert d > 0.7
        if 0.7 < d < 1.3
            ki = groupmap[i]
            kj = groupmap[j]
            ofs .+= offsets[i]
            for l in groups[kj]
                groupmap[l] = ki
                offsets[l] += ofs
            end
            append!(groups[ki], groups[kj])
            empty!(groups[kj])
        end
    end
    for (i, o) in enumerate(offsets)
        poss[i] += o
    end
    filter!(!isempty, groups)
    noangle = any(x -> length(x) == 1, groups)
    # from this point, groupmap is invalid. groups[l] is the list of atom numbers of molecule l
    m = length(groups)
    moleculesID = Vector{Int}(undef, m) # For each molecule, an integer that identifies its formula (e.g. 1 for CO2, 2 for O2, 3 for N2)
    IDmap = Vector{Int}[] # for each molecule ID, the list of individual molecules with this composition
    IDdict = Dict{Vector{Int},Int}() # map a list of atom identifier to a molecule ID
    key = Int[]
    moleculepos = Vector{SVector{3,Float64}}(undef, m) # in fractional coordinates
    moleculeangle = Vector{NTuple{2,Float64}}(undef, ifelse(noangle, 0, m))
    pos = MVector{3,Float64}(undef)
    for (i, mol) in enumerate(groups)
        pos[1] = pos[2] = pos[3] = 0
        numatoms = length(mol)
        resize!(key, numatoms)
        for (j, atom) in enumerate(mol)
            pos .+= poss[atom]
            key[j] = skipmolid ? 0 : frame.atoms[j][1]
        end
        moleculepos[i] = SVector{3,Float64}(pos ./ numatoms)
        θ, ϕ = noangle ? (0.0,0.0) : compute_angle(frame[mol[1]], frame[mol[2]]) # TODO: only works for linear symmetric molecules
        if ϕ > π
            ϕ -= π
            θ = π - θ
        end
        noangle || (moleculeangle[i] = (θ, ϕ))
        molID = get!(IDdict, sort!(key), length(IDmap)+1)
        if molID > length(IDmap)
            push!(IDmap, copy(key))
        end
    end
    moleculepos, moleculeangle, moleculesID, IDmap, groups
end

function normalize_bins(intbins, maxdist, nbins, volume, natoms)
    ϵ = maxdist / nbins
    factor = 3/(4π*ϵ^3) * (2*volume)/natoms^2
    bins = Vector{Float64}(undef, nbins)
    for i in 1:nbins
        bins[i] = (intbins[i] / (3i^2-3i+1)) * factor # δV between sphere of radius i*ϵ and (i-1)*ϵ
    end
    bins, natoms/volume
end

function Frame(f::Chemfiles.Frame, mat, skipmolid)
    p = Chemfiles.positions(f)
    n = size(p, 2)
    atoms = Vector{Tuple{Int,SVector{3,Float64}}}(undef, n)
    for i in 1:n
        id = skipmolid ? 0 : Chemfiles.atomic_number(view(f, i-1))
        atoms[i] = (id, SVector{3,Float64}(@view p[:,i]))
    end
    Frame(atoms, mat)
end

function parse_rdf_chemfiles(file, maxdist=30.0, nbins=1000; skipmolid=true)
    skipmolid || @warn "Using skipmolid=false with chemfiles may be very slow"
    trajectories = [Chemfiles.Trajectory(file) for _ in 1:nthreads()]
    threadbins = zeros(Int, nthreads(), nbins)
    ϵ = maxdist / nbins
    Ns = zeros(Int, nthreads())
    volumes = zeros(Float64, nthreads())
    m = Int(length(trajectories[1]))
    @threads :static for i_frame in 1:m
        trajectory = trajectories[threadid()]
        cframe = Chemfiles.read_step(trajectory, i_frame-1)
        mat = SMatrix{3,3,Cdouble,9}(Chemfiles.matrix(Chemfiles.UnitCell(cframe)))
        buffer, ortho, safemin = prepare_periodic_distance_computations(mat)
        frame = Frame(cframe, mat, skipmolid)
        moleculepos, moleculeangle, moleculesID, IDmap, groups = get_molecules(frame, mat; skipmolid)
        n = length(moleculepos)
        for i in 1:n, j in (i+1):n
            buffer .= moleculepos[i] .- moleculepos[j]
            idx = 1 + floor(Int, periodic_distance!(buffer, mat, ortho, safemin)/ϵ)
            # idx == 69 && (@show(i_frame, i, j, groups[i], groups[j]); throw(""))
            idx > nbins && continue
            threadbins[threadid(), idx] += 1
        end
        Ns[threadid()] += n
        volumes[threadid()] += det(mat)
    end
    foreach(close, trajectories)
    intbins = dropdims(sum(threadbins; dims=1); dims=1)
    normalize_bins(intbins, maxdist, nbins, sum(volumes), sum(Ns))
end

function parse_rdf(file, maxdist=30.0, nbins=1000; skipmolid=true)
    trajectory = LAMMPStrj(file)
    threadbins = zeros(Int, nthreads(), nbins)
    m = length(trajectory)
    Ns = zeros(Int, nthreads())
    volumes = zeros(Float64, nthreads())
    ϵ = maxdist / nbins
    @threads for i_frame in 1:m
        frame = trajectory[i_frame]
        mat = SMatrix{3,3,Cdouble,9}(frame.unitcell)
        if maxdist > max(mat[1,1], mat[2,2], mat[3,3])/2
            error(lazy"maxdist=$maxdist should be lower than half cell lengths $(mat[1,1]), $(mat[2,2]) and $(mat[3,3])")
        end
        buffer, ortho, safemin = prepare_periodic_distance_computations(mat)
        moleculepos, moleculeangle, moleculesID, IDmap, groups = get_molecules(frame, mat; skipmolid)
        n = length(moleculepos)
        for i in 1:n, j in (i+1):n
            buffer .= moleculepos[i] .- moleculepos[j]
            idx = 1 + floor(Int, periodic_distance!(buffer, mat, ortho, safemin)/ϵ)
            idx > nbins && continue
            threadbins[threadid(), idx] += 1
        end
        Ns[threadid()] += n
        volumes[threadid()] += det(mat)
    end
    intbins = dropdims(sum(threadbins; dims=1); dims=1)
    normalize_bins(intbins, maxdist, nbins, sum(volumes), sum(Ns))
end

rdf2tcf(rdf) = rdf ./ mean(rdf[end-div(length(rdf), 10):end]) .- 1


function energy_nocutoff(ff::CEG.ForceField, ffidxi, pos1, pos2)
    energy = 0.0u"K"
    for (k1, p1) in enumerate(pos1), (k2, p2) in enumerate(pos2)
        energy += ff[ffidxi[k1], ffidxi[k2]](CEG.norm2(p1, p2))
    end
    energy
end

function prepare_average_self_potential(mol::AbstractSystem, ff::CEG.ForceField, numrot_hint)
    rots, weights = CEG.get_rotation_matrices(mol, numrot_hint)
    @assert length(rots) == length(weights)
    poss0 = position(mol)
    molposs = [[SVector{3}(r*p) for p in poss0] for r in rots]
    ffidxi = [ff.sdict[CEG.atomic_symbol(mol, k)]::Int for k in 1:length(mol)]
    weights, molposs, ffidxi
end

function _compute_average_self_potential(molposs, ff::CEG.ForceField, ffidxi, offset, weights, buffer=Vector{typeof(1.0u"K")}(undef, length(weights)))
    numrot = length(weights)
    Base.Threads.@threads for j in 1:numrot
        tot = 0.0u"K"
        pos1 = molposs[j]
        pos2 = similar(pos1)
        for (k, weight) in enumerate(weights)
            pos2 .= molposs[k] .+ (offset,)
            tot += weight*energy_nocutoff(ff, ffidxi, pos1, pos2)
        end
        buffer[j] = tot
    end
    tot2 = 0.0u"K"
    for (j, weight) in enumerate(weights)
        tot2 += weight*buffer[j]
    end
    tot2/(4π)^2
end

function compute_average_self_potential(mol::AbstractSystem, ff::CEG.ForceField, dist::Number, numrot_hint=175)
    weights, molposs, ffidxi = prepare_average_self_potential(mol, ff, numrot_hint)
    offset = SA[0.0u"Å", 0.0u"Å", dist isa Quantity ? dist : dist*u"Å"]
    _compute_average_self_potential(molposs, ff, ffidxi, offset, weights)
end


function compute_average_self_potential(mol::AbstractSystem, ff::CEG.ForceField, range, numrot_hint=175)
    weights, molposs, ffidxi = prepare_average_self_potential(mol, ff, numrot_hint)
    n = length(range)
    rs = eltype(range) <: Quantity ? range : range * u"Å"
    T = eltype(rs)
    v = Vector{typeof(1.0u"K")}(undef, n)
    Base.Threads.@threads for i in 1:n
        offset = SVector{3,T}(zero(T), zero(T), rs[i])
        v[i] = _compute_average_self_potential(molposs, ff, ffidxi, offset, weights)
    end
    flag = false
    for i in n:-1:1 # fill the unphysical part close to 0 with value 1e100
        if !flag && v[i] > 1e100u"K"
            flag = true
        end
        if flag
            v[i] = 1e100u"K"
        end
    end
    eltype(range) <: Quantity ? v : ustrip.(u"K", v)
end

"""
    IncrementalSmoothInterpolator{T}

Interpolator divided in three regions:
- a linear interpolation region while values are above 1e4
- a constant region equal to zero above the last entry point
- a cubic spline interpolator in between
"""
struct IncrementalSmoothInterpolator{T}
    flow::LinearInterpolator{T, NoBoundaries}
    mid::T
    fhigh::CubicSplineInterpolator{T, NoBoundaries}
    top::T
end
function (f::IncrementalSmoothInterpolator{T})(x) where {T}
    @assert x ≥ zero(T);
    x > f.top && return zero(T)
    if x < f.mid
        return f.flow(x)
    end
    return f.fhigh(x)
end

function IncrementalSmoothInterpolator(x, y)
    midlimit = 1
    while y[midlimit] == 1e100
        midlimit += 1
    end
    inflimit = max(1, midlimit-1)
    λ = x[inflimit]
    while y[midlimit] > 1e4
        midlimit += 1
    end
    flow = if iszero(λ)
        LinearInterpolator(x[inflimit:midlimit], y[inflimit:midlimit], NoBoundaries())
    else
        LinearInterpolator([-0.0; λ/2; @view x[inflimit:midlimit]], [1e100; 1e100; @view y[inflimit:midlimit]], NoBoundaries())
    end
    fhigh = CubicSplineInterpolator(x[midlimit:end], y[midlimit:end], NoBoundaries())
    IncrementalSmoothInterpolator(flow, x[midlimit], fhigh, last(x))
end

function function_average_self_potential(mol::AbstractSystem, ff::CEG.ForceField, range, numrot_hint=175)
    eltype(range) <: Quantity && error("CubicSplineInterpolator does not support units: please ustrip the input")
    v = compute_average_self_potential(mol, ff, range, numrot_hint)
    IncrementalSmoothInterpolator(range, v)
end
