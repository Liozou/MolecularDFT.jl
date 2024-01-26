import CrystalEnergyGrids as CEG
using Unitful: Quantity, ustrip, @u_str
import Clapeyron
using Base.Threads: @threads
using LinearAlgebra: det, norm

# Compact sparse array used to represent the interactions between two identical species
struct CentroSymmetricTensor{T}
    data::Vector{T}
    inds::Matrix{NTuple{3,Int}}
    default::T
    size::NTuple{3,Int}
end
Base.@propagate_inbounds function Base.getindex(x::CentroSymmetricTensor, i1::Integer, j1::Integer, k1::Integer, i2::Integer, j2::Integer, k2::Integer)
    a, b, c = x.size
    i = mod1(1+i1-i2, a)
    j = mod1(1+j1-j2, b)
    k = mod1(1+k1-k2, c)
    dj, dk = size(x.inds)
    if k > dk
        k = 1 + (k!=1)*(c+1-k)
        k > dk && return x.default
        j = 1 + (j!=1)*(b+1-j)
        i = 1 + (i!=1)*(a+1-i)
    end
    j += dj÷2
    if j > b
        j -= b
    elseif j > dj
        return x.default
    end
    indpos, ofsi, maxi = x.inds[j,k]
    i += ofsi
    if i > a
        i -= a
    elseif i > maxi
        return x.default
    end
    return x.data[indpos + i]
end

"""
    CentroSymmetricTensor(values::AbstractVector{Tuple{Int,<:AbstractVector{Tuple{Int,<:AbstractVector{T}}}}}, (a,b,c)::NTuple{3,Int}, default::T) where T

Return a CentroSymmetricTensor of size `(a,b,c)` and default value `default` representing
an array `x` of identical size and having a central symmetry, i.e. for all `i`, `j`, `k`,
`x[1+i,1+j,1+k] == x[a+1-i, b+1-j, c+1-k]`. The symmetry is with respect to `(1,1,1)`.

`values` is made of `dk` pairs `(midj, values_j)` extracted from the initial array `x` of
size `(a,b,c)`, where `k`-th pair is made of:
- `midj` is the index of `values_j` that corresponds to a `j` index of 1 in `x`.
  Indices lower than `midj` in `values_j` correspond to initial indices between `b÷2` and
  `b`.
- `values_j` is the list of pairs `(midi, values_i)` such that `values_i` is not empty.
  Similarly, its `i`-th value is made of:
  * `midi`, the index of `values_i` that corresponds to a `i` index of 1 in `x`.
  * `values_i`, the list of values of the initial array such that `values_i[i]` is
    `x[mod1(i+1-midi), mod1(j+1-midj, b), k]`
"""
function CentroSymmetricTensor(values::AbstractVector{<:Tuple{Int,AbstractVector{<:Tuple{Int,AbstractVector{T}}}}}, (a,b,c)::NTuple{3,Int}, default::T) where T
    dk = length(values)
    @assert dk ≤ (c+1)÷2
    djneg = djpos = 0
    len = 0
    for (mj, valj) in values
        @assert 1 ≤ mj ≤ length(valj) ≤ b || mj == length(valj) == 0
        djneg = max(djneg, mj-1)
        djpos = max(djpos, length(valj)-mj)
        for (mi, vali) in valj
            @assert 1 ≤ mi ≤ length(vali) ≤ a || mi == length(vali) == 0
            len += length(vali)
        end
    end
    djm = 1 + max(djneg, djpos)
    dj = 1 + 2*(djm-1)
    inds = fill((0,0,0), dj, dk)
    data = Vector{T}(undef, len)
    idx = 1
    for k in 1:dk
        midj, values_j = values[k]
        for (j, (midi, values_i)) in enumerate(values_j)
            n = length(values_i)
            copyto!(data, idx, values_i, 1, n)
            inds[j-midj+djm,k] = (idx-1, midi-1, length(values_i))
            idx += n
        end
    end
    CentroSymmetricTensor{T}(data, inds, default, (a,b,c))
end

"""
    precompute_tailcorrection(setup::CEG.CrystalEnergySetup, ff::CEG.ForceField, mat, Π::Int)

Return a couple of tail corrections `(t1, t2)` such that a system made of `n` molecules has
an energy tail correction of `n * t1 + n^2 * t2`

In other words, adding a molecule to a system of `m` molecules results in an additional
tail correction of `t1 + (2m+1)*t2`.
"""
function precompute_tailcorrection(setup::CEG.CrystalEnergySetup, ff::CEG.ForceField, mat, Π::Int)
    n = length(ff.sdict)
    framework_atoms = zeros(Int, n)
    for at in setup.framework
        framework_atoms[ff.sdict[Symbol(CEG.get_atom_name(atomic_symbol(at)))]] += Π
    end
    m = length(setup.molecule)
    molecule_atoms = zeros(Int, n)
    for k in 1:m
        ix = ff.sdict[atomic_symbol(setup.molecule, k)::Symbol]
        molecule_atoms[ix] += 1
    end
    framework_tcorrection = 0.0u"K"
    self_tcorrection = 0.0u"K"
    for (i, ni) in enumerate(molecule_atoms)
        ni == 0 && continue
        for (j, nj) in enumerate(framework_atoms)
            nj == 0 && continue
            framework_tcorrection += ni*nj*CEG.tailcorrection(ff[i,j], ff.cutoff)
        end
        for (k, nk) in enumerate(molecule_atoms)
            nk == 0 && continue
            self_tcorrection += ni*nk*CEG.tailcorrection(ff[i,k], ff.cutoff)
        end
        # note: there is double counting, but apparently that's how RASPA does it.
    end
    λ = 2π/det(mat)
    (2λ*framework_tcorrection, λ*self_tcorrection)
end

function precompute_interactions(setup::CEG.CrystalEnergySetup, ff_∞::CEG.ForceField, cutoff::Float64, mat, (a,b,c))
    la, lb, lc = norm.(eachcol(mat))
    interp = function_average_self_potential(setup.molecule, ff_∞, 0.0:((la+lb+lc)/(5*(a+b+c))):cutoff)
    maxk = ceil(Int, cutoff*c/lc)
    interactions_data = Vector{Tuple{Int,Vector{Tuple{Int,Vector{Float64}}}}}(undef, maxk)
    invmat = inv(mat)
    cutoff2 = cutoff^2

    @threads for k in 1:maxk
        buffer, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
        safemin2 = safemin^2
        buffer2 = MVector{3,Float64}(undef)
        # last_j is the last j such that distj < cutoff2, and then
        # first_j is the first j such that distj < cutoff2
        last_j = first_j = 0
        values_j = Tuple{Int,Vector{Float64}}[]
        # last_distj and increasing_distj are used for the case where there is no gap
        # between last_j and first_j (and idem for i below)
        last_distj = NaN
        increasing_distj = false
        for j in 1:b
            buffer .= mat[:,3].*((k-1)/c) .+ mat[:,2].*((j-1)/b)
            col = SVector{3,Float64}(buffer)
            distj = CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)
            if !increasing_distj && !isnan(last_distj) && last_distj < distj
                increasing_distj = true
            end
            if last_j == 0
                if distj ≥ cutoff2
                    last_j = j-1
                elseif increasing_distj && distj ≤ last_distj
                    last_j = j-1
                    first_j = j
                end
            elseif last_j != 0 && first_j == 0 && distj < cutoff2
                first_j = j
            end
            if distj < cutoff2
                @assert last_j == 0 || first_j != 0 || (increasing_distj && last_j == first_j-1)
            else
                @assert last_j != 0 && first_j == 0
            end
            last_distj = distj
            distj < cutoff2 || continue
            values_i = Float64[]
            last_i = first_i = 0
            last_disti = NaN
            increasing_disti = false
            for i in 1:a
                buffer .= col .+ mat[:,1].*((i-1)/a)
                disti = CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)
                if !increasing_distj && !isnan(last_distj) && last_distj < distj
                    increasing_distj = true
                end
                if last_i == 0
                    if disti ≥ cutoff2
                        last_i = i-1
                    elseif increasing_disti && disti ≤ last_disti
                        last_i = i-1
                        first_i = i
                    end
                elseif last_i != 0 && first_i == 0 && disti < cutoff2
                    first_i = i
                end
                if disti < cutoff2
                    @assert last_i == 0 || first_i != 0 || (increasing_disti && last_i == first_i-1)
                else
                    @assert last_i != 0 && first_i == 0
                end
                last_disti = disti
                disti < cutoff2 || continue
                push!(values_i, interp(sqrt(disti)))
            end
            circshift!(values_i, -last_i)
            midi = first_i ≤ 1 ? 1 : a + 2 - first_i
            push!(values_j, (midi, values_i))
        end
        circshift!(values_j, -last_j)
        midj = first_j ≤ 1 ? 1 : b + 2 - first_j
        interactions_data[k] = (midj, values_j)
    end
    CentroSymmetricTensor(interactions_data, (a,b,c), 0.0)
end

struct ExplicitAngleInteraction
    mat::SMatrix{3,3,typeof(1.0u"Å"),9}
    invmat::SMatrix{3,3,typeof(1.0u"Å^-1"),9}
    minangles::Array{Int,3}
    gridsize::NTuple{3,Int}
    molposs::Vector{Vector{SVector{3,typeof(1.0u"Å")}}}
    ffidxi::Vector{Int}
    ff::CEG.ForceField
    buffer::MVector{3,typeof(1.0u"Å")}
    buffer2::MVector{3,Float64}
    ortho::Bool
    safemin2::typeof(1.0u"Å^2")
end
function ExplicitAngleInteraction(mol::AbstractSystem{3}, ff::CEG.ForceField, egrid::Array{Float64,4}, mat::AbstractMatrix, gridsize::NTuple{3,Int})
    minangles = argmin.(eachslice(egrid; dims=(2,3,4), drop=true))::Array{Int,3}
    poss0 = position(mol)::Vector{SVector{3,typeof(1.0u"Å")}}
    rots = CEG.get_rotation_matrices(mol, size(egrid, 1), true)[1]
    molposs = [[SVector{3}(r*p) for p in poss0] for r in rots]
    ffidxi = [ff.sdict[CEG.atomic_symbol(mol, k)]::Int for k in 1:length(mol)]
    buffer2, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
    buffer = MVector{3,typeof(1.0u"Å")}(undef)
    invmat = inv(mat).*u"Å^-1"
    ExplicitAngleInteraction(mat.*u"Å", invmat, minangles, gridsize, molposs, ffidxi, ff, buffer, buffer2, ortho, (safemin*u"Å")^2)
end
function Base.getindex(eai::ExplicitAngleInteraction, i1::Integer, j1::Integer, k1::Integer, i2::Integer, j2::Integer, k2::Integer)
    A, B, C = size(eai.minangles)
    l1 = eai.minangles[mod1(i1, A), mod1(j1, B), mod1(k1, C)]
    l2 = eai.minangles[mod1(i2, A), mod1(j2, B), mod1(k2, C)]
    pos1 = eai.molposs[l1]
    pos2 = eai.molposs[l2]
    a, b, c = eai.gridsize
    ofs = ((i2-i1)/a)*eai.mat[:,1] + ((j2-j1)/b)*eai.mat[:,2] + ((k2-k1)/c)*eai.mat[:,3]
    ret = 0.0u"K"
    for n1 in eachindex(pos1), n2 in eachindex(pos2)
        eai.buffer .= pos2[n2] .- pos1[n1] .+ ofs
        d2 = CEG.periodic_distance2_fromcartesian!(eai.buffer, eai.mat, eai.invmat, eai.ortho, eai.safemin2, eai.buffer2)
        ret += eai.ff[eai.ffidxi[n1], eai.ffidxi[n2]](d2)
    end
    ustrip(u"K", ret)
end


struct ExcludedVolumeInteraction
    mat::SMatrix{3,3,typeof(1.0u"Å"),9}
    invmat::SMatrix{3,3,typeof(1.0u"Å^-1"),9}
    gridsize::NTuple{3,Int}
    radius2::typeof(1.0u"Å^2")
    buffer::MVector{3,typeof(1.0u"Å")}
    buffer2::MVector{3,Float64}
    ortho::Bool
    safemin2::typeof(1.0u"Å^2")
end
function ExcludedVolumeInteraction(mat::AbstractMatrix, gridsize::NTuple{3,Int}, radius2)
    buffer2, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
    safemin2 = safemin^2
    buffer = MVector{3,typeof(1.0u"Å")}(undef)
    cell = CEG.CellMatrix(mat)
    ExcludedVolumeInteraction(cell.mat, cell.invmat, gridsize, radius2, buffer, buffer2, ortho, safemin2)
end
function Base.getindex(evi::ExcludedVolumeInteraction, i1::Integer, j1::Integer, k1::Integer, i2::Integer, j2::Integer, k2::Integer)
    a, b, c = evi.gridsize
    evi.buffer .= ((i2-i1)/a)*evi.mat[:,1] + ((j2-j1)/b)*evi.mat[:,2] + ((k2-k1)/c)*evi.mat[:,3]
    d2 = CEG.periodic_distance2_fromcartesian!(evi.buffer, evi.mat, evi.invmat, evi.ortho, evi.safemin2, evi.buffer2)
    d2 ≤ evi.radius2 && return 1e100
    0.0
end

struct GridMCSetup{TModel0,TModel,TInteractions}
    egrid::Array{Float64,4}
    positions::Vector{NTuple{3,Int}}
    volume::typeof(1.0u"Å^3")
    interactions::TInteractions # TODO: replace by the final chosen type, no need to keep a parameter here
    moves::CEG.MCMoves
    model0::TModel0
    model::TModel
    tailcorrection::NTuple{2,typeof(1.0u"K")} # (amount to add for each additional molecule, amount to add after multiplying by the number of molecules)
    input::NTuple{4,String}
    num_unitcell::NTuple{3,Int}
end
function GridMCSetup(grid, mat::AbstractMatrix{typeof(1.0u"Å")}, interactions, moves::CEG.MCMoves, tailcorrection::NTuple{2,typeof(1.0u"K")}, gasname::AbstractString, input::Tuple, num_unitcell::NTuple{3,Int}, positions=NTuple{3,Int}[])
    gaskey = get(GAS_NAMES, gasname, gasname)
    accessible_fraction = count(<(1e100), grid) / length(grid)
    volume = accessible_fraction * prod(CEG.cell_lengths(mat))
    GridMCSetup(grid, positions, volume, interactions, moves, Clapeyron.PR([gaskey]), Clapeyron.GERG2008([gaskey]), tailcorrection, input, num_unitcell)
end

function Base.copy(gmc::GridMCSetup)
    GridMCSetup(gmc.egrid, copy(gmc.positions), gmc.volume, gmc.interactions, gmc.moves, gmc.model0, gmc.model, gmc.tailcorrection, gmc.input, gmc.num_unitcell)
end

function GridMCSetup(framework, forcefield::AbstractString, gasname::AbstractString, mol_ff::AbstractString, step=0.15u"Å", moves=nothing)
    setup = CEG.setup_RASPA(framework, forcefield, gasname, mol_ff; gridstep=step)
    egrid = CEG.energy_grid(setup, step)
    # min_angle = findmin(eachslice(egrid; dims=(2,3)))
    ff = setup.forcefield
    ff_∞ = CEG.parse_forcefield_RASPA(forcefield; cutoff=Inf*u"Å")

    num_unitcell = CEG.find_supercell(setup.framework, ff.cutoff)
    mat = SMatrix{3,3,Float64,9}(ustrip.(u"Å", stack(num_unitcell.*bounding_box(setup.framework))))

    tailcorrection = precompute_tailcorrection(setup, ff, mat, prod(num_unitcell))

    abc = num_unitcell .* size(egrid)[2:end]
    ## one of either...
    interactions = precompute_interactions(setup, ff_∞, ustrip(u"Å", ff.cutoff), mat, abc)

    ## or...
    # interactions = ExplicitAngleInteraction(setup.molecule, ff_∞, egrid, mat, abc)

    ## or...
    # interactions = ExcludedVolumeInteraction(mat.*u"Å", abc, (3.8u"Å")^2)

    cmoves = moves isa Nothing ? CEG.MCMoves(true) : moves
    input = (framework isa AbstractString ? framework : "(empty)", forcefield, gasname, mol_ff)
    GridMCSetup(egrid, mat*u"Å", interactions, cmoves, tailcorrection, gasname, input, num_unitcell)
end


function CEG.ProtoSimulationStep(gmc::GridMCSetup, gmcpositions=gmc.positions)
    # GridMCSetup(framework::AbstractString, forcefield::AbstractString, gasname::AbstractString, mol_ff::AbstractString, step=0.15u"Å", moves=nothing)
    ff = CEG.parse_forcefield_RASPA(gmc.input[2])
    framework = CEG.load_framework_RASPA(gmc.input[1], gmc.input[2])
    molecule = CEG.load_molecule_RASPA(gmc.input[3], gmc.input[4], gmc.input[2], framework)
    rots, _ = CEG.get_rotation_matrices(molecule, 40)
    @assert length(rots) == size(gmc.egrid, 1)

    refpos = position(molecule)::Vector{SVector{3,typeof(1.0u"Å")}}
    m = length(molecule)
    n = length(gmcpositions)

    mat = uconvert.(u"Å", stack(gmc.num_unitcell.*bounding_box(framework)))

    charges = fill(NaN*oneunit(molecule.atomic_charge[1]), length(ff.sdict))
    atoms = [(1,j,k) for j in 1:n for k in 1:m]
    ffidxi = [ff.sdict[molecule.atomic_symbol[k]::Symbol] for k in 1:m]
    for k in 1:m
        ix = ffidxi[k]
        if isnan(charges[ix])
            charges[ix] = molecule.atomic_charge[k]
        else
            @assert charges[ix] == molecule.atomic_charge[k]
        end
    end

    _, _a, _b, _c = size(gmc.egrid)
    a, b, c = gmc.num_unitcell .* (_a, _b, _c)
    stepA, stepB, stepC = eachcol(mat) ./ (a,b,c)
    positions = Vector{SVector{3,typeof(1.0u"Å")}}(undef, n*m)
    buffer = MVector{3,typeof(1.0u"Å")}(undef)
    ofs = MVector{3,typeof(1.0u"Å")}(undef)
    for idx in 1:n
        i, j, k = gmcpositions[idx]
        ofs .= stepA.*(i-1) .+ stepB.*(j-1) .+ stepC.*(k-1)
        l = (idx-1)*m
        rot = rots[findmin(@view gmc.egrid[:, mod1(i, _a), mod1(j, _b), mod1(k, _c)])[2]]
        for p in refpos
            l += 1
            mul!(buffer, rot, p)
            buffer .+= ofs
            positions[l] = SVector{3}(buffer)
        end
    end

    CEG.ProtoSimulationStep(ff, charges, mat, positions, true, atoms, trues(1), [ffidxi])
end


# Energy computation

struct GridMCEnergyReport
    framework::typeof(1.0u"K")
    interaction::typeof(1.0u"K")
    tailcorrection::typeof(1.0u"K")
end
GridMCEnergyReport() = GridMCEnergyReport(0.0u"K", 0.0u"K", 0.0u"K")
Base.Number(x::GridMCEnergyReport) = x.framework + x.interaction + x.tailcorrection
Base.Float64(x::GridMCEnergyReport) = ustrip(u"K", Number(x))
for op in (:+, :-)
    @eval begin
        Base.$op(x::GridMCEnergyReport) = GridMCEnergyReport($op(x.framework), $op(x.interaction), $op(x.tailcorrection))
        function Base.$op(x::GridMCEnergyReport, y::GridMCEnergyReport)
            GridMCEnergyReport($op(x.framework, y.framework), $op(x.interaction, y.interaction), $op(x.tailcorrection, y.tailcorrection))
        end
    end
end
Base.:/(x::GridMCEnergyReport, n::Integer) = GridMCEnergyReport(x.framework/n, x.interaction/n, x.tailcorrection/n)


function interaction_energy(gmc::GridMCSetup, (i1,j1,k1)::NTuple{3,Int}, (i2,j2,k2)::NTuple{3,Int})
    gmc.interactions[i1, j1, k1, i2, j2, k2]
end

function all_interactions(gmc::GridMCSetup, grid::Array{Float64,3}, l::Int, pos::NTuple{3,Int})
    energy = 0.0
    for (idx, other) in enumerate(gmc.positions)
        idx == l && continue
        energy += interaction_energy(gmc, pos, other)
    end
    i,j,k = pos
    a,b,c = size(grid)
    n = length(gmc.positions) - (l!=0)
    tailcorrection = gmc.tailcorrection[1] + gmc.tailcorrection[2]*(2n+1)
    GridMCEnergyReport(grid[mod1(i,a),mod1(j,b),mod1(k,c)]*u"K", energy*u"K", tailcorrection)
end
all_interactions(gmc::GridMCSetup, grid::Array{Float64,3}, insertion_pos::NTuple{3,Int}) = all_interactions(gmc, grid, 0, insertion_pos)


function grid_angle_average(egrid, T0=300.0u"K")
    CEG.meanBoltzmann(egrid, ustrip(u"K", T0), CEG.get_lebedev_direct(size(egrid, 1)).weights)
    # dropdims(minimum(egrid; dims=1); dims=1)
    # TODO: decide whether to keep Boltzmann or minimum
end

function baseline_energy(gmc::GridMCSetup, grid::Array{Float64,3}=grid_angle_average(gmc.egrid))
    framework = 0.0
    energy = 0.0
    a,b,c = size(grid)
    for (l1, (i1,j1,k1)) in enumerate(gmc.positions)
        framework += grid[mod1(i1,a),mod1(j1,b),mod1(k1,c)]
        for l2 in (l1+1):length(gmc.positions)
            energy += interaction_energy(gmc, (i1,j1,k1), gmc.positions[l2])
        end
    end
    n = length(gmc.positions)
    tailcorrection = n*(gmc.tailcorrection[1] + n*gmc.tailcorrection[2])
    GridMCEnergyReport(framework*u"K", energy*u"K", tailcorrection)
end

function choose_newpos!(statistics, gmc::GridMCSetup, grid::Array{Float64,3}, idx, r=(isnothing(idx) ? 1.0 : rand()))
    if idx isa Nothing
        movekind = :random_translation
    else
        movekind = gmc.moves(r)
        CEG.attempt!(statistics, movekind)
        x, y, z = gmc.positions[idx]
    end
    a, b, c = size(grid)
    α, β, γ = gmc.num_unitcell.*(a, b, c)
    for _ in 1:100
        u, v, w = newpos = if movekind === :translation
            (mod1(rand(x-2:x+2), α), mod1(rand(y-2:y+2), β), mod1(rand(z-2:z+2), γ))
        elseif movekind === :random_translation || movekind === :random_reinsertion
            (rand(1:α), rand(1:β), rand(1:γ))
        else
            error(lazy"Unknown move kind: $movekind")
        end
        if grid[mod1(u,a),mod1(v,b),mod1(w,c)] < 1e100
            return newpos, movekind, false
        end
    end
    # failed to find a suitable move
    if movekind === :translation
        return choose_newpos!(statistics, gmc, grid, idx, 1.0)
    end
    @warn "Trapped species did not manage to move out of a blocked situation. This could be caused by an impossible initial configuration."
    return idx, movekind, true # signal that the move was blocked
end

choose_newpos(gmc::GridMCSetup, grid::Array{Float64,3}) = choose_newpos!(nothing, gmc, grid, nothing)

function compute_accept_move(diff, T)
    diff < zero(diff) && return true
    e = exp(Float64(-diff/T))
    return rand() < e
end

function compute_accept_move(diff, temperature, φPV_div_k, N, isinsertion::Bool)
    rand() < if isinsertion
        φPV_div_k/((N+1)*temperature) * exp(-diff/temperature)
    else
        N*temperature/φPV_div_k * exp(-diff/temperature)
    end
end

function perform_swaps!(gmc::GridMCSetup, grid::Array{Float64,3}, temperature, φPV_div_k)
    idx_delete = 0
    insertion_pos = (0,0,0)
    insertion_successes = insertion_attempts = 0
    deletion_successes = deletion_attempts = 0
    newenergy = GridMCEnergyReport()
    for swap_retries in 1:4
        N = length(gmc.positions)
        isinsertion = rand(Bool)
        diff_swap = if isinsertion
            insertion_attempts += 1
            insertion_pos, _, blocked_insertion = choose_newpos(gmc, grid)
            blocked_insertion && continue
            all_interactions(gmc, grid, insertion_pos)
        else # deletion
            deletion_attempts += 1
            N == 0 && continue
            idx_delete = rand(1:N)
            -all_interactions(gmc, grid, idx_delete, gmc.positions[idx_delete])
        end
        if compute_accept_move(Number(diff_swap), temperature, φPV_div_k, N, isinsertion)
            newenergy += diff_swap
            if isinsertion
                push!(gmc.positions, insertion_pos)
                insertion_successes += 1
            else
                if idx_delete != length(gmc.positions)
                    gmc.positions[idx_delete], gmc.positions[end] = gmc.positions[end], gmc.positions[idx_delete]
                end
                pop!(gmc.positions)
                deletion_successes += 1
            end
        end
    end
    (insertion_successes, insertion_attempts, deletion_successes, deletion_attempts), newenergy
end

function run_grid_montecarlo!(gmc::GridMCSetup, simu::CEG.SimulationSetup, pressure)
    time_begin = time()

    @assert allequal(simu.temperatures)
    T0 = first(simu.temperatures)
    grid = grid_angle_average(gmc.egrid, T0)

    # energy initialization
    energy = baseline_energy(gmc, grid)
    if isinf(Float64(energy)) || isnan(Float64(energy))
        @error "Initial energy is not finite, this probably indicates a problem with the initial configuration."
    end
    energies = typeof(energy)[]
    initial_energies = typeof(energy)[]
    Ns = Int[]

    has_initial_output = any(startswith("initial")∘String, simu.outtype)

    statistics = CEG.MoveStatistics(1.3u"Å", 30.0u"°")
    swap_statistics = zero(MVector{4,Int})
    Π = prod(gmc.num_unitcell)


    if simu.ninit == 0
        # put!(output, startstep)
        push!(energies, energy)
        push!(Ns, length(gmc.positions))
        if simu.ncycles == 0 # single-point computation
            @goto end_cleanup
        end
    else
        # put!(initial_output, startstep)
        push!(initial_energies, energy)
    end

    P = pressure isa Quantity ? uconvert(u"Pa", pressure) : pressure*u"Pa"
    PV_div_k = uconvert(u"K", P*gmc.volume/u"k")
    φ = only(Clapeyron.fugacity_coefficient(gmc.model, P, T0; phase=:stable, vol0=Clapeyron.volume(gmc.model0, P, T0)))
    isnan(φ) && error("Specified gas not in gas form at the required temperature ($(first(simu.temperatures))) and pressure ($P)!")
    φPV_div_k = φ*PV_div_k

    # allpositions = [copy(gmc.positions)]

    for (counter_cycle, idx_cycle) in enumerate((-simu.ninit+1):simu.ncycles)
        temperature = simu.temperatures[counter_cycle]

        # φPV_div_k = if constantT
        #     φ
        # else
        #     Clapeyron.fugacity_coefficient!(φ, gmc.model, P, temperature; phase=:stable, vol0=Clapeyron.volume(gmc.model0, P, temperature))
        # end*PV_div_k
        swap_statistics_update, energy_update = perform_swaps!(gmc, grid, temperature, φPV_div_k)
        swap_statistics .+= swap_statistics_update
        energy += energy_update

        nummol = length(gmc.positions)
        numsteps = (nummol!=0)*max(20, nummol)
        for idx_step in 1:numsteps
            # choose the species on which to attempt a move
            idx = rand(1:nummol)

            # newpos is the position after the trial move
            newpos, move, blocked = choose_newpos!(statistics, gmc, grid, idx)

            # If, despite the multiple attempts, the move is blocked, skip early to the
            # next iteration.
            # The refused move is still taken into account in the statistics.
            blocked && continue

            diff = all_interactions(gmc, grid, idx, newpos) - all_interactions(gmc, grid, idx, gmc.positions[idx])

            accepted = compute_accept_move(Number(diff), temperature)
            if accepted
                if Float64(diff) ≥ 1e100
                    @show Number(diff)
                    error("$idx, $newpos")
                end
                gmc.positions[idx] = newpos
                CEG.accept!(statistics, move)
                if abs(Number(diff)) > 1e50u"K" # an atom left a blocked pocket
                    @warn "Unexpected energy difference for pressure $pressure"
                    energy = baseline_energy(gmc, grid) # to avoid underflows
                else
                    energy += diff
                end
            end
        end

        # end of cycle
        report_now = (idx_cycle ≥ 0 && (idx_cycle == 0 || (simu.printevery > 0 && idx_cycle%simu.printevery == 0))) ||
                     (idx_cycle < 0 && has_initial_output && (simu.printevery == 0 || (simu.printevery > 0 && idx_cycle%simu.printevery == 0)))
        if !(simu.record isa Returns) || report_now
            if idx_cycle == 0 # start production with a precise energy
                energy = baseline_energy(gmc, grid)
            end
            if report_now
                if idx_cycle < 0
                    push!(initial_energies, energy)
                else
                    push!(energies, energy)
                    push!(Ns, length(gmc.positions))
                end
            end
            yield()
        end
        # push!(allpositions, copy(gmc.positions))
    end

    @label end_cleanup
    if simu.printevery == 0 && simu.ncycles > 0
        push!(energies, energy)
        push!(Ns, length(gmc.positions))
    end
    time_end = time()
    CEG.print_report(simu, time_begin, time_end, statistics)
    open(joinpath(simu.outdir, "report.txt"), "a") do io
        tot_swap = sum(swap_statistics)
        println(io, "Accepted swap moves: ", swap_statistics[1] + swap_statistics[3], '/', tot_swap, " (attempted) = ", (swap_statistics[1]+swap_statistics[3])/tot_swap, " among which:")
        println(io, " - accepted insertion", swap_statistics[1]>1 ? "s: " : ": ", swap_statistics[1], '/', swap_statistics[2], " (attempted) = ", swap_statistics[1]/swap_statistics[2])
        println(io, " - accepted deletion", swap_statistics[3]>1 ? "s: " : ": ", swap_statistics[3], '/', swap_statistics[4], " (attempted) = ", swap_statistics[3]/swap_statistics[4])
        println(io)
    end
    energies, Ns./Π
end

function make_isotherm(gmc::GridMCSetup, simu::CEG.SimulationSetup, pressures)
    n = length(pressures)
    isotherm = Vector{Float64}(undef, n)
    energies = Vector{GridMCEnergyReport}(undef, n)
    @assert allequal(simu.temperatures)
    @info "The number of cycles is multiplied by 1 + 1e3/sqrt(ustrip(u\"Pa\", pressure))"
    @threads for i in 1:n
        newgmc = copy(gmc)
        pressure = pressures[i]
        P = pressure isa Quantity ? uconvert(u"Pa", pressure) : pressure*u"Pa"
        ncycles = ceil(Int, simu.ncycles*(1+1e3/sqrt(ustrip(u"Pa", P))))
        temperatures = fill(simu.temperatures[1], ncycles+simu.ninit)
        newsimu = CEG.SimulationSetup(temperatures, ncycles, simu.ninit, simu.outdir, simu.printevery, simu.outtype, simu.record)
        es, Ns = run_grid_montecarlo!(newgmc, newsimu, P)
        isotherm[i] = mean(Ns)
        energies[i] = mean(es)
        @show "Finished $pressure Pa"
    end
    isotherm, energies
end

# using Statistics: std

# function make_isotherms(gmcs::Vector{<:GridMCSetup}, simu::CEG.SimulationSetup, pressures)
#     m = length(gmcs)
#     n = length(pressures)
#     isotherms = Matrix{Float64}(undef, n, m)
#     σs = Matrix{Float64}(undef, n, m)
#     @threads for k in 1:m*n
#         i, j = fldmod1(k, m)
#         gmc = copy(gmcs[j])
#         pressure = pressures[i]
#         _, Ns = run_grid_montecarlo!(gmc, simu, pressure)
#         isotherms[i,j] = mean(Ns)
#         σs[i,j] = std(Ns)
#     end
#     isotherms, σs
# end
