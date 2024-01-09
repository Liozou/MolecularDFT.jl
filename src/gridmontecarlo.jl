import CrystalEnergyGrids as CEG
using Unitful: Quantity, ustrip, @u_str
import Clapeyron
using Base.Threads: @threads

# Compact sparse array used to represent the interactions between two identical species
struct CentroSymmetricTensor{T} <: AbstractArray{T,3}
    data::Vector{T}
    inds::Matrix{NTuple{3,Int}}
    default::T
    size::NTuple{3,Int}
end
Base.size(x::CentroSymmetricTensor) = x.size
Base.@propagate_inbounds function Base.getindex(x::CentroSymmetricTensor, i::Int, j::Int, k::Int)
    a, b, c = x.size
    @boundscheck if (i<1)|(j<1)|(k<1)|(i>a)|(j>b)|(k>c)
        throw(BoundsError(x, (i,j,k)))
    end
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

struct GridMCSetup{TModel0,TModel}
    egrid::Array{Float64,4}
    positions::Vector{NTuple{3,Int}}
    volume::typeof(1.0u"Å^3")
    interactions::CentroSymmetricTensor{Float64}
    moves::CEG.MCMoves
    model0::TModel0
    model::TModel
    input::NTuple{4,String}
    num_unitcell::NTuple{3,Int}
end
function GridMCSetup(grid, mat::AbstractMatrix{typeof(1.0u"Å")}, interactions, moves::CEG.MCMoves, gasname::AbstractString, input::Tuple, num_unitcell::NTuple{3,Int}, positions=NTuple{3,Int}[])
    gaskey = get(GAS_NAMES, gasname, gasname)
    accessible_fraction = count(<(1e100), grid) / length(grid)
    volume = accessible_fraction * prod(CEG.cell_lengths(mat))
    GridMCSetup(grid, positions, volume, interactions, moves, Clapeyron.PR([gaskey]), Clapeyron.GERG2008([gaskey]), input, num_unitcell)
end

function Base.copy(gmc::GridMCSetup)
    GridMCSetup(gmc.egrid, copy(gmc.positions), gmc.volume, gmc.interactions, gmc.moves, gmc.model0, gmc.model, gmc.input, gmc.num_unitcell)
end

function GridMCSetup(framework::AbstractString, forcefield::AbstractString, gasname::AbstractString, mol_ff::AbstractString, step=0.15u"Å", moves=nothing)
    setup = CEG.setup_RASPA(framework, forcefield, gasname, mol_ff; gridstep=step)
    egrid = CEG.energy_grid(setup, step)
    # min_angle = findmin(eachslice(egrid; dims=(2,3)))
    ff = setup.forcefield
    ff_∞ = CEG.parse_forcefield_RASPA(forcefield; cutoff=Inf*u"Å")
    interp = function_average_self_potential(setup.molecule, ff_∞, 0.0:(ustrip(u"Å", step)/5):ustrip(u"Å", ff.cutoff))
    num_unitcell = CEG.find_supercell(setup.framework, ff.cutoff)
    mat = ustrip.(u"Å", stack(num_unitcell.*bounding_box(setup.framework)))
    a, b, c = num_unitcell .* size(egrid)[2:end]
    maxk = ceil(Int, ustrip(u"Å", ff.cutoff)/norm(mat[:,3])*c)
    interactions_data = Vector{Tuple{Int,Vector{Tuple{Int,Vector{Float64}}}}}(undef, maxk)
    invmat = inv(mat)
    cutoff2 = ustrip(u"Å", ff.cutoff)^2
    @threads for k in 1:maxk
        buffer, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
        safemin2 = safemin^2
        buffer2 = MVector{3,Float64}(undef)
        last_j = first_j = 0
        values_j = Tuple{Int,Vector{Float64}}[]
        for j in 1:b
            buffer .= mat[:,3].*((k-1)/c) .+ mat[:,2].*((j-1)/b)
            col = SVector{3,Float64}(buffer)
            startdist = CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)
            if last_j == 0 && startdist ≥ cutoff2
                last_j = j-1
            elseif last_j != 0 && first_j == 0 && startdist < cutoff2
                first_j = j
            end
            if startdist < cutoff2
                @assert last_j == 0 || first_j != 0
            else
                @assert last_j != 0 && first_j == 0
            end
            startdist < cutoff2 || continue
            values_i = Float64[]
            last_i = first_i = 0
            for i in 1:a
                buffer .= col .+ mat[:,1].*((i-1)/a)
                dist2 = CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)
                if last_i == 0 && dist2 ≥ cutoff2
                    last_i = i-1
                elseif last_i != 0 && first_i == 0 && dist2 < cutoff2
                    first_i = i
                end
                if dist2 < cutoff2
                    @assert last_i == 0 || first_i != 0
                else
                    @assert last_i != 0 && first_i == 0
                end
                dist2 < cutoff2 || continue
                push!(values_i, interp(sqrt(dist2)))
            end
            circshift!(values_i, -last_i)
            midi = first_i ≤ 1 ? 1 : a + 2 - first_i
            push!(values_j, (midi, values_i))
        end
        circshift!(values_j, -last_j)
        midj = first_j ≤ 1 ? 1 : b + 2 - first_j
        interactions_data[k] = (midj, values_j)
    end
    interactions = CentroSymmetricTensor(interactions_data, (a,b,c), 0.0)
    cmoves = moves isa Nothing ? CEG.MCMoves(true) : moves
    input = (framework::AbstractString, forcefield, gasname, mol_ff)
    GridMCSetup(egrid, mat*u"Å", interactions, cmoves, gasname, input, num_unitcell)
end


# Energy computation

struct GridMCEnergyReport
    framework::typeof(1.0u"K")
    interaction::typeof(1.0u"K")
end
GridMCEnergyReport() = GridMCEnergyReport(0.0u"K", 0.0u"K")
Base.Number(x::GridMCEnergyReport) = x.framework + x.interaction
Base.Float64(x::GridMCEnergyReport) = ustrip(u"K", Number(x))
for op in (:+, :-)
    @eval begin
        Base.$op(x::GridMCEnergyReport) = GridMCEnergyReport($op(x.framework), $op(x.interaction))
        function Base.$op(x::GridMCEnergyReport, y::GridMCEnergyReport)
            GridMCEnergyReport($op(x.framework, y.framework), $op(x.interaction, y.interaction))
        end
    end
end
Base.:/(x::GridMCEnergyReport, n::Integer) = GridMCEnergyReport(x.framework/n, x.interaction/n)


function interaction_energy(gmc::GridMCSetup, (i1,j1,k1)::NTuple{3,Int}, (i2,j2,k2)::NTuple{3,Int})
    a, b, c = size(gmc.interactions)
    gmc.interactions[mod1(i1-i2, a), mod1(j1-j2, b), mod1(k1-k2, c)]
end

function all_interactions(gmc::GridMCSetup, grid::Array{Float64,3}, l::Int, pos::NTuple{3,Int})
    energy = 0.0
    for (idx, other) in enumerate(gmc.positions)
        idx == l && continue
        energy += interaction_energy(gmc, pos, other)
    end
    i,j,k = pos
    a,b,c = size(grid)
    GridMCEnergyReport(grid[mod1(i,a),mod1(j,b),mod1(k,c)]*u"K", energy*u"K")
end
all_interactions(gmc::GridMCSetup, grid::Array{Float64,3}, insertion_pos::NTuple{3,Int}) = all_interactions(gmc, grid, 0, insertion_pos)

function baseline_energy(gmc::GridMCSetup, grid::Array{Float64,3})
    framework = 0.0
    energy = 0.0
    a,b,c = size(grid)
    for (l1, (i1,j1,k1)) in enumerate(gmc.positions)
        framework += grid[mod1(i1,a),mod1(j1,b),mod1(k1,c)]
        for l2 in (l1+1):length(gmc.positions)
            energy += interaction_energy(gmc, (i1,j1,k1), gmc.positions[l2])
        end
    end
    GridMCEnergyReport(framework*u"K", energy*u"K")
end

function choose_newpos!(statistics, gmc::GridMCSetup, grid::Array{Float64,3}, idx)
    if idx isa Nothing
        movekind = :random_translation
    else
        r = rand()
        movekind = gmc.moves(r)
        CEG.attempt!(statistics, movekind)
        x, y, z = gmc.positions[idx]
    end
    a, b, c = size(grid)
    α, β, γ = size(gmc.interactions)
    for _ in 1:100
        u, v, w = newpos = if movekind === :translation
            (mod1(rand(x-2:x+2), α), mod1(rand(y-2:y+2), β), mod1(rand(z-2:z+2), γ))
        elseif movekind === :random_translation
            (rand(1:α), rand(1:β), rand(1:γ))
        else
            error(lazy"Unknown move kind: $movekind")
        end
        if grid[mod1(u,a),mod1(v,b),mod1(w,c)] < 1e100
            return newpos, movekind, false
        end
    end
    @warn "Trapped species did not manage to move out of a blocked situation. This could be caused by an impossible initial configuration."
    return pos, movekind, true # signal that the move was blocked
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
    for swap_retries in 1:10
        N = length(gmc.positions)
        isinsertion = N == 0 ? true : rand(Bool)
        diff_swap = if isinsertion
            insertion_attempts += 1
            insertion_pos, _, blocked_insertion = choose_newpos(gmc, grid)
            blocked_insertion && continue
            all_interactions(gmc, grid, insertion_pos)
        else # deletion
            deletion_attempts += 1
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
        insertion_attempts-insertion_successes + deletion_attempts-deletion_successes ≥ 2 && insertion_successes + deletion_successes ≥ 2 && break
    end
    (insertion_successes, insertion_attempts, deletion_successes, deletion_attempts), newenergy
end

function run_grid_montecarlo!(gmc::GridMCSetup, simu::CEG.SimulationSetup, pressure)
    time_begin = time()

    @assert allequal(simu.temperatures)
    T0 = first(simu.temperatures)
    grid = CEG.meanBoltzmann(gmc.egrid, ustrip(u"K", T0))
    # grid = dropdims(minimum(gmc.egrid; dims=1); dims=1)
    # TODO: decide whether to keep Boltzmann or minimum

    # energy initialization
    energy = baseline_energy(gmc, grid)
    if isinf(Float64(energy)) || isnan(Float64(energy))
        @error "Initial energy is not finite, this probably indicates a problem with the initial configuration."
    end
    energies = typeof(energy)[]
    initial_energies = typeof(energy)[]
    Ns = Int[]

    has_initial_output = any(startswith("initial")∘String, simu.outtype)

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

    statistics = CEG.MoveStatistics(1.3u"Å", 30.0u"°")
    swap_statistics = zero(MVector{4,Int})

    P = pressure isa Quantity ? uconvert(u"Pa", pressure) : pressure*u"Pa"
    PV_div_k = uconvert(u"K", P*gmc.volume/u"k")
    φ = only(Clapeyron.fugacity_coefficient(gmc.model, P, T0; phase=:stable, vol0=Clapeyron.volume(gmc.model0, P, T0)))
    isnan(φ) && error("Specified gas not in gas form at the required temperature ($(first(simu.temperatures))) and pressure ($P)!")
    φPV_div_k = φ*PV_div_k

    Π = prod(gmc.num_unitcell)

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
    @threads for i in 1:n
        newgmc = copy(gmc)
        pressure = pressures[i]
        es, Ns = run_grid_montecarlo!(newgmc, simu, pressure)
        isotherm[i] = mean(Ns)
        energies[i] = mean(es)
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
