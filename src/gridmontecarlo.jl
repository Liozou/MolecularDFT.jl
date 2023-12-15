import CrystalEnergyGrids as CEG
using Unitful: Quantity, ustrip, @u_str
import Clapeyron

struct GridMCSetup{TModel}
    grid::Array{Float64,3}
    positions::Vector{NTuple{3,Int}}
    volume::typeof(1.0u"Å^3")
    interactions::Array{Float64,3}
    moves::CEG.MCMoves
    model::TModel
end
function GridMCSetup(grid, mat::AbstractMatrix{typeof(1.0u"Å")}, interactions, moves::CEG.MCMoves, gasname::AbstractString, positions=NTuple{3,Int}[])
    gaskey = get(GAS_NAMES, gasname, gasname)
    accessible_fraction = count(<(1e100), grid) / length(grid)
    volume = accessible_fraction * prod(CEG.cell_lengths(mat))
    GridMCSetup(grid, positions, volume, interactions, moves, Clapeyron.GERG2008([gaskey]))
end

function GridMCSetup(framework::AbstractString, forcefield::AbstractString, gasname::AbstractString, mol_ff::AbstractString, step=0.15u"Å", moves=nothing)
    setup = CEG.setup_RASPA(framework, forcefield, gasname, mol_ff)
    egrid = dropdims(mean(CEG.energy_grid(setup, step); dims=1); dims=1)
    mat = ustrip.(u"Å", setup.block.csetup.cell.mat)
    ff = setup.forcefield
    interp = function_average_self_potential(setup.molecule, ff, 0.0:1e-3:ustrip(u"Å", ff.cutoff))
    a1, a2, a3 = size(egrid)
    interactions = similar(egrid)
    invmat = inv(mat)
    buffer, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
    safemin2 = safemin^2
    buffer2 = MVector{3,Float64}(undef)
    for i3 in 1:a3, i2 in 1:a2, i1 in 1:a1
        buffer .= mat[:,1].*((i1-1)/a1) .+ mat[:,2].*((i2-1)/a2) .+ mat[:,3].*((i3-1)/a3)
        interactions[i1,i2,i3] = interp(sqrt(CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)))
    end
    cmoves = moves isa Nothing ? CEG.MCMoves(length(setup.molecule) == 1) : moves
    GridMCSetup(egrid, mat*u"Å", interactions, cmoves, gasname)
end

function interaction_energy(gmc::GridMCSetup, (i1,j1,k1)::NTuple{3,Int}, (i2,j2,k2)::NTuple{3,Int})
    a, b, c = size(gmc.interactions)
    gmc.interactions[mod1(i1-i2, a), mod1(j1-j2, b), mod1(k1-k2, c)]
end

function all_interactions(gmc::GridMCSetup, l::Int, pos::NTuple{3,Int})
    i,j,k = pos
    energy = gmc.grid[i,j,k]
    for (idx, other) in enumerate(gmc.positions)
        idx == l && continue
        energy += interaction_energy(gmc, pos, other)
    end
    energy*u"K"
end
all_interactions(gmc::GridMCSetup, insertion_pos::NTuple{3,Int}) = all_interactions(gmc, 0, insertion_pos)

function baseline_energy(gmc::GridMCSetup)
    energy = 0.0
    for (l1, (i1,j1,k1)) in enumerate(gmc.positions)
        energy += gmc.grid[i1,j1,k1]
        for l2 in (l1+1):length(gmc.positions)
            energy += interaction_energy(gmc, (i1,j1,k1), gmc.positions[l2])
        end
    end
    energy*u"K"
end

function choose_newpos!(statistics, gmc::GridMCSetup, idx)
    if idx isa Nothing
        movekind = :random_translation
    else
        r = rand()
        movekind = gmc.moves(r)
        CEG.attempt!(statistics, movekind)
        x, y, z = gmc.positions[idx]
    end
    a, b, c = size(gmc.grid)
    for _ in 1:100
        u, v, w = newpos = if movekind === :translation
            (mod1(rand(x-2:x+2), a), mod1(rand(y-2:y+2), b), mod1(rand(z-2:z+2), c))
        elseif movekind === :random_translation
            (rand(1:a), rand(1:b), rand(1:c))
        else
            error(lazy"Unknown move kind: $movekind")
        end
        if gmc.grid[u,v,w] < 1e100
            return newpos, movekind, false
        end
    end
    @warn "Trapped species did not manage to move out of a blocked situation. This could be caused by an impossible initial configuration."
    return pos, movekind, true # signal that the move was blocked
end

choose_newpos(gmc::GridMCSetup) = choose_newpos!(nothing, gmc, nothing)

function compute_accept_move(diff, T)
    diff < zero(diff) && return true
    e = exp(Float64(diff/T))
    return rand() < e
end

function compute_accept_move(diff, temperature, φPV_div_k, N, isinsertion::Bool)
    rand() < if isinsertion
        φPV_div_k/((N+1)*temperature) * exp(-diff/temperature)
    else
        N*temperature/φPV_div_k * exp(-diff/temperature)
    end
end

function perform_swaps!(gmc, temperature, φPV_div_k)
    idx_delete = 0
    insertion_pos = (0,0,0)
    insertion_successes = insertion_attempts = 0
    deletion_successes = deletion_attempts = 0
    for swap_retries in 1:10
        N = length(gmc.positions)
        isinsertion = N == 0 ? true : rand(Bool)
        diff_swap = if isinsertion
            insertion_attempts += 1
            insertion_pos, _, blocked_insertion = choose_newpos(gmc)
            blocked_insertion && continue
            all_interactions(gmc, insertion_pos)
        else # deletion
            deletion_attempts += 1
            idx_delete = rand(1:N)
            -all_interactions(gmc, idx_delete, gmc.positions[idx_delete])
        end
        if compute_accept_move(diff_swap, temperature, φPV_div_k, N, isinsertion)
            if isinsertion
                push!(gmc.positions, insertion_pos)
                insertion_successes += 1
            else
                if idx_delete != length(gmc.positions)
                    gmc.positions[idx_delete], gmc.positions[end] = gmc.positions[end], gmc.positions[idx_delete]
                    pop!(gmc.positions)
                end
                deletion_successes += 1
            end
        end
        insertion_attempts-insertion_successes + deletion_attempts-deletion_successes ≥ 2 && insertion_successes + deletion_successes ≥ 2 && break
    end
    (insertion_successes, insertion_attempts, deletion_successes, deletion_attempts)
end

function run_grid_montecarlo!(gmc::GridMCSetup, simu::CEG.SimulationSetup, pressure)
    time_begin = time()
    # energy initialization
    energy = baseline_energy(gmc)
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
    φ = only(Clapeyron.fugacity_coefficient(gmc.model, P, first(simu.temperatures)))
    constantT = allequal(simu.temperatures)

    for (counter_cycle, idx_cycle) in enumerate((-simu.ninit+1):simu.ncycles)
        temperature = simu.temperatures[counter_cycle]

        φPV_div_k = (constantT ? φ : Clapeyron.fugacity_coefficient!(φ, gmc.model, P, temperature))*PV_div_k
        swap_statistics .+= perform_swaps!(gmc, temperature, φPV_div_k)

        nummol = length(gmc.positions)
        numsteps = max(20, nummol)
        for idx_step in 1:numsteps
            # choose the species on which to attempt a move
            idx = rand(1:nummol)

            # newpos is the position after the trial move
            newpos, move, blocked = choose_newpos!(statistics, gmc, idx)

            # If, despite the multiple attempts, the move is blocked, skip early to the
            # next iteration.
            # The refused move is still taken into account in the statistics.
            blocked && continue

            diff = all_interactions(gmc, idx, newpos) - all_interactions(gmc, idx, gmc.positions[idx])

            accepted = compute_accept_move(diff, temperature)
            if accepted
                gmc.positions[idx] = newpos
                CEG.accept!(statistics, move)
                if abs(diff) > 1e50u"K" # an atom left a blocked pocket
                    energy = baseline_energy(gmc) # to avoid underflows
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
                energy = baseline_energy(gmc)
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
    energies, Ns
end
