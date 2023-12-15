import LinearAlgebra: norm, mul!
using Base.Threads
using Serialization
using SHA

using StaticArrays
import Optim
using FFTW, Hankel
import BasicInterpolators: CubicSplineInterpolator, NoBoundaries, WeakBoundaries
import Clapeyron

FFTW.set_num_threads(nthreads()÷2)

export isotherm_hnc, IdealGasProblem, MonoAtomic, LinearMolecule
export mdft, finaldensity

abstract type MDFTProblem <: Function end

finaldensity(problem::MDFTProblem) = finaldensity(problem, mdft(problem))

"""
    compute_ρ₀(gasname::AbstractString, T, P)

Compute the density of the given gas at the given temperature and pressure.
If no unit is given, `T` is assumed to be in K and `P` in Pa.
The returned density is in Å⁻³
"""
function compute_ρ₀(gasname::AbstractString, T::Real, P::Real)
    gaskey = get(GAS_NAMES, gasname, gasname)
    if gasname in GERG2008_nameset
        NoUnits(Clapeyron.molar_density(Clapeyron.GERG2008([gaskey]), P*u"Pa", T*u"K")*𝒩ₐ/u"Å^-3")
    else
        NoUnits(Clapeyron.molar_density(Clapeyron.PCSAFT([gaskey]), P*u"Pa", T*u"K")*𝒩ₐ/u"Å^-3")
    end
end
compute_ρ₀(gasname::AbstractString, T::Quantity, P::Quantity) = compute_ρ₀(gasname, NoUnits(T/u"K"), NoUnits(P/u"Pa"))

struct IdealGasProblem <: MDFTProblem
    ρ₀::Float64   # reference number density, in number/Å³ (obtained from VdW coefficients)
    T::Float64    # temperature, in K
    P::Float64    # pressure, in bar
    externalV::Array{Float64,3} # external energy field in K
    mat::SMatrix{3,3,Float64,9} # unit cell
    δv::Float64   # volume of an elementary grid cube, in Å³, derived from externalV and mat
end
function IdealGasProblem(T::Float64, P::Float64, externalV::Array{Float64,3}, mat::SMatrix{3,3,Float64,9})
    IdealGasProblem(P/(0.0831446261815324*T), T, P, externalV, mat, det(mat)/length(externalV))
end
function IdealGasProblem(gasname::String, T::Float64, P::Float64, externalV::Array{Float64,3}, mat::SMatrix{3,3,Float64,9})
    ρ₀ = compute_ρ₀(gasname, T, P)
    IdealGasProblem(ρ₀, T, P, externalV, mat, det(mat)/length(externalV))
end
function IdealGasProblem(ρ₀::Float64, T::Float64, P::Float64, externalV::Array{Float64,3}, mat::AbstractMatrix)
    IdealGasProblem(ρ₀, T, P, externalV, SMatrix{3,3,Float64,9}(mat), det(mat)/length(externalV))
end
function IdealGasProblem(gasname::String, T::Float64, P::Float64, externalV::Array{Float64,3}, mat::AbstractMatrix)
    IdealGasProblem(gasname, T, P, externalV, SMatrix{3,3,Float64,9}(mat))
end


function (igp::IdealGasProblem)(_, flat_∂ψ, flat_ψ::Vector{Float64})
    try
    ψ = reshape(flat_ψ, size(igp.externalV))
    ρ = isnothing(flat_∂ψ) ? similar(ψ) : reshape(flat_∂ψ, size(igp.externalV))
    ρ .= exp.(ψ)
    ψ₀ = log(igp.ρ₀)
    value = igp.δv*sum(ρr*v + igp.T*(ρr*(ψr - ψ₀ - 1) + igp.ρ₀) for (ψr, v, ρr) in zip(ψ, igp.externalV, ρ))
    if !isnothing(flat_∂ψ)
        @. ρ *= igp.δv*(igp.externalV + igp.T*(ψ - ψ₀))
    end
    # isempty(flat_∂ψ) || @show extrema(flat_∂ψ)
    @show value, extrema(ψ), norm(vec(ρ))
    value
    catch e
        Base.showerror(stderr, e)
        Base.show_backtrace(stderr, catch_backtrace())
        rethrow()
    end
end

function mdft(igp::IdealGasProblem)
    ψ₀ = log(igp.ρ₀)
    return exp.(ψ₀ .- vec(igp.externalV)./igp.T)
end

function finaldensity(igp::IdealGasProblem, ψ)
    sum(ψ)*igp.δv
end


struct SemiTruncatedInterpolator{T}
    f::CubicSplineInterpolator{T, NoBoundaries}
    R::T
end
(f::SemiTruncatedInterpolator{T})(x) where {T} = ((@assert x ≥ zero(T)); x > f.R ? zero(T) : f.f(x))
function SemiTruncatedInterpolator(qdht::QDHT, ĉ)
    f = CubicSplineInterpolator(qdht.r, (qdht\ĉ) ./ ((2π)^(3/2)), NoBoundaries())
    SemiTruncatedInterpolator(f, qdht.R)
end
function SemiTruncatedInterpolator(x, y)
    f = CubicSplineInterpolator(x, y, NoBoundaries())
    SemiTruncatedInterpolator(f, last(x))
end


function compute_dcf1D(tcf1D, R, ρ)
    n = length(tcf1D)
    qdht = QDHT{0,2}(R, n)
    hr = CubicSplineInterpolator([0; LinRange(0, R, n+1)[1:end-1] .+ (qdht.R/(2*n))],
                                 [-1.0; tcf1D], WeakBoundaries())
    hk = ((2π)^(3/2)) .* (qdht * hr.(qdht.r))
    ck = hk ./ (1 .+ ρ .* hk)
    qdht, ck
    # return qdht \ ck
end


## MonoAtomic

function expand_correlation(c₂r, (a1, a2, a3)::NTuple{3,Int}, mat)
    δv = det(mat)/(a1*a2*a3)
    invmat = inv(mat)
    buffer, ortho, safemin = CEG.prepare_periodic_distance_computations(mat)
    safemin2 = safemin^2
    buffer2 = MVector{3,Float64}(undef)
    c₂ = Array{Float64}(undef, a1, a2, a3)
    for i3 in 1:a3, i2 in 1:a2, i1 in 1:a1
        buffer .= mat[:,1].*((i1-1)/a1) .+ mat[:,2].*((i2-1)/a2) .+ mat[:,3].*((i3-1)/a3)
        c₂[i1,i2,i3] = δv*c₂r(sqrt(CEG.periodic_distance2_fromcartesian!(buffer, mat, invmat, ortho, safemin2, buffer2)))
    end
    c₂
end

struct MonoAtomic <: MDFTProblem
    igp::IdealGasProblem
    ĉ₂::Array{ComplexF64,3} # ĉ₂(k), Fourier transform of the direct correlation function c₂(r)
    plan::FFTW.rFFTWPlan{Float64, -1, false, 3, NTuple{3,Int}}
    c₂r::SemiTruncatedInterpolator{Float64}
end

function MonoAtomic(gasname_or_ρ₀, T::Float64, P::Float64, externalV::Array{Float64,3}, c₂r::SemiTruncatedInterpolator, _mat::AbstractMatrix{Float64})
    mat = SMatrix{3,3,Float64,9}(_mat)
    c₂ = expand_correlation(c₂r, size(externalV), mat)
    plan = plan_rfft(c₂)
    ĉ₂ = plan * c₂
    igp = IdealGasProblem(gasname_or_ρ₀, T, P, externalV, mat)
    MonoAtomic(igp, ĉ₂, plan, c₂r)
end

function MonoAtomic(gasname_or_ρ₀, T::Float64, P::Float64, withangles::Array{Float64,4}, c₂r::SemiTruncatedInterpolator, _mat::AbstractMatrix{Float64})
    a0, a1, a2, a3 = size(withangles)
    a0 == 1 || @info "Averaging grid over angles to compute MDFT on a model monoatomic species"
    weights = get_lebedev_direct(a0).weights
    externalV = Array{Float64,3}(undef, a1, a2, a3)
    @threads for i3 in 1:a3
        for i2 in 1:a2, i1 in 1:a1
            tot = 0.0
            for i0 in 1:a0
                tot += weights[i0]*withangles[i0,i1,i2,i3]
            end
            externalV[i1,i2,i3] = tot/(4π)
        end
    end
    MonoAtomic(gasname_or_ρ₀, T, P, externalV, c₂r, _mat)
    # MonoAtomic(gasname_or_ρ₀, T, P, meanBoltzmann(withangles, T), c₂r, _mat)
end

function MonoAtomic(gasname_or_ρ₀, T::Float64, P::Float64, externalV::Array{Float64,3}, qdht::QDHT{0,2,Float64}, ĉ₂vec::Vector{Float64}, _mat::AbstractMatrix{Float64})
    c₂r = SemiTruncatedInterpolator(qdht, ĉ₂vec)
    MonoAtomic(gasname_or_ρ₀, T, P, externalV, c₂r, _mat)
end

function (ma::MonoAtomic)(_, flat_∂ψ, flat_ψ::Vector{Float64})
    ψ = reshape(flat_ψ, size(ma.igp.externalV))
    ρ = isnothing(flat_∂ψ) ? similar(ψ) : reshape(flat_∂ψ, size(ma.igp.externalV))
    ρ .= ma.igp.ρ₀ .* ψ.^2
    convol = ρ .- ma.igp.ρ₀
    rfftΔρ = ma.plan * convol
    rfftΔρ .*= ma.ĉ₂
    FFTW.ldiv!(convol, ma.plan, rfftΔρ)
    logρmρ₀ = max.(log.(ρ ./ ma.igp.ρ₀), -1.3407807929942596e154)
    value = ma.igp.δv*sum(ρr*v + ma.igp.T*(ρr*logrmr₀ + (ma.igp.ρ₀-ρr) - CΔρ*(ρr-ma.igp.ρ₀)/2)
                for (logrmr₀, v, ρr, CΔρ) in zip(logρmρ₀, ma.igp.externalV, ρ, convol))
    if !isnothing(flat_∂ψ) # gradient update
        # finaldensity = sum(ρ)*ma.igp.δv
        @. ρ = $(2*ma.igp.ρ₀*ma.igp.δv)*ψ*(ma.igp.externalV + ma.igp.T*(logρmρ₀ - convol))
        # @. ρ = ifelse(abs(ρ) > 1.3407807929942596e100, 0.0, ρ)
        # @show value, finaldensity, maximum(ρ), norm(vec(ρ))
    end
    value
end

function mdft(ma::MonoAtomic, ψ_init=exp.(.-vec(ma.igp.externalV)./(2*ma.igp.T)))
    Optim.optimize(Optim.only_fg!(ma), ψ_init, Optim.LBFGS(linesearch=Optim.BackTracking(order=2, ρ_hi=0.1)),
                   Optim.Options(iterations=1000, f_tol=1e-10))
end

function finaldensity(ma::MonoAtomic, opt)
    if !Optim.converged(opt)
        @error "Optimizer failed to converge; proceeeding with partial result"
    end
    ψ = Optim.minimizer(opt)
    sum(x -> x*x, ψ)*ma.igp.ρ₀*ma.igp.δv
end



struct LinearMolecule <: MDFTProblem
    ρ₀::Float64   # reference number density, in number/Å³ (obtained from VdW coefficients)
    T::Float64    # temperature, in K
    P::Float64    # pressure, in bar
    externalV::Array{Float64,4} # external energy field in K (1st dimension: angles)
    lebedev_weights::Vector{Float64}
    mat::SMatrix{3,3,Float64,9} # unit cell
    δv::Float64   # volume of an elementary grid cube, in Å³, derived from externalV and mat
    ĉ₂::Array{ComplexF64,3} # ĉ₂(k), Fourier transform of the direct correlation function c₂(r)
    plan::FFTW.rFFTWPlan{Float64, -1, false, 3, NTuple{3,Int}}
    c₂r::SemiTruncatedInterpolator{Float64}
end

function LinearMolecule(gasname_or_ρ₀, T::Float64, P::Float64, externalV::Array{Float64,4}, c₂r::SemiTruncatedInterpolator, _mat::AbstractMatrix{Float64})
    mat = SMatrix{3,3,Float64,9}(_mat)
    _, a1, a2, a3 = size(externalV)
    c₂ = expand_correlation(c₂r, (a1, a2, a3), mat)
    plan = plan_rfft(c₂)
    ĉ₂ = plan * c₂
    ρ₀ = gasname_or_ρ₀ isa Float64 ? gasname_or_ρ₀ : compute_ρ₀(gasname_or_ρ₀, T, P)
    δv = det(mat)/(a1*a2*a3)
    weights = get_lebedev_direct(size(externalV, 1)).weights
    LinearMolecule(ρ₀, T, P, externalV, weights, mat, δv, ĉ₂, plan, c₂r)
end
function LinearMolecule(gasname_or_ρ₀, T::Float64, P::Float64, externalV::Array{Float64,4}, qdht::QDHT{0,2,Float64}, ĉ₂vec::Vector{Float64}, _mat::AbstractMatrix{Float64})
    c₂r = SemiTruncatedInterpolator(qdht, ĉ₂vec)
    LinearMolecule(gasname_or_ρ₀, T, P, externalV, c₂r, _mat)
end

function (lm::LinearMolecule)(_, flat_∂ψ, flat_ψ::Vector{Float64})
    a0, a1, a2, a3 = size(lm.externalV)
    ρ₀π = lm.ρ₀ / (4π)
    ψ = reshape(flat_ψ, a0, a1, a2, a3)
    ρ_average = Array{Float64}(undef, a1, a2, a3)
    ρ = isnothing(flat_∂ψ) ? similar(ψ) : reshape(flat_∂ψ, size(lm.externalV))
    logρmρ₀ = similar(ψ)
    Fext_contrib = Vector{Float64}(undef, a3)
    Fid_contrib = Vector{Float64}(undef, a3)
    @threads for i3 in 1:a3
        fext_c = 0.0
        fid_c = 0.0
        for i2 in 1:a2, i1 in 1:a1
            tot = 0.0
            for i0 in 1:a0
                ψ2 = ψ[i0,i1,i2,i3]^2
                wψ2 = lm.lebedev_weights[i0]*ψ2
                tot += wψ2
                fext_c += wψ2*lm.externalV[i0,i1,i2,i3]
                logrmr₀ = max(log(ψ2), -1.3407807929942596e154) # log(ρ / ρ₀)
                logρmρ₀[i0,i1,i2,i3] = logrmr₀
                fid_c += lm.lebedev_weights[i0]*(ψ2*(logrmr₀ - 1) + 1)
            end
            ρ_average[i1,i2,i3] = tot*ρ₀π
        end
        Fext_contrib[i3] = fext_c
        Fid_contrib[i3] = fid_c
    end
    Fext = ρ₀π*sum(Fext_contrib)
    Fid = lm.T*ρ₀π*sum(Fid_contrib)
    # @show Fext, Fid

    # value = ma.igp.δv*sum(ρr*v + ma.igp.T*(ρr*logrmr₀ + (ma.igp.ρ₀-ρr) - CΔρ*(ρr-ma.igp.ρ₀)/2)
    # for (logrmr₀, v, ρr, CΔρ) in zip(logρmρ₀, ma.igp.externalV, ρ, convol))

    convol = ρ_average .- lm.ρ₀
    rfftΔρ = lm.plan * convol
    rfftΔρ .*= lm.ĉ₂
    FFTW.ldiv!(convol, lm.plan, rfftΔρ)

    # ρ .= ρ₀π .* ψ.^2
    # logρmρ₀ = max.(log.(ρ ./ ρ₀π), -1.3407807929942596e154)
    # Fid = lm.T*sum(ρr*(logr - 1) + ρ₀π for (ρr, logr) in zip(ρ, logρmρ₀))
    Fexc = -lm.T*sum(CΔρ*(ρ_ave-lm.ρ₀) for (CΔρ, ρ_ave) in zip(convol, ρ_average))/2
    value = (Fid + Fext + Fexc)*lm.δv
    if !isnothing(flat_∂ψ) # gradient update
        @. ρ = $(2*ρ₀π*lm.δv)*ψ
        @threads for i0 in 1:a0
            @. ρ[i0,:,:,:] *= lm.lebedev_weights[i0]*(lm.externalV[i0,:,:,:] + lm.T*logρmρ₀[i0,:,:,:]) - (4π*lm.T)*convol
        end
        # @. ρ = ifelse(abs(ρ) > 1.3407807929942596e100, 0.0, ρ)
        @show value, maximum(ρ), norm(vec(ρ))
    end
    value
end

function mdft(lm::LinearMolecule, ψ_init=exp.(.-vec(lm.externalV)./(2*lm.T)))
    Optim.optimize(Optim.only_fg!(lm), ψ_init, Optim.LBFGS(linesearch=Optim.BackTracking(order=2, ρ_hi=0.1)),
                   Optim.Options(iterations=1000))
end

function finaldensity(lm::LinearMolecule, opt)
    if !Optim.converged(opt)
        @error "Optimizer failed to converge; proceeeding with partial result"
    end
    ψ = Optim.minimizer(opt)
    a0, a1, a2, a3 = size(lm.externalV)
    ρ = reshape(lm.ρ₀ .* ψ.^2, a0, a1, a2, a3)
    lm.δv*sum(lebedev_average(@view(ρ[:,i1,i2,i3]), lm.lebedev_weights) for i1 in 1:a1, i2 in 1:a2, i3 in 1:a3)
end



function __inter(i, a, b)
    m = a÷b
    (1+(i-1)*m):(i == b ? a : i*m)
end

function iterative_mdft(ma::MonoAtomic)
    rψ_init = exp.(.-ma.igp.externalV./(2*ma.igp.T))
    a1, a2, a3 = size(ma.igp.externalV)
    b1, b2, b3 = (a1, a2, a3) .÷ 20
    ψ = Vector{Float64}(undef, b1*b2*b3)
    rψ = reshape(ψ, b1, b2, b3)
    @threads for i3 in 1:b3
        for i2 in 1:b2, i1 in 1:b1
            rψ[i1,i2,i3] = mean(@view(rψ_init[__inter(i1, a1, b1), __inter(i2, a2, b2), __inter(i3, a3, b3)]))
        end
    end
    opt = nothing
    for M in (20, 15, 11, 8, 5, 3, 1)
        @show M
        c1, c2, c3 = size(rψ)
        (b1, b2, b3) = (a1, a2, a3) .÷ M
        newψ = Vector{Float64}(undef, b1*b2*b3)
        newrψ = reshape(newψ, b1, b2, b3)
        @threads for i3 in 1:c3
            for i2 in 1:c2, i1 in 1:c1
                J1 = __inter(i1, b1, c1)
                J2 = __inter(i2, b2, c2)
                J3 = __inter(i3, b3, c3)
                for j3 in J3, j2 in J2, j1 in J1
                    newrψ[j1,j2,j3] = rψ[i1,i2,i3]
                end
            end
        end
        newexternalV = Array{Float64}(undef, b1, b2, b3)
        @threads for i3 in 1:b3
            for i2 in 1:b2, i1 in 1:b1
                newexternalV[i1,i2,i3] = mean(@view(ma.igp.externalV[__inter(i1, a1, b1), __inter(i2, a2, b2), __inter(i3, a3, b3)]))
            end
        end
        newma = MonoAtomic(ma.igp.ρ₀, ma.igp.T, ma.igp.P, newexternalV, ma.c₂r, ma.igp.mat)
        ψ, newψ = newψ, ψ
        opt = Optim.optimize(Optim.only_fg!(newma), ψ, Optim.LBFGS(linesearch=Optim.BackTracking(order=2, ρ_hi=0.1)),
                   Optim.Options(iterations=10000, f_tol=1e-10))
        display(opt)
        @show finaldensity(newma, opt)
    end
    opt
end





function compute_store_hnc!(potential, ffname, T, ρ₀, qdht)
    sha = bytes2hex(sha256(reinterpret(UInt8, potential)))
    store = "$ffname-$sha-$T-$ρ₀-$(qdht.R)-$(qdht.N)"
    global scratchspace
    path = joinpath(scratchspace, store)
    if isfile(path)
        stored = deserialize(path)
        result = hnc(potential, qdht.R, T, ρ₀; qdht, γ0=copy(stored[2]))
        if result != stored
            @warn "Previously stored hnc computation stale for FF $ffname (T = $T, ρ₀ = $ρ₀, potential $(sha[1:8])..., QDHT($(qdht.R), $(qdht.N))); refreshing."
            serialize(path, result)
        end
        return result
    end
    @info "Generating hnc correlation functions for FF $ffname (T = $T, ρ₀ = $ρ₀, potential $(sha[1:8])..., QDHT($(qdht.R), $(qdht.N)))."
    ret = hnc(potential, qdht.R, T, ρ₀; qdht)
    serialize(path, ret)
    return ret
end


function _isotherm_hnc_igp(_, _, egrid::Array{Float64,3}, temperature, pressures, mat, molname, _)
    m = length(pressures)
    isotherm = Vector{Float64}(undef, m)
    opts = Vector{Vector{Float64}}(undef, m)
    for i in 1:m
        P = pressures[i]
        system = IdealGasProblem(molname, temperature, P, egrid, mat)
        opt = mdft(system)
        opts[i] = opt
        isotherm[i] = finaldensity(system, opt)
        @show P, isotherm[i]
    end
    isotherm, opts
end

function _isotherm_hnc(ff::CEG.ForceField, mol::AbstractSystem, egrid::Array{Float64,N}, temperature, pressures, mat,
                      molname::AbstractString, qdht::QDHT{0,2}) where N
    m = length(pressures)
    if iszero(qdht.R)
        ck_hnc = [Float64[] for _ in 1:m]
    else
        potential = compute_average_self_potential(mol, ff, qdht.r)
        ck_hnc = Vector{Vector{Float64}}(undef, m)
        @threads for i in 1:m
            P = pressures[i]
            ρ₀ = compute_ρ₀(molname, temperature, P)
            c, _ = compute_store_hnc!(potential, ff.name, temperature, ρ₀, qdht)
            ck_hnc[i] = qdht * c
        end
    end
    isotherm = Vector{Float64}(undef, m)
    opts = Vector{Any}(undef, m)
    for i in 1:m
        ck = ck_hnc[i]
        P = pressures[i]
        system = if iszero(qdht.R)
            IdealGasProblem(molname, temperature, P, egrid, mat)
        else
            (N == 3 ? MonoAtomic : LinearMolecule)(molname, temperature, P, egrid, qdht, ck, mat)
        end
        opt = mdft(system)
        opts[i] = opt
        isotherm[i] = finaldensity(system, opt)
        @show P, isotherm[i]
        display(opt)
    end
    isotherm, opts
end

function isotherm_hnc(ff::CEG.ForceField, mol::AbstractSystem, egrid::Array{Float64,N}, temperature, pressures, mat;
                      molname=identify_molecule(atomic_symbol(mol)), qdht=QDHT{0,2}(100, 10000)) where N
    fun = iszero(qdht.R) ? _isotherm_hnc_igp : _isotherm_hnc
    if size(egrid, 1) == 1
        fun(ff, mol, dropdims(egrid; dims=1), temperature, pressures, mat, molname, qdht)
    else
        fun(ff, mol, egrid, temperature, pressures, mat, molname, qdht)
    end
end
