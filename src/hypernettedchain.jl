using Hankel: QDHT
using LinearAlgebra: norm, mul!, ldiv!


"""
    hnc(v::AbstractVector{<:AbstractFloat}, R, T, ρ, ε=1e-8; qdht=QDHT{0,2}(R,length(v)), γ0=nothing)

Takes the interaction potential `v` between two particles on a regularly spaced line
between distances `0` and `R` excluded, i.e. on the points given by
`LinRange(0, R, length(v)+2)[2:end-1]`, expressed in Kelvin.

Returns the pair `(c, γ)` where `c` is the direct correlation function and `c+γ` is the total
correlation function. Both functions are computed on the same points, under the
hypernetted chain approximation, at temperature `T` and density `ρ`, with precision `ε`.

A Hankel transform plan `qdht` may be optionally provided to avoid recomputing it each time.
An estimate `γ0` may also be provided for the initial value of `γ`.

Implements Picard's strategy with a custom adaptative step size.
"""
function hnc(v::AbstractVector{S}, R, T, ρ, ε=1e-8; qdht=QDHT{0,2}(R,length(v)), γ0=nothing) where {S}
    Ϡ = ((2π)^(3/2))
    γ = if γ0 isa Nothing
        zeros(S, length(v))
    else
        copy(γ0)
    end
    γ_min = copy(γ)

    @assert qdht.r ≈ LinRange(0, R, length(v)+2)[2:end-1]
    τ = 1.0

    c = @. expm1(-v/T + γ) - γ
    ck = qdht*c; ck .*= Ϡ
    γ2k = @. ρ*ck*ck/(1 - ρ*ck)
    γ2 = qdht\γ2k; γ2 ./= Ϡ
    η = rmsd(γ, γ2)
    η_min = η
    !(η ≤ ε) && isfinite(η) && (γ .= γ2)

    retries = 0
    global_iter = 0
    outer_iter = floor(Int, max(0, -log(η))*73)
    iter_since_min = 0
    while !(η ≤ ε) # does not stop on NaN
        iter_since_min += 1
        global_iter += 1

        c = @. expm1(-v/T + γ) - γ
        mul!(ck, qdht, c); ck .*= Ϡ
        γ2k = @. ρ*ck*ck/(1 - ρ*ck)
        ldiv!(γ2, qdht, γ2k); γ2 ./= Ϡ
        η = rmsd(γ, γ2)
        if η < η_min
            γ_min .= γ
            η_min = η
            iter_since_min = 0
            outer_iter += 1
            τ = min(1.0, τ*(1+1/outer_iter))
        elseif iter_since_min > 300 || !(η < 20*η_min) # accepts if η == NaN or η == η_min == Inf
            γ .= γ_min
            iter_since_min = -1
            τ = (1 - 1/(2+sqrt(outer_iter*(1+retries) + max(0.0, -log(η_min)))))*τ
            if τ < 0.05
                if retries >= 2
                    error("Failed to find appropriate τ")
                else
                    retries += 1
                    τ = 1.0
                end
            end
            continue
        end
        λ = (1 - 1/(1+iter_since_min))*τ
        @. γ = λ*γ2 + $(1-λ)*γ

        # global_iter % 300 == 0 && @show η, τ, outer_iter
    end
    # @show global_iter
    c, γ
end


#=
function scalenorm(x::Vector, R)
    ret = 0.0
    l = LinRange(0, R, length(x)+2)[2:end-1]
    for (i, r) in enumerate(l)
        ret += (x[i]*r)^2
    end
    ret*step(l)
end

function scaledot(x::Vector, y::Vector, R)
    ret = 0.0
    l = LinRange(0, R, length(x)+2)[2:end-1]
    @assert eachindex(x) == eachindex(y)
    for (i, r) in enumerate(l)
        ret += x[i]*y[i]*r^2
    end
    ret*step(l)
end

# Implements the strategy of [Luc Belloni, J. Chem. Phys., 98 (10), 1993](https://doi.org/10.1063/1.464564),
# paragraph II. B., see [Gilles Zerah, J. Comp. Phys., 61 (2), 1985](https://doi.org/10.1016/0021-9991(85)90087-7).
function hnc_newton(v::AbstractVector{S}, R, T, ρ, ε=1e-12; qdht=QDHT{0,2}(R,length(v))) where {S}
    n = length(v)
    γ = zeros(S, n)
    h = similar(γ)
    Ϡ = ((2π)^(3/2))
    @assert qdht.r ≈ LinRange(0, R, n+2)[2:end-1]
    γk = qdht*γ; γk .*= Ϡ

    r = similar(γ)
    ar = similar(γ)
    old_δγk = similar(γ)

    c = @. expm1(-v/T + γ) - γ
    ck = qdht*c; ck .*= Ϡ
    γk_verif = @. ck/(1 - ρ*ck) - ck
    γ_verif = (qdht\γk_verif); γ_verif ./= Ϡ
    # δγ = γ_verif .- γ
    # δγk = γk_verif .- γk
    # @assert δγk ≈ qdht*δγ .* Ϡ
    δγ = zeros(S, n)
    δγk = zeros(S, n)

    temp = similar(γ)
    hδγk = similar(γ)
    ργk2ck = similar(γ)
    mρck = similar(γ)
    precond = ones(S, n)
    f3k = similar(γ)

    η = 1.0
    while η > ε
        @show η
        old_α = 1.0
        old_normr = 1.0
        old_w = 1.0
        δγk .= 0.0
        δγ .= 0.0
        old_δγk .= 0.0

        @. h = expm1(-v/T + γ)
        @. ργk2ck = ρ*(γk + 2*ck)
        @. temp = h - c - γ # f1
        mul!(f3k, qdht, temp)
        # @. precond = inv(1 - ρ*ck)
        @. f3k = precond*(ργk2ck*f3k*Ϡ + (ρ*(γk+ck)*ck - γk))
        @. mρck = 1 - ρ*ck

        while old_normr > 2ε
            println("old norm: $old_normr, δγk[1:3]: $(δγk[1:3])")
            @. temp = h*δγ
            mul!(hδγk, qdht, temp); hδγk .*= Ϡ
            @. r = precond*(mρck*δγk - ργk2ck*hδγk) - f3k
            @. ar = -ργk2ck*r
            ldiv!(temp, qdht, ar)
            temp .*= h/Ϡ
            mul!(ar, qdht, temp)
            @. ar = precond*(mρck*r + ar*Ϡ)
            normr = scalenorm(r, qdht.K)
            α = normr/scalenorm(ar, qdht.K)
            @show old_α*old_w*old_normr
            w = inv(1 - α*normr/(old_α*old_w*old_normr))
            @. temp = old_δγk + w*(α*ar + δγk - old_δγk)

            old_δγk = δγk
            δγk = temp
            ldiv!(δγ, qdht, δγk); δγ ./= Ϡ

            old_normr = normr
            old_w = w
            old_α = α

            # modelX = inv.(1:n)
            # modelY = sqrt.(1:n)
            # AY = mρck .* modelY .- ργk2ck.*(qdht*(h.*(qdht\modelY)./Ϡ).*Ϡ)
            # ÂX = mρck .* modelX .+ qdht*(h.*(qdht\(.-ργk2ck.*modelX)./Ϡ)).*Ϡ
            # @show scaledot(modelX, modelY, R), scaledot(ÂX, modelY, R) ≈ scaledot(AY, modelX, R)
            # AY = .-ργk2ck.*(qdht*(h.*(qdht\modelY)./Ϡ).*Ϡ)
            # ÂX = qdht*(h.*(qdht\(.-ργk2ck.*modelX)./Ϡ)).*Ϡ
            # @show scaledot(modelX, modelY, R), scaledot(ÂX, modelY, R), scaledot(AY, modelX, R)
        end
        γ .+= δγ
        γk .+= δγk
        # @assert γk ≈ (qdht*γ) .* Ϡ
        if all(isfinite, γ)
            println("SUCCESS!")
        else
            println("Failure... $(count(!isfinite, γ))")
        end
        break

        @. c = h*(1 + δγ) - γ + δγ
        # @. c = expm1(-v/T + γ) - γ
        mul!(ck, qdht, c); ck .*= Ϡ
        @. γk_verif = ck/(1 - ρ*ck) - ck
        ldiv!(γ_verif, qdht, γk_verif); γ_verif ./= Ϡ
        η = rmsd(γ, γ_verif)
    end
    c, γ
end
=#

#=
function hnc_naive_newton(v::AbstractVector{S}, R, T, ρ, ε=1e-12; qdht=QDHT{0,2}(R,length(v)), γ0=.-ones(S,length(v))) where {S}
    γ = copy(γ0)
    γk = zeros(S, length(v))

    @assert qdht.r ≈ LinRange(0, R, length(v)+2)[2:end-1]
    Ϡ = ((2π)^(3/2))

    c = @. expm1(-v/T + γ) - γ
    ck = qdht*c; ck .*= Ϡ
    γk_verif = @. ck/(1 - ρ*ck) - ck
    γ_verif = qdht\γk_verif; γ_verif ./= Ϡ
    η = rmsd(γ, γ_verif)


    expvtm1 = similar(γ)
    f1 = similar(γ)
    ργpck = similar(γ)
    f2 = similar(γ)
    mρck = similar(γ)

    δc = zeros(S, length(v))
    δγ = zeros(S, length(v))
    δck = similar(γ)
    δγk = similar(γ)
    old_δγ = similar(γ)
    old_δc = similar(γ)

    outer_iter = 0
    while η > ε
        outer_iter += 1

        @. expvtm1 = expm1(-v/T + γ)
        @. f1 = expvtm1 - c - γ
        @. ργpck = ρ*(γk + ck)
        @. f2 = ργpck*ck - γk
        @. mρck = 1 - ρ*ck

        # @. γk_verif = ck/(1 - ρ*ck) - ck
        # ldiv!(γ_verif, qdht, γk_verif); γ_verif ./= Ϡ
        # @. δγ = γ - γ_verif
        @. δγ = 0.0

        nδ = 1.0
        nδ_min = Inf
        inner_iter = 0
        while nδ > ε && isfinite(nδ)
            inner_iter += 1
            old_δγ .= δγ
            old_δc .= δc
            # τ = 1 - inv(inner_iter+1)
            τ = 1

            @. δc = old_δc*$(1-τ) + (expvtm1*δγ + f1)*τ
            mul!(δck, qdht, δc); δck .*= Ϡ
            @. δγk = (f2 + ργpck*δck + ρ*ck*δck)/mρck
            ldiv!(δγ, qdht, δγk); δγ ./= Ϡ
            nδ = rmsd(δγ, old_δγ) + rmsd(δc, old_δc)
            nδ_min = min(nδ, nδ_min)
            if !isfinite(nδ) || nδ > 1000*nδ_min
                # @. c = (c + expm1(-v/T + γ) - γ)*$(1 - 1/outer_iter)
                # mul!(ck, qdht, c); ck .*= Ϡ
                # @. γk_verif = ck/(1 - ρ*ck) - ck
                # ldiv!(γ_verif, qdht, γk_verif); γ_verif ./= Ϡ
                # @. δγ = (γk_verif - γ)*$(1 - 1/outer_iter)
                # δc .= 0.0
                error("Inner loop failed to converge")
            end

            @show nδ
        end

        γ .+= δγ
        c .+= δc
        γk .+= δγk
        ck .+= δck

        @. γk_verif = ck/(1 - ρ*ck) - ck
        ldiv!(γ_verif, qdht, γk_verif); γ_verif ./= Ϡ
        η = rmsd(γ, γ_verif)

        @show η
    end
    c, γ
end
=#
