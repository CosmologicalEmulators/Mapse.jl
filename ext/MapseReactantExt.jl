module MapseReactantExt

using Mapse
using Reactant

import Mapse:
    HalofitCosmology,
    _HALOFIT_NEWTON_STEPS,
    _halofit_integrate,
    _halofit_Pmm_one_z_unchecked,
    _halofit_Pmm_unchecked,
    halofit_Pmm

const TracedVec = Reactant.TracedRArray{T,1} where {T}
const TracedMat = Reactant.TracedRArray{T,2} where {T}
const ConcreteVec = Reactant.ConcretePJRTArray{T,1} where {T}
const ConcreteMat = Reactant.ConcretePJRTArray{T,2} where {T}
const ReactantVec = Union{TracedVec,ConcreteVec}
const ReactantMat = Union{TracedMat,ConcreteMat}

# `HalofitCosmology` is scalar configuration for the compiled kernel. Treating
# it as static avoids trying to manufacture tracers for a small immutable struct
# while all large numerical payloads stay explicit Reactant array arguments.
Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:HalofitCosmology}),
    seen,
    mode::Reactant.TraceMode,
    @nospecialize(track_numbers::Type),
    @nospecialize(ndevices),
    @nospecialize(runtime),
)
    return T
end

Base.@nospecializeinfer function Reactant.make_tracer(
    seen,
    @nospecialize(prev::HalofitCosmology),
    @nospecialize(path),
    mode;
    kwargs...,
)
    return prev
end

function _halofit_integrate(x::ReactantVec, y::ReactantVec)
    n = length(x)
    last_simpson = isodd(n) ? n : n - 1

    i0 = 1:2:(last_simpson - 2)
    i1 = 2:2:(last_simpson - 1)
    i2 = 3:2:last_simpson

    x0 = x[i0]
    x1 = x[i1]
    x2 = x[i2]
    y0 = y[i0]
    y1 = y[i1]
    y2 = y[i2]

    h0 = x1 .- x0
    h1 = x2 .- x1
    simpson_terms = (h0 .+ h1) ./ 6 .* (
        (2 .- h1 ./ h0) .* y0 .+
        ((h0 .+ h1) .^ 2 ./ (h0 .* h1)) .* y1 .+
        (2 .- h0 ./ h1) .* y2
    )
    total = sum(simpson_terms)

    if iseven(n)
        xn = x[n:n]
        xprev = x[(n - 1):(n - 1)]
        yn = y[n:n]
        yprev = y[(n - 1):(n - 1)]
        total += sum((xn .- xprev) .* (yn .+ yprev) ./ 2)
    end

    return total
end

function _halofit_integrate_columns(x::ReactantVec, y::ReactantMat)
    n = length(x)
    last_simpson = isodd(n) ? n : n - 1

    i0 = 1:2:(last_simpson - 2)
    i1 = 2:2:(last_simpson - 1)
    i2 = 3:2:last_simpson

    x0 = reshape(x[i0], :, 1)
    x1 = reshape(x[i1], :, 1)
    x2 = reshape(x[i2], :, 1)
    y0 = y[i0, :]
    y1 = y[i1, :]
    y2 = y[i2, :]

    h0 = x1 .- x0
    h1 = x2 .- x1
    simpson_terms = (h0 .+ h1) ./ 6 .* (
        (2 .- h1 ./ h0) .* y0 .+
        ((h0 .+ h1) .^ 2 ./ (h0 .* h1)) .* y1 .+
        (2 .- h0 ./ h1) .* y2
    )
    total = sum(simpson_terms; dims=1)

    if iseven(n)
        xn = reshape(x[n:n], 1, 1)
        xprev = reshape(x[(n - 1):(n - 1)], 1, 1)
        yn = y[n:n, :]
        yprev = y[(n - 1):(n - 1), :]
        total = total .+ (xn .- xprev) .* (yn .+ yprev) ./ 2
    end

    return total
end

function _halofit_σ2_derivs_columns(logk::ReactantVec, k::ReactantVec,
    pk_lin_mm_z::ReactantMat, R_z)

    k_col = reshape(k, :, 1)
    kR2 = (k_col .* R_z) .^ 2
    exp_term = exp.(-kR2)
    integrand_pre = (k_col .^ 3) .* pk_lin_mm_z ./ (2π^2)

    sig2 = _halofit_integrate_columns(logk, integrand_pre .* exp_term)
    dsig2dR = -2 .* R_z .* _halofit_integrate_columns(
        logk, integrand_pre .* (k_col .^ 2) .* exp_term
    )
    d1 = dsig2dR .* R_z ./ sig2

    d2sig2dR2 = _halofit_integrate_columns(
        logk, integrand_pre .* (k_col .^ 2) .* exp_term .* (-2 .+ 4 .* kR2)
    )
    d2 = (R_z .^ 2 ./ sig2) .* d2sig2dR2 .+ d1 .- d1 .^ 2

    return sig2, d1, d2
end

function _halofit_rnl_columns(logk::ReactantVec, k::ReactantVec,
    pk_lin_mm_z::ReactantMat)

    lR = sum(pk_lin_mm_z .* 0; dims=1)
    for _ in 1:_HALOFIT_NEWTON_STEPS
        sig2, d1, _ = _halofit_σ2_derivs_columns(logk, k, pk_lin_mm_z, exp.(lR))
        lR = lR .- log.(sig2) ./ d1
    end

    return exp.(lR)
end

function _halofit_power_columns(cpar::HalofitCosmology, pk_lin_mm_z::ReactantMat,
    k::ReactantVec, z::ReactantVec, rnl, neff, cur, Ωm_z::ReactantVec,
    Ωv_z::ReactantVec)

    T = promote_type(eltype(k), eltype(pk_lin_mm_z), eltype(z), eltype(Ωm_z),
                     eltype(Ωv_z))
    k_col = reshape(k, :, 1)
    z_row = reshape(z, 1, :)
    Ωmz = reshape(Ωm_z, 1, :)
    Ωvz = reshape(Ωv_z, 1, :)

    anorm = inv(T(2) * T(π)^2)
    opz = one(T) .+ z_row
    wz = T(cpar.w0) .+ T(cpar.wa) .* (z_row ./ opz)
    fν = T(cpar.Ωnu0) / T(cpar.Ωm0)
    y = k_col .* rnl

    gam = T(0.1971) .- T(0.0843) .* neff .+ T(0.8460) .* cur
    an = T(10) .^ (T(1.5222) .+ T(2.8553).*neff .+ T(2.3706).*neff.^2 .+
                   T(0.9903).*neff.^3 .+ T(0.2250).*neff.^4 .- T(0.6038).*cur .+
                   T(0.1749).*Ωvz.*(one(T) .+ wz))
    bn = T(10) .^ (-T(0.5642) .+ T(0.5864).*neff .+ T(0.5716).*neff.^2 .-
                   T(1.5474).*cur .+ T(0.2279).*Ωvz.*(one(T) .+ wz))
    cn = T(10) .^ (T(0.3698) .+ T(2.0404).*neff .+ T(0.8161).*neff.^2 .+
                   T(0.5869).*cur)
    xμ = zero(T)
    xν = T(10) .^ (T(5.2105) .+ T(3.6902).*neff)
    α = abs.(T(6.0835) .+ T(1.3373).*neff .- T(0.1959).*neff.^2 .-
             T(5.5274).*cur)
    β = T(2.0379) .- T(0.7354).*neff .+ T(0.3157).*neff.^2 .+
        T(1.2490).*neff.^3 .+ T(0.3980).*neff.^4 .- T(0.1682).*cur .+
        fν .* (T(1.081) .+ T(0.395).*neff.^2)

    use_de = abs.(one(T) .- Ωmz) .> T(0.01)
    denom = ifelse.(use_de, one(T) .- Ωmz, one(T))
    frac = Ωvz ./ denom
    f1a, f2a, f3a = Ωmz .^ (-T(0.0732)), Ωmz .^ (-T(0.1423)), Ωmz .^ T(0.0725)
    f1b, f2b, f3b = Ωmz .^ (-T(0.0307)), Ωmz .^ (-T(0.0585)), Ωmz .^ T(0.0743)
    f1_de = frac .* f1b .+ (one(T) .- frac) .* f1a
    f2_de = frac .* f2b .+ (one(T) .- frac) .* f2a
    f3_de = frac .* f3b .+ (one(T) .- frac) .* f3a
    f1 = ifelse.(use_de, f1_de, one(T))
    f2 = ifelse.(use_de, f2_de, one(T))
    f3 = ifelse.(use_de, f3_de, one(T))

    pk_halo_raw = an .* y .^ (f1 .* T(3)) ./
                  (one(T) .+ bn .* y .^ f2 .+ (f3 .* cn .* y) .^ (T(3) .- gam))
    pk_halo_dim = pk_halo_raw ./ (one(T) .+ xμ ./ y .+ xν ./ y .^ 2)
    pk_halo_dim = pk_halo_dim .* (one(T) + fν * T(0.977))

    Δ2_lin = (k_col .^ 3) .* pk_lin_mm_z .* anorm
    kh = k_col ./ T(cpar.h)
    Δ2_linaa = Δ2_lin .* (one(T) .+ fν .* T(47.48) .* kh .^ 2 ./
                          (one(T) .+ T(1.5) .* kh .^ 2))
    pk_quasi_dim = Δ2_lin .* (one(T) .+ Δ2_linaa) .^ β ./
                   (one(T) .+ Δ2_linaa .* α) .* exp.(-y ./ T(4) .- y .^ 2 ./ T(8))

    return (pk_halo_dim .+ pk_quasi_dim) ./ (k_col .^ 3 .* anorm)
end

function _halofit_Pmm_unchecked(cpar::HalofitCosmology, z::ReactantVec,
    k::ReactantVec, pk_lin_mm_z::ReactantMat, Ωm_z::ReactantVec,
    Ωv_z::ReactantVec)

    logk = log.(k)
    rnl = _halofit_rnl_columns(logk, k, pk_lin_mm_z)
    _, d1, d2 = _halofit_σ2_derivs_columns(logk, k, pk_lin_mm_z, rnl)
    neff = -3 .- d1
    cur = -d2
    return _halofit_power_columns(cpar, pk_lin_mm_z, k, z, rnl, neff, cur, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::ReactantVec, k::ReactantVec,
    pk_lin_mm_z::ReactantMat, Ωm_z::ReactantVec, Ωv_z::ReactantVec)

    return _halofit_Pmm_unchecked(cpar, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::Number, k::ReactantVec,
    pk_lin_mm::ReactantVec, Ωm_z::Number, Ωv_z::Number)

    return _halofit_Pmm_one_z_unchecked(cpar, z, k, pk_lin_mm, Ωm_z, Ωv_z)
end

end # module MapseReactantExt
