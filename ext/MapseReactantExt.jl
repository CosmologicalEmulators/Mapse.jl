module MapseReactantExt

using Mapse
using Reactant
import AbstractCosmologicalEmulators

function Mapse.AbstractCosmologicalEmulators.to_reactant(emu::Mapse.TransferFunctionEmulator)
    compression = if emu.Compression isa Mapse.PCACompression
        Mapse.PCACompression(
            mean=Reactant.to_rarray(emu.Compression.mean),
            basis=Reactant.to_rarray(emu.Compression.basis),
        )
    else
        emu.Compression
    end
    return Mapse.TransferFunctionEmulator(
        TrainedEmulator=Mapse.AbstractCosmologicalEmulators.to_reactant(emu.TrainedEmulator),
        kgrid=Reactant.to_rarray(emu.kgrid),
        InMinMax=Reactant.to_rarray(emu.InMinMax),
        OutMinMax=Reactant.to_rarray(emu.OutMinMax),
        Preprocessing=emu.Preprocessing,
        Postprocessing=emu.Postprocessing,
        Compression=compression,
    )
end

import Mapse:
    HalofitCosmology,
    HMCodeCosmology,
    _HALOFIT_NEWTON_STEPS,
    _halofit_integrate,
    _halofit_σ2_derivs,
    _halofit_rnl,
    _halofit_Pmm_one_z_unchecked,
    _halofit_Pmm_unchecked,
    _validate_halofit_inputs,
    _validate_halofit_background,
    _hmcode_choose_cb,
    halofit_Pmm,
    hmcode_Pmm,
    hmcode_pmm_fast,
    hmcode_pmm_fast_two_splines,
    hmcode_boost,
    predict_baryonic_discontinuity,
    build_smart_coarse_grid,
    build_piecewise_coarse_grid,
    hmcode_pmm_baryonic_smart

const TracedVec = Reactant.TracedRArray{T,1} where {T}
const TracedMat = Reactant.TracedRArray{T,2} where {T}
const ConcreteVec = Reactant.ConcretePJRTArray{T,1} where {T}
const ConcreteMat = Reactant.ConcretePJRTArray{T,2} where {T}
const ReactantVec = Union{TracedVec,ConcreteVec}
const ReactantMat = Union{TracedMat,ConcreteMat}

# Inputs are validated on the host before conversion to device arrays. These
# methods prevent traced comparisons from entering the host-only validators.
Mapse._validate_halofit_inputs(::ReactantVec, ::ReactantVec, ::ReactantMat) = nothing
Mapse._validate_halofit_inputs(::ReactantVec, ::ReactantVec, ::AbstractMatrix) = nothing
Mapse._validate_halofit_background(::ReactantVec, ::ReactantVec, ::ReactantVec) = nothing
Mapse._validate_halofit_background(::ReactantVec, ::ReactantVec, ::AbstractVector) = nothing

function Mapse.lcdm_transfer_function(params::ReactantVec, k::ReactantVec)
    omega_b = sum(params[4:4])
    omega_c = sum(params[5:5])
    Mnu = sum(params[6:6])
    log10_k = log10.(k)
    omega_nu = Mnu / 93.14
    delta_omega = omega_c + omega_nu - omega_b
    omega_m = omega_b + omega_c + omega_nu
    return exp.(0.4971733969600907 .+ (-24.849067935704547 .- log.((((((0.731102574104348 .^ log10_k) .+ delta_omega) ./ 0.17522861267519874) .^ log10_k) .+ ((63.65597287231169 .^ (log10_k .+ 0.0472474783701488)) .* ((0.9899093975978591 .^ (log10_k ./ (cos.(log10_k ./ ((1.1964213875807956 ^ -2.3661897652294015) ./ cos.(log10_k ./ -1.8173117588773222))) ./ 0.20037856443385513))) ./ (delta_omega ^ 0.7767030041348179)))) .+ (0.14823981687164764 * omega_m))))
end

function Mapse.postprocessing_lcdm_transfer_ratio(params::ReactantVec,
                                                  output, D,
                                                  emu::Mapse.TransferFunctionEmulator)
    ln10As = sum(params[1:1])
    ns = sum(params[2:2])
    As = exp(ln10As) * 1e-10
    k = Mapse.get_kgrid(emu)
    P_prim = Mapse.primordial_Pk(As, ns, k)
    transfer = Mapse.lcdm_transfer_function(params, k)
    D2 = reshape(D .^ 2, 1, :)
    return (output .* transfer .^ 2) .* D2 .* P_prim
end

# `HalofitCosmology` is scalar configuration for the compiled kernel. Treating
# it as static avoids trying to manufacture tracers for a small immutable struct
# while all large numerical payloads stay explicit Reactant array arguments.
const StaticReactantCosmology = Union{HalofitCosmology,HMCodeCosmology}

Base.@nospecializeinfer function Reactant.traced_type_inner(
    @nospecialize(T::Type{<:StaticReactantCosmology}),
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
    @nospecialize(prev::StaticReactantCosmology),
    @nospecialize(path),
    mode;
    kwargs...,
)
    return prev
end
function _halofit_r_integrate(x::AbstractVector, y::AbstractVector)
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

function _halofit_integrate_columns(x::AbstractVector, y::AbstractMatrix)
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

function _halofit_σ2_derivs_columns(logk::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix, R_z)

    k_col = reshape(k, :, 1)
    kR2 = (k_col .* R_z) .^ 2
    exp_term = exp.(-kR2)
    integrand_pre = (k_col .^ 3) .* pk_lin_mm_z ./ (2π^2)

    sig2 = _halofit_integrate_columns(logk, integrand_pre .* exp_term)
    dsig2dR = -2 .* R_z .* _halofit_integrate_columns(
        logk, integrand_pre .* (k_col .^ 2) .* exp_term
    )
    is_sig_invalid = (sig2 .<= zero(eltype(sig2))) .| (sig2 .!= sig2) .| (sig2 .== eltype(sig2)(Inf))
    d1 = ifelse.(is_sig_invalid, eltype(sig2)(NaN), dsig2dR .* R_z ./ sig2)

    d2sig2dR2 = _halofit_integrate_columns(
        logk, integrand_pre .* (k_col .^ 2) .* exp_term .* (-2 .+ 4 .* kR2)
    )
    d2 = ifelse.(is_sig_invalid, eltype(sig2)(NaN), (R_z .^ 2 ./ sig2) .* d2sig2dR2 .+ d1 .- d1 .^ 2)

    return sig2, d1, d2
end

function _halofit_rnl_columns(logk::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix)

    lR = sum(pk_lin_mm_z .* 0; dims=1)
    for _ in 1:_HALOFIT_NEWTON_STEPS
        sig2, d1, _ = _halofit_σ2_derivs_columns(logk, k, pk_lin_mm_z, exp.(lR))
        is_invalid = (sig2 .<= zero(eltype(sig2))) .| (sig2 .!= sig2) .| (sig2 .== eltype(sig2)(Inf)) .| (d1 .== zero(eltype(d1))) .| (d1 .!= d1) .| (d1 .== eltype(d1)(Inf)) .| (d1 .== eltype(d1)(-Inf))
        step_raw = ifelse.(is_invalid, eltype(sig2)(NaN), log.(sig2) ./ d1)
        # Guard against a finite but tiny d1 producing an Inf step.
        step = ifelse.(.!isfinite.(step_raw), eltype(sig2)(NaN), step_raw)
        lR = lR .- step
    end

    return exp.(lR)
end

function _halofit_power_columns(cpar::HalofitCosmology, pk_lin_mm_z::AbstractMatrix,
    k::AbstractVector, z::AbstractVector, rnl, neff, cur, Ωm_z::AbstractVector,
    Ωv_z::AbstractVector)

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

function _halofit_r_σ2_derivs(logk::AbstractVector, k::AbstractVector, pk_lin::AbstractVector, R)
    kR2 = (k .* R) .^ 2
    exp_term = exp.(-kR2)
    integrand_pre = (k .^ 3) .* pk_lin ./ (2π^2)

    sig2 = _halofit_r_integrate(logk, integrand_pre .* exp_term)
    dsig2dR = -2 * R * _halofit_r_integrate(logk, integrand_pre .* (k .^ 2) .* exp_term)
    is_sig_invalid = (sig2 .<= zero(eltype(sig2))) .| (sig2 .!= sig2) .| (sig2 .== eltype(sig2)(Inf))
    d1 = ifelse(is_sig_invalid, eltype(sig2)(NaN), dsig2dR * R / sig2)

    d2sig2dR2 = _halofit_r_integrate(logk,
        integrand_pre .* (k .^ 2) .* exp_term .* (-2 .+ 4 .* kR2))
    d2 = ifelse(is_sig_invalid, eltype(sig2)(NaN), (R^2 / sig2) * d2sig2dR2 + d1 - d1^2)

    return sig2, d1, d2
end

function _halofit_r_rnl(logk::AbstractVector, k::AbstractVector, pk_lin::AbstractVector)
    lR = zero(eltype(logk))
    for _ in 1:_HALOFIT_NEWTON_STEPS
        sig2, d1, _ = _halofit_r_σ2_derivs(logk, k, pk_lin, exp(lR))
        is_invalid = (sig2 .<= zero(sig2)) .| (sig2 .!= sig2) .| (sig2 .== eltype(sig2)(Inf)) .| (d1 .== zero(d1)) .| (d1 .!= d1) .| (d1 .== eltype(d1)(Inf)) .| (d1 .== eltype(d1)(-Inf))
        step_raw = ifelse(is_invalid, eltype(sig2)(NaN), log(sig2) / d1)
        # Guard against a finite but tiny d1 producing an Inf step.
        step = ifelse(!isfinite(step_raw), eltype(sig2)(NaN), step_raw)
        lR -= step
    end
    return exp(lR)
end

function _halofit_r_Pmm_unchecked(cpar::HalofitCosmology, z::AbstractVector,
    k::AbstractVector, pk_lin_mm_z::AbstractMatrix, Ωm_z::AbstractVector,
    Ωv_z::AbstractVector)

    logk = log.(k)
    rnl = _halofit_rnl_columns(logk, k, pk_lin_mm_z)
    _, d1, d2 = _halofit_σ2_derivs_columns(logk, k, pk_lin_mm_z, rnl)
    neff = -3 .- d1
    cur = -d2
    return _halofit_power_columns(cpar, pk_lin_mm_z, k, z, rnl, neff, cur, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::ReactantVec, k::ReactantVec,
    pk_lin_mm_z::AbstractMatrix, Ωm_z::ReactantVec, Ωv_z::ReactantVec)

    return _halofit_r_Pmm_unchecked(cpar, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::Number, k::ReactantVec,
    pk_lin_mm::ReactantVec, Ωm_z::Number, Ωv_z::Number)

    pk_mat = reshape(pk_lin_mm, :, 1)
    z_vec = k[1:1] .* 0 .+ Float64(z)
    Ωm_vec = k[1:1] .* 0 .+ Float64(Ωm_z)
    Ωv_vec = k[1:1] .* 0 .+ Float64(Ωv_z)
    out_mat = halofit_Pmm(cpar, z_vec, k, pk_mat, Ωm_vec, Ωv_vec)
    return vec(out_mat)
end






# -----------------------------------------------------------------------------
# Reactant-compatible HMCode2020
# -----------------------------------------------------------------------------

const _HMCODE_RHO_CRITICAL = 2.77536627245708e11
const _HMCODE_DV0 = 18.0 * pi^2
const _HMCODE_DC0 = (3.0 / 20.0) * (12.0 * pi)^(2.0 / 3.0)
const _HMCODE_ND = 2.853
const _HMCODE_ST_A = 0.2161599867112559
const _HMCODE_EULER_GAMMA = 0.5772156649015329

struct _HMCodeReactantParams{A1,A2,A3,A4,A5,A6,A7,A8,A9,A10,A11}
    R_nl::A1
    n_eff::A2
    sigma_v::A3
    Delta_v::A4
    delta_c::A5
    eta::A6
    A::A7
    f_damp::A8
    k_star::A9
    B::A10
    k_damp::A11
end

@inline _hmcode_density(Ωm) = _HMCODE_RHO_CRITICAL * Ωm
@inline _hmcode_lagrangian_radius(M, Ωm) = (3.0 .* M ./ (4.0 * pi * _hmcode_density(Ωm))) .^ (1.0 / 3.0)
@inline _hmcode_scalefactor(z) = 1.0 ./ (1.0 .+ z)
@inline _hmcode_redshift(a) = -1.0 .+ 1.0 ./ a

@inline function _hmcode_w(a, cosmo::HMCodeCosmology, lcdm::Bool)
    w0 = lcdm ? -1.0 : cosmo.w0
    wa = lcdm ? 0.0 : cosmo.wa
    return w0 .+ (1.0 .- a) .* wa
end

@inline function _hmcode_xw(a, cosmo::HMCodeCosmology, lcdm::Bool)
    w0 = lcdm ? -1.0 : cosmo.w0
    wa = lcdm ? 0.0 : cosmo.wa
    return a .^ (-3.0 * (1.0 + w0 + wa)) .* exp.(-3.0 * wa .* (1.0 .- a))
end

@inline function _hmcode_hubble2(a, cosmo::HMCodeCosmology, lcdm::Bool)
    Ωw = lcdm ? (1.0 - cosmo.Omega_m) : (1.0 - cosmo.Omega_m - cosmo.Omega_k)
    Ωtot = lcdm ? 1.0 : (1.0 - cosmo.Omega_k)
    return cosmo.Omega_m .* a .^ -3 .+ Ωw .* _hmcode_xw(a, cosmo, lcdm) .+ (1.0 - Ωtot) .* a .^ -2
end

@inline _hmcode_omega_m_a(a, cosmo::HMCodeCosmology, lcdm::Bool) =
    cosmo.Omega_m .* a .^ -3 ./ _hmcode_hubble2(a, cosmo, lcdm)

@inline function _hmcode_ah(a, cosmo::HMCodeCosmology, lcdm::Bool)
    Ωw = lcdm ? (1.0 - cosmo.Omega_m) : (1.0 - cosmo.Omega_m - cosmo.Omega_k)
    return -0.5 .* (cosmo.Omega_m .* a .^ -3 .+
                    (1.0 .+ 3.0 .* _hmcode_w(a, cosmo, lcdm)) .* Ωw .* _hmcode_xw(a, cosmo, lcdm))
end

function _hmcode_growth_rhs_static(a::Float64, d::Float64, v::Float64,
                                   cosmo::HMCodeCosmology, lcdm::Bool)
    fv = -(2.0 + _hmcode_ah(a, cosmo, lcdm) / _hmcode_hubble2(a, cosmo, lcdm)) * v / a
    fd = 1.5 * _hmcode_omega_m_a(a, cosmo, lcdm) * d / a^2
    return v, fv + fd
end

function _hmcode_growth_tables_static(cosmo::HMCodeCosmology; lcdm::Bool=false, na::Int=2000)
    a_init = 1.0e-4
    a = collect(range(a_init, 1.0, length=na))
    f_init = 1.0 - _hmcode_omega_m_a(a_init, cosmo, lcdm)
    d = a_init^(1.0 - 3.0 * f_init / 5.0)
    v = (1.0 - 3.0 * f_init / 5.0) * a_init^(-3.0 * f_init / 5.0)
    growth = Vector{Float64}(undef, na)
    growth[1] = d
    @inbounds for i in 1:(na - 1)
        a0 = a[i]
        a1 = a[i + 1]
        h = a1 - a0
        k1d, k1v = _hmcode_growth_rhs_static(a0, d, v, cosmo, lcdm)
        k2d, k2v = _hmcode_growth_rhs_static(a0 + 0.5h, d + 0.5h * k1d, v + 0.5h * k1v, cosmo, lcdm)
        k3d, k3v = _hmcode_growth_rhs_static(a0 + 0.5h, d + 0.5h * k2d, v + 0.5h * k2v, cosmo, lcdm)
        k4d, k4v = _hmcode_growth_rhs_static(a1, d + h * k3d, v + h * k3v, cosmo, lcdm)
        d += h * (k1d + 2.0k2d + 2.0k3d + k4d) / 6.0
        v += h * (k1v + 2.0k2v + 2.0k3v + k4v) / 6.0
        growth[i + 1] = d
    end
    agrowth = similar(growth)
    agrowth[1] = growth[1]
    @inbounds for i in 1:(na - 1)
        agrowth[i + 1] = agrowth[i] + 0.5 * (a[i + 1] - a[i]) * (growth[i + 1] / a[i + 1] + growth[i] / a[i])
    end
    return a, growth, agrowth
end


function _hmcode_tophat(x)
    return ifelse.(abs.(x) .< 1.0e-5,
                   1.0 .- x .^ 2 ./ 10.0,
                   3.0 .* (sin.(x) .- x .* cos.(x)) ./ x .^ 3)
end

function _hmcode_trapz_dim1_2d(x, y)
    dx = reshape(x[2:end] .- x[1:(end - 1)], :, 1)
    return vec(sum(0.5 .* dx .* (y[2:end, :] .+ y[1:(end - 1), :]); dims=1))
end

function _hmcode_trapz_dim1_3d(x, y)
    dx = reshape(x[2:end] .- x[1:(end - 1)], :, 1, 1)
    ny, nz = size(y, 2), size(y, 3)
    return reshape(sum(0.5 .* dx .* (y[2:end, :, :] .+ y[1:(end - 1), :, :]); dims=1), ny, nz)
end

function _hmcode_interp_vec(xgrid, ygrid, xvals)
    nout = length(xvals)
    nint = length(xgrid) - 1
    xlo = xgrid[1:1]
    xhi = xgrid[end:end]
    xcl = min.(max.(xvals, xlo), xhi)
    xv = reshape(xcl, nout, 1)
    x0 = reshape(xgrid[1:(end - 1)], 1, nint)
    x1 = reshape(xgrid[2:end], 1, nint)
    y0 = reshape(ygrid[1:(end - 1)], 1, nint)
    y1 = reshape(ygrid[2:end], 1, nint)
    vals = y0 .+ (xv .- x0) .* (y1 .- y0) ./ (x1 .- x0)
    islast = reshape(collect(1:nint) .== nint, 1, nint)
    mask = (xv .>= x0) .& ((xv .< x1) .| (islast .& (xv .<= x1)))
    return vec(sum(ifelse.(mask, vals, zero(vals)); dims=2))
end

function _hmcode_interp_columns_same_x(xgrid, y_mz, x_z)
    nM, nz = size(y_mz)
    nint = nM - 1
    lo = xgrid[1:1]
    hi = xgrid[end:end]
    xcl = min.(max.(x_z, lo), hi)
    xv = reshape(xcl, 1, nz)
    x0 = reshape(xgrid[1:(end - 1)], nint, 1)
    x1 = reshape(xgrid[2:end], nint, 1)
    y0 = y_mz[1:(end - 1), :]
    y1 = y_mz[2:end, :]
    vals = y0 .+ (xv .- x0) .* (y1 .- y0) ./ (x1 .- x0)
    islast = reshape(collect(1:nint) .== nint, nint, 1)
    mask = (xv .>= x0) .& ((xv .< x1) .| (islast .& (xv .<= x1)))
    return vec(sum(ifelse.(mask, vals, zero(vals)); dims=1))
end

function _hmcode_interp_columns_variable_x(xgrid_mz, ygrid_m, x_z)
    nM, nz = size(xgrid_mz)
    nint = nM - 1
    lo = xgrid_mz[1:1, :]
    hi = xgrid_mz[end:end, :]
    xv = reshape(min.(max.(x_z, vec(lo)), vec(hi)), 1, nz)
    x0 = xgrid_mz[1:(end - 1), :]
    x1 = xgrid_mz[2:end, :]
    y0 = reshape(ygrid_m[1:(end - 1)], nint, 1)
    y1 = reshape(ygrid_m[2:end], nint, 1)
    vals = y0 .+ (xv .- x0) .* (y1 .- y0) ./ (x1 .- x0)
    islast = reshape(collect(1:nint) .== nint, nint, 1)
    mask = (xv .>= x0) .& ((xv .< x1) .| (islast .& (xv .<= x1)))
    return vec(sum(ifelse.(mask, vals, zero(vals)); dims=1))
end

function _hmcode_interp_columns_same_x_matrix(xgrid, y_mz, x_mz)
    ngrid, nz = size(y_mz)
    ntarget = size(x_mz, 1)
    nint = ngrid - 1
    lo = xgrid[1:1]
    hi = xgrid[end:end]
    xcl = min.(max.(x_mz, lo), hi)
    xv = reshape(xcl, 1, ntarget, nz)
    x0 = reshape(xgrid[1:(end - 1)], nint, 1, 1)
    x1 = reshape(xgrid[2:end], nint, 1, 1)
    y0 = reshape(y_mz[1:(end - 1), :], nint, 1, nz)
    y1 = reshape(y_mz[2:end, :], nint, 1, nz)
    vals = y0 .+ (xv .- x0) .* (y1 .- y0) ./ (x1 .- x0)
    islast = reshape(collect(1:nint) .== nint, nint, 1, 1)
    mask = (xv .>= x0) .& ((xv .< x1) .| (islast .& (xv .<= x1)))
    return reshape(sum(ifelse.(mask, vals, zero(vals)); dims=1), ntarget, nz)
end

function _hmcode_loglog_interp_columns(k_support, pk_support_kz, k_out)
    logk = log.(k_support)
    logpk = log.(pk_support_kz)
    nout = length(k_out)
    nz = size(pk_support_kz, 2)
    nint = length(k_support) - 1
    lx = reshape(log.(k_out), nout, 1, 1)
    x0 = reshape(logk[1:(end - 1)], 1, nint, 1)
    x1 = reshape(logk[2:end], 1, nint, 1)
    y0 = reshape(logpk[1:(end - 1), :], 1, nint, nz)
    y1 = reshape(logpk[2:end, :], 1, nint, nz)
    vals = y0 .+ (lx .- x0) .* (y1 .- y0) ./ (x1 .- x0)
    islast = reshape(collect(1:nint) .== nint, 1, nint, 1)
    mask = (lx .>= x0) .& ((lx .< x1) .| (islast .& (lx .<= x1)))
    return exp.(reshape(sum(ifelse.(mask, vals, zero(vals)); dims=2), nout, nz))
end

"""Linear interpolation in log(k) for y values that may be negative (e.g. BAO wiggle)."""
function _hmcode_linlogk_interp_columns(k_support, wig_sup_kz, k_out)
    logk = log.(k_support)
    nout = length(k_out)
    nz = size(wig_sup_kz, 2)
    nint = length(k_support) - 1
    lx = reshape(log.(k_out), nout, 1, 1)
    x0 = reshape(logk[1:(end - 1)], 1, nint, 1)
    x1 = reshape(logk[2:end], 1, nint, 1)
    y0 = reshape(wig_sup_kz[1:(end - 1), :], 1, nint, nz)
    y1 = reshape(wig_sup_kz[2:end, :], 1, nint, nz)
    # Linear interpolation of wiggle values (not log), abscissa in log(k)
    t = (lx .- x0) ./ (x1 .- x0)
    vals = (1.0 .- t) .* y0 .+ t .* y1
    islast = reshape(collect(1:nint) .== nint, 1, nint, 1)
    mask = (lx .>= x0) .& ((lx .< x1) .| (islast .& (lx .<= x1)))
    return reshape(sum(ifelse.(mask, vals, zero(vals)); dims=2), nout, nz)
end

function _hmcode_derivative_columns(x_z, xs, fs_mz)
    n = length(xs)
    nz = size(fs_mz, 2)
    dist = abs.(reshape(xs, n, 1) .- reshape(x_z, 1, nz))
    dmin = minimum(dist; dims=1)
    nearest = dist .<= dmin .+ 1.0e-14
    starts_for_i = clamp.(collect(1:n) .- 1, 1, n - 2)
    out = reshape(x_z .* 0, 1, nz)
    @inbounds for s in 1:(n - 2)
        x0 = reshape(xs[s:s], 1, 1)
        x1 = reshape(xs[(s + 1):(s + 1)], 1, 1)
        x2 = reshape(xs[(s + 2):(s + 2)], 1, 1)
        f0 = reshape(fs_mz[s:s, :], 1, nz)
        f1 = reshape(fs_mz[(s + 1):(s + 1), :], 1, nz)
        f2 = reshape(fs_mz[(s + 2):(s + 2), :], 1, nz)
        xv = reshape(x_z, 1, nz)
        deriv = f0 .* (2.0 .* xv .- x1 .- x2) ./ ((x0 .- x1) .* (x0 .- x2)) .+
                f1 .* (2.0 .* xv .- x0 .- x2) ./ ((x1 .- x0) .* (x1 .- x2)) .+
                f2 .* (2.0 .* xv .- x0 .- x1) ./ ((x2 .- x0) .* (x2 .- x1))
        start_mask_i = reshape(starts_for_i .== s, n, 1)
        selected = sum(ifelse.(nearest .& start_mask_i, 1.0, 0.0); dims=1)
        out = out .+ ifelse.(selected .> 0.5, deriv, zero(deriv))
    end
    return vec(out)
end

function _hmcode_tk_eh_nowiggle(k, h, wm, wb)
    rb = wb / wm
    s = 44.5 * log(9.83 / wm) / sqrt(1.0 + 10.0 * wb^0.75)
    alpha = 1.0 - 0.328 * log(431.0 * wm) * rb + 0.38 * log(22.3 * wm) * rb^2
    Γ = (wm / h) .* (alpha .+ (1.0 - alpha) ./ (1.0 .+ (0.43 .* k .* s .* h) .^ 4))
    q = k .* (2.725 / 2.7)^2 ./ Γ
    L = log.(2.0 * exp(1.0) .+ 1.8 .* q)
    C = 14.2 .+ 731.0 ./ (1.0 .+ 62.5 .* q)
    return L ./ (L .+ C .* q .^ 2)
end

function _hmcode_reflect_indices_static(n::Int, off::Int)
    idx = Vector{Int}(undef, n)
    @inbounds for i in 1:n
        j = i + off
        if j < 1
            j = 1 - j
        elseif j > n
            j = 2n - j + 1
        end
        idx[i] = j
    end
    return idx
end

function _hmcode_gaussian_filter_columns(y, sigma; truncate=4.0)
    n, nz = size(y)
    offsets = collect(-(n - 1):(n - 1))
    weights = exp.(-0.5 .* (offsets ./ sigma) .^ 2) .* ifelse.(abs.(offsets) .<= truncate .* sigma, 1.0, 0.0)
    norm = sum(weights)
    out = y .* 0
    @inbounds for (j, off) in enumerate(offsets)
        idx = _hmcode_reflect_indices_static(n, off)
        out = out .+ y[idx, :] .* sum(weights[j:j])
    end
    return out ./ norm
end

function _hmcode_pk_wiggle_columns(k, pk_lin_kz, h, omega_m_h2, omega_b_h2, n_s)
    dlnk = log(sum(k[2:2]) / sum(k[1:1]))
    sigma = 0.25 / dlnk
    tk_nw = _hmcode_tk_eh_nowiggle(k, h, omega_m_h2, omega_b_h2)
    pk_nw = k .^ n_s .* tk_nw .^ 2
    ratio = pk_lin_kz ./ reshape(pk_nw, :, 1)
    smooth = _hmcode_gaussian_filter_columns(ratio, sigma)
    return pk_lin_kz .- smooth .* reshape(pk_nw, :, 1)
end

function _hmcode_sigma_grid(k_support, pk_cb_kz, R)
    logk = log.(k_support)
    nk = length(k_support)
    nM = length(R)
    nz = size(pk_cb_kz, 2)
    k3 = reshape(k_support .^ 3, nk, 1, 1)
    pk = reshape(pk_cb_kz, nk, 1, nz)
    W = _hmcode_tophat(reshape(k_support, nk, 1, 1) .* reshape(R, 1, nM, 1))
    integrand = k3 .* pk .* W .^ 2 ./ (2.0 * pi^2)
    sigma2 = _hmcode_trapz_dim1_3d(logk, integrand)
    return sqrt.(max.(sigma2, 0.0))
end

function _hmcode_sigma_v(k_support, pk_mm_kz)
    logk = log.(k_support)
    integrand = pk_mm_kz .* reshape(k_support, :, 1)
    sigma2 = _hmcode_trapz_dim1_2d(logk, integrand) ./ (2.0 * pi^2)
    return sqrt.(max.(sigma2, 0.0)) ./ sqrt(3.0)
end

function _hmcode_compute_params(z, a_grid, growth, agrowth, sigma_mz, R, k_support,
                                pk_mm_kz, cosmo::HMCodeCosmology)
    fν = cosmo.Omega_nu / cosmo.Omega_m
    a = _hmcode_scalefactor(z)
    Ωmz = _hmcode_omega_m_a(a, cosmo, false)
    g = _hmcode_interp_vec(a_grid, growth, a)
    G = _hmcode_interp_vec(a_grid, agrowth, a)
    dc = _hmcode_dc_mead(a, Ωmz, fν, g, G)
    dv = _hmcode_dv_mead(a, Ωmz, fν, g, G)

    logR = log.(R)
    logsigma = log.(sigma_mz)
    rnl = exp.(_hmcode_interp_columns_variable_x(logsigma[end:-1:1, :], logR[end:-1:1], log.(dc)))
    s8 = exp.(_hmcode_interp_columns_same_x(logR, logsigma, z .* 0 .+ log(8.0)))
    sigv = _hmcode_sigma_v(k_support, pk_mm_kz)
    neff = -3.0 .- 2.0 .* _hmcode_derivative_columns(log.(rnl), logR, logsigma)

    return _HMCodeReactantParams(
        rnl,
        neff,
        sigv,
        dv,
        dc,
        0.1281 .* s8 .^ (-0.3644),
        1.875 .* (1.603) .^ neff,
        0.2696 .* s8 .^ 0.9403,
        0.05618 .* s8 .^ (-1.013),
        z .* 0 .+ 5.196,
        0.05699 .* s8 .^ (-1.089),
    )
end

@inline function _hmcode_f_mead(x, y, p0, p1, p2, p3)
    return p0 .+ p1 .* (1.0 .- x) .+ p2 .* (1.0 .- x) .^ 2 .+ p3 .* (1.0 .- y)
end

function _hmcode_dc_mead(a, Ωm, fν, g, G)
    dc = 1.0 .+
         _hmcode_f_mead(g ./ a, G ./ a, -0.0069, -0.0208, 0.0312, 0.0021) .* log10.(Ωm) .+
         _hmcode_f_mead(g ./ a, G ./ a, 0.0001, -0.0647, -0.0417, 0.0646)
    return dc .* _HMCODE_DC0 .* (1.0 - 0.041 * fν)
end

function _hmcode_dv_mead(a, Ωm, fν, g, G)
    logΩ = log10.(Ωm)
    dv = 1.0 .+
         _hmcode_f_mead(g ./ a, G ./ a, -0.79, -10.17, 2.51, 6.51) .* logΩ .+
         _hmcode_f_mead(g ./ a, G ./ a, -1.89, 0.38, 18.8, -15.87) .* logΩ .^ 2
    return dv .* _HMCODE_DV0 .* (1.0 + 0.763 * fν)
end

function _hmcode_params_notweaks(p::_HMCodeReactantParams)
    z0 = p.eta .* 0
    z1 = z0 .+ 1
    return _HMCodeReactantParams(p.R_nl, p.n_eff, p.sigma_v, p.Delta_v, p.delta_c,
                                 z0, z1, z0, p.k_star, z0 .+ 4.0, z0)
end

function _hmcode_compute_weights(M, nu_mz, amp)
    p_st, q_st = 0.3, 0.707
    g = _HMCODE_ST_A .* (1.0 .+ (q_st .* nu_mz .^ 2) .^ (-p_st)) .* exp.(-q_st .* nu_mz .^ 2 ./ 2.0)
    wt0 = 0.5 .* (nu_mz[2:2, :] .- nu_mz[1:1, :])
    wtm = 0.5 .* (nu_mz[3:end, :] .- nu_mz[1:(end - 2), :])
    wt1 = 0.5 .* (nu_mz[end:end, :] .- nu_mz[(end - 1):(end - 1), :])
    wt = vcat(wt0, wtm, wt1)
    return (g ./ reshape(M, :, 1)) .* reshape(amp .^ 2, :, 1) .* wt
end

const _HMCODE_SI_SMALL = (1.0, -0.05555555555555555, 0.0016666666666666668, -2.834467120181406e-5, 3.0619243582206544e-7, -2.27746439867652e-9, 1.2353110643708935e-11, -5.0981091545465446e-14, 1.6537983849091297e-16, -4.326650129802279e-19, 9.32044812542441e-22, -1.6818131176655147e-24, 2.5787801137537893e-27, -3.401366616220572e-30, 3.8999872022233505e-33, -3.922984005011348e-36, 3.489798672961197e-39, -2.7650265596091116e-42, 1.9636378862575867e-45, -1.257043527311165e-48, 7.291002017420499e-52, -3.849327599400454e-55, 1.8577001882628452e-58, -8.22686917863956e-62, 3.3550504251358757e-65, -1.264109733422975e-68)
const _HMCODE_CI_INT_SMALL = (-0.25, 0.010416666666666666, -0.0002314814814814815, 3.1001984126984127e-6, -2.755731922398589e-8, 1.7397297489890083e-10, -8.193389712664089e-13, 2.9871733327421158e-15, -8.677337204770125e-18, 2.0551588116560825e-20, -4.043996087533e-23, 6.715573212900493e-26, -9.53690870471076e-29, 1.1713890132392279e-31, -1.2566625429386353e-34, 1.1876221108921073e-37, -9.962228045650476e-41, 7.467278517462879e-44, -5.031482118058e-47, 3.0640435978209645e-50, -1.6946206503074855e-53, 8.549642911891504e-57, -3.9506856555684327e-60, 1.6782241814065079e-63, -6.575898833266316e-67)
const _HMCODE_P_F_RAT1 = (0.9999999996217391, 364.510603386319, 44218.54804128844, 2246756.9405961153, 49315316.72305596, 431867952.7967028, 1184799251.9992545, 455732675.9379532)
const _HMCODE_Q_F_RAT1 = (1.0, 366.5106027322935, 44927.56981497069, 2328535.488220404, 53117852.01722826, 503353106.6724187, 1657528501.5623176, 1174653283.7041042)
const _HMCODE_R_G_RAT1 = (0.999999999204849, 513.8550487530732, 92293.48345259381, 7407134.186234174, 281423561.62841356, 4928089035.773462, 35524762685.554024, 79194271662.05495, 17942522624.4139)
const _HMCODE_S_G_RAT1 = (1.0, 519.8550470881487, 95292.61550812594, 7921545.967976676, 319775677.90347815, 6227313470.243901, 54570971054.996445, 182417501666.45703, 154071481488.65445)
const _HMCODE_P_F_ASYM_NUM = (1.9999999999999978, 2220.611938043496, 847490.0762398824, 139592679.54823944, 10197205463.267975, 302298652645.2408, 2750405380428.847, 2181898970468.7498)
const _HMCODE_P_F_ASYM_DEN = (1.0, 1122.3059690217168, 436852.7097485132, 74654702.14065616, 5858003475.188747, 201579803792.09885, 2622914185768.9644, 8785290733498.676)
const _HMCODE_R_G_ASYM_NUM = (5.999999999999999, 9652.774604499714, 5607762.699656884, 1502266771.8927317, 196442710647.33087, 121913682811632.5, 3192438989864569.5, 2.5876053010027484e16, 1.2754978896268878e16)
const _HMCODE_R_G_ASYM_DEN = (1.0, 1628.7957674166142, 966363.0319578709, 268397347.5095067, 37388510548.052925, 2602858566615.2144, 85134283716949.72, 1130407936162795.2, 4251984147948980.0)

function _hmcode_si_fast(x)
    small = x .* evalpoly.(x .* x, Ref(_HMCODE_SI_SMALL))
    t = x .* x
    invt = 1.0 ./ t
    sx, cx = sin.(x), cos.(x)
    f_rat = (evalpoly.(invt, Ref(_HMCODE_P_F_RAT1)) ./ evalpoly.(invt, Ref(_HMCODE_Q_F_RAT1))) ./ x
    g_rat = (evalpoly.(invt, Ref(_HMCODE_R_G_RAT1)) ./ evalpoly.(invt, Ref(_HMCODE_S_G_RAT1))) .* invt
    f_asym = (1.0 .- evalpoly.(invt, Ref(_HMCODE_P_F_ASYM_NUM)) .* invt ./ evalpoly.(invt, Ref(_HMCODE_P_F_ASYM_DEN))) ./ x
    g_asym = (1.0 .- evalpoly.(invt, Ref(_HMCODE_R_G_ASYM_NUM)) .* invt ./ evalpoly.(invt, Ref(_HMCODE_R_G_ASYM_DEN))) .* invt
    f = ifelse.(t .<= 144.0, f_rat, f_asym)
    g = ifelse.(t .<= 144.0, g_rat, g_asym)
    large = pi / 2.0 .- f .* cx .- g .* sx
    return ifelse.(x .<= 4.0, small, large)
end

function _hmcode_ci_fast(x)
    small = _HMCODE_EULER_GAMMA .+ log.(x) .+ x .* x .* evalpoly.(x .* x, Ref(_HMCODE_CI_INT_SMALL))
    t = x .* x
    invt = 1.0 ./ t
    sx, cx = sin.(x), cos.(x)
    f_rat = (evalpoly.(invt, Ref(_HMCODE_P_F_RAT1)) ./ evalpoly.(invt, Ref(_HMCODE_Q_F_RAT1))) ./ x
    g_rat = (evalpoly.(invt, Ref(_HMCODE_R_G_RAT1)) ./ evalpoly.(invt, Ref(_HMCODE_S_G_RAT1))) .* invt
    f_asym = (1.0 .- evalpoly.(invt, Ref(_HMCODE_P_F_ASYM_NUM)) .* invt ./ evalpoly.(invt, Ref(_HMCODE_P_F_ASYM_DEN))) ./ x
    g_asym = (1.0 .- evalpoly.(invt, Ref(_HMCODE_R_G_ASYM_NUM)) .* invt ./ evalpoly.(invt, Ref(_HMCODE_R_G_ASYM_DEN))) .* invt
    f = ifelse.(t .<= 144.0, f_rat, f_asym)
    g = ifelse.(t .<= 144.0, g_rat, g_asym)
    large = f .* sx .- g .* cx
    return ifelse.(x .<= 4.0, small, large)
end

function _hmcode_wnfw_fast(x, c, ln1pc)
    xplus = x .* (1.0 .+ 1.0 ./ c)
    xminus = x ./ c
    dsi = _hmcode_si_fast(xplus) .- _hmcode_si_fast(xminus)
    dci = _hmcode_ci_fast(xplus) .- _hmcode_ci_fast(xminus)
    sinc_xp = sin.(x) ./ xplus
    norm = ln1pc .- c ./ (1.0 .+ c)
    return (dsi .* sin.(xminus) .+ dci .* cos.(xminus) .- sinc_xp) ./ norm
end

function _hmcode_feedback_parameters(T_AGN)
    θ = log10(T_AGN / 10.0^7.8)
    return (
        3.44 - 0.496 * θ,
        -0.0671 - 0.0371 * θ,
        10.0^(13.87 + 1.81 * θ),
        -0.108 + 0.195 * θ,
        (2.01 - 0.3 * θ) * 1.0e-2,
        0.409 + 0.0224 * θ,
    )
end

function _hmcode_assemble_pass(k, z, cosmo::HMCodeCosmology, M, R,
                               params::_HMCodeReactantParams, sigma_mz, nu_mz,
                               pk_lin_kz, pk_wig_kz, a_grid, growth, growth_lcdm;
                               tweaks::Bool=true, include_feedback::Bool=false,
                               T_AGN=10.0^7.8)
    nk, nz = length(k), length(z)
    nM = length(M)
    ρm = _hmcode_density(cosmo.Omega_m)
    Ωm, Ωb, Ων = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    Ωc = Ωm - Ωb - Ων
    fν = Ων / Ωm
    feedback_profile = include_feedback && !tweaks
    amp_factor = feedback_profile ? 1.0 : (1.0 - fν)
    amp = M .* amp_factor ./ ρm
    w1h_mz = _hmcode_compute_weights(M, nu_mz, amp)

    fb_B0, fb_Bz, fb_Mb0, fb_Mbz, fb_f0, fb_fz = _hmcode_feedback_parameters(T_AGN)
    ac_vec = a_grid[1:1] .* 0 .+ _hmcode_scalefactor(10.0)
    g_ac = sum(_hmcode_interp_vec(a_grid, growth, ac_vec))
    gl_ac = sum(_hmcode_interp_vec(a_grid, growth_lcdm, ac_vec))

    a_obs = _hmcode_scalefactor(z)
    g_obs = _hmcode_interp_vec(a_grid, growth, a_obs)
    gl_obs = _hmcode_interp_vec(a_grid, growth_lcdm, a_obs)
    dolag = (g_ac / gl_ac) .* (gl_obs ./ g_obs)

    logR = log.(R)
    rc = _hmcode_lagrangian_radius(0.01 .* M, Ωm)
    x_rc = reshape(log.(rc), nM, 1) .+ reshape(z .* 0, 1, nz)
    sig_rc = exp.(_hmcode_interp_columns_same_x_matrix(logR, log.(sigma_mz), x_rc))

    g_target = reshape(g_obs .* params.delta_c, 1, nz) ./ sig_rc
    af = reshape(_hmcode_interp_vec(growth, a_grid, vec(g_target)), nM, nz)
    zf = ifelse.(g_target .>= reshape(g_obs, 1, nz), reshape(z, 1, nz), _hmcode_redshift(af))

    rv = reshape(R, nM, 1) ./ reshape(params.Delta_v .^ (1.0 / 3.0), 1, nz)
    B = feedback_profile ? (fb_B0 .* 10.0 .^ (z .* fb_Bz)) : params.B
    conc = reshape(B, 1, nz) .* (1.0 .+ zf) ./ reshape(1.0 .+ z, 1, nz) .* reshape(dolag, 1, nz)
    ln1pc = log.(1.0 .+ conc)
    rv_eff = rv .* nu_mz .^ reshape(params.eta, 1, nz)

    W = _hmcode_wnfw_fast(reshape(rv_eff, nM, 1, nz) .* reshape(k, 1, nk, 1),
                          reshape(conc, nM, 1, nz),
                          reshape(ln1pc, nM, 1, nz))
    if feedback_profile
        Mb = fb_Mb0 .* 10.0 .^ (z .* fb_Mbz)
        fstar = min.(fb_f0 .* 10.0 .^ (z .* fb_fz), Ωb / Ωm)
        fg = (Ωb / Ωm .- reshape(fstar, 1, nz)) .* (reshape(M, nM, 1) ./ reshape(Mb, 1, nz)) .^ 2 ./
             (1.0 .+ (reshape(M, nM, 1) ./ reshape(Mb, 1, nz)) .^ 2)
        coeff = Ωc / Ωm .+ fg
        W = reshape(coeff, nM, 1, nz) .* W .+ reshape(fstar, 1, 1, nz)
    end

    I1h = reshape(sum(W .^ 2 .* reshape(w1h_mz, nM, 1, nz); dims=1), nk, nz) .* ρm
    kstar = reshape(params.k_star, 1, nz)
    safe_kstar = ifelse.(kstar .> 0.0, kstar, one.(kstar))
    x4 = (reshape(k, nk, 1) ./ safe_kstar) .^ 4
    p1h_fac = ifelse.(kstar .> 0.0, x4 ./ (1.0 .+ x4), one.(kstar))
    p1h = p1h_fac .* I1h

    if tweaks
        pk_dwl = pk_lin_kz .- (1.0 .- exp.(-(reshape(k, nk, 1) .* reshape(params.sigma_v, 1, nz)) .^ 2)) .* pk_wig_kz
        y = (reshape(k, nk, 1) ./ reshape(params.k_damp, 1, nz)) .^ _HMCODE_ND
        p2h = pk_dwl .* (1.0 .- reshape(params.f_damp, 1, nz) .* y ./ (1.0 .+ y))
        return (p2h .^ reshape(params.A, 1, nz) .+ p1h .^ reshape(params.A, 1, nz)) .^ (1.0 ./ reshape(params.A, 1, nz))
    end
    return pk_lin_kz .+ p1h
end

function _hmcode_Pmm_reactant(cosmo::HMCodeCosmology, z::ReactantVec, k_out::ReactantVec,
                               k_support, pk_mm_support_kz::ReactantMat,
                               pk_cb_support_kz::ReactantMat; T_AGN=10.0^7.8,
                               Mmin=1.0, Mmax=1.0e18, nM=128,
                               growth_cosmo=cosmo)
    nM >= 2 || throw(ArgumentError("HMCode nM must be at least 2."))
    M = exp.(collect(range(log(float(Mmin)), log(float(Mmax)), length=nM)))
    R = _hmcode_lagrangian_radius(M, cosmo.Omega_m)
    pk_mm_out = _hmcode_loglog_interp_columns(k_support, pk_mm_support_kz, k_out)
    sigma_mz = _hmcode_sigma_grid(k_support, pk_cb_support_kz, R)
    a_grid, growth, agrowth = _hmcode_growth_tables_static(growth_cosmo; lcdm=false)
    _, growth_lcdm, _ = _hmcode_growth_tables_static(growth_cosmo; lcdm=true)
    params = _hmcode_compute_params(z, a_grid, growth, agrowth, sigma_mz, R,
                                    k_support, pk_mm_support_kz, cosmo)
    nu_mz = reshape(params.delta_c, 1, :) ./ sigma_mz
    # BAO dewiggling must use the validated log-uniform support grid.
    # _hmcode_pk_wiggle_columns computes dlnk = log(k[2]/k[1]) and applies a
    # fixed index-space Gaussian. Using k_out here would give the wrong
    # smoothing width when k_out has different spacing or is irregular.
    # Compute wiggle on k_support, then interpolate linearly in log(k).
    pk_wig_sup = _hmcode_pk_wiggle_columns(k_support, pk_mm_support_kz, cosmo.h,
                                           cosmo.Omega_m * cosmo.h^2,
                                           cosmo.Omega_b * cosmo.h^2,
                                           cosmo.n_s)
    # Linear-in-logk interpolation of wiggle (can be positive or negative)
    pk_wig = _hmcode_linlogk_interp_columns(k_support, pk_wig_sup, k_out)
    base = _hmcode_assemble_pass(k_out, z, cosmo, M, R, params, sigma_mz, nu_mz,
                                 pk_mm_out, pk_wig, a_grid, growth, growth_lcdm;
                                 tweaks=true, include_feedback=false,
                                 T_AGN=T_AGN === nothing ? 10.0^7.8 : T_AGN)
    if T_AGN !== nothing
        pnot = _hmcode_params_notweaks(params)
        den = _hmcode_assemble_pass(k_out, z, cosmo, M, R, pnot, sigma_mz, nu_mz,
                                    pk_mm_out, pk_wig, a_grid, growth, growth_lcdm;
                                    tweaks=false, include_feedback=false, T_AGN=T_AGN)
        num = _hmcode_assemble_pass(k_out, z, cosmo, M, R, pnot, sigma_mz, nu_mz,
                                    pk_mm_out, pk_wig, a_grid, growth, growth_lcdm;
                                    tweaks=false, include_feedback=true, T_AGN=T_AGN)
        base = base .* (num ./ den)
    end
    return base
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::ReactantVec, k::ReactantVec,
                    pk_mm_z::ReactantMat; pk_cb_z=nothing, k_support=nothing,
                    pk_cb_support_z=nothing, T_AGN=10.0^7.8, Mmin=1.0,
                    Mmax=1.0e18, nM=128, growth_cosmo=cosmo, kwargs...)
    # HMCode's BAO smoothing requires a log-uniform support grid. Native Julia
    # validates that host-side contract before execution. Reactant receives
    # dynamic device arrays here, so materializing them solely to validate grid
    # values would add a host synchronization and defeat compiled execution.
    # The compiled API deliberately relies on the documented input contract.
    k_linear = k_support === nothing ? k : k_support
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    pkcb === nothing && (pkcb = pk_mm_z)
    return _hmcode_Pmm_reactant(cosmo, z, k, k_linear, pk_mm_z, pkcb;
                                T_AGN=T_AGN, Mmin=Mmin, Mmax=Mmax,
                                nM=nM, growth_cosmo=growth_cosmo)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::ReactantVec, k::ReactantVec,
                    pk_mm_z::ReactantMat, pk_cb_z::ReactantMat; kwargs...)
    return hmcode_Pmm(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::ReactantVec, k_out::ReactantVec,
                    k_support::ReactantVec, pk_mm_support_z::ReactantMat;
                    pk_cb_support_z=nothing, pk_cb_z=nothing, kwargs...)
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_Pmm(cosmo, z, k_out, pk_mm_support_z;
                      k_support=k_support, pk_cb_z=pkcb, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::ReactantVec, k_out::ReactantVec,
                    k_support::ReactantVec, pk_mm_support_z::ReactantMat,
                    pk_cb_support_z::ReactantMat; pk_cb_z=nothing, kwargs...)
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_Pmm(cosmo, z, k_out, pk_mm_support_z;
                      k_support=k_support, pk_cb_z=pkcb, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number, k::ReactantVec,
                    pk_mm::ReactantVec; pk_cb=nothing, k_support=nothing,
                    pk_cb_support=nothing, kwargs...)
    k_linear = k_support === nothing ? k : k_support
    zvec = k_linear[1:1] .* 0 .+ float(z)
    pk = reshape(pk_mm, length(k_linear), 1)
    pkcb = _hmcode_choose_cb(pk_cb, pk_cb_support)
    pkcb_mat = pkcb === nothing ? nothing : reshape(pkcb, length(k_linear), 1)
    out = hmcode_Pmm(cosmo, zvec, k, pk; k_support=k_support, pk_cb_z=pkcb_mat, kwargs...)
    return vec(out)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number,
                    k::ReactantVec, pk_mm::ReactantVec,
                    pk_cb::ReactantVec; kwargs...)
    return hmcode_Pmm(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number, k_out::ReactantVec,
                    k_support::ReactantVec, pk_mm_support::ReactantVec,
                    pk_cb_support::ReactantVec; kwargs...)
    return hmcode_Pmm(cosmo, z, k_out, pk_mm_support;
                      k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::ReactantVec, k::ReactantVec,
                      pk_mm_z::ReactantMat; pk_cb_z=nothing, k_support=nothing,
                      pk_cb_support_z=nothing, kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    pk_nl = hmcode_Pmm(cosmo, z, k, pk_mm_z; pk_cb_z=pkcb,
                       k_support=k_support,
                       kwargs...)
    pk_lin_out = k_support === nothing ? pk_mm_z : _hmcode_loglog_interp_columns(k_linear, pk_mm_z, k)
    return pk_nl ./ pk_lin_out
end

function hmcode_boost(cosmo::HMCodeCosmology, z::ReactantVec, k::ReactantVec,
                      pk_mm_z::ReactantMat, pk_cb_z::ReactantMat; kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::ReactantVec, k_out::ReactantVec,
                      k_support::ReactantVec, pk_mm_support_z::ReactantMat;
                      pk_cb_support_z=nothing, pk_cb_z=nothing, kwargs...)
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_boost(cosmo, z, k_out, pk_mm_support_z;
                        k_support=k_support, pk_cb_z=pkcb, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::ReactantVec, k_out::ReactantVec,
                      k_support::ReactantVec, pk_mm_support_z::ReactantMat,
                    pk_cb_support_z::ReactantMat; pk_cb_z=nothing, kwargs...)
    pkcb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_boost(cosmo, z, k_out, pk_mm_support_z;
                        k_support=k_support, pk_cb_z=pkcb, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number, k::ReactantVec,
                      pk_mm::ReactantVec; pk_cb=nothing, k_support=nothing,
                      pk_cb_support=nothing, kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pk_nl = hmcode_Pmm(cosmo, z, k, pk_mm; pk_cb=pk_cb, k_support=k_support,
                       pk_cb_support=pk_cb_support, kwargs...)
    pk_lin_out = if k_support === nothing
        pk_mm
    else
        vec(_hmcode_loglog_interp_columns(k_linear, reshape(pk_mm, length(k_linear), 1), k))
    end
    return pk_nl ./ pk_lin_out
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number,
                      k::ReactantVec, pk_mm::ReactantVec,
                      pk_cb::ReactantVec; kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number, k_out::ReactantVec,
                      k_support::ReactantVec, pk_mm_support::ReactantVec,
                      pk_cb_support::ReactantVec; kwargs...)
    return hmcode_boost(cosmo, z, k_out, pk_mm_support;
                        k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

"""
    hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse, z_fine, k, pk_mm_coarse; kwargs...)

Evaluates HMCode Pmm using Akima interpolation over redshift.
For Reactant tracing, `z_coarse` and `z_fine` must be strictly increasing,
and `z_fine` must lie within the bounds of `z_coarse`.
These preconditions are not checked at trace time; users must validate
host inputs before device conversion. You can use the native Julia method
`Mapse.validate_hmcode_fast_grids(z_coarse, z_fine)` to validate these preconditions.

Note: The Reactant extension explicitly supports `hmcode_pmm_fast` only. It does not
support `hmcode_boost_fast` on traced arrays. For nonlinear boosts, users should evaluate
`hmcode_pmm_fast` and divide by the linear spectrum appropriately.
"""
function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::ReactantVec,
                         z_fine::Union{Number, ReactantVec}, k::ReactantVec,
                         pk_mm_coarse::ReactantMat;
                         pk_cb_coarse=nothing, k_support=nothing,
                         pk_cb_support_coarse=nothing, kwargs...)
    length(z_coarse) >= 5 || throw(ArgumentError("z_coarse must have at least 5 points for Akima interpolation."))
    if z_coarse isa Reactant.ConcretePJRTArray
        z_fine_host = z_fine isa Reactant.ConcretePJRTArray ? Array(z_fine) : z_fine
        Mapse.validate_hmcode_fast_grids(Array(z_coarse), z_fine_host)
    end
    pkcb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    Pk_nl_coarse = hmcode_Pmm(cosmo, z_coarse, k, pk_mm_coarse;
                              pk_cb_z=pkcb, k_support=k_support, kwargs...)
    z_fine_arr = z_fine isa Number ? z_coarse[1:1] .* zero(eltype(z_coarse)) .+ float(z_fine) : z_fine
    Pk_nl_coarse_t = copy(transpose(Pk_nl_coarse))
    Pk_nl_fine_t = Mapse.AbstractCosmologicalEmulators.akima_interpolation(Pk_nl_coarse_t, z_coarse, z_fine_arr)
    Pk_nl_fine = copy(transpose(Pk_nl_fine_t))
    return z_fine isa Number ? vec(Pk_nl_fine) : Pk_nl_fine
end

function hmcode_pmm_fast_two_splines(cosmo::HMCodeCosmology,
                                     z_coarse::ReactantVec,
                                     z_fine::ReactantVec,
                                     k::ReactantVec,
                                     pk_mm_coarse::ReactantMat;
                                     z_split,
                                     split_index::Union{Nothing,Int}=nothing,
                                     pk_cb_coarse=nothing,
                                     k_support=nothing,
                                     pk_cb_support_coarse=nothing,
                                     kwargs...)
    nleft = split_index === nothing ? throw(ArgumentError("split_index is required for compiled two-spline Reactant calls.")) : split_index
    z_host = z_coarse isa Reactant.ConcretePJRTArray ? Array(z_coarse) : nothing
    nleft >= 5 || throw(ArgumentError("The lower two-spline segment needs at least 5 nodes."))
    nleft <= length(z_coarse) - 4 || throw(ArgumentError("The upper two-spline segment needs at least 5 nodes."))
    if z_host !== nothing && z_split isa Real
        isapprox(z_host[nleft], z_split; rtol=0, atol=10eps(Float64)) || throw(ArgumentError("z_split must be a node of z_coarse."))
    end
    pkcb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    pk_nl = hmcode_Pmm(cosmo, z_coarse, k, pk_mm_coarse;
                       pk_cb_z=pkcb, k_support=k_support, kwargs...)
    pk_t = copy(transpose(pk_nl))
    left = Mapse.AbstractCosmologicalEmulators.akima_interpolation(pk_t[1:nleft, :], z_coarse[1:nleft], z_fine)
    right = Mapse.AbstractCosmologicalEmulators.akima_interpolation(pk_t[nleft:end, :], z_coarse[nleft:end], z_fine)
    mask = reshape(z_fine .<= z_split, :, 1)
    return copy(transpose(ifelse.(mask, left, right)))
end

function _hmcode_pmm_baryonic_smart_device(
    params::ReactantVec, z_fine::ReactantVec, z_limits::ReactantVec,
    logT_AGN::ReactantVec, pmm_emu::Mapse.TransferFunctionEmulator,
    pcb_emu::Mapse.TransferFunctionEmulator, cosmo::HMCodeCosmology,
    growth; N_coarse::Int=20, N_left::Int=12, nM::Int=32)
    z_coarse = Mapse.build_baryonic_coarse_grid(
        params, z_limits, logT_AGN, N_coarse, N_left)
    pk_mm = Mapse.get_Pk(params, z_coarse, growth, pmm_emu)
    pk_cb = Mapse.get_Pk(params, z_coarse, growth, pcb_emu)
    h = sum(params[3:3]) / 100
    # The official linear emulators return physical k [Mpc^-1] and P [Mpc^3].
    # HMCode itself is written in h-units, like the JAX implementation:
    # k_h = k_phys / h and P_h = P_phys * h^3. Convert back after the
    # nonlinear calculation so the public smart API remains in physical units.
    k_h = pmm_emu.kgrid ./ h
    pk_mm_h = pk_mm .* h^3
    pk_cb_h = pk_cb .* h^3
    omega_b = sum(params[4:4]) / h^2
    omega_nu = (sum(params[6:6]) / 93.14) / h^2
    omega_m = omega_b + sum(params[5:5]) / h^2 + omega_nu
    dynamic_cosmo = HMCodeCosmology(
        omega_m, omega_b, h, sum(params[2:2]), cosmo.sigma_8,
        sum(params[7:7]), sum(params[8:8]), omega_nu, cosmo.Omega_k)
    dynamic_T_AGN = sum(10.0 .^ logT_AGN)
    z_split = z_coarse[(N_left + 1):(N_left + 1)]
    result_h = hmcode_pmm_fast_two_splines(
        dynamic_cosmo, z_coarse, z_fine, k_h, pk_mm_h;
        pk_cb_coarse=pk_cb_h, z_split=z_split, split_index=N_left + 1,
        T_AGN=dynamic_T_AGN, growth_cosmo=cosmo, nM=nM)
    return result_h ./ h^3
end

"""Fully traced emulator → moving grid → HMCode → two-spline pipeline.

This overload uses unity for the emulator growth factor. The overload accepting
`growth_emu` below evaluates the official background emulator instead.
"""
function hmcode_pmm_baryonic_smart(params::ReactantVec,
                                   z_fine::ReactantVec,
                                   z_limits::ReactantVec,
                                   logT_AGN::ReactantVec,
                                   pmm_emu::Mapse.TransferFunctionEmulator,
                                   pcb_emu::Mapse.TransferFunctionEmulator,
                                   cosmo::HMCodeCosmology;
                                   N_coarse::Int=20, N_left::Int=12,
                                   nM::Int=32)
    z_coarse = Mapse.build_baryonic_coarse_grid(
        params, z_limits, logT_AGN, N_coarse, N_left)
    return _hmcode_pmm_baryonic_smart_device(
        params, z_fine, z_limits, logT_AGN, pmm_emu, pcb_emu, cosmo,
        one.(z_coarse); N_coarse=N_coarse, N_left=N_left, nM=nM)
end

"""Fully traced smart pipeline using an emulator-predicted D(z).

`growth_emu` must be the official 9-input, 7-output generic background
emulator. Its sixth output is D(z). The emulator is evaluated on the moving
coarse grid and remains entirely inside the compiled Reactant graph.
"""
function hmcode_pmm_baryonic_smart(
    params::ReactantVec, z_fine::ReactantVec, z_limits::ReactantVec,
    logT_AGN::ReactantVec, pmm_emu::Mapse.TransferFunctionEmulator,
    pcb_emu::Mapse.TransferFunctionEmulator,
    growth_emu::AbstractCosmologicalEmulators.GenericEmulator,
    cosmo::HMCodeCosmology; N_coarse::Int=20, N_left::Int=12, nM::Int=32)
    z_coarse = Mapse.build_baryonic_coarse_grid(
        params, z_limits, logT_AGN, N_coarse, N_left)
    n_z = length(z_coarse)
    emulator_input = vcat(
        reshape(z_coarse, 1, n_z),
        repeat(reshape(params, :, 1), 1, n_z),
    )
    growth_output = AbstractCosmologicalEmulators.run_emulator(
        emulator_input, growth_emu)
    growth = vec(growth_output[6:6, :])
    return _hmcode_pmm_baryonic_smart_device(
        params, z_fine, z_limits, logT_AGN, pmm_emu, pcb_emu, cosmo,
        growth; N_coarse=N_coarse, N_left=N_left, nM=nM)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::ReactantVec,
                         z_fine::Union{Number, ReactantVec}, k::ReactantVec,
                         pk_mm_coarse::ReactantMat, pk_cb_coarse::ReactantMat;
                         kwargs...)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse;
                           pk_cb_coarse=pk_cb_coarse, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::ReactantVec,
                         z_fine::Union{Number, ReactantVec},
                         k_out::ReactantVec, k_support::ReactantVec,
                         pk_mm_support_coarse::ReactantMat;
                         pk_cb_support_coarse::Union{Nothing,ReactantMat}=nothing,
                         pk_cb_coarse::Union{Nothing,ReactantMat}=nothing,
                         kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k_out, pk_mm_support_coarse;
                           k_support=k_support, pk_cb_coarse=pk_cb, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::ReactantVec,
                         z_fine::Union{Number, ReactantVec},
                         k_out::ReactantVec, k_support::ReactantVec,
                         pk_mm_support_coarse::ReactantMat,
                         pk_cb_support_coarse::ReactantMat;
                         kwargs...)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k_out, k_support,
                           pk_mm_support_coarse;
                           pk_cb_support_coarse=pk_cb_support_coarse, kwargs...)
end

function hmcode_pmm_baryonic_smart(cosmo::HMCodeCosmology, z_fine::ReactantVec,
                                  k::ReactantVec, pk_mm_coarse::ReactantMat;
                                  pk_cb_coarse=nothing, N_coarse::Int=50, T_AGN::Real=10^7.8, kwargs...)
    z_min = minimum(z_fine; dims=1)
    z_max = maximum(z_fine; dims=1)

    z_feature = predict_baryonic_discontinuity(cosmo; T_AGN=T_AGN)
    N_left = N_coarse ÷ 2
    z_coarse = build_piecewise_coarse_grid(
        vcat(z_min, z_max, z_min .* 0 .+ z_feature),
        N_coarse, N_left)

    pk_mm_coarse_eval = if size(pk_mm_coarse, 2) == length(z_fine) && length(z_fine) != length(z_coarse)
        pk_mm_t = copy(transpose(pk_mm_coarse))
        pk_mm_coarse_t = Mapse.AbstractCosmologicalEmulators.akima_interpolation(pk_mm_t, z_fine, z_coarse)
        copy(transpose(pk_mm_coarse_t))
    else
        pk_mm_coarse
    end

    pk_cb_coarse_eval = if !isnothing(pk_cb_coarse) && size(pk_cb_coarse, 2) == length(z_fine) && length(z_fine) != length(z_coarse)
        pk_cb_t = copy(transpose(pk_cb_coarse))
        pk_cb_coarse_t = Mapse.AbstractCosmologicalEmulators.akima_interpolation(pk_cb_t, z_fine, z_coarse)
        copy(transpose(pk_cb_coarse_t))
    else
        pk_cb_coarse
    end

    nleft = N_left + 1
    if nleft >= 5 && nleft <= length(z_coarse) - 4
        return hmcode_pmm_fast_two_splines(cosmo, z_coarse, z_fine, k, pk_mm_coarse_eval;
                                           pk_cb_coarse=pk_cb_coarse_eval,
                                           z_split=z_feature, split_index=nleft,
                                           T_AGN=T_AGN, kwargs...)
    end
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse_eval;
                           pk_cb_coarse=pk_cb_coarse_eval, T_AGN=T_AGN, kwargs...)
end



end # module MapseReactantExt
