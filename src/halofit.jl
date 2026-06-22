"""
    HalofitCosmology(; Ωm0, Ωnu0, h, w0=-1, wa=0, mν=0.06, Ωr0=0, ΩΛ0=1-Ωm0-Ωr0)

Background parameters needed by the Takahashi/Bird Halofit implementation used
for nonlinear total-matter power spectra. Densities are present-day density
fractions and `mν` is the summed massive-neutrino mass in eV.
"""
@kwdef struct HalofitCosmology{T<:Real}
    Ωm0::T
    Ωnu0::T
    h::T
    w0::T = one(T) * -1
    wa::T = zero(T)
    mν::T = T(0.06)
    Ωr0::T = zero(T)
    ΩΛ0::T = one(T) - Ωm0 - Ωr0
end

const _HALOFIT_KB_EV_PER_K = 8.617333262e-5
const _HALOFIT_TNU0_K = 1.945377
const _HALOFIT_OMEGA_GAMMA_H2 = 2.469e-5
const _HALOFIT_MNU_TO_OMEGA_NU_H2 = 93.14
const _HALOFIT_CLASS_N_UR = 2.033
const _HALOFIT_MASSLESS_NU_FACTOR = 0.22710731766023898
const _HALOFIT_NEWTON_STEPS = 8

"""
    halofit_cosmology(input_params; Ωr0=nothing, ΩΛ0=nothing, N_ur=2.033)

Build a `HalofitCosmology` from the MAPSE `mnuw0wacdm` parameter order
`[ln10As, ns, H0, ωb, ωc, Mν, w0, wa]`. `H0` is in km/s/Mpc and `Mν` in eV.
Optional `Ωr0` and `ΩΛ0` override the inferred CLASS-like defaults.
"""
function halofit_cosmology(input_params::AbstractVector; Ωr0=nothing, ΩΛ0=nothing,
    N_ur=_HALOFIT_CLASS_N_UR)

    length(input_params) >= 8 || throw(ArgumentError(
        "MAPSE Halofit expects parameters [ln10As, ns, H0, ωb, ωc, Mν, w0, wa]."
    ))

    T = float(promote_type(typeof(input_params[3]), typeof(input_params[4]),
                           typeof(input_params[5]), typeof(input_params[6]),
                           typeof(input_params[7]), typeof(input_params[8])))
    H0 = T(input_params[3])
    h = H0 > T(10) ? H0 / T(100) : H0
    ωb = T(input_params[4])
    ωc = T(input_params[5])
    mν = T(input_params[6])
    w0 = T(input_params[7])
    wa = T(input_params[8])

    Ωnu0 = (mν / T(_HALOFIT_MNU_TO_OMEGA_NU_H2)) / h^2
    Ωm0 = (ωb + ωc) / h^2 + Ωnu0
    inferred_Ωr0 = T(_HALOFIT_OMEGA_GAMMA_H2) / h^2 *
                   (one(T) + T(_HALOFIT_MASSLESS_NU_FACTOR) * T(N_ur))
    Ωr0_value = isnothing(Ωr0) ? inferred_Ωr0 : T(Ωr0)
    ΩΛ0_value = isnothing(ΩΛ0) ? one(T) - Ωm0 - Ωr0_value : T(ΩΛ0)

    return HalofitCosmology(
        Ωm0=Ωm0,
        Ωnu0=Ωnu0,
        h=h,
        w0=w0,
        wa=wa,
        mν=mν,
        Ωr0=Ωr0_value,
        ΩΛ0=ΩΛ0_value,
    )
end

const _HALOFIT_Fν_SPLINE_TYPE = AkimaSpline{Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}, Vector{Float64}}
const _HALOFIT_Fν_SPLINE = Ref{_HALOFIT_Fν_SPLINE_TYPE}()

function _init_halofit_Fν_spline!()
    itp = ext.F_interpolant[]
    _HALOFIT_Fν_SPLINE[] = AkimaSpline(getfield(itp, :u), getfield(itp, :t))
    return nothing
end

function _halofit_Fν(y::Real)
    yy = float(y)
    T = typeof(yy)
    return convert(T, _HALOFIT_Fν_SPLINE[](yy))::T
end

function _halofit_background_z(cpar::HalofitCosmology, z::Real)
    T = promote_type(typeof(cpar.Ωm0), typeof(z))
    opz = one(T) + T(z)
    a = inv(opz)
    Ωcb0 = T(cpar.Ωm0 - cpar.Ωnu0)

    y = T(cpar.mν) * a / (T(_HALOFIT_KB_EV_PER_K) * T(_HALOFIT_TNU0_K))
    y0 = T(cpar.mν) / (T(_HALOFIT_KB_EV_PER_K) * T(_HALOFIT_TNU0_K))
    Fν_y = convert(T, _halofit_Fν(y))::T
    Fν_y0 = convert(T, _halofit_Fν(y0))::T
    Ωnu_Ez2 = T(cpar.Ωnu0) * (Fν_y / Fν_y0) / a^4

    f_de = a^(-3 * (one(T) + T(cpar.w0) + T(cpar.wa))) * exp(3 * T(cpar.wa) * (a - one(T)))
    Ez2 = Ωcb0 * opz^3 + T(cpar.Ωr0) * opz^4 + T(cpar.ΩΛ0) * f_de + Ωnu_Ez2

    Ωmz = (Ωcb0 * opz^3 + Ωnu_Ez2) / Ez2
    Ωvz = T(cpar.ΩΛ0) * f_de / Ez2
    return Ωmz::T, Ωvz::T
end

"""
    halofit_background(cpar::HalofitCosmology, z)

Compute the Halofit background correction inputs `(Ωm_z, Ωv_z)` for the
redshifts `z`. This host-side helper preserves the CLASS-parity massive
neutrino background convention used by the convenience `halofit_Pmm` wrappers.

For Reactant/XLA workflows, compute these quantities outside Halofit and call
the `halofit_Pmm(..., Ωm_z, Ωv_z)` methods directly.
"""
function halofit_background(cpar::HalofitCosmology, z::AbstractVector)
    T = promote_type(typeof(cpar.Ωm0), eltype(z))
    Ωm_z = Vector{T}(undef, length(z))
    Ωv_z = Vector{T}(undef, length(z))

    for i in eachindex(z)
        Ωm_z[i], Ωv_z[i] = _halofit_background_z(cpar, z[i])
    end

    return Ωm_z, Ωv_z
end

function halofit_background(cpar::HalofitCosmology, z::Number)
    return _halofit_background_z(cpar, z)
end

function halofit_background(input_params::AbstractVector, z; kwargs...)
    return halofit_background(halofit_cosmology(input_params; kwargs...), z)
end

function _halofit_integrate(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("integration grids must have the same length"))
    n >= 2 || throw(ArgumentError("integration requires at least two samples"))

    total = zero(promote_type(eltype(x), eltype(y)))
    last_simpson = isodd(n) ? n : n - 1

    i = 1
    while i + 2 <= last_simpson
        x0, x1, x2 = x[i], x[i + 1], x[i + 2]
        y0, y1, y2 = y[i], y[i + 1], y[i + 2]
        h0 = x1 - x0
        h1 = x2 - x1
        total += (h0 + h1) / 6 * (
            (2 - h1 / h0) * y0 +
            ((h0 + h1)^2 / (h0 * h1)) * y1 +
            (2 - h0 / h1) * y2
        )
        i += 2
    end

    if iseven(n)
        total += (x[end] - x[end - 1]) * (y[end] + y[end - 1]) / 2
    end

    return total
end

function _halofit_σ2_derivs(logk, k, pk_lin, R)
    kR2 = (k .* R) .^ 2
    exp_term = exp.(-kR2)
    integrand_pre = (k .^ 3) .* pk_lin ./ (2π^2)

    sig2 = _halofit_integrate(logk, integrand_pre .* exp_term)
    dsig2dR = -2 * R * _halofit_integrate(logk, integrand_pre .* (k .^ 2) .* exp_term)
    d1 = dsig2dR * R / sig2

    d2sig2dR2 = _halofit_integrate(logk,
        integrand_pre .* (k .^ 2) .* exp_term .* (-2 .+ 4 .* kR2))
    d2 = (R^2 / sig2) * d2sig2dR2 + d1 - d1^2

    return sig2, d1, d2
end

function _halofit_rnl(logk, k, pk_lin)
    lR = zero(eltype(logk))
    for _ in 1:_HALOFIT_NEWTON_STEPS
        sig2, d1, _ = _halofit_σ2_derivs(logk, k, pk_lin, exp(lR))
        lR -= log(sig2) / d1
    end
    return exp(lR)
end

function _halofit_power(cpar::HalofitCosmology, pk_lin_mm, k, z, rnl, neff, cur, Ωmz, Ωvz)
    T = promote_type(eltype(k), eltype(pk_lin_mm), typeof(z), typeof(rnl), typeof(neff),
                     typeof(cur), typeof(Ωmz), typeof(Ωvz))
    rk = k
    anorm = inv(T(2) * T(π)^2)
    opz = one(T) + T(z)
    wz = T(cpar.w0) + T(cpar.wa) * (T(z) / opz)
    fν = T(cpar.Ωnu0) / T(cpar.Ωm0)
    y = rk .* rnl

    gam = T(0.1971) - T(0.0843) * neff + T(0.8460) * cur
    an = T(10) ^ (T(1.5222) + T(2.8553)*neff + T(2.3706)*neff^2 +
                  T(0.9903)*neff^3 + T(0.2250)*neff^4 - T(0.6038)*cur +
                  T(0.1749)*Ωvz*(one(T) + wz))
    bn = T(10) ^ (-T(0.5642) + T(0.5864)*neff + T(0.5716)*neff^2 -
                  T(1.5474)*cur + T(0.2279)*Ωvz*(one(T) + wz))
    cn = T(10) ^ (T(0.3698) + T(2.0404)*neff + T(0.8161)*neff^2 + T(0.5869)*cur)
    xμ = zero(T)
    xν = T(10) ^ (T(5.2105) + T(3.6902)*neff)
    α = abs(T(6.0835) + T(1.3373)*neff - T(0.1959)*neff^2 - T(5.5274)*cur)
    β = T(2.0379) - T(0.7354)*neff + T(0.3157)*neff^2 + T(1.2490)*neff^3 +
        T(0.3980)*neff^4 - T(0.1682)*cur + fν * (T(1.081) + T(0.395)*neff^2)

    use_de = abs(one(T) - Ωmz) > T(0.01)
    denom = ifelse(use_de, one(T) - Ωmz, one(T))
    frac = Ωvz / denom
    f1a, f2a, f3a = Ωmz^-T(0.0732), Ωmz^-T(0.1423), Ωmz^T(0.0725)
    f1b, f2b, f3b = Ωmz^-T(0.0307), Ωmz^-T(0.0585), Ωmz^T(0.0743)
    f1_de = frac * f1b + (one(T) - frac) * f1a
    f2_de = frac * f2b + (one(T) - frac) * f2a
    f3_de = frac * f3b + (one(T) - frac) * f3a
    f1 = ifelse(use_de, f1_de, one(T))
    f2 = ifelse(use_de, f2_de, one(T))
    f3 = ifelse(use_de, f3_de, one(T))

    pk_halo_raw = an .* y .^ (f1 * T(3)) ./
                  (one(T) .+ bn .* y .^ f2 .+ (f3 .* cn .* y) .^ (T(3) - gam))
    pk_halo_dim = pk_halo_raw ./ (one(T) .+ xμ ./ y .+ xν ./ y .^ 2)
    pk_halo_dim = pk_halo_dim .* (one(T) + fν * T(0.977))

    Δ2_lin = (rk .^ 3) .* pk_lin_mm .* anorm
    kh = rk ./ T(cpar.h)
    Δ2_linaa = Δ2_lin .* (one(T) .+ fν .* T(47.48) .* kh .^ 2 ./ (one(T) .+ T(1.5) .* kh .^ 2))
    pk_quasi_dim = Δ2_lin .* (one(T) .+ Δ2_linaa) .^ β ./
                   (one(T) .+ Δ2_linaa .* α) .* exp.(-y ./ T(4) .- y .^ 2 ./ T(8))

    return (pk_halo_dim .+ pk_quasi_dim) ./ (rk .^ 3 .* anorm)
end

function _halofit_Pmm_one_z_unchecked(cpar::HalofitCosmology, z::Number,
    k::AbstractVector, pk_lin_mm::AbstractVector, Ωm_z::Number, Ωv_z::Number)

    logk = log.(k)
    rnl = _halofit_rnl(logk, k, pk_lin_mm)
    _, d1, d2 = _halofit_σ2_derivs(logk, k, pk_lin_mm, rnl)
    neff = -3 - d1
    cur = -d2
    return _halofit_power(cpar, pk_lin_mm, k, z, rnl, neff, cur, Ωm_z, Ωv_z)
end

function _halofit_Pmm_unchecked(cpar::HalofitCosmology, z::AbstractVector,
    k::AbstractVector, pk_lin_mm_z::AbstractMatrix, Ωm_z::AbstractVector,
    Ωv_z::AbstractVector)

    pk_nl = similar(pk_lin_mm_z,
                    promote_type(eltype(pk_lin_mm_z), eltype(k), eltype(z),
                                 eltype(Ωm_z), eltype(Ωv_z)),
                    length(k), length(z))

    for i in eachindex(z)
        pk_nl[:, i] = _halofit_Pmm_one_z_unchecked(cpar, z[i], k,
                                                   @view(pk_lin_mm_z[:, i]),
                                                   Ωm_z[i], Ωv_z[i])
    end

    return pk_nl
end

function _validate_halofit_inputs(z::AbstractVector, k::AbstractVector, pk_lin_mm_z::AbstractMatrix)
    size(pk_lin_mm_z) == (length(k), length(z)) || throw(DimensionMismatch(
        "pk_lin_mm_z must have size (length(k), length(z)); got $(size(pk_lin_mm_z)) " *
        "for length(k)=$(length(k)), length(z)=$(length(z))."
    ))
    all(k .> 0) || throw(ArgumentError("Halofit requires strictly positive k values."))
    all(diff(k) .> 0) || throw(ArgumentError("Halofit requires k values sorted in ascending order."))
    return nothing
end

function _validate_halofit_background(z::AbstractVector, Ωm_z::AbstractVector, Ωv_z::AbstractVector)
    length(Ωm_z) == length(z) || throw(DimensionMismatch(
        "Ωm_z must have length(z)=$(length(z)); got length(Ωm_z)=$(length(Ωm_z))."
    ))
    length(Ωv_z) == length(z) || throw(DimensionMismatch(
        "Ωv_z must have length(z)=$(length(z)); got length(Ωv_z)=$(length(Ωv_z))."
    ))
    return nothing
end

"""
    halofit_Pmm(cpar::HalofitCosmology, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)

Compute the nonlinear total-matter power spectrum using the CLASS-parity
Takahashi/Bird Halofit translation. `k` is in `1/Mpc`, `z` is a vector of
redshifts, and `pk_lin_mm_z` has shape `(length(k), length(z))`.

`Ωm_z` and `Ωv_z` are the matter and dark-energy density fractions evaluated at
each redshift. Supplying them explicitly keeps the Halofit calculator separate
from the background model; this is the preferred API for Reactant/XLA paths.
"""
function halofit_Pmm(cpar::HalofitCosmology, z::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix, Ωm_z::AbstractVector, Ωv_z::AbstractVector)

    _validate_halofit_inputs(z, k, pk_lin_mm_z)
    _validate_halofit_background(z, Ωm_z, Ωv_z)
    return _halofit_Pmm_unchecked(cpar, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix)

    Ωm_z, Ωv_z = halofit_background(cpar, z)
    return halofit_Pmm(cpar, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::Number, k::AbstractVector,
    pk_lin_mm::AbstractVector, Ωm_z::Number, Ωv_z::Number)

    _validate_halofit_inputs([z], k, reshape(pk_lin_mm, :, 1))
    return _halofit_Pmm_one_z_unchecked(cpar, z, k, pk_lin_mm, Ωm_z, Ωv_z)
end

function halofit_Pmm(cpar::HalofitCosmology, z::Number, k::AbstractVector,
    pk_lin_mm::AbstractVector)

    Ωm_z, Ωv_z = halofit_background(cpar, z)
    return halofit_Pmm(cpar, z, k, pk_lin_mm, Ωm_z, Ωv_z)
end

function halofit_Pmm(input_params::AbstractVector, z::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix, Ωm_z::AbstractVector, Ωv_z::AbstractVector; kwargs...)

    return halofit_Pmm(halofit_cosmology(input_params; kwargs...), z, k, pk_lin_mm_z,
                       Ωm_z, Ωv_z)
end

function halofit_Pmm(input_params::AbstractVector, z::Number, k::AbstractVector,
    pk_lin_mm::AbstractVector, Ωm_z::Number, Ωv_z::Number; kwargs...)

    return halofit_Pmm(halofit_cosmology(input_params; kwargs...), z, k, pk_lin_mm,
                       Ωm_z, Ωv_z)
end

function halofit_Pmm(input_params::AbstractVector, z::AbstractVector, k::AbstractVector,
    pk_lin_mm_z::AbstractMatrix; kwargs...)

    return halofit_Pmm(halofit_cosmology(input_params; kwargs...), z, k, pk_lin_mm_z)
end

function halofit_Pmm(input_params::AbstractVector, z::Number, k::AbstractVector,
    pk_lin_mm::AbstractVector; kwargs...)

    return halofit_Pmm(halofit_cosmology(input_params; kwargs...), z, k, pk_lin_mm)
end

"""
    get_halofit_Pmm(input_params, z, D, PkEmu::PkEmulator; kwargs...)

Compute linear `Pmm` with `PkEmu.LinearPmm` and apply Halofit instead of the
emulated nonlinear boost. This is useful when a fast CLASS-like nonlinear
fallback is desired on the linear-emulator k-grid.
"""
function get_halofit_Pmm(input_params, z::AbstractVector, D::AbstractVector,
    PkEmu::PkEmulator; kwargs...)

    k = get_kgrid(PkEmu.LinearPmm)
    pk_lin = get_linear_Pmm(input_params, z, D, PkEmu)
    return halofit_Pmm(input_params, z, k, pk_lin; kwargs...)
end

function get_halofit_Pmm(input_params, z::AbstractVector, D::AbstractVector,
    PkEmu::PkEmulator, Ωm_z::AbstractVector, Ωv_z::AbstractVector; kwargs...)

    k = get_kgrid(PkEmu.LinearPmm)
    pk_lin = get_linear_Pmm(input_params, z, D, PkEmu)
    return halofit_Pmm(input_params, z, k, pk_lin, Ωm_z, Ωv_z; kwargs...)
end

function get_halofit_Pmm(input_params, z::Number, D::Number, PkEmu::PkEmulator; kwargs...)
    k = get_kgrid(PkEmu.LinearPmm)
    pk_lin = get_linear_Pmm(input_params, z, D, PkEmu)
    return halofit_Pmm(input_params, z, k, pk_lin; kwargs...)
end

function get_halofit_Pmm(input_params, z::Number, D::Number, PkEmu::PkEmulator,
    Ωm_z::Number, Ωv_z::Number; kwargs...)

    k = get_kgrid(PkEmu.LinearPmm)
    pk_lin = get_linear_Pmm(input_params, z, D, PkEmu)
    return halofit_Pmm(input_params, z, k, pk_lin, Ωm_z, Ωv_z; kwargs...)
end
