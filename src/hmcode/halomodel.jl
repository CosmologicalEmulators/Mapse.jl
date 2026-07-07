# halomodel.jl
using Trapz
using Integrals
using SpecialFunctions

const TINKER_Z_DEP = false
const TINKER_PBS = false

struct TinkerParams
    alpha::Float64
    beta::Float64
    gamma::Float64
    phi::Float64
    eta::Float64
    A::Float64
    a::Float64
    B::Float64
    b::Float64
    C::Float64
    c::Float64
end

struct STParams
    p::Float64
    q::Float64
    A::Float64
end

struct HaloModel
    z::Float64
    a::Float64
    Om_m::Float64
    name::String
    dc::Float64
    Dv::Float64
    rhom::Float64
    tinker::Union{Nothing, TinkerParams}
    st::Union{Nothing, STParams}
end

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

_is_monotonic(x::AbstractVector) = all(diff(x) .> 0.0)

function _linear_interp_extrap(x::Real, xs::AbstractVector, ys::AbstractVector)
    @assert length(xs) == length(ys)
    if x <= xs[1]
        return ys[1] + (x - xs[1]) * (ys[2] - ys[1]) / (xs[2] - xs[1])
    elseif x >= xs[end]
        n = length(xs)
        return ys[n-1] + (x - xs[n-1]) * (ys[n] - ys[n-1]) / (xs[n] - xs[n-1])
    end
    i = searchsortedlast(xs, x)
    i == length(xs) && return ys[end]
    x0, x1 = xs[i], xs[i+1]
    y0, y1 = ys[i], ys[i+1]
    return y0 + (x - x0) * (y1 - y0) / (x1 - x0)
end

# ------------------------------------------------------------
# Constructor
# ------------------------------------------------------------

function HaloModel(
    z::Real,
    Om_m::Real;
    name::AbstractString = "Tinker et al. (2010)",
    Dv::Real = 200.0,
    dc::Real = 1.686,
)
    a = scalefactor_from_redshift(z)
    rhom = comoving_matter_density(Om_m)

    tinker = nothing
    st = nothing

    if name == "Press & Schecter (1974)"
        # no params
    elseif name == "Sheth & Tormen (1999)" || name == "Sheth, Mo & Tormen (2001)" || name == "Despali et al. (2016)"
        if name == "Despali et al. (2016)"
            p = 0.2536
            q = 0.7689
            A = 0.3295 * sqrt(2.0 * q / pi)
        else
            p = 0.3
            q = 0.707
            A = sqrt(2.0 * q) / (sqrt(pi) + gamma(0.5 - p) / 2.0^p)
        end
        st = STParams(p, q, A)
    elseif name == "Tinker et al. (2010)"
        Dv_array = [200.0, 300.0, 400.0, 600.0, 800.0, 1200.0, 1600.0, 2400.0, 3200.0]
        logDv_array = log.(Dv_array)
        logDv = log(float(Dv))

        alpha_array = [0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327]
        beta_array  = [0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702]
        gamma_array = [0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81]
        phi_array   = [-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49]
        eta_array   = [-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336]

        alpha = _linear_interp_extrap(logDv, logDv_array, alpha_array)
        beta  = _linear_interp_extrap(logDv, logDv_array, beta_array)
        gamma_t = _linear_interp_extrap(logDv, logDv_array, gamma_array)
        phi   = _linear_interp_extrap(logDv, logDv_array, phi_array)
        eta_t = _linear_interp_extrap(logDv, logDv_array, eta_array)

        if TINKER_Z_DEP
            beta *= (1.0 + z)^0.20
            gamma_t *= (1.0 + z)^-0.01
            phi *= (1.0 + z)^-0.08
            eta_t *= (1.0 + z)^0.27
        end

        y = log10(float(Dv))
        exp_term = exp(-(4.0 / y)^4)
        A = 1.0 + 0.24 * y * exp_term
        a_t = 0.44 * y - 0.88
        B = 0.183
        b_t = 1.5
        C = 0.019 + 0.107 * y + 0.19 * exp_term
        c_t = 2.4

        tinker = TinkerParams(alpha, beta, gamma_t, phi, eta_t, A, a_t, B, b_t, C, c_t)
    else
        throw(ArgumentError("Halo model not recognised: $name"))
    end

    return HaloModel(float(z), float(a), float(Om_m), String(name), float(dc), float(Dv), float(rhom), tinker, st)
end


# Backward-compatible constructor used by older local tests
function HaloModel(z::Real, Om_m::Real, Dv::Real, dc::Real; name::AbstractString = "Tinker et al. (2010)")
    HaloModel(z, Om_m; name=name, Dv=Dv, dc=dc)
end

# ------------------------------------------------------------
# ν-space mass function and bias
# ------------------------------------------------------------

function mass_function_nu(hmod::HaloModel, nu::AbstractVector)
    if hmod.name == "Press & Schecter (1974)"
        return @. sqrt(2.0 / pi) * exp(-(nu^2) / 2.0)
    elseif hmod.name == "Sheth & Tormen (1999)" || hmod.name == "Sheth, Mo & Tormen (2001)" || hmod.name == "Despali et al. (2016)"
        st = hmod.st::STParams
        return @. st.A * (1.0 + (st.q * nu^2)^(-st.p)) * exp(-st.q * nu^2 / 2.0)
    elseif hmod.name == "Tinker et al. (2010)"
        tp = hmod.tinker::TinkerParams
        f1 = @. 1.0 + (tp.beta * nu)^(-2.0 * tp.phi)
        f2 = @. nu^(2.0 * tp.eta)
        f3 = @. exp(-tp.gamma * nu^2 / 2.0)
        return @. tp.alpha * f1 * f2 * f3
    else
        throw(ArgumentError("Halo model not recognised in mass_function_nu"))
    end
end
mass_function_nu(hmod::HaloModel, nu::Real) = mass_function_nu(hmod, [float(nu)])[1]

function linear_bias_nu(hmod::HaloModel, nu::AbstractVector)
    if hmod.name == "Press & Schecter (1974)"
        return @. 1.0 + (nu^2 - 1.0) / hmod.dc
    elseif hmod.name == "Sheth & Tormen (1999)" || hmod.name == "Despali et al. (2016)"
        st = hmod.st::STParams
        return @. 1.0 + (st.q * nu^2 - 1.0 + 2.0 * st.p / (1.0 + (st.q * nu^2)^st.p)) / hmod.dc
    elseif hmod.name == "Sheth, Mo & Tormen (2001)"
        # Not required by current HMcode path, but keep for completeness
        st = hmod.st::STParams
        a = 0.707
        b = 0.5
        c = 0.6
        anu2 = @. a * nu^2
        f1 = @. sqrt(a) * anu2
        f2 = @. sqrt(a) * b * anu2^(1.0 - c)
        f3 = @. anu2^c
        f4 = @. anu2^c + b * (1.0 - c) * (1.0 - c / 2.0)
        return @. 1.0 + (f1 + f2 - f3 / f4) / (hmod.dc * sqrt(a))
    elseif hmod.name == "Tinker et al. (2010)"
        tp = hmod.tinker::TinkerParams
        if TINKER_PBS
            f1 = @. (tp.gamma * nu^2 - (1.0 + 2.0 * tp.eta)) / hmod.dc
            f2 = @. (2.0 * tp.phi / hmod.dc) / (1.0 + (tp.beta * nu)^(2.0 * tp.phi))
            return @. 1.0 + f1 + f2
        else
            fA = @. tp.A * nu^tp.a / (nu^tp.a + hmod.dc^tp.a)
            fB = @. tp.B * nu^tp.b
            fC = @. tp.C * nu^tp.c
            return @. 1.0 - fA + fB + fC
        end
    else
        throw(ArgumentError("Halo model not recognised in linear_bias_nu"))
    end
end
linear_bias_nu(hmod::HaloModel, nu::Real) = linear_bias_nu(hmod, [float(nu)])[1]

# ------------------------------------------------------------
# Mass-space wrappers
# ------------------------------------------------------------

_peak_height(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector) = hmod.dc ./ sigmaM

Lagrangian_radius(hmod::HaloModel, M::AbstractVector) = Lagrangian_radius(M, hmod.Om_m)
virial_radius(hmod::HaloModel, M::AbstractVector) = Lagrangian_radius(hmod, M) ./ cbrt(hmod.Dv)

function multiplicity_function(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector)
    nu = _peak_height(hmod, M, sigmaM)
    R = Lagrangian_radius(hmod, M)
    logR, logsigmaM = log.(R), log.(sigmaM)
    deriv = similar(logR)
    @inbounds for i in eachindex(logR)
        deriv[i] = 2.0 * derivative_from_samples(logR[i], logR, logsigmaM)
    end
    dnu_dlnm = @. -(nu / 6.0) * deriv
    return mass_function_nu(hmod, nu) .* dnu_dlnm
end

function mass_function(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector)
    F = multiplicity_function(hmod, M, sigmaM)
    return @. F * hmod.rhom / M^2
end

function linear_bias(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector)
    nu = _peak_height(hmod, M, sigmaM)
    return linear_bias_nu(hmod, nu)
end

function average(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector, func::AbstractVector)
    _is_monotonic(M) || throw(ArgumentError("Halo mass array must be increasing monotonically"))
    nu = _peak_height(hmod, M, sigmaM)
    integrand = (func ./ M) .* mass_function_nu(hmod, nu)
    return trapz(nu, integrand) * hmod.rhom
end

"""
Missing low-mass contribution to integral b(nu) g(nu) dnu in pyhalomodel.
"""
function _missing_bias_mass(hmod::HaloModel, nu_min::Real)
    prob = IntegralProblem((nu, p) -> mass_function_nu(hmod, nu) * linear_bias_nu(hmod, nu), (float(nu_min), Inf))
    val = solve(prob, QuadGKJL()).u
    return 1.0 - val
end

# ------------------------------------------------------------
# pyhalomodel-like power internals
# ------------------------------------------------------------

function _I_2h(
    hmod::HaloModel,
    M::AbstractVector,
    nu::AbstractVector,
    W::AbstractVector,
    mass::Bool,
    A::Real,
)
    integrand = W .* linear_bias_nu(hmod, nu) .* mass_function_nu(hmod, nu) ./ M
    I_2h = trapz(nu, integrand)
    if mass
        I_2h += A * W[1] / M[1]
    end
    return I_2h * hmod.rhom
end

function _Pk_2h(
    hmod::HaloModel,
    Pk_lin::Real,
    M::AbstractVector,
    nu::AbstractVector,
    Wu::AbstractVector,
    Wv::AbstractVector,
    mass_u::Bool,
    mass_v::Bool,
    A::Real,
)
    Iu = _I_2h(hmod, M, nu, Wu, mass_u, A)
    Iv = _I_2h(hmod, M, nu, Wv, mass_v, A)
    return (Iu * Iv) * Pk_lin
end

function _Pk_1h(
    hmod::HaloModel,
    M::AbstractVector,
    nu::AbstractVector,
    WuWv::AbstractVector,
)
    integrand = WuWv .* mass_function_nu(hmod, nu) ./ M
    return trapz(nu, integrand) * hmod.rhom
end

"""
Compute halo-model power spectra for provided tracer profiles.
Returns `(Pk_2h_dict, Pk_1h_dict, Pk_hm_dict)` with keys `"u-v"`.

This is a Julia equivalent of `pyhalomodel.model.power_spectrum` for the subset
used by HMcode (`simple_twohalo=true`, beta omitted).
"""
function power_spectrum(
    hmod::HaloModel,
    k::AbstractVector,
    Pk_lin::AbstractVector,
    M::AbstractVector,
    sigmaM::AbstractVector,
    profiles::Dict{String, HaloProfile};
    simple_twohalo::Bool = true,
    subtract_shotnoise::Bool = true,
    correct_discrete::Bool = true,
    k_trunc::Union{Nothing, Real} = nothing,
)
    _is_monotonic(M) || throw(ArgumentError("Halo mass array must be increasing monotonically"))

    nu = _peak_height(hmod, M, sigmaM)
    A = _missing_bias_mass(hmod, nu[1])

    # shot-noise bookkeeping
    Pk_SN = Dict{String, Float64}()
    for (name, profile) in profiles
        if profile.discrete_tracer
            Wsn = (profile.amplitude ./ profile.normalisation) .^ 2
            Pk_SN[name] = _Pk_1h(hmod, M, nu, Wsn)
        else
            Pk_SN[name] = 0.0
        end
    end

    Pk_2h_dict = Dict{String, Vector{Float64}}()
    Pk_1h_dict = Dict{String, Vector{Float64}}()
    Pk_hm_dict = Dict{String, Vector{Float64}}()

    names = collect(keys(profiles))
    for (iu, name_u) in enumerate(names)
        profile_u = profiles[name_u]
        for (iv, name_v) in enumerate(names)
            profile_v = profiles[name_v]
            power_name = string(name_u, "-", name_v)
            reverse_name = string(name_v, "-", name_u)

            if iu > iv
                Pk_2h_dict[power_name] = Pk_2h_dict[reverse_name]
                Pk_1h_dict[power_name] = Pk_1h_dict[reverse_name]
                Pk_hm_dict[power_name] = Pk_hm_dict[reverse_name]
                continue
            end

            P2 = zeros(Float64, length(k))
            P1 = zeros(Float64, length(k))

            if simple_twohalo
                Wu = profile_u.amplitude ./ profile_u.normalisation
                Wv = profile_v.amplitude ./ profile_v.normalisation
                @inbounds for ik in eachindex(k)
                    P2[ik] = _Pk_2h(hmod, Pk_lin[ik], M, nu, Wu, Wv, profile_u.mass_tracer, profile_v.mass_tracer, A)
                end
            else
                @inbounds for ik in eachindex(k)
                    P2[ik] = _Pk_2h(
                        hmod,
                        Pk_lin[ik],
                        M,
                        nu,
                        view(profile_u.Wk, ik, :),
                        view(profile_v.Wk, ik, :),
                        profile_u.mass_tracer,
                        profile_v.mass_tracer,
                        A,
                    )
                end
            end

            @inbounds for ik in eachindex(k)
                if name_u == name_v
                    if correct_discrete && profile_u.discrete_tracer
                        Wfac = profile_u.amplitude .* (profile_u.amplitude .- 1.0)
                    else
                        Wfac = profile_u.amplitude .^ 2
                    end
                    if profile_u.variance !== nothing
                        Wfac .+= profile_u.variance
                    end
                    Wprod = Wfac .* (view(profile_u.Uk, ik, :) ./ profile_u.normalisation) .^ 2
                else
                    Wprod = view(profile_u.Wk, ik, :) .* view(profile_v.Wk, ik, :)
                end
                P1[ik] = _Pk_1h(hmod, M, nu, Wprod)
            end

            if (name_u == name_v) && profile_u.discrete_tracer
                if correct_discrete && !subtract_shotnoise
                    P1 .+= Pk_SN[name_u]
                elseif !correct_discrete && subtract_shotnoise
                    P1 .-= Pk_SN[name_u]
                end
            end

            if k_trunc !== nothing
                P1 .*= (1 .- exp.(-(k ./ float(k_trunc)).^2))
            end

            Pk_2h_dict[power_name] = copy(P2)
            Pk_1h_dict[power_name] = copy(P1)
            Pk_hm_dict[power_name] = P2 .+ P1
        end
    end

    return Pk_2h_dict, Pk_1h_dict, Pk_hm_dict
end


# Backward-compatible alias
get_peak_height(hmod::HaloModel, M::AbstractVector, sigmaM::AbstractVector) = _peak_height(hmod, M, sigmaM)
