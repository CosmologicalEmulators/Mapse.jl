# cosmology.jl
using Interpolations
using Integrals
using OrdinaryDiffEqTsit5
using OrdinaryDiffEqCore

struct HMcodeCosmology
    Omega_m::Float64
    Omega_b::Float64
    h::Float64
    n_s::Float64
    sigma_8::Float64
    w0::Float64
    wa::Float64
    Omega_nu::Float64
    Omega_k::Float64
end

const rho_critical = 2.77536627245708E11 # Msun/h / (Mpc/h)^3
const Dv0 = 18.0 * pi^2
const dc0 = (3.0 / 20.0) * (12.0 * pi)^(2.0 / 3.0)

comoving_matter_density(Om_m::Float64) = rho_critical * Om_m

Lagrangian_radius(M::Float64, Om_m::Float64) = cbrt(3.0 * M / (4.0 * pi * comoving_matter_density(Om_m)))
Lagrangian_radius(M::AbstractArray, Om_m::Float64) = Lagrangian_radius.(M, Om_m)

redshift_from_scalefactor(a::Float64) = -1.0 + 1.0 / a
scalefactor_from_redshift(z::Float64) = 1.0 / (1.0 + z)

function _f_Mead(x::Float64, y::Float64, p0::Float64, p1::Float64, p2::Float64, p3::Float64)
    return p0 + p1*(1.0 - x) + p2*(1.0 - x)^2 + p3*(1.0 - y)
end

function dc_Mead(a::Float64, Om_m::Float64, f_nu::Float64, g::Float64, G::Float64)
    p10, p11, p12, p13 = -0.0069, -0.0208, 0.0312, 0.0021
    p20, p21, p22, p23 = 0.0001, -0.0647, -0.0417, 0.0646
    
    dc = 1.0
    dc += _f_Mead(g/a, G/a, p10, p11, p12, p13) * log10(Om_m)
    dc += _f_Mead(g/a, G/a, p20, p21, p22, p23)
    return dc * dc0 * (1.0 - 0.041 * f_nu)
end

function Dv_Mead(a::Float64, Om_m::Float64, f_nu::Float64, g::Float64, G::Float64)
    p30, p31, p32, p33 = -0.79, -10.17, 2.51, 6.51
    p40, p41, p42, p43 = -1.89, 0.38, 18.8, -15.87
    
    Dv = 1.0
    Dv += _f_Mead(g/a, G/a, p30, p31, p32, p33) * log10(Om_m)
    Dv += _f_Mead(g/a, G/a, p40, p41, p42, p43) * log10(Om_m)^2
    return Dv * Dv0 * (1.0 + 0.763 * f_nu)
end

function _w(a::Float64, cosmo::HMcodeCosmology; LCDM::Bool=false)
    w0, wa = LCDM ? (-1.0, 0.0) : (cosmo.w0, cosmo.wa)
    return w0 + (1.0 - a) * wa
end

function _X_w(a::Float64, cosmo::HMcodeCosmology; LCDM::Bool=false)
    w0, wa = LCDM ? (-1.0, 0.0) : (cosmo.w0, cosmo.wa)
    return a^(-3.0 * (1.0 + w0 + wa)) * exp(-3.0 * wa * (1.0 - a))
end

function _Hubble2(a::Float64, cosmo::HMcodeCosmology; LCDM::Bool=false)
    Om_w = LCDM ? (1.0 - cosmo.Omega_m) : (1.0 - cosmo.Omega_m - cosmo.Omega_k)
    Om = LCDM ? 1.0 : (1.0 - cosmo.Omega_k)
    return cosmo.Omega_m * a^-3 + Om_w * _X_w(a, cosmo, LCDM=LCDM) + (1.0 - Om) * a^-2
end

function _Omega_m_a(a::Float64, cosmo::HMcodeCosmology; LCDM::Bool=false)
    return cosmo.Omega_m * a^-3 / _Hubble2(a, cosmo, LCDM=LCDM)
end

function _AH(a::Float64, cosmo::HMcodeCosmology; LCDM::Bool=false)
    Om_w = LCDM ? (1.0 - cosmo.Omega_m) : (1.0 - cosmo.Omega_m - cosmo.Omega_k)
    return -0.5 * (cosmo.Omega_m * a^-3 + (1.0 + 3.0 * _w(a, cosmo, LCDM=LCDM)) * Om_w * _X_w(a, cosmo, LCDM=LCDM))
end

function get_growth_interpolator(cosmo::HMcodeCosmology; LCDM::Bool=false)
    a_init = 1e-4
    na = 129
    a_vals = range(a_init, 1.0, length=na)
    f_init = 1.0 - _Omega_m_a(a_init, cosmo, LCDM=LCDM)
    d = a_init^(1.0 - 3.0 * f_init / 5.0)
    v = (1.0 - 3.0 * f_init / 5.0) * a_init^(-3.0 * f_init / 5.0)
    
    function growth_ode!(du, u, p, a)
        d_val, v_val = u
        cosmo, LCDM = p
        fv = -(2.0 + _AH(a, cosmo, LCDM=LCDM) / _Hubble2(a, cosmo, LCDM=LCDM)) * v_val / a
        fd = 1.5 * _Omega_m_a(a, cosmo, LCDM=LCDM) * d_val / a^2
        du[1] = v_val
        du[2] = fv + fd
    end
    
    u0 = [d, v]
    prob = ODEProblem(growth_ode!, u0, (a_init, 1.0), (cosmo, LCDM))
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8)
    
    g_vals = [sol(a)[1] for a in a_vals]
    
    itp = scale(interpolate(g_vals, BSpline(Cubic(Line(OnGrid())))), a_vals)
    return extrapolate(itp, Line())
end

function get_accumulated_growth(a::Float64, g_func)
    a_init = 1e-4
    missing_val = g_func(a_init)
    prob = IntegralProblem((x, p) -> g_func(x) / x, (a_init, a))
    G = solve(prob, QuadGKJL(), reltol=1e-6).u
    return G + missing_val
end
