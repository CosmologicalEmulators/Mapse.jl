# profiles.jl
using SpecialFunctions
using Roots

"""
Container for halo profiles in Fourier space.

Fields follow pyhalomodel semantics:
- `Uk` stores dimensionless profile transforms
- `Wk` stores dimensionful profile transforms after amplitude/normalisation
"""
struct HaloProfile
    k::Vector{Float64}
    M::Vector{Float64}
    Uk::Matrix{Float64}           # [nk, nM]
    Wk::Matrix{Float64}           # [nk, nM]
    amplitude::Vector{Float64}
    normalisation::Float64
    variance::Union{Nothing, Vector{Float64}}
    mass_tracer::Bool
    discrete_tracer::Bool
end

"""
Create a Fourier-space halo profile equivalent to `pyhalomodel.profile.Fourier`.

If `amplitude === nothing`, `Uk` is treated as dimensionful and amplitudes are inferred
from low-k values (first k-bin), then converted to dimensionless internally.
"""
function Fourier(
    k::AbstractVector,
    M::AbstractVector,
    Uk_in::AbstractMatrix;
    amplitude::Union{Nothing, AbstractVector} = nothing,
    normalisation::Real = 1.0,
    variance::Union{Nothing, AbstractVector} = nothing,
    mass_tracer::Bool = false,
    discrete_tracer::Bool = false,
)
    nk, nM = length(k), length(M)
    size(Uk_in) == (nk, nM) || throw(ArgumentError("Uk shape must be (length(k), length(M))"))

    Uk = Array{Float64}(Uk_in)
    Wk = similar(Uk)

    amp = if amplitude === nothing
        vec(Uk[1, :])
    else
        length(amplitude) == nM || throw(ArgumentError("amplitude must have length(M) entries"))
        collect(Float64.(amplitude))
    end

    if amplitude === nothing
        Wk .= Uk
        @inbounds for j in 1:nM
            Uk[:, j] ./= amp[j]
        end
    else
        @inbounds for j in 1:nM
            Wk[:, j] .= (amp[j] .* Uk[:, j]) ./ float(normalisation)
        end
    end

    var = variance === nothing ? nothing : collect(Float64.(variance))
    return HaloProfile(collect(Float64.(k)), collect(Float64.(M)), Uk, Wk, amp, float(normalisation), var, mass_tracer, discrete_tracer)
end

# ------------------------------------------------------------
# Fast Si/Ci approximations (MacLeod 1996)
# ------------------------------------------------------------

const EULER_GAMMA = 0.5772156649015329

const SI_SMALL = (1.0, -0.05555555555555555, 0.0016666666666666668, -2.834467120181406e-5, 3.0619243582206544e-7, -2.27746439867652e-9, 1.2353110643708935e-11, -5.0981091545465446e-14, 1.6537983849091297e-16, -4.326650129802279e-19, 9.32044812542441e-22, -1.6818131176655147e-24, 2.5787801137537893e-27, -3.401366616220572e-30, 3.8999872022233505e-33, -3.922984005011348e-36, 3.489798672961197e-39, -2.7650265596091116e-42, 1.9636378862575867e-45, -1.257043527311165e-48, 7.291002017420499e-52, -3.849327599400454e-55, 1.8577001882628452e-58, -8.22686917863956e-62, 3.3550504251358757e-65, -1.264109733422975e-68)

@inline si_small(x) = x * evalpoly(x*x, SI_SMALL)

const CI_INT_SMALL = (-0.25, 0.010416666666666666, -0.0002314814814814815, 3.1001984126984127e-6, -2.755731922398589e-8, 1.7397297489890083e-10, -8.193389712664089e-13, 2.9871733327421158e-15, -8.677337204770125e-18, 2.0551588116560825e-20, -4.0439960874775335e-23, 6.715573212900493e-26, -9.53690870471076e-29, 1.1713890132392279e-31, -1.2566625429386353e-34, 1.1876221108921073e-37, -9.962228045650476e-41, 7.467278517462879e-44, -5.031482118527058e-47, 3.0640435978209645e-50, -1.6946206503074855e-53, 8.549642911891504e-57, -3.9506856555684327e-60, 1.6782241814065079e-63, -6.575898833266316e-67)

@inline ci_int_small(x) = x*x * evalpoly(x*x, CI_INT_SMALL)

# Rational approximations for f(x) and g(x) for x > 4.0
const P_F_RAT1 = (0.99999999962173909991E0, 0.36451060338631902917E3, 0.44218548041288440874E5, 0.22467569405961151887E7, 0.49315316723035561922E8, 0.43186795279670283193E9, 0.11847992519956804350E10, 0.45573267593795103181E9)
const Q_F_RAT1 = (1.0, 0.36651060273229347594E3, 0.44927569814970692777E5, 0.23285354882204041700E7, 0.53117852017228262911E8, 0.50335310667241870372E9, 0.16575285015623175410E10, 0.11746532837038341076E10)
const R_G_RAT1 = (0.99999999920484901956E0, 0.51385504875307321394E3, 0.92293483452013810811E5, 0.74071341863359841727E7, 0.28142356162841356551E9, 0.49280890357734623984E10, 0.35524762685554302472E11, 0.79194271662085049376E11, 0.17942522624413898907E11)
const S_G_RAT1 = (1.0, 0.51985504708814870209E3, 0.95292615508125947321E5, 0.79215459679762667578E7, 0.31977567790733781460E9, 0.62273134702439012114E10, 0.54570971054996441467E11, 0.18241750166645704670E12, 0.15407148148861454434E12)

const P_F_ASYM_NUM = (0.19999999999999978257E1, 0.22206119380434958727E4, 0.84749007623988236808E6, 0.13959267954823943232E9, 0.10197205463267975592E11, 0.30229865264524075951E12, 0.27504053804288471142E13, 0.21818989704686874983E13)
const P_F_ASYM_DEN = (1.0, 0.11223059690217167788E4, 0.43685270974851313242E6, 0.74654702140658116258E8, 0.58580034751805687471E10, 0.20157980379272098841E12, 0.26229141857684496445E13, 0.87852907334918467516E13)
const R_G_ASYM_NUM = (0.59999999999999993089E1, 0.96527746044997139158E4, 0.56077626996568834185E7, 0.15022667718927317198E10, 0.19644271064733088465E12, 0.12191368281163225043E14, 0.31924389898645609533E15, 0.25876053010027485934E16, 0.12754978896268878403E16)
const R_G_ASYM_DEN = (1.0, 0.16287957674166143196E4, 0.96636303195787870963E6, 0.26839734750950667021E9, 0.37388510548029219241E11, 0.26028585666152144496E13, 0.85134283716950697226E14, 0.11304079361627952930E16, 0.42519841479489798424E16)

@inline function si_fast(x::Float64)::Float64
    if x <= 4.0
        return si_small(x)
    end
    t = x*x
    invt = inv(t)
    sx, cx = sin(x), cos(x)
    if t <= 144.0
        f = (evalpoly(invt, P_F_RAT1) / evalpoly(invt, Q_F_RAT1)) / x
        g = (evalpoly(invt, R_G_RAT1) / evalpoly(invt, S_G_RAT1)) * invt
    else
        f = (1.0 - evalpoly(invt, P_F_ASYM_NUM) * invt / evalpoly(invt, P_F_ASYM_DEN)) / x
        g = (1.0 - evalpoly(invt, R_G_ASYM_NUM) * invt / evalpoly(invt, R_G_ASYM_DEN)) * invt
    end
    return pi/2.0 - f*cx - g*sx
end

@inline function ci_fast(x::Float64)::Float64
    if x <= 4.0
        return EULER_GAMMA + log(x) + ci_int_small(x)
    end
    t = x*x
    invt = inv(t)
    sx, cx = sin(x), cos(x)
    if t <= 144.0
        f = (evalpoly(invt, P_F_RAT1) / evalpoly(invt, Q_F_RAT1)) / x
        g = (evalpoly(invt, R_G_RAT1) / evalpoly(invt, S_G_RAT1)) * invt
    else
        f = (1.0 - evalpoly(invt, P_F_ASYM_NUM) * invt / evalpoly(invt, P_F_ASYM_DEN)) / x
        g = (1.0 - evalpoly(invt, R_G_ASYM_NUM) * invt / evalpoly(invt, R_G_ASYM_DEN)) * invt
    end
    return f*sx - g*cx
end

# ------------------------------------------------------------
# NFW window functions
# ------------------------------------------------------------

"""Exact scalar NFW Fourier window from x ≡ k*rv and concentration c."""
@inline function wnfw_xc(x::Float64, c::Float64)
    xs = x / c
    Sisv = sinint(xs + x)
    Cisv = cosint(xs + x)
    Sis = sinint(xs)
    Cis = cosint(xs)

    f1 = cos(xs) * (Cisv - Cis)
    f2 = sin(xs) * (Sisv - Sis)
    f3 = sin(x) / (xs + x)
    f4 = log(1.0 + c) - c / (1.0 + c)
    return (f1 + f2 - f3) / f4
end

"""Fast scalar NFW window from x ≡ k*rv, c and precomputed ln(1+c)."""
@inline function wnfw_fast(x::Float64, c::Float64, ln1pc::Float64)::Float64
    x_plus  = x * (1.0 + 1.0/c)
    x_minus = x / c
    ΔSi = si_fast(x_plus) - si_fast(x_minus)
    ΔCi = ci_fast(x_plus) - ci_fast(x_minus)
    s, cv = sin(x_minus), cos(x_minus)
    sinc_xp = sin(x) / x_plus
    norm = ln1pc - c/(1.0 + c)
    return (ΔSi*s + ΔCi*cv - sinc_xp) / norm
end

using LoopVectorization

function win_NFW_fast!(out::AbstractVector{Float64}, k::AbstractVector{Float64}, rv_eff::Float64, c::Float64, ln1pc::Float64)
    @turbo for i in eachindex(k)
        out[i] = wnfw_fast(k[i] * rv_eff, c, ln1pc)
    end
    return out
end

function win_NFW_fast!(W_buf::AbstractMatrix{Float64}, k::AbstractVector{Float64}, rv_eff::AbstractVector{Float64}, c_vec::AbstractVector{Float64}, ln1pc::AbstractVector{Float64})
    nM, nk = size(W_buf)
    @turbo warn_check_args=false for ik in 1:nk, iM in 1:nM
        x = k[ik] * rv_eff[iM]
        W_buf[iM, ik] = wnfw_fast(x, c_vec[iM], ln1pc[iM])
    end
    return W_buf
end

function win_NFW!(out::AbstractVector{Float64}, k::AbstractVector, rv_eff::Float64, c::Float64)
    @inbounds for i in eachindex(k)
        out[i] = wnfw_xc(k[i] * rv_eff, c)
    end
    return out
end

"""Normalised Fourier transform for an NFW profile (scalar rv, c)."""
function win_NFW(k::AbstractVector, rv::Real, c::Real)
    out = Vector{Float64}(undef, length(k))
    return win_NFW!(out, k, float(rv), float(c))
end

"""
In-place NFW+baryons profile.
"""
function win_NFW_baryons!(
    out::AbstractVector{Float64},
    k::AbstractVector,
    rv_eff::Float64,
    c::Float64,
    ln1pc::Float64,
    M::Float64,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
)
    win_NFW_fast!(out, k, rv_eff, c, ln1pc)
    fg = (Om_b / Om_m - fstar) * (M / Mb)^2 / (1.0 + (M / Mb)^2)
    coeff = Om_c / Om_m + fg
    @inbounds for i in eachindex(out)
        out[i] = coeff * out[i] + fstar
    end
    return out
end

function win_NFW_baryons!(
    out::AbstractVector{Float64},
    k::AbstractVector,
    rv_eff::Float64,
    c::Float64,
    M::Float64,
    Mb::Float64,
    fstar::Float64,
    Om_m::Float64,
    Om_c::Float64,
    Om_b::Float64,
)
    win_NFW!(out, k, rv_eff, c)
    fg = (Om_b / Om_m - fstar) * (M / Mb)^2 / (1.0 + (M / Mb)^2)
    coeff = Om_c / Om_m + fg
    @inbounds for i in eachindex(out)
        out[i] = coeff * out[i] + fstar
    end
    return out
end

"""
Normalised Fourier transform for NFW profile including baryonic corrections
(Equation 25 in Mead et al. 2021).
"""
function win_NFW_baryons(
    k::AbstractVector,
    rv::Real,
    c::Real,
    M::Real,
    Mb::Real,
    fstar::Real,
    Om_m::Real,
    Om_c::Real,
    Om_b::Real,
)
    out = Vector{Float64}(undef, length(k))
    return win_NFW_baryons!(
        out,
        k,
        float(rv),
        float(c),
        float(M),
        float(Mb),
        float(fstar),
        float(Om_m),
        float(Om_c),
        float(Om_b),
    )
end

"""
Bullock et al. (2001)-style halo collapse redshifts used by HMcode concentration model.
"""
function get_halo_collapse_redshifts(
    M::AbstractVector,
    z::Real,
    dc::Real,
    Om_m::Real,
    growth,
    sigmaR_func,
)
    gamma = 0.01
    a = scalefactor_from_redshift(z)

    zf = similar(collect(Float64.(M)))
    @inbounds for (i, m) in enumerate(M)
        Mc = gamma * m
        Rc = Lagrangian_radius(Mc, Om_m)
        sigma = sigmaR_func(Rc)
        fac = growth(a) * dc / sigma

        af = if fac >= growth(a)
            a
        else
            find_zero(af -> growth(af) - fac, (1e-3, 1.0), A42())
        end
        zf[i] = redshift_from_scalefactor(af)
    end

    return zf
end
