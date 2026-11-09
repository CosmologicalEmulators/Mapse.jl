module MapseMooncakeExt

using Mapse, Mooncake, ForwardDiff, Integrals, Polyester

const H = Mapse.HMcode
const Q = Integrals.QuadGK

# Differentiate the actual approximation polynomials, not ideal Si/Ci functions.
# Constant derivative coefficients keep evalpoly's unrolled scalar evaluation.
_derivative_coefficients(p::NTuple{N,Float64}) where {N} =
    ntuple(i -> i * p[i + 1], Val(N - 1))

const D_SI_SMALL = _derivative_coefficients(H.SI_SMALL)
const D_CI_INT_SMALL = _derivative_coefficients(H.CI_INT_SMALL)
const D_P_F_RAT1 = _derivative_coefficients(H.P_F_RAT1)
const D_Q_F_RAT1 = _derivative_coefficients(H.Q_F_RAT1)
const D_R_G_RAT1 = _derivative_coefficients(H.R_G_RAT1)
const D_S_G_RAT1 = _derivative_coefficients(H.S_G_RAT1)
const D_P_F_ASYM_NUM = _derivative_coefficients(H.P_F_ASYM_NUM)
const D_P_F_ASYM_DEN = _derivative_coefficients(H.P_F_ASYM_DEN)
const D_R_G_ASYM_NUM = _derivative_coefficients(H.R_G_ASYM_NUM)
const D_R_G_ASYM_DEN = _derivative_coefficients(H.R_G_ASYM_DEN)

@inline function _ratio_with_derivative(u, p, q, dp, dq)
    numerator, denominator = evalpoly(u, p), evalpoly(u, q)
    ratio = numerator / denominator
    derivative = (evalpoly(u, dp) - ratio * evalpoly(u, dq)) / denominator
    return numerator, denominator, ratio, derivative
end

@inline function _sici_with_derivatives(x::Float64, remove_si_limit::Bool=false)
    t = x * x
    if x <= 4.0
        p, q = evalpoly(t, H.SI_SMALL), evalpoly(t, H.CI_INT_SMALL)
        si = x * p
        ci = H.EULER_GAMMA + log(x) + t * q
        dsi = p + 2t * evalpoly(t, D_SI_SMALL)
        dci = inv(x) + 2x * (q + t * evalpoly(t, D_CI_INT_SMALL))
        return si, ci, dsi, dci
    end
    u = inv(t)
    du = -2u / x
    if t <= 144.0
        _, _, rf, drf = _ratio_with_derivative(u, H.P_F_RAT1, H.Q_F_RAT1,
                                              D_P_F_RAT1, D_Q_F_RAT1)
        _, _, rg, drg = _ratio_with_derivative(u, H.R_G_RAT1, H.S_G_RAT1,
                                              D_R_G_RAT1, D_S_G_RAT1)
        f, g = rf / x, rg * u
        df, dg = (drf * du - f) / x, (drg * u + rg) * du
    else
        pf, qf, rf, drf = _ratio_with_derivative(u, H.P_F_ASYM_NUM, H.P_F_ASYM_DEN,
                                                D_P_F_ASYM_NUM, D_P_F_ASYM_DEN)
        pg, qg, rg, drg = _ratio_with_derivative(u, H.R_G_ASYM_NUM, H.R_G_ASYM_DEN,
                                                D_R_G_ASYM_NUM, D_R_G_ASYM_DEN)
        # Keep the value evaluation order used by sici_fast.
        f, g = (1.0 - pf * u / qf) / x, (1.0 - pg * u / qg) * u
        df = (-(drf * u + rf) * du - f) / x
        dg = (1.0 - 2rg * u - drg * u * u) * du
    end
    sx, cx = sin(x), cos(x)
    # For a difference of two large-argument Si values, omit the shared pi/2
    # before subtraction. Its rounding error otherwise contaminates the tail.
    si = remove_si_limit ? -f * cx - g * sx : pi / 2.0 - f * cx - g * sx
    ci = f * sx - g * cx
    dsi = (f - dg) * sx - (df + g) * cx
    dci = (df + g) * sx + (f - dg) * cx
    return si, ci, dsi, dci
end

@inline function _ci_regular_derivative(x)
    t = x * x
    return 2x * (evalpoly(t, H.CI_INT_SMALL) + t * evalpoly(t, D_CI_INT_SMALL))
end

@inline function _sinc_derivative(x)
    # d(sin(x)/x)/dx. Direct subtraction loses the small numerator near zero.
    # At |x| <= 1e-3 the first omitted term is bounded by |x|^9/3991680.
    if abs(x) <= 1e-3
        return x * evalpoly(x*x, (-1/3, 1/30, -1/840, 1/45360))
    end
    return (cos(x) - sin(x)/x) / x
end

# Recompute analytical scalar partials without Dual numbers or a reverse tape.
# Keep the matrix primal, restoration and shared-cotangent accumulation unchanged.
@inline function _nfw_partials(x::Float64, c::Float64, l::Float64)
    ic = inv(c)
    xp, xm = x * (1.0 + ic), x / c
    large_pair = xp > 4.0 && xm > 4.0
    sp, cp, dsp, dcp = _sici_with_derivatives(xp, large_pair)
    sm, cm, dsm, dcm = _sici_with_derivatives(xm, large_pair)
    ds, dc = sp - sm, cp - cm
    s, cv = sin(xm), cos(xm)
    sinc_xp = sin(x) / xp
    norm = l - c / (1.0 + c)
    value = (ds * s + dc * cv - sinc_xp) / norm
    phase_derivative = ds * cv - dc * s

    dxp_dx, dxm_dx = 1.0 + ic, ic
    dxp_dc, dxm_dc = -x * (ic / c), -xm / c
    if xp <= 4.0 && xm <= 4.0
        # Ci(q)=gamma+log(q)+regular(q). Cancel the logarithmic derivative
        # symbolically: log(xp/xm)=log(1+c), independent of x.
        rp, rm = _ci_regular_derivative(xp), _ci_regular_derivative(xm)
        dci_dx = rp * dxp_dx - rm * dxm_dx
        dci_dc = inv(1.0 + c) + rp * dxp_dc - rm * dxm_dc
    else
        dci_dx = dcp * dxp_dx - dcm * dxm_dx
        dci_dc = dcp * dxp_dc - dcm * dxm_dc
    end
    numerator_dx = (dsp * dxp_dx - dsm * dxm_dx) * s +
                   dci_dx * cv + phase_derivative * dxm_dx -
                   _sinc_derivative(x) / dxp_dx
    numerator_dc = (dsp * dxp_dc - dsm * dxm_dc) * s +
                   dci_dc * cv +
                   phase_derivative * dxm_dc + sinc_xp * dxp_dc / xp
    return (numerator_dx / norm,
            (numerator_dc + value / (1.0 + c)^2) / norm,
            -value / norm)
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H.win_NFW_fast!),
    StridedMatrix{Float64}, StridedVector{Float64}, StridedVector{Float64},
    StridedVector{Float64}, StridedVector{Float64}}

function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H.win_NFW_fast!)},
                         W::Mooncake.CoDual{<:StridedMatrix{Float64}},
                         k::Mooncake.CoDual{<:StridedVector{Float64}},
                         r::Mooncake.CoDual{<:StridedVector{Float64}},
                         c::Mooncake.CoDual{<:StridedVector{Float64}},
                         l::Mooncake.CoDual{<:StridedVector{Float64}})
    wv, dw = Mooncake.arrayify(W)
    kv, dk = Mooncake.arrayify(k)
    rv, dr = Mooncake.arrayify(r)
    cv, dc = Mooncake.arrayify(c)
    lv, dl = Mooncake.arrayify(l)
    # Read-only inputs may share storage and therefore cotangents. Overlap
    # with the overwritten output has no certified SIMD semantics.
    any(v -> Base.mightalias(wv, v), (kv, rv, cv, lv)) && throw(ArgumentError(
        "NFW reverse rule requires output storage disjoint from input storage"))
    original = copy(wv)
    H.win_NFW_fast!(wv, kv, rv, cv, lv)
    function nfw_pullback(_)
        for j in axes(wv, 2), i in axes(wv, 1)
            seed = dw[i, j]
            partials = _nfw_partials(kv[j] * rv[i], cv[i], lv[i])
            dk[j] += seed * partials[1] * rv[i]
            dr[i] += seed * partials[1] * kv[j]
            dc[i] += seed * partials[2]
            dl[i] += seed * partials[3]
        end
        fill!(dw, 0.0)
        copyto!(wv, original)
        return ntuple(_ -> Mooncake.NoRData(), 6)
    end
    return W, nfw_pullback
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H._one_halo_integral!),
    StridedVector{Float64}, StridedMatrix{Float64}, StridedMatrix{Float64},
    StridedVector{Float64}, Float64}

function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H._one_halo_integral!)},
                         integral::Mooncake.CoDual{<:StridedVector{Float64}},
                         squared::Mooncake.CoDual{<:StridedMatrix{Float64}},
                         profile::Mooncake.CoDual{<:StridedMatrix{Float64}},
                         weights::Mooncake.CoDual{<:StridedVector{Float64}},
                         density::Mooncake.CoDual{Float64})
    I, dI = Mooncake.arrayify(integral)
    W2, dW2 = Mooncake.arrayify(squared)
    W, dW = Mooncake.arrayify(profile)
    w, dw = Mooncake.arrayify(weights)
    rho = Mooncake.primal(density)
    # This rule supports disjoint output buffers. Read-only inputs may alias;
    # arrayify preserves their shared cotangent storage.
    if Base.mightalias(I, W2) || Base.mightalias(I, W) || Base.mightalias(I, w) ||
       Base.mightalias(W2, W) || Base.mightalias(W2, w)
        throw(ArgumentError("One-halo reverse rule requires disjoint output storage"))
    end
    old_I, old_W2 = copy(I), copy(W2)
    H._one_halo_integral!(I, W2, W, w, rho)
    function one_halo_pullback(_)
        drho = 0.0
        for j in axes(W, 2), i in axes(W, 1)
            value = W[i, j]
            value2 = value * value
            seed = dI[j]
            dW[i, j] += 2value * (dW2[i, j] + rho * seed * w[i])
            dw[i] += rho * seed * value2
            drho += seed * value2 * w[i]
        end
        fill!(dI, 0.0)
        fill!(dW2, 0.0)
        copyto!(I, old_I)
        copyto!(W2, old_W2)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
               Mooncake.NoRData(), Mooncake.NoRData(), drho
    end
    return integral, one_halo_pullback
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H.apply_baryonic_transform!),
    StridedMatrix{Float64}, StridedVector{Float64}, Float64, Float64,
    Float64, Float64, Float64}

function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H.apply_baryonic_transform!)},
                         profile::Mooncake.CoDual{<:StridedMatrix{Float64}},
                         masses::Mooncake.CoDual{<:StridedVector{Float64}},
                         mass_scale::Mooncake.CoDual{Float64},
                         stellar::Mooncake.CoDual{Float64},
                         matter::Mooncake.CoDual{Float64},
                         cold::Mooncake.CoDual{Float64},
                         baryons::Mooncake.CoDual{Float64})
    W, dW = Mooncake.arrayify(profile)
    M, dM = Mooncake.arrayify(masses)
    Mb, fs, Om, Oc, Ob = Mooncake.primal.((mass_scale, stellar, matter, cold, baryons))
    Base.mightalias(W, M) && throw(ArgumentError(
        "Baryonic reverse rule requires profile and mass storage to be disjoint"))
    original = copy(W)
    H.apply_baryonic_transform!(W, M, Mb, fs, Om, Oc, Ob)
    function baryonic_pullback(_)
        dMb = dfs = dOm = dOc = dOb = 0.0
        gas = Ob / Om - fs
        for i in axes(W, 1)
            ratio = M[i] / Mb
            t = ratio * ratio
            inverse_denominator = inv(1.0 + t)
            q = t / (1.0 + t)
            coefficient = Oc / Om + gas * q
            coefficient_bar = direct_stellar_bar = 0.0
            for j in axes(W, 2)
                seed = dW[i, j]
                coefficient_bar += seed * original[i, j]
                direct_stellar_bar += seed
                # Same storage represents the incoming output cotangent and
                # the outgoing input cotangent of this in-place affine map.
                dW[i, j] = seed * coefficient
            end
            qbar = coefficient_bar * gas
            dM[i] += qbar * 2ratio / Mb * inverse_denominator^2
            dMb -= qbar * 2t / Mb * inverse_denominator^2
            dfs += direct_stellar_bar - coefficient_bar * q
            dOm -= coefficient_bar * (Oc + Ob * q) / Om^2
            dOc += coefficient_bar / Om
            dOb += coefficient_bar * q / Om
        end
        copyto!(W, original)
        return Mooncake.NoRData(), Mooncake.NoRData(), Mooncake.NoRData(),
               dMb, dfs, dOm, dOc, dOb
    end
    return profile, baryonic_pullback
end

const GrowthBackground = NamedTuple{(:Omega_m, :Omega_k, :w0, :wa), NTuple{4, Float64}}
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H._growth_values), GrowthBackground, Bool}
function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H._growth_values)},
                         background::Mooncake.CoDual{GrowthBackground}, lcdm::Mooncake.CoDual{Bool})
    bg, flag = Mooncake.primal(background), Mooncake.primal(lcdm)
    values = H._growth_values(bg, flag)
    # Explicitly mixed mode: only four background coordinates, never a full
    # HMCode Jacobian. Keep the Float64 adaptive primal rather than replacing
    # it by the value part of a dual solve with a different step history.
    x = collect(Tuple(bg))
    jac = ForwardDiff.jacobian(x) do p
        H._growth_values((Omega_m=p[1], Omega_k=p[2], w0=p[3], wa=p[4]), flag)
    end
    output = Mooncake.zero_fcodual(values)
    function growth_pullback(_)
        dy = Mooncake.tangent(output)
        bars = ntuple(4) do j
            total = 0.0
            for i in axes(jac, 1)
                total += jac[i, j] * dy[i]
            end
            total
        end
        return Mooncake.NoRData(), NamedTuple{keys(bg)}(bars), Mooncake.NoRData()
    end
    return output, growth_pullback
end

# Differentiate sigma(R)=dc implicitly, not the bracketing solver's decisions.
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{
    typeof(H.get_nonlinear_radius), Float64, Float64, Float64, Any,
}

function Mooncake.rrule!!(
    ::Mooncake.CoDual{typeof(H.get_nonlinear_radius)},
    lower::Mooncake.CoDual{Float64}, upper::Mooncake.CoDual{Float64},
    threshold::Mooncake.CoDual{Float64}, sigma::Mooncake.CoDual,
)
    lo, hi, dc, fun = Mooncake.primal.((lower, upper, threshold, sigma))
    flo, fhi = fun(lo) - dc, fun(hi) - dc
    lower_branch = flo == 0 || (fhi != 0 && signbit(flo) == signbit(fhi) && flo < 0)
    upper_branch = !lower_branch && (fhi == 0 || signbit(flo) == signbit(fhi))
    if lower_branch || upper_branch
        value = lower_branch ? lo : hi
        function endpoint_pullback(dy)
            return Mooncake.NoRData(), lower_branch ? dy : 0.0,
                   upper_branch ? dy : 0.0, 0.0, Mooncake.zero_rdata(fun)
        end
        return Mooncake.zero_fcodual(value), endpoint_pullback
    end

    radius = H.get_nonlinear_radius(lo, hi, dc, fun)
    slope = ForwardDiff.derivative(fun, radius)
    rule = Mooncake.build_rrule(fun, radius)
    _, sigma_pullback = rule(sigma, Mooncake.zero_fcodual(radius))
    function interior_pullback(dy)
        sigma_bar, _ = sigma_pullback(-dy / slope)
        return Mooncake.NoRData(), 0.0, 0.0, dy / slope, sigma_bar
    end
    return Mooncake.zero_fcodual(radius), interior_pullback
end

function _weighted_samples(fun, nodes, weights)
    total = zero(eltype(weights))
    for i in eachindex(nodes, weights)
        total += weights[i] * fun(nodes[i])
    end
    return total
end

_displacement_integrand(R, pk, k) = R == 0 ? pk(k) : pk(k) * H._Tophat_k(k * R)^2
function _displacement_samples(R, pk, nodes, weights)
    total = zero(R)
    for i in eachindex(nodes, weights)
        total += weights[i] * _displacement_integrand(R, pk, nodes[i])
    end
    return total
end

Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H._sigmaV_integral), Float64, Any, Float64, Float64}
function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H._sigmaV_integral)},
                         R::Mooncake.CoDual{Float64}, pk::Mooncake.CoDual,
                         lower::Mooncake.CoDual{Float64}, upper::Mooncake.CoDual{Float64})
    radius, fun, lo, hi = Mooncake.primal.((R, pk, lower, upper))
    # Public spectrum-grid HMCode uses finite support. Do not silently use
    # finite-panel weights for QuadGK's transformed infinite-domain panels.
    isfinite(lo) && isfinite(hi) || throw(ArgumentError(
        "Mooncake displacement quadrature currently requires finite integration bounds"))
    integral = H._sigmaV_integral(radius, fun, lo, hi)
    _, _, segments = Q.quadgk_segbuf(k -> _displacement_integrand(radius, fun, k), lo, hi;
                                    rtol=1e-3, atol=1e-8, order=7)
    x, w, _ = Q.kronrod(7)
    nodes, weights = Float64[], Float64[]
    for segment in segments
        half = (segment.b - segment.a) / 2
        push!(nodes, segment.a + half)
        push!(weights, half * w[end])
        for i in 1:length(x)-1
            push!(nodes, segment.a + (1+x[i])*half, segment.a + (1-x[i])*half)
            push!(weights, half*w[i], half*w[i])
        end
    end
    rule = Mooncake.build_rrule(_displacement_samples, radius, fun, nodes, weights)
    _, back = rule(Mooncake.zero_fcodual(_displacement_samples), R, pk,
                   Mooncake.zero_fcodual(nodes), Mooncake.zero_fcodual(weights))
    low_value = _displacement_integrand(radius, fun, lo)
    high_value = _displacement_integrand(radius, fun, hi)
    function displacement_pullback(dy)
        _, rbar, pbar, _, _ = back(dy)
        return Mooncake.NoRData(), rbar, pbar, -dy*low_value, dy*high_value
    end
    return Mooncake.zero_fcodual(integral), displacement_pullback
end

# Integrals' default reverse path does not propagate this callable's captured
# fields. Differentiate samples on the accepted primal panels, with a separate
# Leibniz term for the moving upper endpoint.
Mooncake.@is_primitive Mooncake.DefaultCtx Tuple{typeof(H.get_accumulated_growth), Float64, Any}
function Mooncake.rrule!!(::Mooncake.CoDual{typeof(H.get_accumulated_growth)},
                         a::Mooncake.CoDual{Float64}, g::Mooncake.CoDual)
    endpoint, fun = Mooncake.primal(a), Mooncake.primal(g)
    initial = 1.0e-4
    value = H.get_accumulated_growth(endpoint, fun)
    _, _, segments = Q.quadgk_segbuf(x -> fun(x) / x, initial, endpoint;
                                    rtol=1.0e-6, atol=1.0e-8, order=7)
    x, w, _ = Q.kronrod(7)
    nodes, weights = Float64[initial], Float64[1.0]
    for segment in segments
        half = (segment.b - segment.a) / 2
        center = segment.a + half
        push!(nodes, center)
        push!(weights, half * w[end] / center)
        for i in 1:length(x)-1
            left = segment.a + (1 + x[i]) * half
            right = segment.a + (1 - x[i]) * half
            push!(nodes, left, right)
            push!(weights, half * w[i] / left, half * w[i] / right)
        end
    end
    rule = Mooncake.build_rrule(_weighted_samples, fun, nodes, weights)
    _, back = rule(Mooncake.zero_fcodual(_weighted_samples), g,
                   Mooncake.zero_fcodual(nodes), Mooncake.zero_fcodual(weights))
    boundary = fun(endpoint) / endpoint
    function accumulated_pullback(dy)
        _, gbar, _, _ = back(dy)
        return Mooncake.NoRData(), dy * boundary, gbar
    end
    return Mooncake.zero_fcodual(value), accumulated_pullback
end

include("hmcode_threaded_mooncake.jl")

end
