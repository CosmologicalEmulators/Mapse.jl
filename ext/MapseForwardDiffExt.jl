module MapseForwardDiffExt

using ForwardDiff
using Integrals
using Mapse

const HMcode = Mapse.HMcode

@inline _primal(x::ForwardDiff.Dual) = ForwardDiff.value(x)
@inline _primal(x::Real) = x

function HMcode.get_nonlinear_radius(
    Rmin::ForwardDiff.Dual, Rmax::Real, dc::Real, sigmaR_func,
)
    lo, hi, threshold = _primal(Rmin), _primal(Rmax), _primal(dc)
    primal_sigma(r) = _primal(sigmaR_func(r))
    flo, fhi = primal_sigma(lo) - threshold, primal_sigma(hi) - threshold
    # Preserve the native endpoint/clamping convention, including tangents
    # of a moving endpoint. Interior roots use the implicit-function rule;
    # differentiating a bracket solver's stopping decisions is unreliable.
    flo == 0 && return Rmin
    fhi == 0 && return Rmax
    signbit(flo) == signbit(fhi) && return flo < 0 ? Rmin : Rmax
    radius = HMcode.get_nonlinear_radius(lo, hi, threshold, primal_sigma)
    residual = sigmaR_func(radius) - dc
    slope = _primal(ForwardDiff.derivative(sigmaR_func, radius))
    return radius - (residual - _primal(residual)) / slope
end

function HMcode.sigmaV(
    R::Real,
    Pk::HMcode.PkLinEval{<:Mapse.HMCodeLinearInterpolator};
    kmin::Real=0.0,
    kmax::Real=Inf,
)
    integrand = if R == 0
        k -> Pk(k)
    else
        k -> Pk(k) * HMcode._Tophat_k(k * R)^2
    end

    primal_kmin = _primal(kmin)
    primal_kmax = _primal(kmax)
    integral = HMcode._sigmaV_integral(R, Pk, primal_kmin, primal_kmax)
    # First-order Leibniz boundary terms. Do not evaluate an infinite fixed
    # endpoint merely to multiply it by zero.
    if kmax isa ForwardDiff.Dual
        integral += integrand(primal_kmax) * (kmax - primal_kmax)
    end
    if kmin isa ForwardDiff.Dual
        integral -= integrand(primal_kmin) * (kmin - primal_kmin)
    end
    return sqrt(integral / (2.0 * pi^2)) / sqrt(3.0)
end

function HMcode.get_accumulated_growth(a::ForwardDiff.Dual, g_func)
    a_init = 1.0e-4
    primal_a = ForwardDiff.value(a)
    integrand(x) = g_func(x) / x
    prob = IntegralProblem((x, p) -> integrand(x), (a_init, primal_a))
    integral = solve(prob, QuadGKJL(), reltol=1.0e-6).u
    return integral + integrand(primal_a) * (a - primal_a) + g_func(a_init)
end

end
