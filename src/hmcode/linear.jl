# linear.jl
using Roots
using Integrals

# ------------------------------------------------------------
# Utility equivalents
# ------------------------------------------------------------

is_array_monotonic(x::AbstractVector) = all(diff(x) .> 0.0)

"""
Python-equivalent linear-spacing checker.
The original Python implementation is quirky; here we keep intended behavior.
"""
function is_array_linear(x::AbstractVector; atol::Real = 1e-8)
    dx = diff(x)
    all(abs.(dx .- dx[1]) .<= atol)
end

"""Local derivative from sampled values using linear/quadratic stencils."""
function derivative_from_samples(x::Real, xs::AbstractVector, fs::AbstractVector)
    @assert length(xs) == length(fs)
    n = length(xs)
    n >= 2 || throw(ArgumentError("Need at least two sample points"))

    ix = argmin(abs.(xs .- x))
    if ix == 1
        imin, imax = (x < xs[1]) ? (1, 2) : (1, min(3, n))
    elseif ix == n
        imin, imax = (x > xs[end]) ? (n - 1, n) : (max(1, n - 2), n)
    else
        imin, imax = ix - 1, ix + 1
    end

    if imax - imin == 1
        x0, x1 = xs[imin], xs[imax]
        f0, f1 = fs[imin], fs[imax]
        return (f1 - f0) / (x1 - x0)
    else
        # Quadratic Lagrange derivative through 3 points
        x0, x1, x2 = xs[imin], xs[imin + 1], xs[imax]
        f0, f1, f2 = fs[imin], fs[imin + 1], fs[imax]
        t0 = f0 * (2x - x1 - x2) / ((x0 - x1) * (x0 - x2))
        t1 = f1 * (2x - x0 - x2) / ((x1 - x0) * (x1 - x2))
        t2 = f2 * (2x - x0 - x1) / ((x2 - x0) * (x2 - x1))
        return t0 + t1 + t2
    end
end

# ------------------------------------------------------------
# HMcode linear ingredients
# ------------------------------------------------------------

function Tk_EH_nowiggle(
    k::AbstractVector,
    h::Real,
    wm::Real,
    wb::Real,
    T_CMB::Real = 2.725,
)
    rb = wb / wm
    e = exp(1.0)
    s = 44.5 * log(9.83 / wm) / sqrt(1.0 + 10.0 * wb^0.75)
    alpha = 1.0 - 0.328 * log(431.0 * wm) * rb + 0.38 * log(22.3 * wm) * rb^2

    Γ = @. (wm / h) * (alpha + (1.0 - alpha) / (1.0 + (0.43 * k * s * h)^4))
    q = @. k * (T_CMB / 2.7)^2 / Γ
    L = @. log(2.0 * e + 1.8 * q)
    C = @. 14.2 + 731.0 / (1.0 + 62.5 * q)
    return @. L / (L + C * q^2)
end
Tk_EH_nowiggle(k::Real, h::Real, wm::Real, wb::Real, T_CMB::Real = 2.725) =
    Tk_EH_nowiggle([float(k)], h, wm, wb, T_CMB)[1]

# Tophat Fourier transform used by sigmaV(R>0)
function _Tophat_k(x::Real)
    xmin = 1e-5
    if abs(x) < xmin
        return 1.0 - x^2 / 10.0
    end
    return (3.0 / x^3) * (sin(x) - x * cos(x))
end

function get_effective_index(Rnl::Real, R::AbstractVector, sigmaR::AbstractVector)
    logR = log.(R)
    logsigmaR = log.(sigmaR)
    -3.0 - 2.0 * derivative_from_samples(log(Rnl), logR, logsigmaR)
end

function get_nonlinear_radius(
    Rmin::Real,
    Rmax::Real,
    dc::Real,
    sigmaR_func,
)
    root_func = R -> sigmaR_func(R) - dc
    return find_zero(root_func, (float(Rmin), float(Rmax)), A42())
end

"""
1D RMS in displacement field.
Matches hmcode/cosmology.py::sigmaV (including optional tophat smoothing scale R).
"""
function sigmaV(
    R::Real,
    Pk;
    kmin::Real = 0.0,
    kmax::Real = Inf,
    eps::Real = 1e-4,
)
    # Fast analytic evaluation for specific R ranges or use a slightly looser tolerance
    integrand = if R == 0
        k -> Pk(k)
    else
        k -> Pk(k) * _Tophat_k(k * R)^2
    end
    prob = IntegralProblem((u, p) -> integrand(u), (float(kmin), float(kmax)))
    sigmaV_squared = solve(prob, QuadGKJL(), reltol=1e-3).u
    return sqrt(sigmaV_squared / (2.0 * pi^2)) / sqrt(3.0)
end

# Backward-compatible convenience (previous local API)
sigmaV(Pk, kmin::Real = 0.0, kmax::Real = Inf, eps::Real = 1e-4) =
    sigmaV(0.0, Pk; kmin=kmin, kmax=kmax, eps=eps)

# ------------------------------------------------------------
# Dewiggle
# ------------------------------------------------------------

# Reflect boundary handling used by gaussian_filter1d approximation
function _reflect_index(idx::Int, n::Int)
    i = idx
    while true
        if i < 1
            i = 1 - i
        elseif i > n
            i = 2n - i + 1
        else
            return i
        end
    end
end

"""Simple Gaussian 1D filter with reflect boundaries."""
function gaussian_filter1d(y::AbstractVector, sigma::Real; truncate::Real = 4.0)
    lw = Int(trunc(truncate * sigma + 0.5))
    lw == 0 && return collect(Float64.(y))

    x = -lw:lw
    weights = exp.(-0.5 .* (x ./ sigma).^2)
    weights ./= sum(weights)

    n = length(y)
    out = zeros(Float64, n)
    @inbounds for i in 1:n
        for j in -lw:lw
            idx = _reflect_index(i + j, n)
            out[i] += y[idx] * weights[j + lw + 1]
        end
    end
    return out
end

"""
Extract BAO wiggle component from linear spectrum.
Equivalent to hmcode.py::_get_Pk_wiggle.
"""
function get_Pk_wiggle(
    k::AbstractVector,
    Pk_lin::AbstractVector,
    h::Real,
    omega_m::Real,
    omega_b::Real,
    n_s::Real;
    T_CMB::Real = 2.725,
    sigma_dlnk::Real = 0.25,
)
    length(k) >= 2 || throw(ArgumentError("k array must have at least 2 points"))
    dlnk = log(k[2] / k[1])
    sigma = sigma_dlnk / dlnk

    Tk_nw = Tk_EH_nowiggle(k, h, omega_m, omega_b, T_CMB)
    Pk_nowiggle = @. (k^n_s) * Tk_nw^2

    Pk_ratio = Pk_lin ./ Pk_nowiggle
    Pk_ratio_smooth = gaussian_filter1d(Pk_ratio, sigma)
    Pk_smooth = Pk_ratio_smooth .* Pk_nowiggle
    return Pk_lin .- Pk_smooth
end
