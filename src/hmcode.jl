"""
    HMCodeCosmology(Ωm, Ωb, h, ns, σ8, w0, wa, Ων, Ωk)

Cosmology container used by the embedded HMCode2020 implementation. Densities are
fractions today. The linear input spectra supplied to `hmcode_Pmm`/`hmcode_boost`
should use the same distance convention as `k`.
"""
const HMCodeCosmology = HMcode.HMcodeCosmology
const HMCODE_LOG_GRID_RTOL = 1.0e-10

function _validate_hmcode_log_grid(k::AbstractVector)
    dlogk = diff(log.(k))
    all(isapprox.(dlogk, first(dlogk); rtol=HMCODE_LOG_GRID_RTOL, atol=0)) ||
        throw(ArgumentError(
            "HMCode support k values must be uniformly spaced in log(k) " *
            "(relative tolerance $(HMCODE_LOG_GRID_RTOL))."
        ))
    return nothing
end

function _validate_hmcode_inputs(z::AbstractVector, k::AbstractVector, pk_lin_z::AbstractMatrix;
                                 name::AbstractString="pk_lin_z")
    size(pk_lin_z) == (length(k), length(z)) || throw(DimensionMismatch(
        "$name must have size (length(k), length(z)); got $(size(pk_lin_z)) " *
        "for length(k)=$(length(k)), length(z)=$(length(z))."
    ))
    length(k) >= 2 || throw(ArgumentError("HMCode requires at least two k values."))
    all(k .> 0) || throw(ArgumentError("HMCode requires strictly positive k values."))
    all(pk_lin_z .> 0) || throw(ArgumentError("HMCode requires strictly positive $name values."))
    all(isfinite, pk_lin_z) || throw(ArgumentError("HMCode requires finite $name values."))
    all(diff(k) .> 0) || throw(ArgumentError("HMCode requires k values sorted in ascending order."))
    _validate_hmcode_log_grid(k)
    return nothing
end

function _validate_hmcode_output_grid(k_out::AbstractVector, k_support::AbstractVector)
    length(k_out) >= 2 || throw(ArgumentError("HMCode requires at least two output k values."))
    all(k_out .> 0) || throw(ArgumentError("HMCode requires strictly positive output k values."))
    all(diff(k_out) .> 0) || throw(ArgumentError("HMCode requires output k values sorted in ascending order."))
    first(k_out) >= first(k_support) && last(k_out) <= last(k_support) || throw(ArgumentError(
        "HMCode output k range [$(first(k_out)), $(last(k_out))] must lie inside " *
        "support k range [$(first(k_support)), $(last(k_support))]."
    ))
    return nothing
end

function _hmcode_loglog_interp(logx::AbstractVector, logy::AbstractVector, x::Real)
    lx = log(float(x))
    if lx <= first(logx)
        i = firstindex(logx)
    elseif lx >= last(logx)
        i = lastindex(logx) - 1
    else
        i = searchsortedlast(logx, lx)
        i == lastindex(logx) && (i -= 1)
    end
    slope = (logy[i + 1] - logy[i]) / (logx[i + 1] - logx[i])
    return exp(logy[i] + slope * (lx - logx[i]))
end

function _hmcode_z_index(zs::AbstractVector, z::Real)
    iz = findfirst(==(z), zs)
    isnothing(iz) || return iz
    iz = findfirst(zi -> isapprox(zi, z; rtol=0, atol=1e-10), zs)
    isnothing(iz) && throw(ArgumentError("redshift $z was not found in the HMCode redshift grid."))
    return iz
end

function _hmcode_trapz(x::AbstractVector, y::AbstractVector)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length."))
    total = zero(promote_type(eltype(x), eltype(y)))
    @inbounds for i in firstindex(x):(lastindex(x) - 1)
        total += (x[i + 1] - x[i]) * (y[i] + y[i + 1]) / 2
    end
    return total
end

struct HMCodeLinearInterpolator{T_logk, T_logpk, T_z}
    logk::T_logk
    logpk::T_logpk
    z::T_z
end

@inline function (itp::HMCodeLinearInterpolator)(kval::Real, zval::Real)
    iz = _hmcode_z_index(itp.z, zval)
    return itp(kval, iz)
end

@inline function (itp::HMCodeLinearInterpolator)(kval::Real, iz::Int)
    return _hmcode_loglog_interp(itp.logk, view(itp.logpk, :, iz), kval)
end

struct HMCodeSigmaInterpolator{T_logk, T_logpk, T_k, T_z}
    logk::T_logk
    logpk::T_logpk
    k::T_k
    z::T_z
end

@inline function (itp::HMCodeSigmaInterpolator)(R::Real, zval::Real)
    iz = _hmcode_z_index(itp.z, zval)
    return itp(R, iz)
end

@inline function (itp::HMCodeSigmaInterpolator)(R::Real, iz::Int)
    logpk_col = view(itp.logpk, :, iz)
    integrand = similar(itp.logk)
    @inbounds for i in eachindex(itp.logk)
        kval = itp.k[i]
        pkval = exp(logpk_col[i])
        W = HMcode._Tophat_k(kval * R)
        integrand[i] = kval^3 * pkval * W^2 / (2π^2)
    end
    return sqrt(max(_hmcode_trapz(itp.logk, integrand), 0.0))
end

HMcode._eval_sigma(f::HMCodeSigmaInterpolator, R, z, iz) = f(R, iz)
HMcode._eval_pk(f::HMCodeLinearInterpolator, k, z, iz) = f(k, iz)

@inline (s::HMcode.SigmaREval{<:HMCodeSigmaInterpolator})(R::Float64) = s.sigma_R(R, s.iz)
@inline (s::HMcode.PkLinEval{<:HMCodeLinearInterpolator})(k::Float64) = s.Pk_lin(k, s.iz)

function _hmcode_linear_interpolators(z::AbstractVector, k::AbstractVector, pk_lin_z::AbstractMatrix)
    logk = log.(Float64.(k))
    logpk = log.(Float64.(pk_lin_z))
    pk_lin = HMCodeLinearInterpolator(logk, logpk, z)
    sigma_R = HMCodeSigmaInterpolator(logk, logpk, k, z)
    return pk_lin, sigma_R
end

function _hmcode_matrix_from_interpolator(pk, z::AbstractVector, k::AbstractVector)
    out = Matrix{Float64}(undef, length(k), length(z))
    @inbounds for iz in eachindex(z), ik in eachindex(k)
        out[ik, iz] = pk(k[ik], z[iz])
    end
    return out
end

function _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    if pk_cb_support_z !== nothing
        pk_cb_z === nothing || throw(ArgumentError(
            "Pass either pk_cb_z or pk_cb_support_z, not both."
        ))
        return pk_cb_support_z
    end
    return pk_cb_z
end

"""
    hmcode_Pmm(cosmo, z, k, pk_mm_z; pk_cb_z=nothing, kwargs...)
    hmcode_Pmm(cosmo, z, k, pk_mm_z, pk_cb_z; kwargs...)
    hmcode_Pmm(cosmo, z, k_out, pk_mm_support_z; k_support, pk_cb_z=nothing, kwargs...)
    hmcode_Pmm(cosmo, z, k_out, k_support, pk_mm_support_z; pk_cb_support_z=nothing, kwargs...)
    hmcode_Pmm(cosmo, z::Number, k, pk_mm; pk_cb=nothing, k_support=nothing, pk_cb_support=nothing, kwargs...)

Compute the HMCode2020 nonlinear matter power spectrum from a sampled linear
matter power spectrum. For the vector-redshift methods, `z` and `k` are vectors,
and `pk_mm_z` must have shape `(length(k), length(z))`, matching the `halofit_Pmm` convention.
For scalar redshifts `z::Number`, the corresponding linear spectra are 1D vectors of length `length(k)`.

For massive-neutrino cosmologies, pass the cold+baryon linear spectrum with
`pk_cb_z` (or `pk_cb` for scalar forms). HMCode2020 uses the total matter spectrum for the returned nonlinear
`Pmm`, but uses cold+baryon σ(R,z) quantities internally for halo collapse and
transition parameters. If `pk_cb_z` is omitted, the total spectrum is used for
both roles, preserving the old behaviour.

Use `k_support` when the linear spectra are sampled on a wider grid than the
requested output. In that case `pk_mm_z` and `pk_cb_z` are interpreted on
`k_support`, while nonlinear power is returned on `k_out` (or `k`). This lets σ(R,z) and
HMCode internal parameters use a high-k support grid without evaluating the final
halo-model spectrum on every support-grid point.

When using a support grid with a scalar redshift, you MUST use the keyword-only
form (e.g., `hmcode_Pmm(cosmo, z_scalar, k_out, pk_mm_support; k_support=...)`).
The 5-positional argument form `(cosmo, z, k, pk_mm, pk_cb)` is strictly reserved
for passing the cold+baryon spectrum without a support grid.

The supplied `k` grid is also used to compute the σ(R,z) integrals required by
HMCode. It should therefore cover the linear spectrum over a sufficiently broad
range for the requested cosmology and must be uniformly spaced in `log(k)`.
When `k_support` is supplied, this requirement applies to `k_support`; `k` may
be an arbitrary strictly increasing output grid within the support range.

Keyword arguments are forwarded to the embedded HMCode implementation. Common
options include `T_AGN=10^7.8`, `Mmin=1e0`, `Mmax=1e18`, `nM=128`, and `threaded=false`.
"""
function hmcode_Pmm(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                    pk_mm_z::AbstractMatrix; pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                    k_support::Union{Nothing,AbstractVector}=nothing,
                    pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                    kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pk_cb_linear_z = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)

    _validate_hmcode_inputs(z, k_linear, pk_mm_z; name="pk_mm_z")
    if k_support !== nothing
        _validate_hmcode_output_grid(k, k_support)
    end
    if pk_cb_linear_z !== nothing
        _validate_hmcode_inputs(z, k_linear, pk_cb_linear_z; name="pk_cb_z")
    end

    pk_mm, _ = _hmcode_linear_interpolators(z, k_linear, pk_mm_z)
    _, sigma_R_cb = _hmcode_linear_interpolators(
        z, k_linear, pk_cb_linear_z === nothing ? pk_mm_z : pk_cb_linear_z
    )
    return HMcode.hmcode_power(k, z, pk_mm, sigma_R_cb, cosmo; k_support=k_support, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                    pk_mm_z::AbstractMatrix, pk_cb_z::AbstractMatrix; kwargs...)
    return hmcode_Pmm(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::AbstractVector,
                    k_out::AbstractVector, k_support::AbstractVector,
                    pk_mm_support_z::AbstractMatrix;
                    pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                    pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                    kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_Pmm(cosmo, z, k_out, pk_mm_support_z;
                      k_support=k_support, pk_cb_z=pk_cb, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                    pk_mm::AbstractVector; pk_cb::Union{Nothing,AbstractVector}=nothing,
                    k_support::Union{Nothing,AbstractVector}=nothing,
                    pk_cb_support::Union{Nothing,AbstractVector}=nothing,
                    kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pkcb = _hmcode_choose_cb(pk_cb, pk_cb_support)
    pk = reshape(pk_mm, length(k_linear), 1)
    pkcb_mat = pkcb === nothing ? nothing : reshape(pkcb, length(k_linear), 1)
    return vec(hmcode_Pmm(cosmo, [float(z)], k, pk;
                          k_support=k_support, pk_cb_z=pkcb_mat, kwargs...))
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number,
                    k::AbstractVector, pk_mm::AbstractVector,
                    pk_cb::AbstractVector; kwargs...)
    return hmcode_Pmm(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number,
                    k_out::AbstractVector, k_support::AbstractVector,
                    pk_mm_support::AbstractVector, pk_cb_support::AbstractVector;
                    kwargs...)
    return hmcode_Pmm(cosmo, z, k_out, pk_mm_support;
                      k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

"""
    hmcode_boost(cosmo, z, k, pk_mm_z; pk_cb_z=nothing, kwargs...)
    hmcode_boost(cosmo, z, k, pk_mm_z, pk_cb_z; kwargs...)
    hmcode_boost(cosmo, z::Number, k, pk_mm; pk_cb=nothing, k_support=nothing, pk_cb_support=nothing, kwargs...)

Return the HMCode2020 nonlinear boost, defined as `hmcode_Pmm(...) ./ pk_mm_z`.
Input shape conventions and scalar/support-grid API behavior match `hmcode_Pmm`.
"""
function hmcode_boost(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                      pk_mm_z::AbstractMatrix; pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                      k_support::Union{Nothing,AbstractVector}=nothing,
                      pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                      kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pk_cb_linear_z = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    pk_mm, _ = _hmcode_linear_interpolators(z, k_linear, pk_mm_z)
    pk_mm_out_z = k_support === nothing ? pk_mm_z : _hmcode_matrix_from_interpolator(pk_mm, z, k)
    return hmcode_Pmm(cosmo, z, k, pk_mm_z;
                      k_support=k_support, pk_cb_z=pk_cb_linear_z, kwargs...) ./ pk_mm_out_z
end

function hmcode_boost(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                      pk_mm_z::AbstractMatrix, pk_cb_z::AbstractMatrix; kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::AbstractVector,
                      k_out::AbstractVector, k_support::AbstractVector,
                      pk_mm_support_z::AbstractMatrix;
                      pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                      pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                      kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_boost(cosmo, z, k_out, pk_mm_support_z;
                        k_support=k_support, pk_cb_z=pk_cb, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                      pk_mm::AbstractVector; pk_cb::Union{Nothing,AbstractVector}=nothing,
                      k_support::Union{Nothing,AbstractVector}=nothing,
                      pk_cb_support::Union{Nothing,AbstractVector}=nothing,
                      kwargs...)
    k_linear = k_support === nothing ? k : k_support
    pk_mm_mat = reshape(pk_mm, length(k_linear), 1)
    pk_mm_itp, _ = _hmcode_linear_interpolators([float(z)], k_linear, pk_mm_mat)
    pk_mm_out = k_support === nothing ? pk_mm : vec(_hmcode_matrix_from_interpolator(pk_mm_itp, [float(z)], k))
    return hmcode_Pmm(cosmo, z, k, pk_mm;
                      k_support=k_support, pk_cb=pk_cb, pk_cb_support=pk_cb_support,
                      kwargs...) ./ pk_mm_out
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number,
                      k::AbstractVector, pk_mm::AbstractVector,
                      pk_cb::AbstractVector; kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end

function hmcode_boost(cosmo::HMCodeCosmology, z::Number,
                      k_out::AbstractVector, k_support::AbstractVector,
                      pk_mm_support::AbstractVector, pk_cb_support::AbstractVector;
                      kwargs...)
    return hmcode_boost(cosmo, z, k_out, pk_mm_support;
                        k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

"""
    hmcode_pmm_fast(cosmo, z, k, pk_mm_z, N_z_coarse; [pk_cb_z, k_support, pk_cb_support_z, kwargs...])
    hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse; [pk_cb_coarse, k_support, pk_cb_support_coarse, kwargs...])

Compute HMCode2020 nonlinear matter power spectrum P_mm(k,z) using a coarse redshift grid and Akima interpolation.

Warning:
    This fast API is an approximation. Instead of evaluating the full non-linear
    HMCode equations on the high-fidelity target grid, it solves HMCode on
    the coarse grid and uses Akima splines to reconstruct the results.
    
    - Typical Errors: Redshift interpolation errors are generally small but largest
      at high k, high redshift, and in regions where the nonlinear boost factor
      evolves rapidly.
    - Recommended Coarse Grid Size:
        - `N_z_coarse = 10` is NOT precision-safe and can introduce percent-level artifacts.
        - `N_z_coarse = 50` is a reasonable compromise between speed and accuracy.
        - `N_z_coarse = 100` is safer, typically guaranteeing sub-percent worst-case
          accuracy compared to full direct evaluation in studied regimes.
    - Validation: Users are advised to validate the accuracy of this fast
      path against the direct `hmcode_pmm` function for their specific redshift and
      k ranges.

Preferred Production Pattern:
    Using the smart/coarse-grid API is the preferred production pattern:
    1. Choose `z_coarse` (typically 50-100 nodes linearly spaced).
    2. Evaluate linear transfer functions/emulators only on `z_coarse`.
    3. Run HMCode via `hmcode_pmm_fast` on `z_coarse`.
    4. Akima interpolate the final non-linear spectrum to `z_fine`.
    This avoids running linear/transfer emulators on the dense fine redshift grid.

Interpolation Target Note:
    `hmcode_pmm_fast` interpolates the full nonlinear power spectrum P(k,z),
    whereas `hmcode_boost_fast` interpolates the non-linear boost factor B(k,z).
    Because these interpolation targets differ, they are not numerically equivalent.
    For workflows where linear theory is smooth and high accuracy on the nonlinear
    power spectrum is desired, interpolating P(k,z) directly via `hmcode_pmm_fast`
    is generally recommended.
"""
function hmcode_pmm_fast(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                         pk_mm_z::AbstractMatrix, N_z_coarse::Int;
                         pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                         k_support::Union{Nothing,AbstractVector}=nothing,
                         pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                         kwargs...)
    N_z_coarse >= 5 || throw(ArgumentError("N_z_coarse must be at least 5."))
    issorted(z) && all(diff(z) .> 0) || throw(ArgumentError("z must be strictly increasing."))

    pk_cb_linear_z = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)

    if length(z) <= N_z_coarse || length(z) < 5
        return hmcode_Pmm(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_linear_z, k_support=k_support, kwargs...)
    end

    z_coarse = collect(LinRange(minimum(z), maximum(z), N_z_coarse))

    # Interpolate linear input spectra along the redshift axis to z_coarse
    pk_mm_t = copy(transpose(pk_mm_z))
    pk_mm_coarse_t = AbstractCosmologicalEmulators.akima_interpolation(pk_mm_t, z, z_coarse)
    pk_mm_coarse = copy(transpose(pk_mm_coarse_t))

    pk_cb_coarse = nothing
    if pk_cb_linear_z !== nothing
        pk_cb_t = copy(transpose(pk_cb_linear_z))
        pk_cb_coarse_t = AbstractCosmologicalEmulators.akima_interpolation(pk_cb_t, z, z_coarse)
        pk_cb_coarse = copy(transpose(pk_cb_coarse_t))
    end

    # Solve non-linear HMCode2020 on the coarse grid
    Pk_nl_coarse = hmcode_Pmm(cosmo, z_coarse, k, pk_mm_coarse;
                              pk_cb_z=pk_cb_coarse, k_support=k_support, kwargs...)

    # Interpolate output non-linear power spectra back to the fine z grid
    Pk_nl_coarse_t = copy(transpose(Pk_nl_coarse))
    Pk_nl_fine_t = AbstractCosmologicalEmulators.akima_interpolation(Pk_nl_coarse_t, z_coarse, z)
    return copy(transpose(Pk_nl_fine_t))
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z::AbstractVector,
                         k_out::AbstractVector, k_support::AbstractVector,
                         pk_mm_support_z::AbstractMatrix, N_z_coarse::Int;
                         pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                         pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                         kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_pmm_fast(cosmo, z, k_out, pk_mm_support_z, N_z_coarse;
                           k_support=k_support, pk_cb_z=pk_cb, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                         pk_mm::AbstractVector, N_z_coarse::Int;
                         pk_cb::Union{Nothing,AbstractVector}=nothing,
                         k_support::Union{Nothing,AbstractVector}=nothing,
                         pk_cb_support::Union{Nothing,AbstractVector}=nothing,
                         kwargs...)
    return hmcode_pmm(cosmo, z, k, pk_mm; pk_cb=pk_cb, k_support=k_support,
                      pk_cb_support=pk_cb_support, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z::Number,
                         k::AbstractVector, pk_mm::AbstractVector,
                         pk_cb::AbstractVector, N_z_coarse::Int; kwargs...)
    return hmcode_pmm_fast(cosmo, z, k, pk_mm, N_z_coarse; pk_cb=pk_cb, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z::Number,
                         k_out::AbstractVector, k_support::AbstractVector,
                         pk_mm_support::AbstractVector, pk_cb_support::AbstractVector,
                         N_z_coarse::Int; kwargs...)
    return hmcode_pmm_fast(cosmo, z, k_out, pk_mm_support, N_z_coarse;
                           k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

function validate_hmcode_fast_grids(z_coarse::AbstractVector, z_fine::Union{Number, AbstractVector})
    length(z_coarse) >= 5 || throw(ArgumentError("z_coarse must have at least 5 points for Akima interpolation."))
    issorted(z_coarse) && all(diff(z_coarse) .> 0) || throw(ArgumentError("z_coarse must be strictly increasing."))

    if z_fine isa AbstractVector
        issorted(z_fine) && all(diff(z_fine) .> 0) || throw(ArgumentError("z_fine must be strictly increasing."))
        all(z_fine .>= minimum(z_coarse)) && all(z_fine .<= maximum(z_coarse)) || throw(ArgumentError("z_fine must lie within the range of z_coarse."))
    else
        z_coarse[1] <= z_fine <= z_coarse[end] || throw(ArgumentError("z_fine must lie within the range of z_coarse."))
    end
end

# Smart signatures: User provides inputs directly on the coarse grid
function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector,
                         z_fine::Union{Number, AbstractVector}, k::AbstractVector,
                         pk_mm_coarse::AbstractMatrix;
                         pk_cb_coarse::Union{Nothing,AbstractMatrix}=nothing,
                         k_support::Union{Nothing,AbstractVector}=nothing,
                         pk_cb_support_coarse::Union{Nothing,AbstractMatrix}=nothing,
                         kwargs...)
    validate_hmcode_fast_grids(z_coarse, z_fine)

    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    Pk_nl_coarse = hmcode_Pmm(cosmo, z_coarse, k, pk_mm_coarse;
                              pk_cb_z=pk_cb, k_support=k_support, kwargs...)

    z_fine_arr = z_fine isa Number ? [float(z_fine)] : z_fine
    Pk_nl_coarse_t = copy(transpose(Pk_nl_coarse))
    Pk_nl_fine_t = AbstractCosmologicalEmulators.akima_interpolation(Pk_nl_coarse_t, z_coarse, z_fine_arr)
    Pk_nl_fine = copy(transpose(Pk_nl_fine_t))
    return z_fine isa Number ? vec(Pk_nl_fine) : Pk_nl_fine
end

"""Evaluate HMCode on a coarse grid and interpolate Pmm with two Akima splines.

The coarse grid must contain `z_split`. One spline is fitted on the nodes below
the split and one on the nodes above it. This is intended for the baryonic
feedback feature, where a single spline can smooth across the transition.
"""
function hmcode_pmm_fast_two_splines(cosmo::HMCodeCosmology,
                                     z_coarse::AbstractVector,
                                     z_fine::AbstractVector,
                                     k::AbstractVector,
                                     pk_mm_coarse::AbstractMatrix;
                                     z_split::Real,
                                     pk_cb_coarse::Union{Nothing,AbstractMatrix}=nothing,
                                     k_support::Union{Nothing,AbstractVector}=nothing,
                                     pk_cb_support_coarse::Union{Nothing,AbstractMatrix}=nothing,
                                     kwargs...)
    validate_hmcode_fast_grids(z_coarse, z_fine)
    nleft = searchsortedlast(z_coarse, z_split)
    nleft >= 5 || throw(ArgumentError("The lower two-spline segment needs at least 5 nodes."))
    nleft <= length(z_coarse) - 4 || throw(ArgumentError("The upper two-spline segment needs at least 5 nodes."))
    isapprox(z_coarse[nleft], z_split; rtol=0, atol=10eps(Float64)) || throw(ArgumentError("z_split must be a node of z_coarse."))

    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    pk_nl = hmcode_Pmm(cosmo, z_coarse, k, pk_mm_coarse;
                       pk_cb_z=pk_cb, k_support=k_support, kwargs...)
    pk_t = copy(transpose(pk_nl))
    left = AbstractCosmologicalEmulators.akima_interpolation(pk_t[1:nleft, :], z_coarse[1:nleft], z_fine)
    right = AbstractCosmologicalEmulators.akima_interpolation(pk_t[nleft:end, :], z_coarse[nleft:end], z_fine)
    return copy(transpose(ifelse.(reshape(z_fine .<= z_split, :, 1), left, right)))
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector,
                         z_fine::Union{Number, AbstractVector}, k::AbstractVector,
                         pk_mm_coarse::AbstractMatrix, pk_cb_coarse::AbstractMatrix;
                         kwargs...)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse;
                           pk_cb_coarse=pk_cb_coarse, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector,
                         z_fine::Union{Number, AbstractVector},
                         k_out::AbstractVector, k_support::AbstractVector,
                         pk_mm_support_coarse::AbstractMatrix;
                         pk_cb_support_coarse::Union{Nothing,AbstractMatrix}=nothing,
                         pk_cb_coarse::Union{Nothing,AbstractMatrix}=nothing,
                         kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k_out, pk_mm_support_coarse;
                           k_support=k_support, pk_cb_coarse=pk_cb, kwargs...)
end

function hmcode_pmm_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector,
                         z_fine::Union{Number, AbstractVector},
                         k_out::AbstractVector, k_support::AbstractVector,
                         pk_mm_support_coarse::AbstractMatrix,
                         pk_cb_support_coarse::AbstractMatrix;
                         kwargs...)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k_out, k_support,
                           pk_mm_support_coarse;
                           pk_cb_support_coarse=pk_cb_support_coarse, kwargs...)
end

# Boost fast implementations
"""
    hmcode_boost_fast(cosmo, z, k, pk_mm_z, N_z_coarse; [pk_cb_z, k_support, pk_cb_support_z, kwargs...])
    hmcode_boost_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse; [pk_cb_coarse, k_support, pk_cb_support_coarse, kwargs...])

Compute HMCode2020 nonlinear matter power spectrum boost factor B(k,z) using a coarse redshift grid and Akima interpolation.

Warning:
    This fast API is an approximation. Instead of evaluating the full non-linear
    HMCode equations on the high-fidelity target grid, it solves HMCode on
    the coarse grid and uses Akima splines to reconstruct the results.
    
    - Typical Errors: Redshift interpolation errors are generally small but largest
      at high k, high redshift, and in regions where the nonlinear boost factor
      evolves rapidly.
    - Recommended Coarse Grid Size:
        - `N_z_coarse = 10` is NOT precision-safe and can introduce percent-level artifacts.
        - `N_z_coarse = 50` is a reasonable compromise between speed and accuracy.
        - `N_z_coarse = 100` is safer, typically guaranteeing sub-percent worst-case
          accuracy compared to full direct evaluation in studied regimes.
    - Validation: Users are advised to validate the accuracy of this fast
      path against the direct `hmcode_boost` function for their specific redshift and
      k ranges.

Preferred Production Pattern:
    Using the smart/coarse-grid API is the preferred production pattern:
    1. Choose `z_coarse` (typically 50-100 nodes linearly spaced).
    2. Evaluate linear transfer functions/emulators only on `z_coarse`.
    3. Run HMCode via `hmcode_boost_fast` on `z_coarse`.
    4. Akima interpolate the final non-linear boost to `z_fine`.
    This avoids running linear/transfer emulators on the dense fine redshift grid.

Interpolation Target Note:
    `hmcode_pmm_fast` interpolates the full nonlinear power spectrum P(k,z),
    whereas `hmcode_boost_fast` interpolates the non-linear boost factor B(k,z).
    Because these interpolation targets differ, they are not numerically equivalent.
    For workflows where linear theory is smooth and high accuracy on the nonlinear
    power spectrum is desired, interpolating P(k,z) directly via `hmcode_pmm_fast`
    is generally recommended.
"""
function hmcode_boost_fast(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                           pk_mm_z::AbstractMatrix, N_z_coarse::Int;
                           pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                           k_support::Union{Nothing,AbstractVector}=nothing,
                           pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                           kwargs...)
    N_z_coarse >= 5 || throw(ArgumentError("N_z_coarse must be at least 5."))
    issorted(z) && all(diff(z) .> 0) || throw(ArgumentError("z must be strictly increasing."))

    k_linear = k_support === nothing ? k : k_support
    pk_cb_linear_z = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    pk_mm, _ = _hmcode_linear_interpolators(z, k_linear, pk_mm_z)
    pk_mm_out_z = k_support === nothing ? pk_mm_z : _hmcode_matrix_from_interpolator(pk_mm, z, k)
    return hmcode_pmm_fast(cosmo, z, k, pk_mm_z, N_z_coarse;
                           k_support=k_support, pk_cb_z=pk_cb_linear_z, kwargs...) ./ pk_mm_out_z
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z::AbstractVector, k::AbstractVector,
                           pk_mm_z::AbstractMatrix, pk_cb_z::AbstractMatrix, N_z_coarse::Int; kwargs...)
    return hmcode_boost_fast(cosmo, z, k, pk_mm_z, N_z_coarse; pk_cb_z=pk_cb_z, kwargs...)
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z::AbstractVector,
                           k_out::AbstractVector, k_support::AbstractVector,
                           pk_mm_support_z::AbstractMatrix, N_z_coarse::Int;
                           pk_cb_support_z::Union{Nothing,AbstractMatrix}=nothing,
                           pk_cb_z::Union{Nothing,AbstractMatrix}=nothing,
                           kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_z, pk_cb_support_z)
    return hmcode_boost_fast(cosmo, z, k_out, pk_mm_support_z, N_z_coarse;
                             k_support=k_support, pk_cb_z=pk_cb, kwargs...)
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                           pk_mm::AbstractVector, N_z_coarse::Int;
                           pk_cb::Union{Nothing,AbstractVector}=nothing,
                           k_support::Union{Nothing,AbstractVector}=nothing,
                           pk_cb_support::Union{Nothing,AbstractVector}=nothing,
                           kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm; pk_cb=pk_cb, k_support=k_support,
                        pk_cb_support=pk_cb_support, kwargs...)
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z::Number,
                           k::AbstractVector, pk_mm::AbstractVector,
                           pk_cb::AbstractVector, N_z_coarse::Int; kwargs...)
    return hmcode_boost_fast(cosmo, z, k, pk_mm, N_z_coarse; pk_cb=pk_cb, kwargs...)
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z::Number,
                           k_out::AbstractVector, k_support::AbstractVector,
                           pk_mm_support::AbstractVector, pk_cb_support::AbstractVector,
                           N_z_coarse::Int; kwargs...)
    return hmcode_boost_fast(cosmo, z, k_out, pk_mm_support, N_z_coarse;
                             k_support=k_support, pk_cb_support=pk_cb_support, kwargs...)
end

# Smart signatures for boost: User provides inputs directly on the coarse grid
function hmcode_boost_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector{<:Real},
                           z_fine::Union{Real, AbstractVector{<:Real}}, k::AbstractVector{<:Real},
                           pk_mm_coarse::AbstractMatrix{<:Real};
                           pk_cb_coarse::Union{Nothing,AbstractMatrix}=nothing,
                           k_support::Union{Nothing,AbstractVector}=nothing,
                           pk_cb_support_coarse::Union{Nothing,AbstractMatrix}=nothing,
                           kwargs...)
    length(z_coarse) >= 5 || throw(ArgumentError("z_coarse must have at least 5 points for Akima interpolation."))
    issorted(z_coarse) && all(diff(z_coarse) .> 0) || throw(ArgumentError("z_coarse must be strictly increasing."))

    if z_fine isa AbstractVector
        issorted(z_fine) && all(diff(z_fine) .> 0) || throw(ArgumentError("z_fine must be strictly increasing."))
        all(z_fine .>= minimum(z_coarse)) && all(z_fine .<= maximum(z_coarse)) || throw(ArgumentError("z_fine must lie within the range of z_coarse."))
    else
        z_coarse[1] <= z_fine <= z_coarse[end] || throw(ArgumentError("z_fine must lie within the range of z_coarse."))
    end

    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    boost_coarse = hmcode_boost(cosmo, z_coarse, k, pk_mm_coarse;
                                pk_cb_z=pk_cb, k_support=k_support, kwargs...)

    z_fine_arr = z_fine isa Number ? [float(z_fine)] : z_fine
    boost_coarse_t = copy(transpose(boost_coarse))
    boost_fine_t = AbstractCosmologicalEmulators.akima_interpolation(boost_coarse_t, z_coarse, z_fine_arr)
    boost_fine = copy(transpose(boost_fine_t))
    return z_fine isa Number ? vec(boost_fine) : boost_fine
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector{<:Real},
                           z_fine::Union{Real, AbstractVector{<:Real}}, k::AbstractVector{<:Real},
                           pk_mm_coarse::AbstractMatrix{<:Real}, pk_cb_coarse::AbstractMatrix{<:Real};
                           kwargs...)
    return hmcode_boost_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse;
                             pk_cb_coarse=pk_cb_coarse, kwargs...)
end

function hmcode_boost_fast(cosmo::HMCodeCosmology, z_coarse::AbstractVector{<:Real},
                           z_fine::Union{Real, AbstractVector{<:Real}},
                           k_out::AbstractVector{<:Real}, k_support::AbstractVector{<:Real},
                           pk_mm_support_coarse::AbstractMatrix{<:Real};
                           pk_cb_support_coarse::Union{Nothing,AbstractMatrix}=nothing,
                           pk_cb_coarse::Union{Nothing,AbstractMatrix}=nothing,
                           kwargs...)
    pk_cb = _hmcode_choose_cb(pk_cb_coarse, pk_cb_support_coarse)
    return hmcode_boost_fast(cosmo, z_coarse, z_fine, k_out, pk_mm_support_coarse;
                             k_support=k_support, pk_cb_coarse=pk_cb, kwargs...)
end


"""
    predict_baryonic_discontinuity(cosmo::HMCodeCosmology; T_AGN::Real=10^7.8)
    predict_baryonic_discontinuity(params::AbstractVector; T_AGN::Real=10^7.8)

Predict the baryonic feedback threshold/feature redshift z_discontinuity.
"""
function predict_baryonic_discontinuity(cosmo::HMCodeCosmology; T_AGN::Real=10^7.8)
    logT_AGN = log10(Float64(T_AGN))
    sbar = -0.0030 * (logT_AGN - 7.8) + 0.0201
    sbarz = 0.0224 * (logT_AGN - 7.8) + 0.409
    om_b = cosmo.Omega_b
    om_m = cosmo.Omega_m
    return (log10(om_b / om_m) - log10(sbar)) / sbarz
end

function predict_baryonic_discontinuity(params::AbstractVector; T_AGN::Real=10^7.8)
    H0 = params[3]
    h = H0 > 10.0 ? H0 / 100.0 : H0
    ombh2 = params[4]
    omch2 = params[5]
    Mnu = params[6]
    om_b = ombh2 / h^2
    om_c = omch2 / h^2
    om_nu = (Mnu / 93.14) / h^2
    om_m = om_b + om_c + om_nu
    logT_AGN = log10(Float64(T_AGN))
    sbar = -0.0030 * (logT_AGN - 7.8) + 0.0201
    sbarz = 0.0224 * (logT_AGN - 7.8) + 0.409
    return (log10(om_b / om_m) - log10(sbar)) / sbarz
end

"""
    build_smart_coarse_grid(z_min::Real, z_max::Real, N_coarse::Int, z_feature::Union{Nothing, Real}=nothing; min_spacing::Real=1e-4)

Construct a coarse redshift grid of length N_coarse, placing N_coarse - 1 uniform nodes and inserting z_feature if within (z_min, z_max).
"""
function build_smart_coarse_grid(z_min::Real, z_max::Real, N_coarse::Int, z_feature::Union{Nothing, Real}=nothing; min_spacing::Real=1e-4)
    N_coarse >= 5 || throw(ArgumentError("N_coarse must be at least 5 for Akima interpolation."))
    zmin = Float64(z_min)
    zmax = Float64(z_max)
    if zmin >= zmax
        return [zmin]
    end

    if !isnothing(z_feature) && (zmin + min_spacing) < z_feature < (zmax - min_spacing)
        grid_base = collect(LinRange(zmin, zmax, N_coarse - 1))
        dists = abs.(grid_base .- z_feature)
        if minimum(dists) < min_spacing
            return collect(LinRange(zmin, zmax, N_coarse))
        end
        smart_grid = sort(vcat(grid_base, Float64(z_feature)))
        return smart_grid
    else
        return collect(LinRange(zmin, zmax, N_coarse))
    end
end

"""
    build_piecewise_coarse_grid(bounds, N_coarse, N_left)

Construct a fixed-shape coarse grid around a moving feature without sorting or
host-side control flow. `bounds` is the three-element vector
`[z_min, z_max, z_feature]`. The output contains `N_left` nodes below the
feature, the feature itself, and the remaining nodes above it.

`N_coarse` and `N_left` determine array shapes and must remain static across a
compiled call. The values in `bounds` may be traced and can therefore change at
every MCMC step.
"""
function build_piecewise_coarse_grid(bounds::AbstractVector,
                                     N_coarse::Int, N_left::Int)
    N_left >= 4 || throw(ArgumentError("N_left must be at least 4 for Akima interpolation."))
    N_right = N_coarse - N_left - 1
    N_right >= 4 || throw(ArgumentError("N_right must be at least 4 for Akima interpolation."))

    # Keep bounds as one-element arrays. This avoids scalar indexing when
    # `bounds` is a Reactant traced array.
    z_min = bounds[1:1]
    z_max = bounds[2:2]
    z_feature = bounds[3:3]

    T = eltype(bounds)
    left_fraction = T.(collect(0:(N_left - 1))) ./ T(N_left)
    right_fraction = T.(collect(1:N_right)) ./ T(N_right)
    left = z_min .+ left_fraction .* (z_feature .- z_min)
    right = z_feature .+ right_fraction .* (z_max .- z_feature)
    return vcat(left, z_feature, right)
end

"""
    build_baryonic_coarse_grid(params, z_limits, logT_AGN, N_coarse, N_left)

Compute the HMCode baryonic feature redshift and its piecewise coarse grid using
array operations only. `params` follows the package convention
`[ln10As, ns, H0, ωb, ωc, Mν, w0, wa]`; `z_limits` is `[z_min, z_max]`; and
`logT_AGN` is a one-element array. All three inputs may be traced device arrays.
"""
function build_baryonic_coarse_grid(params::AbstractVector,
                                    z_limits::AbstractVector,
                                    logT_AGN::AbstractVector,
                                    N_coarse::Int, N_left::Int)
    H0 = params[3:3]
    h = ifelse.(H0 .> 10, H0 ./ 100, H0)
    omega_b = params[4:4] ./ h.^2
    omega_c = params[5:5] ./ h.^2
    omega_nu = (params[6:6] ./ 93.14) ./ h.^2
    omega_m = omega_b .+ omega_c .+ omega_nu
    sbar = -0.0030 .* (logT_AGN .- 7.8) .+ 0.0201
    sbarz = 0.0224 .* (logT_AGN .- 7.8) .+ 0.409
    z_feature = (log10.(omega_b ./ omega_m) .- log10.(sbar)) ./ sbarz
    bounds = vcat(z_limits[1:1], z_limits[2:2], z_feature)
    return build_piecewise_coarse_grid(bounds, N_coarse, N_left)
end

"""
    hmcode_pmm_baryonic_smart(cosmo, z_fine, k, pk_mm_coarse; [pk_cb_coarse, N_coarse=50, T_AGN=10^7.8, kwargs...])

Baryonic-only HMCode fast solver using an automated smart coarse grid.
"""
function hmcode_pmm_baryonic_smart(cosmo::HMCodeCosmology, z_fine::Union{Real, AbstractVector},
                                  k::AbstractVector, pk_mm_coarse::AbstractMatrix;
                                  pk_cb_coarse::Union{Nothing, AbstractMatrix}=nothing,
                                  N_coarse::Int=50, T_AGN::Real=10^7.8, kwargs...)
    z_min = z_fine isa Real ? Float64(z_fine) : minimum(z_fine)
    z_max = z_fine isa Real ? Float64(z_fine) : maximum(z_fine)
    if z_min >= z_max || (z_fine isa AbstractVector && length(z_fine) <= N_coarse)
        if z_fine isa Real
            pk_mm_vec = size(pk_mm_coarse, 2) == 1 ? vec(pk_mm_coarse) : vec(pk_mm_coarse[:, 1])
            pk_cb_vec = isnothing(pk_cb_coarse) ? nothing : (size(pk_cb_coarse, 2) == 1 ? vec(pk_cb_coarse) : vec(pk_cb_coarse[:, 1]))
            return hmcode_Pmm(cosmo, Float64(z_fine), k, pk_mm_vec; pk_cb=pk_cb_vec, T_AGN=T_AGN, kwargs...)
        else
            return hmcode_Pmm(cosmo, z_fine, k, pk_mm_coarse; pk_cb_z=pk_cb_coarse, T_AGN=T_AGN, kwargs...)
        end
    end

    z_feature = predict_baryonic_discontinuity(cosmo; T_AGN=T_AGN)
    z_coarse = build_smart_coarse_grid(z_min, z_max, N_coarse, z_feature)

    pk_mm_coarse_eval = if size(pk_mm_coarse, 2) == length(z_fine) && length(z_fine) != length(z_coarse)
        pk_mm_t = copy(transpose(pk_mm_coarse))
        pk_mm_coarse_t = AbstractCosmologicalEmulators.akima_interpolation(pk_mm_t, z_fine, z_coarse)
        copy(transpose(pk_mm_coarse_t))
    else
        pk_mm_coarse
    end

    pk_cb_coarse_eval = if !isnothing(pk_cb_coarse) && size(pk_cb_coarse, 2) == length(z_fine) && length(z_fine) != length(z_coarse)
        pk_cb_t = copy(transpose(pk_cb_coarse))
        pk_cb_coarse_t = AbstractCosmologicalEmulators.akima_interpolation(pk_cb_t, z_fine, z_coarse)
        copy(transpose(pk_cb_coarse_t))
    else
        pk_cb_coarse
    end

    nleft = searchsortedlast(z_coarse, z_feature)
    if nleft >= 5 && nleft <= length(z_coarse) - 4
        return hmcode_pmm_fast_two_splines(cosmo, z_coarse, z_fine, k, pk_mm_coarse_eval;
                                           pk_cb_coarse=pk_cb_coarse_eval,
                                           z_split=z_feature, T_AGN=T_AGN, kwargs...)
    end
    # The requested interval is too short on one side of the feature for two
    # Akima segments. Preserve the existing API by falling back to one spline.
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse_eval;
                           pk_cb_coarse=pk_cb_coarse_eval, T_AGN=T_AGN, kwargs...)
end

"""Evaluate HMCode using physical-unit inputs and return physical P(k,z).

The public unit contract is `k` in Mpc⁻¹ and spectra in Mpc³. The internal
HMCode implementation continues to use h-units.
"""
function hmcode_pmm_physical(cosmo::HMCodeCosmology, z::AbstractVector,
                             k::AbstractVector, pk_mm_z::AbstractMatrix;
                             pk_cb_z=nothing, k_support=nothing,
                             pk_cb_support_z=nothing, kwargs...)
    h = cosmo.h
    k_phys_support = k_support === nothing ? k : k_support
    pk_cb = pk_cb_support_z === nothing ? pk_cb_z : pk_cb_support_z
    pk_cb === nothing && (pk_cb = pk_mm_z)
    return hmcode_Pmm(cosmo, z, k ./ h, pk_mm_z .* h^3;
                      pk_cb_z=pk_cb .* h^3,
                      k_support=k_phys_support ./ h, kwargs...) ./ h^3
end

function hmcode_pmm_fast_physical(cosmo::HMCodeCosmology,
                                  z_coarse::AbstractVector,
                                  z_fine::AbstractVector,
                                  k::AbstractVector,
                                  pk_mm_coarse::AbstractMatrix;
                                  pk_cb_coarse=nothing, k_support=nothing,
                                  pk_cb_support_coarse=nothing, kwargs...)
    h = cosmo.h
    k_phys_support = k_support === nothing ? k : k_support
    pk_cb = pk_cb_support_coarse === nothing ? pk_cb_coarse : pk_cb_support_coarse
    pk_cb === nothing && (pk_cb = pk_mm_coarse)
    return hmcode_pmm_fast(cosmo, z_coarse, z_fine, k ./ h,
                           pk_mm_coarse .* h^3;
                           pk_cb_coarse=pk_cb .* h^3,
                           k_support=k_phys_support ./ h, kwargs...) ./ h^3
end
