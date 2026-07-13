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

Compute the HMCode2020 nonlinear matter power spectrum from a sampled linear
matter power spectrum. `z` and `k` are vectors, and `pk_mm_z` must have shape
`(length(k), length(z))`, matching the `halofit_Pmm` convention.

For massive-neutrino cosmologies, pass the cold+baryon linear spectrum with
`pk_cb_z`. HMCode2020 uses the total matter spectrum for the returned nonlinear
`Pmm`, but uses cold+baryon σ(R,z) quantities internally for halo collapse and
transition parameters. If `pk_cb_z` is omitted, the total spectrum is used for
both roles, preserving the old behaviour.

Use `k_support` when the linear spectra are sampled on a wider grid than the
requested output. In that case `pk_mm_z` and `pk_cb_z` are interpreted on
`k_support`, while nonlinear power is returned on `k`. This lets σ(R,z) and
HMCode internal parameters use a high-k support grid without evaluating the final
halo-model spectrum on every support-grid point.

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
    return HMcode.hmcode_power(k, z, pk_mm, sigma_R_cb, cosmo; kwargs...)
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

function hmcode_Pmm(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                    pk_mm::AbstractVector, pk_cb::AbstractVector; kwargs...)
    return hmcode_Pmm(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end

"""
    hmcode_boost(cosmo, z, k, pk_mm_z; pk_cb_z=nothing, kwargs...)
    hmcode_boost(cosmo, z, k, pk_mm_z, pk_cb_z; kwargs...)

Return the HMCode2020 nonlinear boost, defined as `hmcode_Pmm(...) ./ pk_mm_z`.
Input shape conventions match `hmcode_Pmm`.
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

function hmcode_boost(cosmo::HMCodeCosmology, z::Number, k::AbstractVector,
                      pk_mm::AbstractVector, pk_cb::AbstractVector; kwargs...)
    return hmcode_boost(cosmo, z, k, pk_mm; pk_cb=pk_cb, kwargs...)
end
