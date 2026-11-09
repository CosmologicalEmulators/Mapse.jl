# power_spectrum.jl
using LinearAlgebra
using Base.Threads
using Polyester
using LoopVectorization

struct SigmaREval{F,Tz}
    sigma_R::F
    z::Tz
    iz::Int
end
@inline (s::SigmaREval)(R::Real) = s.sigma_R(R, s.z)

struct PkLinEval{F,Tz}
    Pk_lin::F
    z::Tz
    iz::Int
end
@inline (s::PkLinEval)(k::Real) = s.Pk_lin(k, s.z)

_eval_sigma(f, R, z, iz) = f(R, z)
_eval_pk(f, k, z, iz) = f(k, z)

@inline _hmcode_matter_density(omega_m::Float64) =
    comoving_matter_density(omega_m)
@inline _hmcode_matter_density(omega_m::Real) = rho_critical * omega_m

@inline _hmcode_lagrangian_radius(M::Float64, omega_m::Float64) =
    Lagrangian_radius(M, omega_m)
@inline _hmcode_lagrangian_radius(M::Real, omega_m::Real) =
    cbrt(3.0 * M / (4.0 * pi * _hmcode_matter_density(omega_m)))

@inline _hmcode_redshift_from_scalefactor(a::Float64) =
    redshift_from_scalefactor(a)
@inline _hmcode_redshift_from_scalefactor(a::Real) = -1.0 + 1.0 / a

@inline _hmcode_feedback_parameters(T_AGN::Float64) =
    get_feedback_parameters(T_AGN)
@inline function _hmcode_feedback_parameters(T_AGN::Real)
    theta = log10(T_AGN / 10.0^7.8)
    return (
        B0 = 3.44 - 0.496 * theta,
        Bz = -0.0671 - 0.0371 * theta,
        Mb0 = 10.0^(13.87 + 1.81 * theta),
        Mbz = -0.108 + 0.195 * theta,
        f0 = (2.01 - 0.3 * theta) * 1.0e-2,
        fz = 0.409 + 0.0224 * theta,
    )
end

@inline _hmcode_dc_mead(
    a::Float64, omega_m::Float64, f_nu::Float64,
    growth::Float64, accumulated_growth::Float64,
) = dc_Mead(a, omega_m, f_nu, growth, accumulated_growth)

@inline function _hmcode_dc_mead(
    a::Real, omega_m::Real, f_nu::Real,
    growth::Real, accumulated_growth::Real,
)
    f_mead(x, y, p0, p1, p2, p3) =
        p0 + p1 * (1.0 - x) + p2 * (1.0 - x)^2 + p3 * (1.0 - y)
    dc = 1.0
    dc += f_mead(growth / a, accumulated_growth / a,
                 -0.0069, -0.0208, 0.0312, 0.0021) * log10(omega_m)
    dc += f_mead(growth / a, accumulated_growth / a,
                 0.0001, -0.0647, -0.0417, 0.0646)
    return dc * dc0 * (1.0 - 0.041 * f_nu)
end

@inline _hmcode_dv_mead(
    a::Float64, omega_m::Float64, f_nu::Float64,
    growth::Float64, accumulated_growth::Float64,
) = Dv_Mead(a, omega_m, f_nu, growth, accumulated_growth)

@inline function _hmcode_dv_mead(
    a::Real, omega_m::Real, f_nu::Real,
    growth::Real, accumulated_growth::Real,
)
    f_mead(x, y, p0, p1, p2, p3) =
        p0 + p1 * (1.0 - x) + p2 * (1.0 - x)^2 + p3 * (1.0 - y)
    log_omega_m = log10(omega_m)
    delta_v = 1.0
    delta_v += f_mead(growth / a, accumulated_growth / a,
                      -0.79, -10.17, 2.51, 6.51) * log_omega_m
    delta_v += f_mead(growth / a, accumulated_growth / a,
                      -1.89, 0.38, 18.8, -15.87) * log_omega_m^2
    return delta_v * Dv0 * (1.0 + 0.763 * f_nu)
end

const ND_HMCODE = 2.853

struct HMcodeParams{T}
    sigma_v::Vector{T}
    Delta_v::Vector{T}
    delta_c::Vector{T}
    eta::Vector{T}
    A::Vector{T}
    f_damp::Vector{T}
    k_star::Vector{T}
    B::Vector{T}
    k_damp::Vector{T}
end

mutable struct HMcodeWorkspace{T,Tz,G,GL,P<:HMcodeParams{T}}
    M::Vector{T}
    R::Vector{T}
    k::Vector{T}
    zs::Vector{Tz}
    growth_itp::G
    growth_LCDM_itp::GL
    params_tweaks::P
    params_notweaks::P
    nu_mat::Matrix{T}
    Pk_lin_mat::Matrix{T}
    Pk_wig_mat::Matrix{T}
    Ptmp1::Matrix{T}
    Ptmp2::Matrix{T}
    w1h_mat::Matrix{T}
    w1h_mat_fb::Matrix{T}
    rv::Matrix{T}
    cc::Matrix{T}
    ln1pc::Matrix{T}
    zf_mat::Matrix{T}
    Mb_vec::Vector{T}
    fstar_vec::Vector{T}
    W_buf::Vector{Matrix{T}}
    W2_buf::Vector{Matrix{T}}
    I1h_buf::Vector{Vector{T}}
    Pk1h_buf::Vector{Vector{T}}
    rv_eff_buf::Vector{Vector{T}}
    gcol_buf::Vector{Vector{T}}
    wtcol_buf::Vector{Vector{T}}
end

function compute_weights_inplace!(w1h_mat, M, nu_mat, amp, gcol, wtcol)
    nM, nz = size(nu_mat)
    p_st, q_st = 0.3, 0.707
    A_st = sqrt(2.0 * q_st) / (sqrt(pi) + gamma(0.5 - p_st) / 2.0^p_st)
    for iz in 1:nz
        ncol = view(nu_mat, :, iz)
        @inbounds @fastmath for i in 1:nM
            ν2 = ncol[i] * ncol[i]
            gcol[i] = A_st * (1.0 + (q_st * ν2)^(-p_st)) * exp(-q_st * ν2 / 2.0)
        end
        wtcol[1] = 0.5 * (ncol[2] - ncol[1])
        @inbounds @fastmath for i in 2:nM-1; wtcol[i] = 0.5 * (ncol[i+1] - ncol[i-1]); end
        wtcol[nM] = 0.5 * (ncol[nM] - ncol[nM-1])
        @inbounds @fastmath for i in 1:nM
            w1h_mat[i, iz] = (gcol[i] / M[i]) * amp[i]^2 * wtcol[i]
        end
    end
    return w1h_mat
end

function HMcodeWorkspace(k, zs, M_grid, cosmo, sigma_R_interp, Pk_lin_interp; nthreads=Threads.maxthreadid(), k_support=nothing)
    nk, nz, nM = length(k), length(zs), length(M_grid)
    T = promote_type(
        eltype(k), eltype(zs), eltype(M_grid),
        typeof(cosmo.Omega_m), typeof(cosmo.Omega_b), typeof(cosmo.h),
        typeof(cosmo.n_s), typeof(cosmo.sigma_8), typeof(cosmo.w0),
        typeof(cosmo.wa), typeof(cosmo.Omega_nu), typeof(cosmo.Omega_k),
    )
    M = collect(T.(M_grid))
    R = _hmcode_lagrangian_radius.(M, Ref(cosmo.Omega_m))
    sigma_fast, Pk_fast = sigma_R_interp, Pk_lin_interp
    growth_itp = get_growth_interpolator(cosmo, LCDM=false)
    growth_LCDM_itp = get_growth_interpolator(cosmo, LCDM=true)
    a_grid_inv = range(1e-4, 1.0, length=2000) |> collect
    g_grid_inv = growth_itp.(a_grid_inv)
    # Knots are active growth values. Use the existing gridded linear API,
    # whose reverse path retains knot cotangents. Exact interior knots use
    # the left interval; outside-support queries still throw.
    growth_itp_inverse = Interpolations.linear_interpolation(g_grid_inv, a_grid_inv)
    Sigma = zeros(T, nM, nz)
    for iz in 1:nz, iM in 1:nM; Sigma[iM, iz] = _eval_sigma(sigma_fast, R[iM], zs[iz], iz); end
    Pk_lin_mat = zeros(T, nk, nz)
    for iz in 1:nz, ik in 1:nk; Pk_lin_mat[ik, iz] = _eval_pk(Pk_fast, k[ik], zs[iz], iz); end
    params_tweaks = compute_hmcode_params(k, zs, Pk_fast, sigma_fast, Sigma, R, cosmo, growth_itp; tweaks=true, k_support=k_support)
    params_notweaks = _hmcode_notweaks_params(params_tweaks)
    nu_mat = zeros(T, nM, nz)
    for iz in 1:nz; @views nu_mat[:, iz] .= params_tweaks.delta_c[iz] ./ Sigma[:, iz]; end

    # BAO dewiggling must use the validated log-uniform support grid.
    # When k_support is provided, compute the wiggle component on k_support
    # (guaranteed log-uniform by _validate_hmcode_log_grid) and then
    # interpolate to k_out using log-log interpolation.  Using k_out directly
    # would give a physically wrong smoothing width for irregular or
    # differently-strided output grids.
    Pk_wig_mat = zeros(T, nk, nz)
    om, ob = cosmo.Omega_m*cosmo.h^2, cosmo.Omega_b*cosmo.h^2
    if k_support === nothing
        # No separate support grid: k_out = k_support, compute directly.
        for iz in 1:nz
            Pk_wig_mat[:, iz] .= get_Pk_wiggle(k, view(Pk_lin_mat, :, iz), cosmo.h, om, ob, cosmo.n_s)
        end
    else
        # Support grid available: compute wiggle on support grid, then
        # interpolate (log-log) each column to the output grid k.
        k_sup = collect(T.(k_support))
        nk_sup = length(k_sup)
        logk_out = log.(collect(T.(k)))
        logk_sup = log.(k_sup)
        Pk_sup_mat = zeros(T, nk_sup, nz)
        for iz in 1:nz, ik in 1:nk_sup
            Pk_sup_mat[ik, iz] = _eval_pk(Pk_fast, k_sup[ik], zs[iz], iz)
        end
        Pk_wig_sup = zeros(T, nk_sup, nz)
        for iz in 1:nz
            Pk_wig_sup[:, iz] .= get_Pk_wiggle(k_sup, view(Pk_sup_mat, :, iz), cosmo.h, om, ob, cosmo.n_s)
        end
        # Interpolate wiggle component (can be positive or negative) to k_out.
        # Use linear interpolation in log(k) space, preserving the sign.
        for iz in 1:nz
            wig_col = Pk_wig_sup[:, iz]
            for ik in 1:nk
                lk = logk_out[ik]
                # binary search in logk_sup
                ilo = searchsortedfirst(logk_sup, lk) - 1
                ilo = clamp(ilo, 1, nk_sup - 1)
                t = (lk - logk_sup[ilo]) / (logk_sup[ilo+1] - logk_sup[ilo])
                # linear interpolation in log-k of the (possibly negative) wiggle
                Pk_wig_mat[ik, iz] = (1 - t) * wig_col[ilo] + t * wig_col[ilo+1]
            end
        end
    end

    rhom = _hmcode_matter_density(cosmo.Omega_m)
    amp_no = M .* (1.0 - cosmo.Omega_nu/cosmo.Omega_m) ./ rhom
    w1h_mat = Matrix{T}(undef, nM, nz)
    compute_weights_inplace!(w1h_mat, M, nu_mat, amp_no, zeros(T, nM), zeros(T, nM))
    
    zf_mat = zeros(T, nM, nz)
    for iz in 1:nz
        compute_collapse_redshifts_fast!(view(zf_mat, :, iz), M, zs[iz], params_tweaks.delta_c[iz], cosmo.Omega_m, growth_itp, growth_itp_inverse, SigmaREval(sigma_fast, zs[iz], iz))
    end
    
    return HMcodeWorkspace(
        M, R, collect(T.(k)), collect(zs), growth_itp, growth_LCDM_itp,
        params_tweaks, params_notweaks, nu_mat, Pk_lin_mat, Pk_wig_mat,
        zeros(T, nk, nz), zeros(T, nk, nz), w1h_mat,
        zeros(T, nM, nz), zeros(T, nM, nz), zeros(T, nM, nz),
        zeros(T, nM, nz), zf_mat, zeros(T, nz), zeros(T, nz),
        [zeros(T, nM, nk) for _ in 1:nthreads],
        [zeros(T, nM, nk) for _ in 1:nthreads],
        [zeros(T, nk) for _ in 1:nthreads],
        [zeros(T, nk) for _ in 1:nthreads],
        [zeros(T, nM) for _ in 1:nthreads],
        [zeros(T, nM) for _ in 1:nthreads],
        [zeros(T, nM) for _ in 1:nthreads],
    )
end

function compute_hmcode_params(k, zs, Pk_lin, sigma_R, sigma_grid, R_grid, cosmo, growth_itp; tweaks=true, k_support=nothing)
    nz = length(zs); Om_m = cosmo.Omega_m; f_nu = cosmo.Omega_nu/Om_m
    T = promote_type(
        eltype(k), eltype(sigma_grid), eltype(R_grid),
        typeof(cosmo.Omega_m), typeof(cosmo.Omega_b), typeof(cosmo.h),
        typeof(cosmo.n_s), typeof(cosmo.sigma_8), typeof(cosmo.w0),
        typeof(cosmo.wa), typeof(cosmo.Omega_nu), typeof(cosmo.Omega_k),
    )
    R_nl, n_eff, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp =
        (zeros(T, nz) for _ in 1:11)
    kmin = k_support === nothing ? k[1] : k_support[1]
    kmax = k_support === nothing ? k[end] : k_support[end]
    @inbounds for iz in 1:nz
        z = zs[iz]; a = scalefactor_from_redshift(z); Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth_itp(a); G = get_accumulated_growth(a, growth_itp); dc = _hmcode_dc_mead(a, Om_mz, f_nu, g, G); Dv = _hmcode_dv_mead(a, Om_mz, f_nu, g, G)
        delta_c[iz], Delta_v[iz] = dc, Dv; Rnl = get_nonlinear_radius(R_grid[1], R_grid[end], dc, SigmaREval(sigma_R, z, iz)); s8 = _eval_sigma(sigma_R, 8.0, z, iz); sv = sigmaV(0.0, PkLinEval(Pk_lin, z, iz); kmin=kmin, kmax=kmax); neff = get_effective_index(Rnl, R_grid, view(sigma_grid, :, iz))
        R_nl[iz], sigma_v[iz], n_eff[iz], k_star[iz] = Rnl, sv, neff, 0.05618 * s8^(-1.013)
        if tweaks
            k_damp[iz] = 0.05699 * s8^(-1.089)
            f_damp[iz] = clamp(0.2696 * s8^0.9403, 1.0e-3, 0.99)
            eta[iz] = 0.1281 * s8^(-0.3644)
            B[iz] = 5.196
            A[iz] = clamp(1.875 * (1.603)^neff, 0.5, 2.0)
        else
            B[iz] = 4.0
            A[iz] = 1.0
        end
    end
    return HMcodeParams(sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp)
end

"""
Construct the unfitted/baryonic HMCode-2020 response parameters.

The response legs intentionally disable the fitted transition/dewiggling tweaks, but
must retain the HMCode-2020 `k_star` quartic one-halo cutoff. See CAMB PR #136.
"""
function _hmcode_notweaks_params(params::HMcodeParams)
    nz = length(params.k_star)
    T = eltype(params.k_star)
    return HMcodeParams(
        params.sigma_v,
        params.Delta_v,
        params.delta_c,
        zeros(T, nz),          # eta
        ones(T, nz),           # alpha/A
        zeros(T, nz),          # f_damp
        copy(params.k_star),    # REQUIRED: CAMB PR #136 low-k response fix
        fill(one(params.k_star[1]) * 4.0, nz), # concentration amplitude B
        zeros(T, nz),          # k_damp
    )
end

function compute_collapse_redshifts_fast!(zf_out, M_grid, z, dc, Om_m, growth_itp, growth_itp_inverse, sigmaR_func; gamma=0.01)
    g_obs = growth_itp(scalefactor_from_redshift(z))
    @inbounds for iM in eachindex(M_grid)
        sigma = sigmaR_func(_hmcode_lagrangian_radius(gamma*M_grid[iM], Om_m))
        g_target = g_obs * dc / sigma
        # Convert on each branch, as storing into zf_out already does. Keeping
        # the branch result concrete also supports mixed-precision fixed z grids.
        zf_out[iM] = (g_target >= g_obs) ? convert(eltype(zf_out), z) :
            convert(eltype(zf_out), _hmcode_redshift_from_scalefactor(growth_itp_inverse(g_target)))
    end
    return zf_out
end

function apply_baryonic_transform!(Wbuf, M, Mb, fstar, Om_m, Om_c, Om_b)
    nM, nk = size(Wbuf)
    # Applying @turbo to the nested loop
    @turbo warn_check_args=false for iM in 1:nM
        fg = (Om_b/Om_m - fstar) * (M[iM]/Mb)^2 / (1.0 + (M[iM]/Mb)^2)
        coeff = Om_c/Om_m + fg
        for ik in 1:nk
            Wbuf[iM, ik] = coeff * Wbuf[iM, ik] + fstar
        end
    end
    return Wbuf
end

# A boundary for the analytical reverse rule; preserve the optimized primal
# loops rather than differentiating LoopVectorization's LLVM intrinsics.
Base.@noinline function _one_halo_integral!(I1h, W2buf, Wbuf, w1h_iz, rhom)
    nM, nk = size(Wbuf)
    @turbo warn_check_args=false for ik in 1:nk, iM in 1:nM
        wv = Wbuf[iM, ik]
        W2buf[iM, ik] = wv * wv
    end
    @turbo warn_check_args=false for ik in 1:nk
        val = 0.0
        for iM in 1:nM
            val += W2buf[iM, ik] * w1h_iz[iM]
        end
        I1h[ik] = val * rhom
    end
    return I1h
end

function _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, tweaks, T_AGN, nu_iz, w1h_iz, rv_iz, rv_eff_tmp, c_iz, ln1pc_iz, Mb, fstar, Om_m, Om_c, Om_b, Wbuf, W2buf, I1h, Pk1h; use_fast_specials=true)
    nk, nM = length(k), length(ws.M); η = hmpars.eta[iz]
    @inbounds for iM in 1:nM; rv_eff_tmp[iM] = rv_iz[iM] * (nu_iz[iM]^η); end
    if use_fast_specials; win_NFW_fast!(Wbuf, k, rv_eff_tmp, c_iz, ln1pc_iz); else; for ik in 1:nk, iM in 1:nM; Wbuf[iM, ik] = wnfw_xc(k[ik]*rv_eff_tmp[iM], c_iz[iM]); end; end
    if (T_AGN !== nothing) && (!tweaks); apply_baryonic_transform!(Wbuf, ws.M, Mb, fstar, Om_m, Om_c, Om_b); end
    _one_halo_integral!(I1h, W2buf, Wbuf, w1h_iz, rhom)
    ks = hmpars.k_star[iz]
    safe_ks = ks > 0.0 ? ks : 1.0
    @inbounds @fastmath for ik in 1:nk
        if ks > 0.0
            x = k[ik] / safe_ks
            x4 = x * x * x * x
            fac = x4 / (1.0 + x4)
        else
            fac = 1.0
        end
        Pk1h[ik] = fac * I1h[ik]
    end
    if tweaks
        kd, f, alpha = hmpars.k_damp[iz], hmpars.f_damp[iz], hmpars.A[iz]
        Pk_lin_k, Pk_wig = view(ws.Pk_lin_mat, :, iz), view(ws.Pk_wig_mat, :, iz)
        @inbounds @fastmath for ik in 1:nk
            Pk_dwl = Pk_lin_k[ik] - (1.0 - exp(-(k[ik] * hmpars.sigma_v[iz])^2)) * Pk_wig[ik]
            y = (k[ik] / kd)^ND_HMCODE
            Pk_2h = Pk_dwl * (1.0 - f * y / (1.0 + y))
            Pk_out[ik, iz] = (Pk_2h^alpha + Pk1h[ik]^alpha)^(1.0 / alpha)
        end
    else
        @inbounds @fastmath for ik in 1:nk
            Pk_out[ik, iz] = ws.Pk_lin_mat[ik, iz] + Pk1h[ik]
        end
    end
    return nothing
end

# Logical chunks own buffers independently of the task/thread executing them.
# The serial chunk body can be differentiated without tracing the scheduler.
function _assemble_slice_chunk!(Pk_out, k, zs, ws, hmpars, rhom, tweaks,
        T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials, chunk, nchunks)
    buffers = (W_buf=ws.W_buf[chunk], W2_buf=ws.W2_buf[chunk],
               I1h_buf=ws.I1h_buf[chunk], Pk1h_buf=ws.Pk1h_buf[chunk],
               rv_eff_buf=ws.rv_eff_buf[chunk])
    return _assemble_owned_slice_chunk!(Pk_out, k, zs, ws, hmpars, rhom,
        tweaks, T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials,
        buffers, chunk, nchunks)
end

# The owned buffers are separate from read-only workspace inputs so reverse
# workers need not allocate cotangents for other chunks or unused model state.
function _assemble_owned_slice_chunk!(Pk_out, k, zs, ws, hmpars, rhom, tweaks,
        T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials, buffers, chunk, nchunks)
    first_z = fld((chunk - 1) * length(zs), nchunks) + 1
    last_z = fld(chunk * length(zs), nchunks)
    for iz in first_z:last_z
        _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, tweaks, T_AGN,
            view(ws.nu_mat, :, iz), view(w1h_ptr, :, iz), view(ws.rv, :, iz),
            buffers.rv_eff_buf, view(ws.cc, :, iz), view(ws.ln1pc, :, iz),
            ws.Mb_vec[iz], ws.fstar_vec[iz], Om_m, Om_c, Om_b,
            buffers.W_buf, buffers.W2_buf, buffers.I1h_buf,
            buffers.Pk1h_buf; use_fast_specials=use_fast_specials)
    end
    return nothing
end

function _assembly_chunk_count(zs, ws)
    isempty(zs) && return 0
    n = min(length(zs), Threads.nthreads(), length(ws.W_buf))
    n > 0 || throw(ArgumentError("Threaded assembly requires scratch buffers"))
    all(length(b) >= n for b in (ws.W2_buf, ws.I1h_buf, ws.Pk1h_buf, ws.rv_eff_buf)) ||
        throw(ArgumentError("Threaded assembly scratch buffer counts do not match"))
    return n
end

# Keep Polyester's llvmcall machinery outside the serial method's IR.
Base.@noinline function _assemble_threaded_slices!(Pk_out, k, zs, ws, hmpars,
        rhom, tweaks, T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials)
    nchunks = _assembly_chunk_count(zs, ws)
    @batch for chunk in 1:nchunks
        _assemble_slice_chunk!(Pk_out, k, zs, ws, hmpars, rhom, tweaks,
            T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials, chunk, nchunks)
    end
    return nothing
end

function hmcode_power_single!(Pk_out, k, zs, cosmo, ws; T_AGN=nothing, tweaks=true, threaded=false, use_fast_specials=true, hmpars=tweaks ? ws.params_tweaks : ws.params_notweaks)
    nz, nM = length(zs), length(ws.M); Om_m, Om_b, Om_nu = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    Om_c, rhom = Om_m - Om_b - Om_nu, _hmcode_matter_density(Om_m)
    zc = 10.0; ac = scalefactor_from_redshift(zc); growth, growth_LCDM = ws.growth_itp, ws.growth_LCDM_itp
    feedback_params = (T_AGN === nothing) ? (B0=0.0, Bz=0.0, Mb0=0.0, Mbz=0.0, f0=0.0, fz=0.0) : _hmcode_feedback_parameters(T_AGN)
    w1h_ptr = ws.w1h_mat
    if (T_AGN !== nothing && !tweaks)
        amp_fb = ws.M ./ rhom; compute_weights_inplace!(ws.w1h_mat_fb, ws.M, ws.nu_mat, amp_fb, ws.gcol_buf[1], ws.wtcol_buf[1]); w1h_ptr = ws.w1h_mat_fb
    end
    if (T_AGN !== nothing) && (!tweaks)
        for iz in 1:nz
            z = zs[iz]
            ws.Mb_vec[iz] = feedback_params.Mb0 * 10.0^(z * feedback_params.Mbz)
            ws.fstar_vec[iz] = feedback_stellar_fraction(feedback_params, z, Om_b, Om_m)
        end
    end
    for iz in 1:nz
        z = zs[iz]; Dv, B = hmpars.Delta_v[iz], hmpars.B[iz]
        if (T_AGN !== nothing) && (!tweaks); B = feedback_params.B0 * 10.0^(z * feedback_params.Bz); end
        
        a_obs = scalefactor_from_redshift(z); dolag = (growth(ac)/growth_LCDM(ac)) * (growth_LCDM(a_obs)/growth(a_obs))
        rvs = cbrt(Dv)
        @inbounds @fastmath for iM in 1:nM
            zfv = ws.zf_mat[iM, iz]
            ws.rv[iM, iz] = ws.R[iM] / rvs
            cc_v = B * (1.0 + zfv)/(1.0 + z) * dolag
            ws.cc[iM, iz] = cc_v
            ws.ln1pc[iM, iz] = log(1.0 + cc_v)
        end
    end
    if threaded
        _assemble_threaded_slices!(Pk_out, k, zs, ws, hmpars, rhom, tweaks,
            T_AGN, w1h_ptr, Om_m, Om_c, Om_b, use_fast_specials)
    else
        for iz in 1:nz; _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_ptr, :, iz), view(ws.rv, :, iz), ws.rv_eff_buf[1], view(ws.cc, :, iz), view(ws.ln1pc, :, iz), ws.Mb_vec[iz], ws.fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[1], ws.W2_buf[1], ws.I1h_buf[1], ws.Pk1h_buf[1]; use_fast_specials=use_fast_specials); end
    end
    return Pk_out
end

function hmcode_power!(Pk_out, k, zs, cosmo, ws; T_AGN=10^7.8, threaded=true, use_fast_specials=true)
    hmcode_power_single!(Pk_out, k, zs, cosmo, ws; T_AGN=nothing, tweaks=true, threaded=threaded, use_fast_specials=use_fast_specials)
    if T_AGN !== nothing
        hmcode_power_single!(ws.Ptmp1, k, zs, cosmo, ws; T_AGN=nothing, tweaks=false, threaded=threaded, use_fast_specials=use_fast_specials)
        hmcode_power_single!(ws.Ptmp2, k, zs, cosmo, ws; T_AGN=T_AGN, tweaks=false, threaded=threaded, use_fast_specials=use_fast_specials)
        @inbounds @fastmath for i in eachindex(Pk_out)
            Pk_out[i] = Pk_out[i] * (ws.Ptmp2[i] / ws.Ptmp1[i])
        end
    end
    return Pk_out
end

function hmcode_power(k, zs, Pk_lin, sigma_R, cosmo; T_AGN=10^7.8, Mmin=1e0, Mmax=1e18, nM=128, threaded=false, use_fast_specials=true, k_support=nothing)
    nM >= 2 || throw(ArgumentError("HMCode nM must be at least 2."))
    feedback_type = T_AGN === nothing ? Float64 : typeof(T_AGN)
    T = promote_type(
        eltype(k), eltype(zs), feedback_type,
        typeof(cosmo.Omega_m), typeof(cosmo.Omega_b), typeof(cosmo.h),
        typeof(cosmo.n_s), typeof(cosmo.sigma_8), typeof(cosmo.w0),
        typeof(cosmo.wa), typeof(cosmo.Omega_nu), typeof(cosmo.Omega_k),
    )
    M = T.(exp.(range(log(Mmin), log(Mmax), length=nM)))
    ws = HMcodeWorkspace(
        collect(T.(k)), collect(zs), M, cosmo, sigma_R, Pk_lin;
        nthreads=Threads.maxthreadid(), k_support=k_support)
    Pk_out = zeros(T, length(k), length(zs))
    hmcode_power!(Pk_out, ws.k, ws.zs, cosmo, ws; T_AGN=T_AGN, threaded=threaded, use_fast_specials=use_fast_specials)
    return Pk_out
end
