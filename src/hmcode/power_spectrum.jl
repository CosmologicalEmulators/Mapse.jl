# power_spectrum.jl
using LinearAlgebra
using Base.Threads
using Polyester
using LoopVectorization
import DataInterpolations
const DI = DataInterpolations

struct SigmaREval{F}
    sigma_R::F
    z::Float64
    iz::Int
end
@inline (s::SigmaREval)(R::Float64) = s.sigma_R(R, s.z)

struct PkLinEval{F}
    Pk_lin::F
    z::Float64
    iz::Int
end
@inline (s::PkLinEval)(k::Float64) = s.Pk_lin(k, s.z)

_eval_sigma(f, R, z, iz) = f(R, z)
_eval_pk(f, k, z, iz) = f(k, z)

const ND_HMCODE = 2.853

struct HMcodeParams
    sigma_v::Vector{Float64}
    Delta_v::Vector{Float64}
    delta_c::Vector{Float64}
    eta::Vector{Float64}
    A::Vector{Float64}
    f_damp::Vector{Float64}
    k_star::Vector{Float64}
    B::Vector{Float64}
    k_damp::Vector{Float64}
end

mutable struct HMcodeWorkspace
    M::Vector{Float64}
    R::Vector{Float64}
    k::Vector{Float64}
    zs::Vector{Float64}
    growth_itp::Any
    growth_LCDM_itp::Any
    params_tweaks::HMcodeParams
    params_notweaks::HMcodeParams
    nu_mat::Matrix{Float64}
    Pk_lin_mat::Matrix{Float64}
    Pk_wig_mat::Matrix{Float64}
    Ptmp1::Matrix{Float64}
    Ptmp2::Matrix{Float64}
    w1h_mat::Matrix{Float64}
    w1h_mat_fb::Matrix{Float64}
    rv::Matrix{Float64}
    cc::Matrix{Float64}
    ln1pc::Matrix{Float64}
    zf_mat::Matrix{Float64}
    Mb_vec::Vector{Float64}
    fstar_vec::Vector{Float64}
    W_buf::Vector{Matrix{Float64}}
    W2_buf::Vector{Matrix{Float64}}
    I1h_buf::Vector{Vector{Float64}}
    Pk1h_buf::Vector{Vector{Float64}}
    rv_eff_buf::Vector{Vector{Float64}}
    gcol_buf::Vector{Vector{Float64}}
    wtcol_buf::Vector{Vector{Float64}}
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
    M = collect(Float64.(M_grid))
    R = Lagrangian_radius(M, cosmo.Omega_m)
    sigma_fast, Pk_fast = sigma_R_interp, Pk_lin_interp
    growth_itp = get_growth_interpolator(cosmo, LCDM=false)
    growth_LCDM_itp = get_growth_interpolator(cosmo, LCDM=true)
    a_grid_inv = range(1e-4, 1.0, length=2000) |> collect
    g_grid_inv = growth_itp.(a_grid_inv)
    growth_itp_inverse = DI.LinearInterpolation(a_grid_inv, g_grid_inv)
    Sigma = zeros(nM, nz)
    for iz in 1:nz, iM in 1:nM; Sigma[iM, iz] = _eval_sigma(sigma_fast, R[iM], zs[iz], iz); end
    Pk_lin_mat = zeros(nk, nz)
    for iz in 1:nz, ik in 1:nk; Pk_lin_mat[ik, iz] = _eval_pk(Pk_fast, k[ik], zs[iz], iz); end
    params_tweaks = compute_hmcode_params(k, zs, Pk_fast, sigma_fast, Sigma, R, cosmo, growth_itp; tweaks=true, k_support=k_support)
    params_notweaks = _hmcode_notweaks_params(params_tweaks)
    nu_mat = zeros(nM, nz)
    for iz in 1:nz; @views nu_mat[:, iz] .= params_tweaks.delta_c[iz] ./ Sigma[:, iz]; end

    # BAO dewiggling must use the validated log-uniform support grid.
    # When k_support is provided, compute the wiggle component on k_support
    # (guaranteed log-uniform by _validate_hmcode_log_grid) and then
    # interpolate to k_out using log-log interpolation.  Using k_out directly
    # would give a physically wrong smoothing width for irregular or
    # differently-strided output grids.
    Pk_wig_mat = zeros(nk, nz)
    om, ob = cosmo.Omega_m*cosmo.h^2, cosmo.Omega_b*cosmo.h^2
    if k_support === nothing
        # No separate support grid: k_out = k_support, compute directly.
        for iz in 1:nz
            Pk_wig_mat[:, iz] .= get_Pk_wiggle(k, view(Pk_lin_mat, :, iz), cosmo.h, om, ob, cosmo.n_s)
        end
    else
        # Support grid available: compute wiggle on support grid, then
        # interpolate (log-log) each column to the output grid k.
        k_sup = collect(Float64.(k_support))
        nk_sup = length(k_sup)
        logk_out = log.(collect(Float64.(k)))
        logk_sup = log.(k_sup)
        Pk_sup_mat = zeros(nk_sup, nz)
        for iz in 1:nz, ik in 1:nk_sup
            Pk_sup_mat[ik, iz] = _eval_pk(Pk_fast, k_sup[ik], zs[iz], iz)
        end
        Pk_wig_sup = zeros(nk_sup, nz)
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

    rhom = comoving_matter_density(cosmo.Omega_m)
    amp_no = M .* (1.0 - cosmo.Omega_nu/cosmo.Omega_m) ./ rhom
    w1h_mat = Matrix{Float64}(undef, nM, nz)
    compute_weights_inplace!(w1h_mat, M, nu_mat, amp_no, zeros(nM), zeros(nM))
    
    zf_mat = zeros(nM, nz)
    for iz in 1:nz
        compute_collapse_redshifts_fast!(view(zf_mat, :, iz), M, zs[iz], params_tweaks.delta_c[iz], cosmo.Omega_m, growth_itp, growth_itp_inverse, SigmaREval(sigma_fast, zs[iz], iz))
    end
    
    return HMcodeWorkspace(M, R, collect(Float64.(k)), collect(Float64.(zs)), growth_itp, growth_LCDM_itp, params_tweaks, params_notweaks, nu_mat, Pk_lin_mat, Pk_wig_mat, zeros(nk, nz), zeros(nk, nz), w1h_mat, zeros(nM, nz), zeros(nM, nz), zeros(nM, nz), zeros(nM, nz), zf_mat, zeros(nz), zeros(nz), [zeros(nM, nk) for _ in 1:nthreads], [zeros(nM, nk) for _ in 1:nthreads], [zeros(nk) for _ in 1:nthreads], [zeros(nk) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads])
end

function compute_hmcode_params(k, zs, Pk_lin, sigma_R, sigma_grid, R_grid, cosmo, growth_itp; tweaks=true, k_support=nothing)
    nz = length(zs); Om_m = cosmo.Omega_m; f_nu = cosmo.Omega_nu/Om_m
    R_nl, n_eff, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp = (zeros(nz) for _ in 1:11)
    kmin = k_support === nothing ? k[1] : k_support[1]
    kmax = k_support === nothing ? k[end] : k_support[end]
    @inbounds for iz in 1:nz
        z = zs[iz]; a = scalefactor_from_redshift(z); Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth_itp(a); G = get_accumulated_growth(a, growth_itp); dc = dc_Mead(a, Om_mz, f_nu, g, G); Dv = Dv_Mead(a, Om_mz, f_nu, g, G)
        delta_c[iz], Delta_v[iz] = dc, Dv; Rnl = get_nonlinear_radius(R_grid[1], R_grid[end], dc, SigmaREval(sigma_R, z, iz)); s8 = _eval_sigma(sigma_R, 8.0, z, iz); sv = sigmaV(0.0, PkLinEval(Pk_lin, z, iz); kmin=kmin, kmax=kmax); neff = get_effective_index(Rnl, R_grid, view(sigma_grid, :, iz))
        R_nl[iz], sigma_v[iz], n_eff[iz], k_star[iz] = Rnl, sv, neff, 0.05618 * s8^(-1.013)
        if tweaks
            k_damp[iz] = 0.05699 * s8^(-1.089)
            f_damp[iz] = 0.2696 * s8^0.9403
            eta[iz] = 0.1281 * s8^(-0.3644)
            B[iz] = 5.196
            A[iz] = 1.875 * (1.603)^neff
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
    return HMcodeParams(
        params.sigma_v,
        params.Delta_v,
        params.delta_c,
        zeros(nz),             # eta
        ones(nz),              # alpha/A
        zeros(nz),             # f_damp
        copy(params.k_star),    # REQUIRED: CAMB PR #136 low-k response fix
        fill(4.0, nz),          # concentration amplitude B
        zeros(nz),             # k_damp
    )
end

function compute_collapse_redshifts_fast!(zf_out, M_grid, z, dc, Om_m, growth_itp, growth_itp_inverse, sigmaR_func; gamma=0.01)
    g_obs = growth_itp(scalefactor_from_redshift(z))
    @inbounds for iM in eachindex(M_grid)
        sigma = sigmaR_func(Lagrangian_radius(gamma*M_grid[iM], Om_m))
        g_target = g_obs * dc / sigma
        zf_out[iM] = (g_target >= g_obs) ? z : redshift_from_scalefactor(growth_itp_inverse(g_target))
    end
    return zf_out
end

function apply_baryonic_transform!(Wbuf, M, Mb, fstar, Om_m, Om_c, Om_b)
    nM, nk = size(Wbuf)
    # Applying @turbo to the nested loop
    @turbo for iM in 1:nM
        fg = (Om_b/Om_m - fstar) * (M[iM]/Mb)^2 / (1.0 + (M[iM]/Mb)^2)
        coeff = Om_c/Om_m + fg
        for ik in 1:nk
            Wbuf[iM, ik] = coeff * Wbuf[iM, ik] + fstar
        end
    end
    return Wbuf
end

function _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, tweaks, T_AGN, nu_iz, w1h_iz, rv_iz, rv_eff_tmp, c_iz, ln1pc_iz, Mb, fstar, Om_m, Om_c, Om_b, Wbuf, W2buf, I1h, Pk1h; use_fast_specials=true)
    nk, nM = length(k), length(ws.M); η = hmpars.eta[iz]
    @inbounds for iM in 1:nM; rv_eff_tmp[iM] = rv_iz[iM] * (nu_iz[iM]^η); end
    if use_fast_specials; win_NFW_fast!(Wbuf, k, rv_eff_tmp, c_iz, ln1pc_iz); else; for ik in 1:nk, iM in 1:nM; Wbuf[iM, ik] = wnfw_xc(k[ik]*rv_eff_tmp[iM], c_iz[iM]); end; end
    if (T_AGN !== nothing) && (!tweaks); apply_baryonic_transform!(Wbuf, ws.M, Mb, fstar, Om_m, Om_c, Om_b); end
    @turbo for ik in 1:nk, iM in 1:nM
        wv = Wbuf[iM, ik]
        W2buf[iM, ik] = wv * wv
    end
    # mul!(I1h, transpose(W2buf), w1h_iz); @. I1h = I1h * rhom
    # Manual loop for I1h to avoid potential mul!/transpose issues
    @turbo for ik in 1:nk
        val = 0.0
        for iM in 1:nM
            val += W2buf[iM, ik] * w1h_iz[iM]
        end
        I1h[ik] = val * rhom
    end
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

function hmcode_power_single!(Pk_out, k, zs, cosmo, ws; T_AGN=nothing, tweaks=true, threaded=false, use_fast_specials=true, hmpars=tweaks ? ws.params_tweaks : ws.params_notweaks)
    nz, nM = length(zs), length(ws.M); Om_m, Om_b, Om_nu = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    Om_c, rhom = Om_m - Om_b - Om_nu, comoving_matter_density(Om_m)
    zc = 10.0; ac = scalefactor_from_redshift(zc); growth, growth_LCDM = ws.growth_itp, ws.growth_LCDM_itp
    feedback_params = (T_AGN === nothing) ? (B0=0.0, Bz=0.0, Mb0=0.0, Mbz=0.0, f0=0.0, fz=0.0) : get_feedback_parameters(T_AGN)
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
        @batch for iz in 1:nz
            tid = threadid(); _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_ptr, :, iz), view(ws.rv, :, iz), ws.rv_eff_buf[tid], view(ws.cc, :, iz), view(ws.ln1pc, :, iz), ws.Mb_vec[iz], ws.fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[tid], ws.W2_buf[tid], ws.I1h_buf[tid], ws.Pk1h_buf[tid]; use_fast_specials=use_fast_specials)
        end
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
    M = exp.(range(log(Mmin), log(Mmax), length=nM))
    ws = HMcodeWorkspace(collect(Float64.(k)), collect(Float64.(zs)), M, cosmo, sigma_R, Pk_lin; nthreads=Threads.maxthreadid(), k_support=k_support)
    Pk_out = zeros(length(k), length(zs))
    hmcode_power!(Pk_out, ws.k, ws.zs, cosmo, ws; T_AGN=T_AGN, threaded=threaded, use_fast_specials=use_fast_specials)
    return Pk_out
end
