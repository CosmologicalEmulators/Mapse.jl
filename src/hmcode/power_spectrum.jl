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
end
@inline (s::SigmaREval)(R::Float64) = s.sigma_R(R, s.z)

struct PkLinEval{F}
    Pk_lin::F
    z::Float64
end
@inline (s::PkLinEval)(k::Float64) = s.Pk_lin(k, s.z)

const ND_HMCODE = 2.853

struct HMcodeParams
    R_nl::Vector{Float64}
    n_eff::Vector{Float64}
    C_curv::Vector{Float64}
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
    sigma_fast::Any
    Pk_fast::Any
    growth_itp::Any
    growth_LCDM_itp::Any
    growth_itp_inverse::Any
    params_tweaks::HMcodeParams
    params_notweaks::HMcodeParams
    Sigma::Matrix{Float64}
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
    Mb_vec::Vector{Float64}
    fstar_vec::Vector{Float64}
    W_buf::Vector{Matrix{Float64}}
    W2_buf::Vector{Matrix{Float64}}
    I1h_buf::Vector{Vector{Float64}}
    Pk1h_buf::Vector{Vector{Float64}}
    Pkwig_buf::Vector{Vector{Float64}}
    rv_eff_buf::Vector{Vector{Float64}}
    zf_buf::Vector{Vector{Float64}}
    rv_buf::Vector{Vector{Float64}}
    cc_buf::Vector{Vector{Float64}}
    ln1pc_buf::Vector{Vector{Float64}}
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

function HMcodeWorkspace(k, zs, M_grid, cosmo, sigma_R_interp, Pk_lin_interp; nthreads=Threads.nthreads())
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
    for iz in 1:nz, iM in 1:nM; Sigma[iM, iz] = sigma_fast(R[iM], zs[iz]); end
    Pk_lin_mat = zeros(nk, nz)
    for iz in 1:nz, ik in 1:nk; Pk_lin_mat[ik, iz] = Pk_fast(k[ik], zs[iz]); end
    params_tweaks = compute_hmcode_params(k, zs, Pk_fast, sigma_fast, Sigma, R, cosmo, growth_itp; tweaks=true)
    params_notweaks = HMcodeParams(params_tweaks.R_nl, params_tweaks.n_eff, params_tweaks.C_curv, params_tweaks.sigma_v, params_tweaks.Delta_v, params_tweaks.delta_c, zeros(nz), ones(nz), zeros(nz), params_tweaks.k_star, fill(4.0, nz), zeros(nz))
    nu_mat = zeros(nM, nz)
    for iz in 1:nz; @views nu_mat[:, iz] .= params_tweaks.delta_c[iz] ./ Sigma[:, iz]; end
    Pk_wig_mat = zeros(nk, nz)
    om, ob = cosmo.Omega_m*cosmo.h^2, cosmo.Omega_b*cosmo.h^2
    for iz in 1:nz; Pk_wig_mat[:, iz] .= get_Pk_wiggle(k, view(Pk_lin_mat, :, iz), cosmo.h, om, ob, cosmo.n_s); end
    
    rhom = comoving_matter_density(cosmo.Omega_m)
    amp_no = M .* (1.0 - cosmo.Omega_nu/cosmo.Omega_m) ./ rhom
    w1h_mat = Matrix{Float64}(undef, nM, nz)
    compute_weights_inplace!(w1h_mat, M, nu_mat, amp_no, zeros(nM), zeros(nM))
    
    return HMcodeWorkspace(M, R, collect(Float64.(k)), collect(Float64.(zs)), sigma_fast, Pk_fast, growth_itp, growth_LCDM_itp, growth_itp_inverse, params_tweaks, params_notweaks, Sigma, nu_mat, Pk_lin_mat, Pk_wig_mat, zeros(nk, nz), zeros(nk, nz), w1h_mat, zeros(nM, nz), zeros(nM, nz), zeros(nM, nz), zeros(nM, nz), zeros(nz), zeros(nz), [zeros(nM, nk) for _ in 1:nthreads], [zeros(nM, nk) for _ in 1:nthreads], [zeros(nk) for _ in 1:nthreads], [zeros(nk) for _ in 1:nthreads], [zeros(nk) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads], [zeros(nM) for _ in 1:nthreads])
end

function compute_sigma_grid!(Sigma, R_grid, zs, sigma_R)
    @inbounds for iz in eachindex(zs), iM in eachindex(R_grid); Sigma[iM, iz] = sigma_R(R_grid[iM], zs[iz]); end
    return Sigma
end

function compute_hmcode_params(k, zs, Pk_lin, sigma_R, sigma_grid, R_grid, cosmo, growth_itp; tweaks=true, kmin_sigmaV=1e-5)
    nz = length(zs); Om_m = cosmo.Omega_m; f_nu = cosmo.Omega_nu/Om_m
    R_nl, n_eff, C_curv, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp = (zeros(nz) for _ in 1:12)
    @inbounds for iz in 1:nz
        z = zs[iz]; a = scalefactor_from_redshift(z); Om_mz = _Omega_m_a(a, cosmo, LCDM=false)
        g = growth_itp(a); G = get_accumulated_growth(a, growth_itp); dc = dc_Mead(a, Om_mz, f_nu, g, G); Dv = Dv_Mead(a, Om_mz, f_nu, g, G)
        delta_c[iz], Delta_v[iz] = dc, Dv; Rnl = get_nonlinear_radius(R_grid[1], R_grid[end], dc, SigmaREval(sigma_R, z)); s8 = sigma_R(8.0, z); sv = sigmaV(0.0, PkLinEval(Pk_lin, z); kmin=kmin_sigmaV); neff = get_effective_index(Rnl, R_grid, view(sigma_grid, :, iz))
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
    return HMcodeParams(R_nl, n_eff, C_curv, sigma_v, Delta_v, delta_c, eta, A, f_damp, k_star, B, k_damp)
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

function _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, cosmo, tweaks, T_AGN, nu_iz, w1h_iz, rv_iz, rv_eff_tmp, c_iz, ln1pc_iz, Mb, fstar, Om_m, Om_c, Om_b, Wbuf, W2buf, I1h, Pk1h, Pkwig_tmp; use_fast_specials=true)
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
    @inbounds @fastmath for ik in 1:nk
        x = k[ik] / ks
        x4 = x * x * x * x
        Pk1h[ik] = (x4 / (1.0 + x4)) * I1h[ik]
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
    nk, nz, nM = length(k), length(zs), length(ws.M); Om_m, Om_b, Om_nu = cosmo.Omega_m, cosmo.Omega_b, cosmo.Omega_nu
    Om_c, f_nu, rhom = Om_m - Om_b - Om_nu, Om_nu/Om_m, comoving_matter_density(Om_m)
    zc = 10.0; ac = scalefactor_from_redshift(zc); growth, growth_LCDM = ws.growth_itp, ws.growth_LCDM_itp
    feedback_params = (T_AGN === nothing) ? Dict{Symbol, Float64}() : get_feedback_parameters(T_AGN)
    w1h_ptr = ws.w1h_mat
    if (T_AGN !== nothing && !tweaks)
        amp_fb = ws.M ./ rhom; compute_weights_inplace!(ws.w1h_mat_fb, ws.M, ws.nu_mat, amp_fb, ws.gcol_buf[1], ws.wtcol_buf[1]); w1h_ptr = ws.w1h_mat_fb
    end
    if (T_AGN !== nothing) && (!tweaks)
        for iz in 1:nz
            z = zs[iz]
            ws.Mb_vec[iz] = feedback_params[:Mb0] * 10.0^(z * feedback_params[:Mbz])
            ws.fstar_vec[iz] = feedback_stellar_fraction(feedback_params, z, Om_b, Om_m)
        end
    end
    for iz in 1:nz
        z = zs[iz]; dc, Dv, B = hmpars.delta_c[iz], hmpars.Delta_v[iz], hmpars.B[iz]
        if (T_AGN !== nothing) && (!tweaks); B = feedback_params[:B0] * 10.0^(z * feedback_params[:Bz]); end
        
        # Only compute cc if we are in tweaks mode, or if it hasn't been computed?
        # Actually, cc depends on B which changes if T_AGN !== nothing && !tweaks
        # compute_collapse_redshifts_fast! is expensive. We only need to compute it once per cosmology if we cache it correctly.
        # But wait, it doesn't depend on tweaks or T_AGN! It only depends on z, dc, Om_m, growth, etc.
        # So we can skip it if it's already computed.
        # For now, let's leave it as is to ensure correctness.
        compute_collapse_redshifts_fast!(view(ws.cc, :, iz), ws.M, z, dc, Om_m, growth, ws.growth_itp_inverse, SigmaREval(ws.sigma_fast, z))
        a_obs = scalefactor_from_redshift(z); dolag = (growth(ac)/growth_LCDM(ac)) * (growth_LCDM(a_obs)/growth(a_obs))
        rvs = cbrt(Dv)
        @inbounds @fastmath for iM in 1:nM
            zfv = ws.cc[iM, iz]
            ws.rv[iM, iz] = ws.R[iM] / rvs
            cc_v = B * (1.0 + zfv)/(1.0 + z) * dolag
            ws.cc[iM, iz] = cc_v
            ws.ln1pc[iM, iz] = log(1.0 + cc_v)
        end
    end
    if threaded
        @batch for iz in 1:nz
            tid = threadid(); _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, cosmo, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_ptr, :, iz), view(ws.rv, :, iz), ws.rv_eff_buf[tid], view(ws.cc, :, iz), view(ws.ln1pc, :, iz), ws.Mb_vec[iz], ws.fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[tid], ws.W2_buf[tid], ws.I1h_buf[tid], ws.Pk1h_buf[tid], ws.Pkwig_buf[tid]; use_fast_specials=use_fast_specials)
        end
    else
        for iz in 1:nz; _assemble_slice!(Pk_out, iz, k, ws, hmpars, rhom, cosmo, tweaks, T_AGN, view(ws.nu_mat, :, iz), view(w1h_ptr, :, iz), view(ws.rv, :, iz), ws.rv_eff_buf[1], view(ws.cc, :, iz), view(ws.ln1pc, :, iz), ws.Mb_vec[iz], ws.fstar_vec[iz], Om_m, Om_c, Om_b, ws.W_buf[1], ws.W2_buf[1], ws.I1h_buf[1], ws.Pk1h_buf[1], ws.Pkwig_buf[1]; use_fast_specials=use_fast_specials); end
    end
    return Pk_out
end

function hmcode_power!(Pk_out, k, zs, Pk_lin, sigma_R, cosmo, ws; T_AGN=10^7.8, threaded=true, use_fast_specials=true)
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

function _hmcode_mass_steps(nM, accuracy)
    accuracy > 0 || throw(ArgumentError("HMCode accuracy must be positive."))
    if nM === nothing
        return max(16, ceil(Int, 256 * float(accuracy)))
    end
    accuracy == 1.0 || throw(ArgumentError("Pass either HMCode accuracy or nM, not both."))
    nM >= 2 || throw(ArgumentError("HMCode nM must be at least 2."))
    return Int(nM)
end

function hmcode_power(k, zs, Pk_lin, sigma_R, cosmo; T_AGN=10^7.8, Mmin=1e0, Mmax=1e18, nM=nothing, accuracy=1.0, threaded=false, use_fast_specials=true)
    nM = _hmcode_mass_steps(nM, accuracy)
    M = exp.(range(log(Mmin), log(Mmax), length=nM))
    ws = HMcodeWorkspace(collect(Float64.(k)), collect(Float64.(zs)), M, cosmo, sigma_R, Pk_lin; nthreads=Threads.nthreads())
    Pk_out = zeros(length(k), length(zs))
    hmcode_power!(Pk_out, ws.k, ws.zs, Pk_lin, sigma_R, cosmo, ws; T_AGN=T_AGN, threaded=threaded, use_fast_specials=use_fast_specials)
    return Pk_out
end
