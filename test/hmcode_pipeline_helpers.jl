# Shared live-input targets for ForwardDiff and Mooncake validation.
function _di_native_hmcode_cosmology(params, sigma_8)
    h = params[3] / 100
    omega_b = params[4] / h^2
    omega_nu = (params[6] / 93.14) / h^2
    omega_m = omega_b + params[5] / h^2 + omega_nu
    return Mapse.HMCodeCosmology(
        omega_m, omega_b, h, params[2], sigma_8,
        params[7], params[8], omega_nu, zero(h))
end

function _di_native_linear_spectra(params, z, pmm_emulator, pcb_emulator,
                                   growth_emulator)
    emulator_input = vcat(
        reshape(z, 1, :), repeat(reshape(params, :, 1), 1, length(z)))
    growth = vec(Mapse.AbstractCosmologicalEmulators.run_emulator(
        emulator_input, growth_emulator)[6:6, :])
    return (
        Mapse.get_Pk(params, z, growth, pmm_emulator),
        Mapse.get_Pk(params, z, growth, pcb_emulator),
    )
end

function _di_native_hmcode_prediction(
    parameters, z, k, pmm_emulator, pcb_emulator, growth_emulator, sigma_8;
    smart::Bool, feedback::Bool, nM::Int, N_coarse::Int=24, threaded::Bool=false,
)
    params = parameters[1:8]
    logT_AGN = parameters[9]
    cosmo = _di_native_hmcode_cosmology(params, sigma_8)
    temperature = feedback ? 10.0^logT_AGN : nothing
    z_coarse = if smart
        if feedback
            feature = Mapse.predict_baryonic_discontinuity(
                cosmo; T_AGN=temperature)
            Mapse.build_smart_coarse_grid(
                first(z), last(z), N_coarse, feature)
        else
            collect(range(first(z), last(z); length=N_coarse))
        end
    else
        z
    end
    pk_mm, pk_cb = _di_native_linear_spectra(
        params, z_coarse, pmm_emulator, pcb_emulator, growth_emulator)
    k_support = collect(Mapse.get_kgrid(pmm_emulator))

    if !smart
        return Mapse.hmcode_pmm(
            cosmo, z, k, pk_mm;
            pk_cb_z=pk_cb, k_support=k_support,
            T_AGN=temperature, nM=nM, threaded=threaded)
    elseif feedback
        return Mapse.hmcode_pmm_baryonic_smart(
            cosmo, z, k, pk_mm;
            pk_cb_coarse=pk_cb, k_support=k_support,
            N_coarse=N_coarse, T_AGN=temperature,
            nM=nM, threaded=threaded)
    else
        return Mapse.hmcode_pmm_fast(
            cosmo, z_coarse, z, k, pk_mm;
            pk_cb_coarse=pk_cb, k_support=k_support,
            T_AGN=nothing, nM=nM, threaded=threaded)
    end
end
