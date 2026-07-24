using Test
using DelimitedFiles
using Enzyme
using Reactant
using Mapse

function _native_hmcode_cosmology(params, sigma_8)
    h = params[3] / 100
    omega_b = params[4] / h^2
    omega_nu = (params[6] / 93.14) / h^2
    omega_m = omega_b + params[5] / h^2 + omega_nu
    return Mapse.HMCodeCosmology(
        omega_m, omega_b, h, params[2], sigma_8,
        params[7], params[8], omega_nu, 0.0)
end

function _native_linear_spectra(params, z, pmm_emu, pcb_emu, growth_emu)
    emulator_input = vcat(
        reshape(z, 1, :), repeat(reshape(params, :, 1), 1, length(z)))
    growth = vec(Mapse.AbstractCosmologicalEmulators.run_emulator(
        emulator_input, growth_emu)[6:6, :])
    return (
        Mapse.get_Pk(params, z, growth, pmm_emu),
        Mapse.get_Pk(params, z, growth, pcb_emu),
    )
end

function _native_hmcode_prediction(
    params, z, k, logT_AGN, pmm_emu, pcb_emu, growth_emu, sigma_8;
    smart::Bool, feedback::Bool,
)
    cosmo = _native_hmcode_cosmology(params, sigma_8)
    h = cosmo.h
    temperature = feedback ? 10.0^logT_AGN : nothing
    z_coarse = if smart
        if feedback
            feature = Mapse.predict_baryonic_discontinuity(
                cosmo; T_AGN=temperature)
            Mapse.build_smart_coarse_grid(first(z), last(z), 24, feature)
        else
            collect(range(first(z), last(z); length=24))
        end
    else
        z
    end
    pk_mm, pk_cb = _native_linear_spectra(
        params, z_coarse, pmm_emu, pcb_emu, growth_emu)
    k_support_h = collect(Mapse.get_kgrid(pmm_emu)) ./ h

    if !smart
        return Mapse.hmcode_pmm_physical(
            cosmo, z, k, pk_mm;
            pk_cb_z=pk_cb,
            k_support=collect(Mapse.get_kgrid(pmm_emu)),
            T_AGN=temperature,
            nM=128,
            threaded=false,
        )
    elseif feedback
        result_h = Mapse.hmcode_pmm_baryonic_smart(
            cosmo, z, k ./ h, pk_mm .* h^3;
            pk_cb_coarse=pk_cb .* h^3,
            k_support=k_support_h,
            N_coarse=24,
            T_AGN=temperature,
            nM=128,
            threaded=false,
        )
    else
        result_h = Mapse.hmcode_pmm_fast(
            cosmo, z_coarse, z, k ./ h, pk_mm .* h^3;
            pk_cb_coarse=pk_cb .* h^3,
            k_support=k_support_h,
            T_AGN=nothing,
            nM=128,
            threaded=false,
        )
    end
    return result_h ./ h^3
end

@testset "Full Reactant smart HMCode pipeline" begin
    Reactant.set_default_backend("cpu")
    ext = Base.get_extension(Mapse, :MapseReactantExt)
    @test !isnothing(ext)

    sigma_k_host = collect(10.0 .^ range(-3.0, 2.0; length=300))
    sigma_power_host = reshape(sigma_k_host .^ -2.8, :, 1)
    sigma_radius_host = [1.0e-4]
    sigma_kR = Reactant.to_rarray(sigma_k_host)
    sigma_powerR = Reactant.to_rarray(sigma_power_host)
    sigma_radiusR = Reactant.to_rarray(sigma_radius_host)
    sigma_kernel(k, p, radius) = ext._hmcode_sigma_grid(k, p, radius)
    compiled_sigma = Reactant.@compile sync=true sigma_kernel(
        sigma_kR, sigma_powerR, sigma_radiusR)
    sigma_extended = compiled_sigma(sigma_kR, sigma_powerR, sigma_radiusR)
    Reactant.synchronize(sigma_extended)
    sigma_extended_host = Array(sigma_extended)[1]
    window = ext._hmcode_tophat(
        reshape(sigma_k_host, :, 1, 1) .* reshape(sigma_radius_host, 1, :, 1))
    truncated_integrand = reshape(sigma_k_host .^ 3, :, 1, 1) .*
        reshape(sigma_power_host, :, 1, 1) .* window .^ 2 ./ (2π^2)
    sigma_truncated = sqrt(ext._hmcode_trapz_dim1_3d(
        log.(sigma_k_host), truncated_integrand)[1])
    @test sigma_extended_host > 1.1sigma_truncated
    _, native_sigma = Mapse._hmcode_linear_interpolators(
        [0.0], sigma_k_host, sigma_power_host)
    @test native_sigma(first(sigma_radius_host), 0.0) > 1.1sigma_truncated

    boundsR = Reactant.to_rarray([0.0, 3.0, 10.0])
    grid_kernel(bounds) = Mapse.build_piecewise_coarse_grid(bounds, 24, 14)
    compiled_grid = Reactant.@compile sync=true grid_kernel(boundsR)
    clipped_grid = Array(compiled_grid(boundsR))
    @test length(clipped_grid) == 24
    @test all(diff(clipped_grid) .> 0.0)
    @test clipped_grid[15] ≈ 0.95 * 3.0

    artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"))
    pcb_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_cb"))
    growth_emu = Mapse.AbstractCosmologicalEmulators.trained_emulators[
        "ACE_mnuw0wacdm_ln10As_basis"
    ]
    pmm_emuR = Mapse.AbstractCosmologicalEmulators.to_reactant(pmm_emu)
    pcb_emuR = Mapse.AbstractCosmologicalEmulators.to_reactant(pcb_emu)
    growth_emuR = Mapse.AbstractCosmologicalEmulators.to_reactant(growth_emu)

    params_host = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    params_low_host = [3.044, 0.9649, 67.36, 0.0202, 0.178, 0.06, -1.0, 0.0]
    params_high_host = [3.044, 0.9649, 67.36, 0.0248, 0.082, 0.06, -1.0, 0.0]
    paramsR = Reactant.to_rarray(params_host)
    params_lowR = Reactant.to_rarray(params_low_host)
    params_highR = Reactant.to_rarray(params_high_host)
    logT_R = Reactant.to_rarray([7.8])
    logT_lowR = Reactant.to_rarray([7.6])
    logT_highR = Reactant.to_rarray([8.2])

    h = 0.6736
    Ωb = 0.02237 / h^2
    Ων = 0.06 / (93.14 * h^2)
    Ωm = Ωb + 0.12 / h^2 + Ων
    cosmo = Mapse.HMCodeCosmology(
        Ωm, Ωb, h, 0.9649, 0.8109118, -1.0, 0.0, Ων, 0.0)

    # The internal HMCode growth table is integrated from the dynamic
    # cosmology. Linear emulators use the separate growth emulator below.
    growth_kernel(p, c) = ext._hmcode_dynamic_growth_tables(p, c; na=64)
    compiled_growth = Reactant.@compile sync=true growth_kernel(
        paramsR, cosmo)
    growth_ref = compiled_growth(paramsR, cosmo)
    growth_shifted = compiled_growth(params_highR, cosmo)
    Reactant.synchronize(growth_ref)
    Reactant.synchronize(growth_shifted)
    a_grid, growth, agrowth, growth_lcdm = map(Array, growth_ref)
    _, shifted_growth, shifted_agrowth, shifted_growth_lcdm =
        map(Array, growth_shifted)
    a_static, growth_static, agrowth_static =
        ext._hmcode_growth_tables_static(cosmo)
    growth_expected = ext._hmcode_interp_vec(a_static, growth_static, a_grid)
    agrowth_expected = ext._hmcode_interp_vec(a_static, agrowth_static, a_grid)
    @test growth ≈ growth_expected rtol=5.0e-4
    @test agrowth ≈ agrowth_expected rtol=5.0e-4
    @test all(isfinite, vcat(growth, agrowth, growth_lcdm))
    @test growth != shifted_growth
    @test agrowth != shifted_agrowth
    @test growth_lcdm != shifted_growth_lcdm

    # Compile the production-size N=24 path once and reuse it across large
    # feature shifts. The growth emulator is only queried over 0 <= z <= 3.
    z_all = collect(range(0.0, 3.5; length=150))
    keep = findall(<=(3.0), z_all)
    z_host = z_all[keep]
    zR = Reactant.to_rarray(z_host)
    z_limitsR = Reactant.to_rarray([0.0, 3.0])
    k_host = collect(10.0 .^ range(-3.0, 1.0; length=128)) .* h
    kR = Reactant.to_rarray(k_host)
    camb_feedback = permutedims(readdlm(
        joinpath(@__DIR__, "data", "camb_pk_hmcode_fb.txt")))[:, keep]

    function smart_pipeline(p, z, z_limits, k, logT, pmm, pcb, growth, c)
        return Mapse.hmcode_pmm_baryonic_smart(
            p, z, z_limits, k, logT, pmm, pcb, growth, c;
            N_coarse=24, N_left=14, nM=128, growth_na=64)
    end
    compiled_smart = Reactant.@compile sync=true smart_pipeline(
        paramsR, zR, z_limitsR, kR, logT_R,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    output = compiled_smart(
        paramsR, zR, z_limitsR, kR, logT_R,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    output_low = compiled_smart(
        params_lowR, zR, z_limitsR, kR, logT_lowR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    output_high = compiled_smart(
        params_highR, zR, z_limitsR, kR, logT_highR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    Reactant.synchronize(output)
    Reactant.synchronize(output_low)
    Reactant.synchronize(output_high)
    output_host = Array(output)
    output_low_host = Array(output_low)
    output_high_host = Array(output_high)
    @test size(output_host) == size(camb_feedback)
    @test all(isfinite, output_host)
    @test all(isfinite, output_low_host)
    @test all(isfinite, output_high_host)
    @test output_host != output_low_host
    @test output_host != output_high_host
    @test output_low_host != output_high_host
    relative_error = abs.(output_host .- camb_feedback) ./
        max.(abs.(camb_feedback), eps(Float64))
    @test maximum(relative_error) < 0.0051

    # The same executable must remain accurate across the ten CAMB cosmologies
    # used by jaxmapse. Fixture rows above z=3 are outside the growth-emulator
    # contract and are deliberately excluded.
    multicosmo_cases = (
        ("fiducial_lcdm", params_host, 7.8),
        ("low_baryon_high_cdm", [3.0, 0.95, 64.0, 0.0202, 0.178, 0.1, -1.0, 0.0], 7.6),
        ("high_baryon_low_cdm", [3.1, 0.98, 72.0, 0.0248, 0.082, 0.1, -1.0, 0.0], 8.2),
        ("evolving_quintessence", [3.08, 0.99, 74.0, 0.0235, 0.105, 0.15, -0.8, -0.4], 7.9),
        ("near_early_de_boundary", [2.95, 0.93, 69.0, 0.021, 0.14, 0.02, -0.65, 0.55], 7.7),
        ("phantom_positive_wa", [3.15, 1.0, 61.0, 0.024, 0.13, 0.25, -1.4, 0.6], 8.1),
        ("phantom_negative_wa", [2.9, 0.94, 76.0, 0.022, 0.1, 0.3, -1.2, -1.0], 7.65),
        ("high_neutrino_mass", [3.05, 0.97, 68.0, 0.023, 0.125, 0.45, -0.9, 0.15], 8.0),
        ("low_h0_strong_evolution", [3.18, 0.92, 55.0, 0.0215, 0.15, 0.2, -2.2, 1.3], 7.75),
        ("high_h0_negative_wa", [2.92, 1.01, 87.0, 0.0245, 0.09, 0.35, -0.55, -1.1], 8.15),
    )
    function dmo_pipeline(p, z, z_limits, k, pmm, pcb, growth, c)
        return Mapse.hmcode_pmm_dmo_smart(
            p, z, z_limits, k, pmm, pcb, growth, c;
            N_coarse=24, N_left=14, nM=128, growth_na=64)
    end
    compiled_dmo = Reactant.@compile sync=true dmo_pipeline(
        paramsR, zR, z_limitsR, kR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    fixture_root = joinpath(@__DIR__, "data", "hmcode_multicosmo")
    for (index, (name, case_params, case_logT)) in enumerate(multicosmo_cases)
        @test case_params[7] + case_params[8] < 0.0
        fixture = readdlm(joinpath(
            fixture_root,
            "case_$(lpad(index - 1, 2, '0'))_$(name).txt",
        ); comments=true)
        dmo_reference = permutedims(fixture[:, 1:128])[:, keep]
        feedback_reference = permutedims(fixture[:, 129:256])[:, keep]
        case_paramsR = Reactant.to_rarray(case_params)
        case_logT_R = Reactant.to_rarray([case_logT])
        case_k_host = collect(10.0 .^ range(-3.0, 1.0; length=128)) .*
            case_params[3] / 100
        case_kR = Reactant.to_rarray(case_k_host)
        prediction = compiled_smart(
            case_paramsR, zR, z_limitsR, case_kR, case_logT_R,
            pmm_emuR, pcb_emuR, growth_emuR, cosmo)
        dmo_prediction = compiled_dmo(
            case_paramsR, zR, z_limitsR, case_kR,
            pmm_emuR, pcb_emuR, growth_emuR, cosmo)
        Reactant.synchronize(prediction)
        Reactant.synchronize(dmo_prediction)
        prediction_host = Array(prediction)
        dmo_prediction_host = Array(dmo_prediction)
        @test size(prediction_host) == size(feedback_reference)
        @test size(dmo_prediction_host) == size(dmo_reference)
        @test all(isfinite, prediction_host)
        @test all(isfinite, dmo_prediction_host)
        @test all(>(0), prediction_host)
        @test all(>(0), dmo_prediction_host)
        case_relative_error = abs.(prediction_host .- feedback_reference) ./
            max.(abs.(feedback_reference), eps(Float64))
        dmo_relative_error = abs.(dmo_prediction_host .- dmo_reference) ./
            max.(abs.(dmo_reference), eps(Float64))
        @test maximum(case_relative_error) < 0.011
        @test maximum(dmo_relative_error) < 0.011

        for (smart, feedback, reference) in (
            (false, false, dmo_reference),
            (true, false, dmo_reference),
            (false, true, feedback_reference),
            (true, true, feedback_reference),
        )
            native_prediction = _native_hmcode_prediction(
                case_params, z_host, case_k_host, case_logT,
                pmm_emu, pcb_emu, growth_emu, cosmo.sigma_8;
                smart=smart, feedback=feedback)
            @test size(native_prediction) == size(reference)
            @test all(isfinite, native_prediction)
            @test all(>(0), native_prediction)
            native_relative_error = abs.(native_prediction .- reference) ./
                max.(abs.(reference), eps(Float64))
            @test maximum(native_relative_error) < 0.011
        end
    end

    # Reverse mode must compile through the complete small smart pipeline.
    z_gradientR = Reactant.to_rarray(collect(range(0.0, 3.0; length=5)))
    k_gradientR = Reactant.to_rarray(collect(Mapse.get_kgrid(pmm_emu))[1:100:end])
    function smart_loss(p, logT, z, z_limits, k, pmm, pcb, growth, c)
        prediction = Mapse.hmcode_pmm_baryonic_smart(
            p, z, z_limits, k, logT, pmm, pcb, growth, c;
            N_coarse=10, N_left=5, nM=4, growth_na=8)
        return sum(log.(prediction)) / length(prediction)
    end
    function smart_gradient(p, logT, z, z_limits, k, pmm, pcb, growth, c)
        derivatives = Enzyme.gradient(
            Reverse,
            smart_loss,
            p,
            logT,
            Const(z),
            Const(z_limits),
            Const(k),
            Const(pmm),
            Const(pcb),
            Const(growth),
            Const(c),
        )
        return derivatives[1], derivatives[2]
    end
    compiled_gradient = Reactant.@compile sync=true smart_gradient(
        paramsR, logT_R, z_gradientR, z_limitsR, k_gradientR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    params_gradient, temperature_gradient = compiled_gradient(
        paramsR, logT_R, z_gradientR, z_limitsR, k_gradientR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    shifted_params_gradient, shifted_temperature_gradient = compiled_gradient(
        params_highR, logT_highR, z_gradientR, z_limitsR, k_gradientR,
        pmm_emuR, pcb_emuR, growth_emuR, cosmo)
    Reactant.synchronize(params_gradient)
    Reactant.synchronize(temperature_gradient)
    Reactant.synchronize(shifted_params_gradient)
    Reactant.synchronize(shifted_temperature_gradient)
    params_gradient_host = Array(params_gradient)
    temperature_gradient_host = Array(temperature_gradient)
    shifted_params_gradient_host = Array(shifted_params_gradient)
    shifted_temperature_gradient_host = Array(shifted_temperature_gradient)
    @test all(isfinite, params_gradient_host)
    @test all(isfinite, temperature_gradient_host)
    @test all(isfinite, shifted_params_gradient_host)
    @test all(isfinite, shifted_temperature_gradient_host)
    @test any(!=(0), params_gradient_host)
    @test any(!=(0), temperature_gradient_host)
    @test params_gradient_host != shifted_params_gradient_host
    @test temperature_gradient_host != shifted_temperature_gradient_host
end
