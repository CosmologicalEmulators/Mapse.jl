@testset "Smart HMCode vs CAMB Benchmarks" begin
    # exact cosmology
    local H = 0.6736
    local ΩB = 0.02237 / H^2
    local Ων = 0.06 / (93.14 * H^2)
    local ΩM = ΩB + 0.12 / H^2 + Ων
    local NS = 0.9649
    local SIGMA8 = 0.8109118
    local T_AGN = 10.0^7.8
    local NM = 128
    local PARAMS = [3.044, NS, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    local COSMO = Mapse.HMCodeCosmology(ΩM, ΩB, H, NS, SIGMA8, -1.0, 0.0, Ων, 0.0)
    local K_LABEL = collect(10.0 .^ range(-3.0, 1.0; length=128))
    local K_OUT = K_LABEL .* H
    local Z_FINE = collect(range(0.0, 3.5; length=150))
    local N_COARSE = (24, 32, 40)

    # load fixtures
    camb_dmo = readdlm(joinpath(@__DIR__, "data", "camb_pk_hmcode_dmo.txt"))
    camb_fb = readdlm(joinpath(@__DIR__, "data", "camb_pk_hmcode_fb.txt"))
    ref_dmo = permutedims(camb_dmo)
    ref_fb = permutedims(camb_fb)

    # load emulators
    artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"))
    pcb_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_cb"))
    growth_emu = Mapse.AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_ln10As_basis"]
    k_support = collect(Mapse.get_kgrid(pmm_emu))

    function growth_at(z, params, growth_emu)
        input = vcat(reshape(z, 1, :), repeat(reshape(params, :, 1), 1, length(z)))
        return vec(Mapse.AbstractCosmologicalEmulators.run_emulator(input, growth_emu)[6:6, :])
    end

    # smart DMO pipeline vs CAMB
    for n in N_COARSE
        z_feature = Mapse.predict_baryonic_discontinuity(COSMO; T_AGN=10.0^7.8)
        zc = Mapse.build_smart_coarse_grid(first(Z_FINE), last(Z_FINE), n, z_feature)
        growth = growth_at(zc, PARAMS, growth_emu)
        pmm = Mapse.get_Pk(PARAMS, zc, growth, pmm_emu)
        pcb = Mapse.get_Pk(PARAMS, zc, growth, pcb_emu)
        h = COSMO.h
        result_h = Mapse.hmcode_pmm_fast(COSMO, zc, Z_FINE, K_OUT ./ h, pmm .* h^3;
            pk_cb_coarse=pcb .* h^3, k_support=k_support ./ h, T_AGN=nothing, nM=NM)
        smart_dmo = result_h ./ h^3

        δ = abs.(smart_dmo .- ref_dmo)
        rel = δ ./ max.(abs.(ref_dmo), eps(Float64))
        max_err = maximum(rel)
        @test max_err < 0.0051 # Ensure max relative diff is below 0.51%
    end

    # smart feedback pipeline vs CAMB
    for n in N_COARSE
        z_feature = Mapse.predict_baryonic_discontinuity(COSMO; T_AGN=T_AGN)
        zc = Mapse.build_smart_coarse_grid(first(Z_FINE), last(Z_FINE), n, z_feature)
        growth = growth_at(zc, PARAMS, growth_emu)
        pmm = Mapse.get_Pk(PARAMS, zc, growth, pmm_emu)
        pcb = Mapse.get_Pk(PARAMS, zc, growth, pcb_emu)
        h = COSMO.h
        result_h = Mapse.hmcode_pmm_baryonic_smart(
            COSMO, Z_FINE, K_OUT ./ h, pmm .* h^3;
            pk_cb_coarse=pcb .* h^3, k_support=k_support ./ h,
            N_coarse=n, T_AGN=T_AGN, nM=NM)
        smart_fb = result_h ./ h^3

        δ = abs.(smart_fb .- ref_fb)
        rel = δ ./ max.(abs.(ref_fb), eps(Float64))
        max_err = maximum(rel)
        @test max_err < 0.0051 # Ensure max relative diff is below 0.51%
    end
end
