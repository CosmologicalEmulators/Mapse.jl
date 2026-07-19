using Test
using DelimitedFiles
using Reactant
using Mapse

@testset "HMCode Reactant support" begin
    Reactant.set_default_backend("cpu")

    ext = Base.get_extension(Mapse, :MapseReactantExt)
    @test !isnothing(ext)
    @test !occursin("@allowscalar", read(joinpath(pkgdir(Mapse), "ext", "MapseReactantExt.jl"), String))

    hmcode_reference = readdlm(joinpath(@__DIR__, "data", "hmcode_camb_reference.txt"), comments=true)
    z_all = unique(hmcode_reference[:, 1])
    k_support = collect(hmcode_reference[hmcode_reference[:, 1] .== z_all[1], 2])
    z = collect(z_all[1:1])
    k_out = k_support[1:8:end]

    pk_mm_all = reshape(hmcode_reference[:, 3], length(k_support), length(z_all))
    pk_cb_all = reshape(hmcode_reference[:, 4], length(k_support), length(z_all))
    pk_mm = Array(pk_mm_all[:, 1:1])
    pk_cb = Array(pk_cb_all[:, 1:1])

    h = 0.6736
    Ωb = 0.02237 / h^2
    Ων = 0.06 / (93.14 * h^2)
    Ωm = Ωb + 0.12 / h^2 + Ων
    cosmo = Mapse.HMCodeCosmology(Ωm, Ωb, h, 0.9649, 0.8109118,
                                  -1.0, 0.0, Ων, 0.0)

    native_pmm = Mapse.hmcode_Pmm(cosmo, z, k_out, k_support, pk_mm;
                                  pk_cb_support_z=pk_cb, nM=32,
                                  threaded=false)
    native_boost = Mapse.hmcode_boost(cosmo, z, k_out, k_support, pk_mm;
                                      pk_cb_support_z=pk_cb, nM=32,
                                      threaded=false)

    zR = Reactant.to_rarray(z)
    k_outR = Reactant.to_rarray(k_out)
    k_supportR = Reactant.to_rarray(k_support)
    pk_mmR = Reactant.to_rarray(pk_mm)
    pk_cbR = Reactant.to_rarray(pk_cb)

    @test_throws ArgumentError Mapse.hmcode_Pmm(cosmo, zR, k_outR, k_supportR, pk_mmR;
                                                 pk_cb_z=pk_cbR,
                                                 pk_cb_support_z=pk_cbR, nM=32)
    @test_throws ArgumentError Mapse.hmcode_Pmm(cosmo, zR, k_outR, k_supportR,
                                                 pk_mmR, pk_cbR;
                                                 pk_cb_z=pk_cbR, nM=32)
    @test_throws ArgumentError Mapse.hmcode_boost(cosmo, zR, k_outR, k_supportR, pk_mmR;
                                                   pk_cb_z=pk_cbR,
                                                   pk_cb_support_z=pk_cbR, nM=32)

    compiled = Reactant.@compile sync=true Mapse.hmcode_Pmm(
        cosmo, zR, k_outR, k_supportR, pk_mmR, pk_cbR; nM=32
    )
    pmmR = compiled(cosmo, zR, k_outR, k_supportR, pk_mmR, pk_cbR)
    Reactant.synchronize(pmmR)
    pmm = Array(pmmR)
    @test size(pmm) == size(native_pmm)
    @test all(isfinite, pmm)
    @test pmm ≈ native_pmm rtol=3e-3

    pmm_fullR = Mapse.hmcode_Pmm(cosmo, zR, k_supportR, k_supportR, pk_mmR; pk_cb_z=pk_cbR, nM=32)
    Reactant.synchronize(pmm_fullR)
    @test pmm ≈ Array(pmm_fullR)[1:8:end, :] rtol=1e-12

    k_out_irr = copy(k_out)
    k_out_irr[5] = (k_out[4] + k_out[5]) / 2
    k_out_irrR = Reactant.to_rarray(k_out_irr)
    pmm_irrR = Mapse.hmcode_Pmm(cosmo, zR, k_out_irrR, k_supportR, pk_mmR; pk_cb_z=pk_cbR, nM=32)
    Reactant.synchronize(pmm_irrR)
    pmm_irr = Array(pmm_irrR)
    @test size(pmm_irr) == (length(k_out_irr), length(z))
    @test all(isfinite, pmm_irr)

    native_pmm_irr = Mapse.hmcode_Pmm(cosmo, z, k_out_irr, k_support, pk_mm;
                                      pk_cb_support_z=pk_cb, nM=32, threaded=false)
    @test pmm_irr ≈ native_pmm_irr rtol=3e-3


    boostR = Mapse.hmcode_boost(cosmo, zR, k_outR, k_supportR, pk_mmR, pk_cbR;
                                nM=32)
    Reactant.synchronize(boostR)
    boost = Array(boostR)
    @test size(boost) == size(native_boost)
    @test all(isfinite, boost)
    @test boost ≈ native_boost rtol=3e-3

    curved_reference = readdlm(joinpath(@__DIR__, "data", "hmcode_curved_parity_reference.txt"), comments=true)
    curved_z = unique(curved_reference[:, 1])
    curved_k = collect(curved_reference[curved_reference[:, 1] .== curved_z[1], 2])
    curved_pmm = reshape(curved_reference[:, 3], length(curved_k), length(curved_z))
    curved_pcb = reshape(curved_reference[:, 4], length(curved_k), length(curved_z))
    curved_dmo = reshape(curved_reference[:, 5], length(curved_k), length(curved_z))
    curved_feedback = reshape(curved_reference[:, 6], length(curved_k), length(curved_z))
    curved_cosmo = Mapse.HMCodeCosmology(Ωm, Ωb, h, 0.9649, 0.8109118,
                                         -0.9, 0.2, Ων, 0.01)

    curved_zR = Reactant.to_rarray(collect(curved_z))
    curved_kR = Reactant.to_rarray(curved_k)
    curved_pmmR = Reactant.to_rarray(curved_pmm)
    curved_pcbR = Reactant.to_rarray(curved_pcb)
    curved_dmoR = Mapse.hmcode_Pmm(curved_cosmo, curved_zR, curved_kR, curved_pmmR;
                                   pk_cb_z=curved_pcbR, T_AGN=nothing, nM=32)
    curved_feedbackR = Mapse.hmcode_Pmm(curved_cosmo, curved_zR, curved_kR, curved_pmmR;
                                        pk_cb_z=curved_pcbR, T_AGN=10.0^7.8, nM=32)
    Reactant.synchronize(curved_dmoR)
    Reactant.synchronize(curved_feedbackR)
    # Native HMCode uses adaptive root finding/integration; Reactant uses the
    # fixed compiled grid algorithm. This curved w0waCDM case needs 1.5%.
    @test Array(curved_dmoR) ≈ curved_dmo rtol=1.5e-2
    @test Array(curved_feedbackR) ≈ curved_feedback rtol=1.5e-2
    # Fast path numerical parity tests
    z_coarse_arr = collect(range(0.0, 3.0, length=5))
    z_fine_arr = collect(range(0.0, 3.0, length=20))
    z_coarseR_arr = Reactant.to_rarray(z_coarse_arr)
    z_fineR_arr = Reactant.to_rarray(z_fine_arr)
    # create synthetic pk_mm with correct dimensions (128, 5)
    pk_mm_synth = repeat(pk_mm_all[:, 1], 1, 5)
    pk_cb_synth = repeat(pk_cb_all[:, 1], 1, 5)
    pk_mm_allR = Reactant.to_rarray(pk_mm_synth)
    pk_cb_allR = Reactant.to_rarray(pk_cb_synth)

    # Native fast DMO
    native_fast_dmo = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, z_fine_arr, k_support, pk_mm_synth; pk_cb_coarse=pk_cb_synth, T_AGN=nothing, nM=32, threaded=false)
    # Reactant fast DMO
    reactant_fast_dmo = Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)
    Reactant.synchronize(reactant_fast_dmo)

    @test size(Array(reactant_fast_dmo)) == size(native_fast_dmo)
    @test Array(reactant_fast_dmo) ≈ native_fast_dmo rtol=1.5e-2

    # Native fast Baryonic
    native_fast_bar = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, z_fine_arr, k_support, pk_mm_synth; pk_cb_coarse=pk_cb_synth, T_AGN=10^7.8, nM=32, threaded=false)
    # Reactant fast Baryonic
    reactant_fast_bar = Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=10^7.8, nM=32)
    Reactant.synchronize(reactant_fast_bar)

    @test size(Array(reactant_fast_bar)) == size(native_fast_bar)
    @test Array(reactant_fast_bar) ≈ native_fast_bar rtol=1.5e-2

    # Compiled vector check (DMO)
    f_vec_dmo(c, zc, zf, k, pm, pc) = Mapse.hmcode_pmm_fast(c, zc, zf, k, pm; pk_cb_coarse=pc, T_AGN=nothing, nM=32)
    compiled_vec_dmo = Reactant.@compile f_vec_dmo(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR, pk_cb_allR)
    out_vec_dmo = compiled_vec_dmo(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR, pk_cb_allR)
    Reactant.synchronize(out_vec_dmo)
    @test size(Array(out_vec_dmo)) == size(native_fast_dmo)
    @test Array(out_vec_dmo) ≈ native_fast_dmo rtol=1.5e-2

    # Compiled vector check (Feedback)
    f_vec_bar(c, zc, zf, k, pm, pc) = Mapse.hmcode_pmm_fast(c, zc, zf, k, pm; pk_cb_coarse=pc, T_AGN=10^7.8, nM=32)
    compiled_vec_bar = Reactant.@compile f_vec_bar(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR, pk_cb_allR)
    out_vec_bar = compiled_vec_bar(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, pk_mm_allR, pk_cb_allR)
    Reactant.synchronize(out_vec_bar)
    @test all(isfinite, Array(out_vec_bar))
    @test size(Array(out_vec_bar)) == size(native_fast_bar)
    @test Array(out_vec_bar) ≈ native_fast_bar rtol=1.5e-2

    # Compiled scalar check (DMO)
    f_scalar_dmo(c, zc, zf, k, pm, pc) = Mapse.hmcode_pmm_fast(c, zc, zf, k, pm; pk_cb_coarse=pc, T_AGN=nothing, nM=32)
    compiled_scalar_dmo = Reactant.@compile f_scalar_dmo(cosmo, z_coarseR_arr, 1.5, k_supportR, pk_mm_allR, pk_cb_allR)
    out_scalar_dmo = compiled_scalar_dmo(cosmo, z_coarseR_arr, 1.5, k_supportR, pk_mm_allR, pk_cb_allR)
    Reactant.synchronize(out_scalar_dmo)
    native_scalar_dmo = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, 1.5, k_support, pk_mm_synth; pk_cb_coarse=pk_cb_synth, T_AGN=nothing, nM=32, threaded=false)
    @test size(Array(out_scalar_dmo)) == size(native_scalar_dmo)
    @test Array(out_scalar_dmo) ≈ native_scalar_dmo rtol=1.5e-2

    # Compiled scalar check (Feedback)
    f_scalar_bar(c, zc, zf, k, pm, pc) = Mapse.hmcode_pmm_fast(c, zc, zf, k, pm; pk_cb_coarse=pc, T_AGN=10^7.8, nM=32)
    compiled_scalar_bar = Reactant.@compile f_scalar_bar(cosmo, z_coarseR_arr, 1.5, k_supportR, pk_mm_allR, pk_cb_allR)
    out_scalar_bar = compiled_scalar_bar(cosmo, z_coarseR_arr, 1.5, k_supportR, pk_mm_allR, pk_cb_allR)
    Reactant.synchronize(out_scalar_bar)
    @test all(isfinite, Array(out_scalar_bar))
    native_scalar_bar = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, 1.5, k_support, pk_mm_synth; pk_cb_coarse=pk_cb_synth, T_AGN=10^7.8, nM=32, threaded=false)
    @test size(Array(out_scalar_bar)) == size(native_scalar_bar)
    @test Array(out_scalar_bar) ≈ native_scalar_bar rtol=1.5e-2

    # Compiled positional support-grid check (DMO)
    f_pos_dmo(c, zc, zf, k_out, k_sup, pm_sup) = Mapse.hmcode_pmm_fast(c, zc, zf, k_out, k_sup, pm_sup; T_AGN=nothing, nM=32)
    compiled_pos_dmo = Reactant.@compile f_pos_dmo(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, k_supportR, pk_mm_allR)
    out_pos_dmo = compiled_pos_dmo(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, k_supportR, pk_mm_allR)
    Reactant.synchronize(out_pos_dmo)
    native_pos_dmo = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, z_fine_arr, k_support, k_support, pk_mm_synth; T_AGN=nothing, nM=32, threaded=false)
    @test size(Array(out_pos_dmo)) == size(native_pos_dmo)
    @test Array(out_pos_dmo) ≈ native_pos_dmo rtol=1.5e-2

    # Compiled positional support-grid check (Feedback, with positional pk_cb_coarse)
    f_pos_bar(c, zc, zf, k_out, k_sup, pm_sup, pc_sup) = Mapse.hmcode_pmm_fast(c, zc, zf, k_out, k_sup, pm_sup, pc_sup; T_AGN=10^7.8, nM=32)
    compiled_pos_bar = Reactant.@compile f_pos_bar(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, k_supportR, pk_mm_allR, pk_cb_allR)
    out_pos_bar = compiled_pos_bar(cosmo, z_coarseR_arr, z_fineR_arr, k_supportR, k_supportR, pk_mm_allR, pk_cb_allR)
    Reactant.synchronize(out_pos_bar)
    @test all(isfinite, Array(out_pos_bar))
    native_pos_bar = Mapse.hmcode_pmm_fast(cosmo, z_coarse_arr, z_fine_arr, k_support, k_support, pk_mm_synth, pk_cb_synth; T_AGN=10^7.8, nM=32, threaded=false)
    @test size(Array(out_pos_bar)) == size(native_pos_bar)
    @test Array(out_pos_bar) ≈ native_pos_bar rtol=1.5e-2

    # Fast path Boundary error checks
    z_coarse_short = Reactant.to_rarray(collect(range(0.0, 3.0, length=4)))
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarse_short, z_fineR_arr, k_supportR, Reactant.to_rarray(ones(length(k_support), 4)))

    z_coarse_unordered = Reactant.to_rarray([0.0, 1.0, 0.5, 2.0, 3.0])
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarse_unordered, z_fineR_arr, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    z_fine_out_of_bounds = Reactant.to_rarray(collect(range(0.0, 4.0, length=20)))
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, z_fine_out_of_bounds, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    z_fine_out_of_bounds_lower = Reactant.to_rarray([-0.5, 0.5, 1.0])
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, z_fine_out_of_bounds_lower, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    z_fine_unordered = Reactant.to_rarray([0.0, 1.0, 0.5, 2.0, 3.0])
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, z_fine_unordered, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, 4.0, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    @test_throws ArgumentError Mapse.hmcode_pmm_fast(cosmo, z_coarseR_arr, -0.5, k_supportR, pk_mm_allR; pk_cb_coarse=pk_cb_allR, T_AGN=nothing, nM=32)

    # UCF-3: Patched-CLASS low-k Reactant coverage
    class_reference = readdlm(joinpath(@__DIR__, "data", "hmcode_class_feedback_reference.txt"), comments=true)
    z_class = unique(class_reference[:, 1])
    k_class = class_reference[class_reference[:, 1] .== z_class[1], 2]
    pk_mm_class = zeros(length(k_class), length(z_class))
    pk_cb_class = zeros(length(k_class), length(z_class))
    for iz in 1:length(z_class)
        mask = class_reference[:, 1] .== z_class[iz]
        pk_mm_class[:, iz] = class_reference[mask, 3]
        pk_cb_class[:, iz] = class_reference[mask, 4]
    end
    cosmo_class = Mapse.HMCodeCosmology(
        0.315192, 0.04930, 0.6736, 0.9649, 0.8109118, -1.0, 0.0, 0.001422, 0.0
    )
    z_classR = Reactant.to_rarray(z_class)
    k_classR = Reactant.to_rarray(k_class)
    pk_mm_classR = Reactant.to_rarray(pk_mm_class)
    pk_cb_classR = Reactant.to_rarray(pk_cb_class)

    compiled_class_fb = Reactant.@compile Mapse.hmcode_Pmm(cosmo_class, z_classR, k_classR, pk_mm_classR, pk_cb_classR; T_AGN=10.0^7.8, nM=32)
    out_class_fb = compiled_class_fb(cosmo_class, z_classR, k_classR, pk_mm_classR, pk_cb_classR)
    Reactant.synchronize(out_class_fb)

    compiled_class_dmo = Reactant.@compile Mapse.hmcode_Pmm(cosmo_class, z_classR, k_classR, pk_mm_classR, pk_cb_classR; T_AGN=nothing, nM=32)
    out_class_dmo = compiled_class_dmo(cosmo_class, z_classR, k_classR, pk_mm_classR, pk_cb_classR)
    Reactant.synchronize(out_class_dmo)

    out_class_fb_arr = Array(out_class_fb)
    out_class_dmo_arr = Array(out_class_dmo)

    mask_low_k_class = k_class .<= 3e-4
    boost_class = out_class_fb_arr ./ pk_mm_class
    dmo_boost_class = out_class_dmo_arr ./ pk_mm_class

    @test boost_class[mask_low_k_class, :] ≈ dmo_boost_class[mask_low_k_class, :] rtol=1e-3
    @test boost_class[mask_low_k_class, :] ≈ ones(count(mask_low_k_class), length(z_class)) rtol=1e-3

end
