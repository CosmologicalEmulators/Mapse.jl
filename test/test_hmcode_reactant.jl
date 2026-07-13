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
end
