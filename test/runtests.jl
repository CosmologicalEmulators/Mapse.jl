using Test
using NPZ
using DelimitedFiles
using SimpleChains
using Static
using Mapse
using DataInterpolations

mlpd = SimpleChain(
  static(6),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(tanh, 64),
  TurboDense(identity, 40)
)

k_test = Array(LinRange(0,200, 40))
z_test = Array(LinRange(0, 5, 40))
weights = SimpleChains.init_params(mlpd)
inminmax = rand(6,2)
outminmax = rand(40,2)
a, Ωcb0, mν, h, w0, wa = [1., 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0., 3., 100))

emu = Mapse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

postprocessing = (input, output, D, Pkemu) -> output
preprocessing = (input) -> input

effort_emu = Mapse.TransferFunctionEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing)

x = [Ωcb0, h, mν, w0, wa]

n = 64
x1 = vcat([0.], sort(rand(n-2)), [1.])
x2 = 2 .* vcat([0.], sort(rand(n-2)), [1.])
y = rand(n)

function D_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

function f_z_x(z, x)
    Ωcb0, h, mν, w0, wa = x
    sum(Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
end

myx = Array(LinRange(0., 1., 100))
monotest = sin.(myx)
quadtest = 0.5.*cos.(myx)
hexatest = 0.1.*cos.(2 .* myx)
q_par = 1.4
q_perp = 0.6

x3 = Array(LinRange(-1., 1., 100))

@testset "Mapse tests" begin
    @test isapprox(Mapse.E_a(1, Ωcb0, h), 1.)
    D, f = Mapse.D_f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa)
    @test isapprox(D, Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox(f, Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa))
    @test isapprox([Mapse.f_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse.f_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    @test isapprox([Mapse.D_z(myz, Ωcb0, h; mν =mν, w0=w0, wa=wa) for myz in z],  Mapse.D_z(z, Ωcb0, h; mν =mν, w0=w0, wa=wa), rtol=1e-10)
    mycosmo = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.636, ωb=0.02237, ωc = 0.1, mν=0.06, w0=-2., wa=1.)
    mycosmo_ref = Mapse.w0waCDMCosmology(ln10Aₛ=3., nₛ=0.96, h=0.6736, ωb=0.02237, ωc = 0.12, mν=0.06, w0=-1., wa=0.)

    # Test get_Pk
    pk_scalar = Mapse.get_Pk(x, 1.0, 1.0, effort_emu)
    @test size(pk_scalar) == (40,)

    pk_vector = Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], effort_emu)
    @test size(pk_vector) == (40, 2)

    # Test PCA reconstruction used by compressed MAPSE artifacts.
    compression = Mapse.PCACompression(
        mean = [10.0, 20.0, 30.0],
        basis = [
            1.0 0.0
            0.0 1.0
            1.0 1.0
        ]
    )
    @test Mapse.reconstruct([2.0, 3.0], compression) ≈ [12.0, 23.0, 35.0]
    @test Mapse.reconstruct([2.0 3.0; 4.0 5.0], compression) ≈ [12.0 13.0; 24.0 25.0; 36.0 38.0]

    @test Mapse.preprocessing_drop_primordial_parameters(collect(1:8)) == collect(3:8)
    @test Mapse.BUILTIN_PREPROCESSING["drop_primordial_parameters"] === Mapse.preprocessing_drop_primordial_parameters
    @test Mapse.BUILTIN_POSTPROCESSING["lcdm_transfer_ratio"] === Mapse.postprocessing_lcdm_transfer_ratio
    @test Mapse.DEFAULT_EMULATOR_NAME == "mnuw0wacdm_class"
    @test Mapse.DEFAULT_EMULATOR_ARTIFACT == "mnuw0wacdm_class"

    # Artifact-based component loading tests
    artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    # Stricter load_emulator validation tests
    @test_throws ArgumentError Mapse.load_emulator(artifact_root)
    @test_throws ArgumentError Mapse.load_emulator(tempname())

    artifact_pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"))
    artifact_pcb_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_cb"))

    artifact_params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    artifact_pk_mm = Mapse.get_Pk(artifact_params, 0.0, 1.0, artifact_pmm_emu)
    @test size(artifact_pk_mm) == (length(Mapse.get_kgrid(artifact_pmm_emu)),)
    @test all(isfinite, artifact_pk_mm)

    artifact_pk_cb = Mapse.get_Pk(artifact_params, 0.0, 1.0, artifact_pcb_emu)
    @test size(artifact_pk_cb) == (length(Mapse.get_kgrid(artifact_pcb_emu)),)
    @test all(isfinite, artifact_pk_cb)

    # Halofit on linear emulator output
    artifact_k_lin = Mapse.get_kgrid(artifact_pmm_emu)
    artifact_halofit_pk = Mapse.halofit_pmm(artifact_params, 0.0, artifact_k_lin, artifact_pk_mm)
    @test size(artifact_halofit_pk) == (length(artifact_k_lin),)
    @test all(isfinite, artifact_halofit_pk)
    @test all(>(0), artifact_halofit_pk)

    # Vector redshift workflow
    artifact_z_vec = [0.0, 0.5]
    artifact_D_vec = [1.0, 0.75]
    artifact_pk_vec = Mapse.get_Pk(artifact_params, artifact_z_vec, artifact_D_vec, artifact_pmm_emu)
    @test size(artifact_pk_vec) == (length(artifact_k_lin), length(artifact_z_vec))
    @test all(isfinite, artifact_pk_vec)

    artifacts_toml = read(Mapse.ARTIFACTS_TOML, String)
    @test occursin("git-tree-sha1 = \"38a05969d61632358bf4981957f397e45f88107f\"", artifacts_toml)
    @test occursin("zenodo.org/records/21328528", artifacts_toml)

    primitive_output = ones(3)
    primitive_emu = Mapse.TransferFunctionEmulator(TrainedEmulator = emu, kgrid=[0.1, 0.2, 0.3],
                                        InMinMax = inminmax, OutMinMax = outminmax,
                                        Preprocessing = preprocessing,
                                        Postprocessing = postprocessing)
    primitive_params = [3.0, 0.96, 67.0, 0.0224, 0.12, 0.06, -1.0, 0.0]
    scalar_post = Mapse.postprocessing_lcdm_transfer_ratio(primitive_params, primitive_output, 2.0, primitive_emu)
    vector_post = Mapse.postprocessing_lcdm_transfer_ratio(primitive_params, repeat(primitive_output, 1, 2), [2.0, 3.0], primitive_emu)
    @test size(scalar_post) == (3,)
    @test size(vector_post) == (3, 2)
    @test vector_post[:, 1] ≈ scalar_post
    @test vector_post[:, 2] ≈ scalar_post .* (3.0 / 2.0)^2

    halofit_params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    halofit_cosmo = @inferred Mapse.halofit_cosmology(halofit_params)
    @test halofit_cosmo.h ≈ 0.6736
    @test halofit_cosmo.Ωm0 ≈ (0.02237 + 0.12 + 0.06 / 93.14) / 0.6736^2

    halofit_k = exp.(range(log(1e-4), log(40.0), length=101))
    halofit_z = [0.0, 1.0]
    halofit_pk_lin = [1e4 * (k / 0.1)^0.96 * exp(-k / 5) / (1 + z)^2
                      for k in halofit_k, z in halofit_z]
    halofit_Ωm_z, halofit_Ωv_z = @inferred Mapse.halofit_background(halofit_cosmo, halofit_z)
    @test size(halofit_Ωm_z) == size(halofit_z)
    @test size(halofit_Ωv_z) == size(halofit_z)
    @test Mapse.halofit_background(halofit_cosmo, halofit_z[1]) == (halofit_Ωm_z[1], halofit_Ωv_z[1])

    halofit_pk_nl = @inferred Mapse.halofit_pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin)
    halofit_pk_nl_unchecked = @inferred Mapse._halofit_Pmm_unchecked(halofit_cosmo,
                                                                     halofit_z,
                                                                     halofit_k,
                                                                     halofit_pk_lin,
                                                                     halofit_Ωm_z,
                                                                     halofit_Ωv_z)
    halofit_pk_nl_one_z = @inferred Mapse._halofit_Pmm_one_z_unchecked(halofit_cosmo,
                                                                       halofit_z[1],
                                                                       halofit_k,
                                                                       view(halofit_pk_lin, :, 1),
                                                                       halofit_Ωm_z[1],
                                                                       halofit_Ωv_z[1])
    halofit_pk_nl_external_bg = @inferred Mapse.halofit_pmm(halofit_cosmo, halofit_z,
                                                           halofit_k, halofit_pk_lin,
                                                           halofit_Ωm_z, halofit_Ωv_z)
    @test size(halofit_pk_nl) == size(halofit_pk_lin)
    @test halofit_pk_nl_unchecked ≈ halofit_pk_nl
    @test halofit_pk_nl_one_z ≈ halofit_pk_nl[:, 1]
    @test halofit_pk_nl_external_bg ≈ halofit_pk_nl
    @test all(isfinite, halofit_pk_nl)
    @test all(>(0), halofit_pk_nl)
    @test (@inferred Mapse.halofit_pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1])) ≈ halofit_pk_nl[:, 1]
    @test Mapse.halofit_pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1],
                            halofit_Ωm_z[1], halofit_Ωv_z[1]) ≈ halofit_pk_nl[:, 1]
    @test size(@inferred(Mapse.halofit_pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin))) == size(halofit_pk_lin)
    @test Mapse.halofit_pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin,
                            halofit_Ωm_z, halofit_Ωv_z) ≈ halofit_pk_nl
    @test_throws ArgumentError Mapse.halofit_pmm(halofit_cosmo, halofit_z, reverse(halofit_k), halofit_pk_lin)
    @test_throws DimensionMismatch Mapse.halofit_pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin[:, 1:1])
    @test_throws DimensionMismatch Mapse.halofit_pmm(halofit_cosmo, halofit_z, halofit_k,
                                                     halofit_pk_lin, halofit_Ωm_z[1:1],
                                                     halofit_Ωv_z)

    class_reference = readdlm(joinpath(@__DIR__, "data", "halofit_class_reference.txt"), comments=true)
    class_z = unique(class_reference[:, 1])
    class_k = class_reference[class_reference[:, 1] .== class_z[0+1], 2]
    class_pk_lin = reshape(class_reference[:, 3], length(class_k), length(class_z))
    class_Ωm_z = [class_reference[findfirst(==(z), class_reference[:, 1]), 4] for z in class_z]
    class_Ωv_z = [class_reference[findfirst(==(z), class_reference[:, 1]), 5] for z in class_z]
    class_pk_nl = reshape(class_reference[:, 6], length(class_k), length(class_z))
    class_params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    mapse_class_pk_nl = Mapse.halofit_pmm(class_params, class_z, class_k, class_pk_lin,
                                          class_Ωm_z, class_Ωv_z)
    @test mapse_class_pk_nl ≈ class_pk_nl rtol=6e-3

    hmcode_reference = readdlm(joinpath(@__DIR__, "data", "hmcode_camb_reference.txt"), comments=true)
    hmcode_z = unique(hmcode_reference[:, 1])
    hmcode_k = hmcode_reference[hmcode_reference[:, 1] .== hmcode_z[1], 2]
    hmcode_pk_mm = reshape(hmcode_reference[:, 3], length(hmcode_k), length(hmcode_z))
    hmcode_pk_cb = reshape(hmcode_reference[:, 4], length(hmcode_k), length(hmcode_z))
    hmcode_boost_ref = reshape(hmcode_reference[:, 6], length(hmcode_k), length(hmcode_z))
    hmcode_h = 0.6736
    hmcode_Ωb = 0.02237 / hmcode_h^2
    hmcode_Ων = 0.06 / (93.14 * hmcode_h^2)
    hmcode_Ωm = hmcode_Ωb + 0.12 / hmcode_h^2 + hmcode_Ων
    hmcode_cosmo = Mapse.HMCodeCosmology(hmcode_Ωm, hmcode_Ωb, hmcode_h, 0.9649,
                                         0.8109118, -1.0, 0.0, hmcode_Ων, 0.0)
    hmcode_boost = Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                                      hmcode_pk_cb;
                                      nM=256, threaded=false)
    @test size(hmcode_boost) == size(hmcode_pk_mm)
    @test all(isfinite, hmcode_boost)
    @test hmcode_boost ≈ hmcode_boost_ref rtol=3e-3
    @test Mapse.hmcode_pmm(hmcode_cosmo, 0.0, hmcode_k, hmcode_pk_mm[:, 1],
                           hmcode_pk_cb[:, 1]; nM=64, threaded=false) ./ hmcode_pk_mm[:, 1] ≈
          Mapse.hmcode_boost(hmcode_cosmo, 0.0, hmcode_k, hmcode_pk_mm[:, 1],
                             hmcode_pk_cb[:, 1];
                             nM=64, threaded=false)
    @test Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm;
                             pk_cb_z=hmcode_pk_cb, nM=64, threaded=false) ≈
          Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; nM=64, threaded=false)
    @test Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_k,
                             hmcode_pk_mm; pk_cb_support_z=hmcode_pk_cb,
                             nM=64, threaded=false) ≈
          Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; nM=64, threaded=false)
    hmcode_kout = hmcode_k[1:2:end]
    hmcode_pk_nl_kout = Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z, hmcode_kout,
                                         hmcode_k, hmcode_pk_mm;
                                         pk_cb_support_z=hmcode_pk_cb,
                                         nM=64, threaded=false)
    hmcode_boost_kout = Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_kout,
                                           hmcode_k, hmcode_pk_mm;
                                           pk_cb_support_z=hmcode_pk_cb,
                                           nM=64, threaded=false)
    @test size(hmcode_pk_nl_kout) == (length(hmcode_kout), length(hmcode_z))
    @test hmcode_boost_kout ≈ hmcode_pk_nl_kout ./ hmcode_pk_mm[1:2:end, :]

    hmcode_pk_nl = Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm;
                                    pk_cb_z=hmcode_pk_cb, nM=64, threaded=false)
    @test hmcode_pk_nl_kout ≈ hmcode_pk_nl[1:2:end, :]

    hmcode_kout_irr = copy(hmcode_kout)
    hmcode_kout_irr[10] = (hmcode_kout[9] + hmcode_kout[10]) / 2
    hmcode_pk_nl_kout_irr = Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z, hmcode_kout_irr,
                                             hmcode_k, hmcode_pk_mm;
                                             pk_cb_support_z=hmcode_pk_cb,
                                             nM=64, threaded=false)
    @test size(hmcode_pk_nl_kout_irr) == (length(hmcode_kout_irr), length(hmcode_z))

    # We test that hmcode_pk_nl_kout_irr is close to interpolating hmcode_pk_nl to kout_irr
    for iz in 1:length(hmcode_z)
        itp = AkimaInterpolation(hmcode_pk_nl[:, iz], log.(hmcode_k))
        @test itp.(log.(hmcode_kout_irr)) ≈ hmcode_pk_nl_kout_irr[:, iz] rtol=5e-3
    end

    @test Mapse.hmcode_pmm(hmcode_cosmo, 0.0, hmcode_kout, hmcode_pk_mm[:, 1];
                           k_support=hmcode_k, pk_cb_support=hmcode_pk_cb[:, 1],
                           nM=64, threaded=false) ≈ hmcode_pk_nl_kout[:, 1]
    @test_throws ArgumentError Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z, reverse(hmcode_k), hmcode_pk_mm)
    hmcode_k_irregular = copy(hmcode_k)
    hmcode_k_irregular[10] = (hmcode_k[9] + hmcode_k[10]) / 2
    @test_throws ArgumentError Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z,
                                                hmcode_k_irregular, hmcode_pk_mm)
    @test_throws ArgumentError Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z,
                                                hmcode_kout, hmcode_k_irregular,
                                                hmcode_pk_mm;
                                                pk_cb_support_z=hmcode_pk_cb)
    @test_throws DimensionMismatch Mapse.hmcode_pmm(hmcode_cosmo, [0.0, 1.0], hmcode_k, hmcode_pk_mm)
    @test_throws DimensionMismatch Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z, hmcode_k,
                                                    hmcode_pk_mm, hmcode_pk_cb[1:end-1, :])
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k,
                                                  hmcode_pk_mm, hmcode_pk_cb; nM=0)
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k,
                                                  hmcode_pk_mm, hmcode_pk_cb; nM=1)
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z,
                                                  vcat(hmcode_kout, 20.0),
                                                  hmcode_k, hmcode_pk_mm;
                                                  pk_cb_support_z=hmcode_pk_cb)

    # Fast HMCode API tests
    hmcode_z_fine = collect(LinRange(minimum(hmcode_z), maximum(hmcode_z), 100))
    pk_mm_fine = zeros(length(hmcode_k), length(hmcode_z_fine))
    pk_cb_fine = zeros(length(hmcode_k), length(hmcode_z_fine))
    for ik in 1:length(hmcode_k)
        itp_mm = AkimaInterpolation(view(hmcode_pk_mm, ik, :), hmcode_z)
        itp_cb = AkimaInterpolation(view(hmcode_pk_cb, ik, :), hmcode_z)
        for iz in 1:length(hmcode_z_fine)
            pk_mm_fine[ik, iz] = itp_mm(hmcode_z_fine[iz])
            pk_cb_fine[ik, iz] = itp_cb(hmcode_z_fine[iz])
        end
    end

    pk_nl_direct = Mapse.hmcode_pmm(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine;
                                    pk_cb_z=pk_cb_fine, nM=64, threaded=false)
    pk_nl_fast = Mapse.hmcode_pmm_fast(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine, 20;
                                       pk_cb_z=pk_cb_fine, nM=64, threaded=false)
    @test size(pk_nl_fast) == size(pk_nl_direct)
    @test all(isfinite, pk_nl_fast)
    @test pk_nl_fast ≈ pk_nl_direct rtol=5e-3

    # Test boost fast
    boost_direct = Mapse.hmcode_boost(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine, pk_cb_fine;
                                      nM=64, threaded=false)
    boost_fast = Mapse.hmcode_boost_fast(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine, pk_cb_fine, 20;
                                         nM=64, threaded=false)
    @test size(boost_fast) == size(boost_direct)
    @test boost_fast ≈ boost_direct rtol=5e-3

    # Test support k
    pk_nl_fast_kout = Mapse.hmcode_pmm_fast(hmcode_cosmo, hmcode_z_fine, hmcode_kout, hmcode_k, pk_mm_fine, 20;
                                            pk_cb_support_z=pk_cb_fine, nM=64, threaded=false)
    @test size(pk_nl_fast_kout) == (length(hmcode_kout), length(hmcode_z_fine))
    @test all(isfinite, pk_nl_fast_kout)

    # Test scalar fallback
    pk_nl_fast_scalar = Mapse.hmcode_pmm_fast(hmcode_cosmo, 1.0, hmcode_k, pk_mm_fine[:, 1], 20;
                                             pk_cb=pk_cb_fine[:, 1], nM=64, threaded=false)
    pk_nl_direct_scalar = Mapse.hmcode_pmm(hmcode_cosmo, 1.0, hmcode_k, pk_mm_fine[:, 1];
                                           pk_cb=pk_cb_fine[:, 1], nM=64, threaded=false)
    @test pk_nl_fast_scalar ≈ pk_nl_direct_scalar

    # Test smart signatures (inputs already on the coarse grid)
    z_coarse = collect(LinRange(minimum(hmcode_z), maximum(hmcode_z), 20))
    pk_mm_coarse = zeros(length(hmcode_k), length(z_coarse))
    pk_cb_coarse = zeros(length(hmcode_k), length(z_coarse))
    for ik in 1:length(hmcode_k)
        itp_mm = AkimaInterpolation(view(hmcode_pk_mm, ik, :), hmcode_z)
        itp_cb = AkimaInterpolation(view(hmcode_pk_cb, ik, :), hmcode_z)
        for iz in 1:length(z_coarse)
            pk_mm_coarse[ik, iz] = itp_mm(z_coarse[iz])
            pk_cb_coarse[ik, iz] = itp_cb(z_coarse[iz])
        end
    end

    # Verify smart pmm_fast output matches downsampled pmm_fast exactly
    pk_nl_smart = Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse, hmcode_z_fine, hmcode_k, pk_mm_coarse;
                                        pk_cb_coarse=pk_cb_coarse, nM=64, threaded=false)
    @test size(pk_nl_smart) == size(pk_nl_direct)
    @test pk_nl_smart ≈ pk_nl_fast rtol=1e-12

    # Verify smart boost_fast output matches direct calculation to high accuracy
    boost_smart = Mapse.hmcode_boost_fast(hmcode_cosmo, z_coarse, hmcode_z_fine, hmcode_k, pk_mm_coarse;
                                          pk_cb_coarse=pk_cb_coarse, nM=64, threaded=false)
    @test size(boost_smart) == size(boost_direct)
    @test boost_smart ≈ boost_direct rtol=5e-3

    # Verify smart support-k output matches
    pk_nl_smart_kout = Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse, hmcode_z_fine, hmcode_kout, hmcode_k, pk_mm_coarse;
                                             pk_cb_coarse=pk_cb_coarse, nM=64, threaded=false)
    @test size(pk_nl_smart_kout) == size(pk_nl_fast_kout)
    @test pk_nl_smart_kout ≈ pk_nl_fast_kout rtol=1e-12

    # Verify smart scalar z_fine output
    pk_nl_smart_scalar = Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse, 1.0, hmcode_k, pk_mm_coarse;
                                               pk_cb_coarse=pk_cb_coarse, nM=64, threaded=false)
    @test length(pk_nl_smart_scalar) == length(hmcode_k)

    # Grid and parameter validation checks for fast APIs
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine, 4; pk_cb_z=pk_cb_fine)
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, reverse(hmcode_z_fine), hmcode_k, pk_mm_fine, 20; pk_cb_z=pk_cb_fine)
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse[1:4], hmcode_z_fine, hmcode_k, pk_mm_coarse[:, 1:4]; pk_cb_coarse=pk_cb_coarse[:, 1:4])
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, reverse(z_coarse), hmcode_z_fine, hmcode_k, pk_mm_coarse; pk_cb_coarse=pk_cb_coarse)
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse, [z_coarse[end] + 0.1], hmcode_k, pk_mm_coarse; pk_cb_coarse=pk_cb_coarse)
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, z_coarse, hmcode_z_fine, hmcode_k, pk_mm_coarse; pk_cb_coarse=pk_cb_coarse, pk_cb_support_coarse=pk_cb_coarse)
    @test_throws ArgumentError Mapse.hmcode_pmm_fast(hmcode_cosmo, hmcode_z_fine, hmcode_k, pk_mm_fine, 20; pk_cb_z=pk_cb_fine, pk_cb_support_z=pk_cb_fine)


    hmcode_curved_reference = readdlm(joinpath(@__DIR__, "data", "hmcode_curved_parity_reference.txt"), comments=true)
    hmcode_curved_z = unique(hmcode_curved_reference[:, 1])
    hmcode_curved_k = hmcode_curved_reference[hmcode_curved_reference[:, 1] .== hmcode_curved_z[1], 2]
    hmcode_curved_pmm = reshape(hmcode_curved_reference[:, 3], length(hmcode_curved_k), length(hmcode_curved_z))
    hmcode_curved_pcb = reshape(hmcode_curved_reference[:, 4], length(hmcode_curved_k), length(hmcode_curved_z))
    hmcode_curved_dmo = reshape(hmcode_curved_reference[:, 5], length(hmcode_curved_k), length(hmcode_curved_z))
    hmcode_curved_feedback = reshape(hmcode_curved_reference[:, 6], length(hmcode_curved_k), length(hmcode_curved_z))
    hmcode_curved_cosmo = Mapse.HMCodeCosmology(hmcode_Ωm, hmcode_Ωb, hmcode_h, 0.9649,
                                                0.8109118, -0.9, 0.2, hmcode_Ων, 0.01)
    @test Mapse.hmcode_pmm(hmcode_curved_cosmo, hmcode_curved_z, hmcode_curved_k,
                           hmcode_curved_pmm; pk_cb_z=hmcode_curved_pcb,
                           T_AGN=nothing, nM=32, threaded=false) ≈ hmcode_curved_dmo rtol=1e-12
    @test Mapse.hmcode_pmm(hmcode_curved_cosmo, hmcode_curved_z, hmcode_curved_k,
                           hmcode_curved_pmm; pk_cb_z=hmcode_curved_pcb,
                           T_AGN=10.0^7.8, nM=32, threaded=false) ≈ hmcode_curved_feedback rtol=1e-12

    mktempdir() do dir
        Mapse.save_pca_metadata(dir, compression.mean, compression.basis)
        @test isfile(joinpath(dir, "pca_mean.npy"))
        @test isfile(joinpath(dir, "pca_projection.npy"))
        @test npzread(joinpath(dir, "pca_mean.npy")) ≈ compression.mean
        @test npzread(joinpath(dir, "pca_projection.npy")) ≈ compression.basis
    end

    mktempdir() do dir
        open(joinpath(dir, "nn_setup.json"), "w") do io
            write(io, """
            {
                "n_input_features": 6,
                "n_output_features": 2,
                "n_hidden_layers": 1,
                "layers": {
                    "layer_1": {
                        "n_neurons": 4,
                        "activation_function": "tanh"
                    }
                },
                "preprocessing_name": "identity",
                "postprocessing_name": "identity"
            }
            """)
        end

        npzwrite(joinpath(dir, "weights.npy"), zeros(Float32, 6 * 4 + 4 + 4 * 2 + 2))
        npzwrite(joinpath(dir, "k.npy"), [0.1, 0.2, 0.3])
        npzwrite(joinpath(dir, "inminmax.npy"), hcat(zeros(6), ones(6)))
        npzwrite(joinpath(dir, "outminmax.npy"), hcat(zeros(2), ones(2)))
        npzwrite(joinpath(dir, "pca_mean.npy"), compression.mean)
        npzwrite(joinpath(dir, "pca_basis.npy"), compression.basis)

        loaded_component = Mapse.load_emulator(dir; emu=Mapse.SimpleChainsEmulator)

        @test loaded_component.TrainedEmulator.Description["emulator_description"] == Dict{String, Any}()
        @test loaded_component.Compression isa Mapse.PCACompression
        @test Mapse.get_Pk([0.1, 0.2, 0.3, 0.4, 0.5], 0.0, 1.0, loaded_component) ≈ compression.mean
    end

    mktempdir() do dir
        open(joinpath(dir, "nn_setup.json"), "w") do io
            write(io, """
            {
                "n_input_features": 6,
                "n_output_features": 2,
                "n_hidden_layers": 1,
                "layers": {
                    "layer_1": {
                        "n_neurons": 4,
                        "activation_function": "tanh"
                    }
                }
            }
            """)
        end

        npzwrite(joinpath(dir, "weights.npy"), zeros(Float32, 6 * 4 + 4 + 4 * 2 + 2))
        npzwrite(joinpath(dir, "k.npy"), [0.1, 0.2])
        npzwrite(joinpath(dir, "inminmax.npy"), hcat(zeros(6), ones(6)))
        npzwrite(joinpath(dir, "outminmax.npy"), hcat(zeros(2), ones(2)))

        loaded_component = Mapse.load_emulator(dir; emu=Mapse.SimpleChainsEmulator,
            preprocessing_name = :identity,
            postprocessing_name = :identity)

        @test loaded_component.Preprocessing === identity
        @test loaded_component.Postprocessing === Mapse.BUILTIN_POSTPROCESSING["identity"]
        @test_throws ArgumentError Mapse.load_emulator(dir; emu=Mapse.SimpleChainsEmulator,
            preprocessing_name = :not_a_registered_function,
            postprocessing_name = :identity)
    end
end

include("test_halofit_reactant.jl")
include("test_hmcode_reactant.jl")

@testset "HMCode-2020 Baryonic Response vs Patched CLASS" begin
    # UCF-2: Direct helper invariant test
    dummy_params = Mapse.HMcode.HMcodeParams(ones(10), ones(10), ones(10), ones(10), ones(10), ones(10), ones(10), ones(10), ones(10), fill(1.23, 10), ones(10), ones(10))
    notweaks_params = Mapse.HMcode._hmcode_notweaks_params(dummy_params)
    @test notweaks_params.k_star == fill(1.23, 10)
    @test notweaks_params.eta == zeros(10)
    @test notweaks_params.A == ones(10)
    @test notweaks_params.f_damp == zeros(10)
    @test notweaks_params.B == fill(4.0, 10)
    @test notweaks_params.k_damp == zeros(10)

    fixture_path_ref = joinpath(@__DIR__, "data", "hmcode_class_feedback_reference.txt")
    fixture_path_sup = joinpath(@__DIR__, "data", "hmcode_class_linear_support.txt")

    # Canonical fixture: schema, headers, shape, domain
    raw_header_ref = readlines(fixture_path_ref)[1:9]
    @test occursin("HMCode2020 feedback reference", raw_header_ref[1])
    @test occursin("fixture_schema_version: 1", raw_header_ref[2])
    @test occursin("class_patch: CAMB_PR_136_low_k_response_cutoff", raw_header_ref[4])
    @test occursin("domain: canonical test region k <= 10 h/Mpc", join(raw_header_ref, "\n"))

    data_ref = readdlm(fixture_path_ref, comments=true)
    @test size(data_ref) == (705, 8)

    # Support fixture: schema, headers, shape, domain
    raw_header_sup = readlines(fixture_path_sup)[1:9]
    @test occursin("fixture_schema_version: 1", raw_header_sup[2])
    @test occursin("domain: linear support grid k <= 50 h/Mpc", join(raw_header_sup, "\n"))

    data_sup = readdlm(fixture_path_sup, comments=true)
    @test size(data_sup) == (805, 4)

    # Domain bounds and per-redshift counts
    k_ref_z0 = data_ref[data_ref[:, 1] .== data_ref[1, 1], 2]
    k_sup_z0 = data_sup[data_sup[:, 1] .== data_sup[1, 1], 2]
    @test length(k_ref_z0) == 141
    @test length(k_sup_z0) == 161
    @test maximum(k_ref_z0) ≈ 9.696137237434288 rtol=1e-5
    @test maximum(k_sup_z0) ≈ 50.0 rtol=1e-5

    # Finite and positive values
    @test all(isfinite, data_ref)
    @test all(isfinite, data_sup)
    @test all(>(0), k_ref_z0)
    @test all(>(0), k_sup_z0)

    # Matching redshift grids
    z_ref = unique(data_ref[:, 1])
    z_sup_all = unique(data_sup[:, 1])
    @test z_ref == z_sup_all

    # Canonical k-grid is a subset of support k-grid
    k_sup_set = Set(round.(k_sup_z0, digits=12))
    @test all(k -> round(k, digits=12) ∈ k_sup_set, k_ref_z0)

    z_vals = z_sup_all
    k_vals = k_sup_z0
    k_ref_vals = k_ref_z0
    nz = length(z_vals)
    nk = length(k_vals)
    nk_ref = length(k_ref_vals)

    pk_mm_lin = zeros(nk, nz)
    pk_cb_lin = zeros(nk, nz)
    boost_ref = zeros(nk_ref, nz)
    dmo_boost_ref = zeros(nk_ref, nz)

    for iz in 1:nz
        mask_sup = data_sup[:, 1] .== z_vals[iz]
        mask_ref = data_ref[:, 1] .== z_vals[iz]
        pk_mm_lin[:, iz] = data_sup[mask_sup, 3]
        pk_cb_lin[:, iz] = data_sup[mask_sup, 4]
        boost_ref[:, iz] = data_ref[mask_ref, 6]
        dmo_boost_ref[:, iz] = data_ref[mask_ref, 8]
    end

    cosmo = Mapse.HMcode.HMcodeCosmology(
        0.315192,
        0.04930,
        0.6736,
        0.9649,
        0.8109118,
        -1.0,
        0.0,
        0.001422,
        0.0
    )

    dmo = Mapse.hmcode_pmm(cosmo, z_vals, k_vals, pk_mm_lin; pk_cb_z=pk_cb_lin, T_AGN=nothing)
    feedback = Mapse.hmcode_pmm(cosmo, z_vals, k_vals, pk_mm_lin; pk_cb_z=pk_cb_lin, T_AGN=10.0^7.8)

    boost = feedback ./ pk_mm_lin
    dmo_boost = dmo ./ pk_mm_lin

    mask10 = k_vals .<= 10.0
    boost_masked = boost[mask10, :]

    err_boost = abs.(boost_masked .- boost_ref) ./ boost_ref

    @test maximum(err_boost) < 5.0e-3

    err_dmo = abs.(dmo_boost[mask10, :] .- dmo_boost_ref) ./ dmo_boost_ref
    @test maximum(err_dmo) < 3.0e-3

    mask_low_k = k_vals .<= 3e-4
    @test boost[mask_low_k, :] ≈ dmo_boost[mask_low_k, :] rtol=1e-3
    @test boost[mask_low_k, :] ≈ ones(count(mask_low_k), nz) rtol=1e-3

    z_fine = collect(LinRange(minimum(z_vals), maximum(z_vals), 25))
    fast_dmo = Mapse.hmcode_pmm_fast(cosmo, z_vals, z_fine, k_vals, pk_mm_lin; pk_cb_coarse=pk_cb_lin, T_AGN=nothing, nM=64, threaded=false)
    fast_feedback = Mapse.hmcode_pmm_fast(cosmo, z_vals, z_fine, k_vals, pk_mm_lin; pk_cb_coarse=pk_cb_lin, T_AGN=10.0^7.8, nM=64, threaded=false)
    fast_response = fast_feedback ./ fast_dmo
    @test fast_response[mask_low_k, :] ≈ ones(count(mask_low_k), length(z_fine)) rtol=1e-3
end
