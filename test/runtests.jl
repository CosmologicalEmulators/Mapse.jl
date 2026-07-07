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

    @test Mapse.preprocessing_linear_pk_mnuw0wacdm(collect(1:8)) == collect(3:8)
    @test haskey(Mapse.LOAD_PRESETS, :mnuw0wacdm_linear)
    @test Mapse.BUILTIN_PREPROCESSING["linear_pk_mnuw0wacdm"] === Mapse.preprocessing_linear_pk_mnuw0wacdm
    @test Mapse.BUILTIN_POSTPROCESSING["linear_pk_mnuw0wacdm_sym_ratio"] === Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio
    @test Mapse.DEFAULT_EMULATOR_NAME == "mnuw0wacdm_class"
    @test Mapse.DEFAULT_EMULATOR_ARTIFACT == "mnuw0wacdm_class"

    # Artifact-based component loading tests
    artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    # Stricter load_emulator validation tests
    @test_throws ArgumentError Mapse.load_emulator(artifact_root)
    @test_throws ArgumentError Mapse.load_emulator(tempname())

    artifact_pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"); preset=:mnuw0wacdm_linear)
    artifact_pcb_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_cb"); preset=:mnuw0wacdm_linear)

    artifact_params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    artifact_pk_mm = Mapse.get_Pk(artifact_params, 0.0, 1.0, artifact_pmm_emu)
    @test size(artifact_pk_mm) == (length(Mapse.get_kgrid(artifact_pmm_emu)),)
    @test all(isfinite, artifact_pk_mm)

    artifact_pk_cb = Mapse.get_Pk(artifact_params, 0.0, 1.0, artifact_pcb_emu)
    @test size(artifact_pk_cb) == (length(Mapse.get_kgrid(artifact_pcb_emu)),)
    @test all(isfinite, artifact_pk_cb)

    # Halofit on linear emulator output
    artifact_k_lin = Mapse.get_kgrid(artifact_pmm_emu)
    artifact_halofit_pk = Mapse.halofit_Pmm(artifact_params, 0.0, artifact_k_lin, artifact_pk_mm)
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
    @test occursin("git-tree-sha1 = \"c1a93f08faafd81f6c62ac3ee97bb9fe37f8cf2e\"", artifacts_toml)
    @test occursin("zenodo.org/records/20646263", artifacts_toml)

    primitive_output = ones(3)
    primitive_emu = Mapse.TransferFunctionEmulator(TrainedEmulator = emu, kgrid=[0.1, 0.2, 0.3],
                                        InMinMax = inminmax, OutMinMax = outminmax,
                                        Preprocessing = preprocessing,
                                        Postprocessing = postprocessing)
    primitive_params = [3.0, 0.96, 67.0, 0.0224, 0.12, 0.06, -1.0, 0.0]
    scalar_post = Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio(primitive_params, primitive_output, 2.0, primitive_emu)
    vector_post = Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio(primitive_params, repeat(primitive_output, 1, 2), [2.0, 3.0], primitive_emu)
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

    halofit_pk_nl = @inferred Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin)
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
    halofit_pk_nl_external_bg = @inferred Mapse.halofit_Pmm(halofit_cosmo, halofit_z,
                                                           halofit_k, halofit_pk_lin,
                                                           halofit_Ωm_z, halofit_Ωv_z)
    @test size(halofit_pk_nl) == size(halofit_pk_lin)
    @test halofit_pk_nl_unchecked ≈ halofit_pk_nl
    @test halofit_pk_nl_one_z ≈ halofit_pk_nl[:, 1]
    @test halofit_pk_nl_external_bg ≈ halofit_pk_nl
    @test all(isfinite, halofit_pk_nl)
    @test all(>(0), halofit_pk_nl)
    @test (@inferred Mapse.halofit_Pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1])) ≈ halofit_pk_nl[:, 1]
    @test Mapse.halofit_Pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1],
                            halofit_Ωm_z[1], halofit_Ωv_z[1]) ≈ halofit_pk_nl[:, 1]
    @test size(@inferred(Mapse.halofit_Pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin))) == size(halofit_pk_lin)
    @test Mapse.halofit_Pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin,
                            halofit_Ωm_z, halofit_Ωv_z) ≈ halofit_pk_nl
    @test_throws ArgumentError Mapse.halofit_Pmm(halofit_cosmo, halofit_z, reverse(halofit_k), halofit_pk_lin)
    @test_throws DimensionMismatch Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin[:, 1:1])
    @test_throws DimensionMismatch Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k,
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
    mapse_class_pk_nl = Mapse.halofit_Pmm(class_params, class_z, class_k, class_pk_lin,
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
    @test Mapse.hmcode_Pmm(hmcode_cosmo, 0.0, hmcode_k, hmcode_pk_mm[:, 1],
                           hmcode_pk_cb[:, 1]; nM=64, threaded=false) ./ hmcode_pk_mm[:, 1] ≈
          Mapse.hmcode_boost(hmcode_cosmo, 0.0, hmcode_k, hmcode_pk_mm[:, 1],
                             hmcode_pk_cb[:, 1];
                             nM=64, threaded=false)
    @test Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm;
                             pk_cb_z=hmcode_pk_cb, nM=64, threaded=false) ≈
          Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; nM=64, threaded=false)
    @test Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; accuracy=0.25, threaded=false) ≈
          Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; nM=64, threaded=false)
    @test Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_k,
                             hmcode_pk_mm; pk_cb_support_z=hmcode_pk_cb,
                             nM=64, threaded=false) ≈
          Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k, hmcode_pk_mm,
                             hmcode_pk_cb; nM=64, threaded=false)
    hmcode_kout = hmcode_k[1:2:end]
    hmcode_pk_nl_kout = Mapse.hmcode_Pmm(hmcode_cosmo, hmcode_z, hmcode_kout,
                                         hmcode_k, hmcode_pk_mm;
                                         pk_cb_support_z=hmcode_pk_cb,
                                         nM=64, threaded=false)
    hmcode_boost_kout = Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_kout,
                                           hmcode_k, hmcode_pk_mm;
                                           pk_cb_support_z=hmcode_pk_cb,
                                           nM=64, threaded=false)
    @test size(hmcode_pk_nl_kout) == (length(hmcode_kout), length(hmcode_z))
    @test hmcode_boost_kout ≈ hmcode_pk_nl_kout ./ hmcode_pk_mm[1:2:end, :]
    @test Mapse.hmcode_Pmm(hmcode_cosmo, 0.0, hmcode_kout, hmcode_pk_mm[:, 1];
                           k_support=hmcode_k, pk_cb_support=hmcode_pk_cb[:, 1],
                           nM=64, threaded=false) ≈ hmcode_pk_nl_kout[:, 1]
    @test_throws ArgumentError Mapse.hmcode_Pmm(hmcode_cosmo, hmcode_z, reverse(hmcode_k), hmcode_pk_mm)
    @test_throws DimensionMismatch Mapse.hmcode_Pmm(hmcode_cosmo, [0.0, 1.0], hmcode_k, hmcode_pk_mm)
    @test_throws DimensionMismatch Mapse.hmcode_Pmm(hmcode_cosmo, hmcode_z, hmcode_k,
                                                    hmcode_pk_mm, hmcode_pk_cb[1:end-1, :])
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k,
                                                  hmcode_pk_mm, hmcode_pk_cb; accuracy=0.0)
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z, hmcode_k,
                                                  hmcode_pk_mm, hmcode_pk_cb; nM=64,
                                                  accuracy=0.25)
    @test_throws ArgumentError Mapse.hmcode_boost(hmcode_cosmo, hmcode_z,
                                                  vcat(hmcode_kout, 20.0),
                                                  hmcode_k, hmcode_pk_mm;
                                                  pk_cb_support_z=hmcode_pk_cb)

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
