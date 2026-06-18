using Test
using NPZ
using SimpleChains
using Static
using Mapse
using DataInterpolations

const RUN_REACTANT_TESTS = lowercase(get(ENV, "MAPSE_TEST_REACTANT", "false")) in
                           ("1", "true", "yes")

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
#a, Ωcb0, mν, h, w0, wa = [1., 0.3, 0.06, 0.67, -1.1, 0.2]
z = Array(LinRange(0., 3., 100))

emu = Mapse.SimpleChainsEmulator(Architecture = mlpd, Weights = weights)

postprocessing = (input, output, D, Pkemu) -> output
preprocessing = (input) -> input

effort_emu = Mapse.MapseEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing)

pkcb_emu = Mapse.MapseEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing)

postprocessing_boost = (input, output, D, emu) -> output
boost_emu = Mapse.NonLinearBoostPkEmulator(TrainedEmulator = emu, kgrid=k_test,
                                InMinMax = inminmax, OutMinMax = outminmax,
                                Preprocessing = preprocessing,
                                Postprocessing = postprocessing_boost)

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
    #@test isapprox(Zygote.gradient(x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->D_z_x(z, x), x)[1], ForwardDiff.gradient(x->D_z_x(z, x), x), rtol=1e-3)
    #@test isapprox(Zygote.gradient(x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-5)
    #@test isapprox(grad(central_fdm(5,1), x->f_z_x(z, x), x)[1], ForwardDiff.gradient(x->f_z_x(z, x), x), rtol=1e-4)
    #@test isapprox(grad(central_fdm(5,1), x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-7)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], ForwardDiff.gradient(x->r_z_x(3., x), x), rtol=1e-6)
    #@test isapprox(Zygote.gradient(x->r_z_x(3., x), x)[1], Zygote.gradient(x->r_z_check_x(3., x), x)[1], rtol=1e-7)
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

    # Test get_Pk for Boost
    boost_scalar = Mapse.get_Pk(x, 1.0, boost_emu)
    @test size(boost_scalar) == (40,)

    boost_vector = Mapse.get_Pk(x, [0.5, 1.0], boost_emu)
    @test size(boost_vector) == (40, 2)

    # Test PkEmulator (Combined)
    full_emu = Mapse.PkEmulator(LinearPmm = effort_emu, LinearPcb = pkcb_emu, Boost = boost_emu)

    full_pk_scalar = Mapse.get_Pk(x, 1.0, 1.0, full_emu)
    @test size(full_pk_scalar) == (40,)

    full_pk_vector = Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], full_emu)
    @test size(full_pk_vector) == (40, 2)

    bad_boost_emu = Mapse.NonLinearBoostPkEmulator(TrainedEmulator = emu, kgrid=k_test[1:end-1],
                                    InMinMax = inminmax, OutMinMax = outminmax,
                                    Preprocessing = preprocessing,
                                    Postprocessing = postprocessing_boost)
    bad_full_emu = Mapse.PkEmulator(LinearPmm = effort_emu, LinearPcb = pkcb_emu, Boost = bad_boost_emu)
    @test_throws DimensionMismatch Mapse.get_Pk(x, 1.0, 1.0, bad_full_emu)

    # Test get_linear_Pmm and get_linear_Pkcb
    linear_pmm_scalar = Mapse.get_linear_Pmm(x, 1.0, 1.0, full_emu)
    @test linear_pmm_scalar == Mapse.get_Pk(x, 1.0, 1.0, effort_emu)

    linear_pkcb_vector = Mapse.get_linear_Pkcb(x, [0.5, 1.0], [1.0, 1.0], full_emu)
    @test linear_pkcb_vector == Mapse.get_Pk(x, [0.5, 1.0], [1.0, 1.0], pkcb_emu)

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
    @test Mapse.preprocessing_boost_mnuw0wacdm(collect(1:8)) == collect(1:8)
    @test haskey(Mapse.LOAD_PRESETS, :mnuw0wacdm_class)
    @test Mapse.BUILTIN_PREPROCESSING["linear_pk_mnuw0wacdm"] === Mapse.preprocessing_linear_pk_mnuw0wacdm
    @test Mapse.BUILTIN_PREPROCESSING["boost_mnuw0wacdm"] === Mapse.preprocessing_boost_mnuw0wacdm
    @test Mapse.BUILTIN_POSTPROCESSING["linear_pk_mnuw0wacdm_sym_ratio"] === Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio
    @test Mapse.BUILTIN_POSTPROCESSING["boost_log10"] === Mapse.postprocessing_boost_log10
    @test Mapse.DEFAULT_EMULATOR_NAME == "mnuw0wacdm_class"
    @test Mapse.DEFAULT_EMULATOR_ARTIFACT == "mnuw0wacdm_class"
    @test Mapse.TRAINED_EMULATOR_ARTIFACTS[Mapse.DEFAULT_EMULATOR_NAME] == Mapse.DEFAULT_EMULATOR_ARTIFACT
    @test haskey(Mapse.trained_emulators, Mapse.DEFAULT_EMULATOR_NAME)
    @test Mapse.trained_emulators[Mapse.DEFAULT_EMULATOR_NAME] isa Mapse.PkEmulator
    artifacts_toml = read(Mapse.ARTIFACTS_TOML, String)
    @test occursin("git-tree-sha1 = \"c1a93f08faafd81f6c62ac3ee97bb9fe37f8cf2e\"", artifacts_toml)
    @test occursin("zenodo.org/records/20646263", artifacts_toml)
    @test !occursin("lazy = true", artifacts_toml)

    primitive_output = ones(3)
    primitive_emu = Mapse.MapseEmulator(TrainedEmulator = emu, kgrid=[0.1, 0.2, 0.3],
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
    @test Mapse.postprocessing_boost_log10(primitive_params, [0.0, 1.0, 2.0], nothing, primitive_emu) ≈ [1.0, 10.0, 100.0]

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

    halofit_pk_nl = Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin)
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
    @test Mapse.halofit_Pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1]) ≈ halofit_pk_nl[:, 1]
    @test Mapse.halofit_Pmm(halofit_cosmo, 0.0, halofit_k, halofit_pk_lin[:, 1],
                            halofit_Ωm_z[1], halofit_Ωv_z[1]) ≈ halofit_pk_nl[:, 1]
    @test size(Mapse.halofit_Pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin)) == size(halofit_pk_lin)
    @test Mapse.halofit_Pmm(halofit_params, halofit_z, halofit_k, halofit_pk_lin,
                            halofit_Ωm_z, halofit_Ωv_z) ≈ halofit_pk_nl
    @test_throws ArgumentError Mapse.halofit_Pmm(halofit_cosmo, halofit_z, reverse(halofit_k), halofit_pk_lin)
    @test_throws DimensionMismatch Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k, halofit_pk_lin[:, 1:1])
    @test_throws DimensionMismatch Mapse.halofit_Pmm(halofit_cosmo, halofit_z, halofit_k,
                                                     halofit_pk_lin, halofit_Ωm_z[1:1],
                                                     halofit_Ωv_z)

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

        loaded_component = Mapse.load_component_emulator(dir; emu=Mapse.SimpleChainsEmulator)

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

        loaded_component = Mapse.load_component_emulator(dir; emu=Mapse.SimpleChainsEmulator,
            preprocessing_name = :identity,
            postprocessing_name = :identity)

        @test loaded_component.Preprocessing === identity
        @test loaded_component.Postprocessing === Mapse.BUILTIN_POSTPROCESSING["identity"]
        @test_throws ArgumentError Mapse.load_component_emulator(dir; emu=Mapse.SimpleChainsEmulator,
            preprocessing_name = :not_a_registered_function,
            postprocessing_name = :identity)
    end

    mktempdir() do dir
        function write_minimal_component(component_dir, n_input_features)
            mkpath(component_dir)
            open(joinpath(component_dir, "nn_setup.json"), "w") do io
                write(io, """
                {
                    "n_input_features": $(n_input_features),
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

            npzwrite(joinpath(component_dir, "weights.npy"), zeros(Float32, n_input_features * 4 + 4 + 4 * 2 + 2))
            npzwrite(joinpath(component_dir, "k.npy"), [0.1, 0.2])
            npzwrite(joinpath(component_dir, "inminmax.npy"), hcat(zeros(n_input_features), ones(n_input_features)))
            npzwrite(joinpath(component_dir, "outminmax.npy"), hcat(zeros(2), ones(2)))
        end

        write_minimal_component(joinpath(dir, "Pk_lin_mm"), 7)
        write_minimal_component(joinpath(dir, "Pk_lin_cb"), 7)
        write_minimal_component(joinpath(dir, "Boost"), 9)

        preset_emu = Mapse.load_emulator(dir; emu=Mapse.SimpleChainsEmulator,
            preset = :mnuw0wacdm_class)

        @test preset_emu.LinearPmm.Preprocessing === Mapse.preprocessing_linear_pk_mnuw0wacdm
        @test preset_emu.LinearPcb.Preprocessing === Mapse.preprocessing_linear_pk_mnuw0wacdm
        @test preset_emu.Boost.Preprocessing === Mapse.preprocessing_boost_mnuw0wacdm
        @test preset_emu.LinearPmm.Postprocessing === Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio
        @test preset_emu.LinearPcb.Postprocessing === Mapse.postprocessing_linear_pk_mnuw0wacdm_sym_ratio
        @test preset_emu.Boost.Postprocessing === Mapse.postprocessing_boost_log10
    end
end

if RUN_REACTANT_TESTS
    include("test_halofit_reactant.jl")
end
