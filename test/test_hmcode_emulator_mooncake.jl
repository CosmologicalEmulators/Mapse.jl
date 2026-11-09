using Test, Mapse, Mooncake, ForwardDiff, FiniteDiff, DifferentiationInterface
using DelimitedFiles

# The ordinary full suite has already loaded these through the ForwardDiff
# tests; the focused Mooncake subset needs the same helpers explicitly.
if !isdefined(@__MODULE__, :_di_native_linear_spectra)
    include("hmcode_pipeline_helpers.jl")
end

@testset "Live-emulator Mooncake gradients" begin
    root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    mm = Mapse.load_emulator(joinpath(root, "Pk_lin_mm"))
    cb = Mapse.load_emulator(joinpath(root, "Pk_lin_cb"))
    growth = Mapse.AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_ln10As_basis"]
    for mode in ("linear", "direct", "smart")
        @testset "$mode" begin
            fixtures = joinpath(@__DIR__, "fixtures", "hmcode_mooncake_emulators", mode)
            x = vec(readdlm(joinpath(fixtures, "parameters.txt"), Float64))
            shifted = vec(readdlm(joinpath(fixtures, "perturbed_parameters.txt"), Float64))
            frozen = vec(readdlm(joinpath(fixtures, "forwarddiff_gradient.txt"), Float64))
            z = mode == "smart" ? collect(range(0.0, 3.0; length=32)) :
                                  collect(range(0.0, 1.0; length=5))
            k = mode == "smart" ? collect(10.0 .^ range(-3, 0.5; length=8)) .* 0.6736 :
                                  collect(10.0 .^ range(-3, 1; length=16)) .* 0.6736
            if mode == "smart"
                for point in (x, shifted)
                    feature = Mapse.predict_baryonic_discontinuity(point[1:8]; T_AGN=10.0^point[9])
                    grid = Mapse.build_smart_coarse_grid(first(z), last(z), 24, feature)
                    split = searchsortedlast(grid, feature)
                    @test 5 <= split <= length(grid)-4 && grid[split] == feature
                end
            end
            f = if mode == "linear"
                p -> begin
                    pmm, pcb = _di_native_linear_spectra(p, z, mm, cb, growth)
                    (sum(log, pmm) + sum(log, pcb)) / (length(pmm) + length(pcb))
                end
            else
                p -> begin
                    result = _di_native_hmcode_prediction(p, z, k, mm, cb, growth, 0.8109118;
                        smart=mode == "smart", feedback=true, nM=8, N_coarse=24)
                    sum(log, result) / length(result)
                end
            end
            mc, fd = AutoMooncake(), AutoForwardDiff()
            original_x = copy(x)
            mp, fp = prepare_gradient(f, mc, x), prepare_gradient(f, fd, x)
            initial_value, initial_buffer = value_and_gradient(f, mp, mc, x)
            initial_gradient = copy(initial_buffer)
            @test all(isapprox.(initial_gradient, frozen; rtol=1e-5, atol=1e-7))
            for point in (x, shifted, x)
                original = copy(point)
                value, g = value_and_gradient(f, mp, mc, point)
                reference = gradient(f, fp, fd, point)
                @test value ≈ f(point) rtol=1e-12 atol=1e-12
                @test all(isfinite, g)
                @test all(isapprox.(g, reference; rtol=1e-5, atol=1e-7))
                @test point == original
            end
            last_value, last_gradient = value_and_gradient(f, mp, mc, x)
            @test last_value == initial_value
            @test last_gradient == initial_gradient
            @test x == original_x
            finite = AutoFiniteDiff(; fdtype=Val(:central))
            finite_prep = prepare_gradient(f, finite, x)
            finite_rtol = mode == "smart" ? 5e-4 : 1e-4
            @test all(isapprox.(initial_gradient, gradient(f, finite_prep, finite, x);
                                rtol=finite_rtol, atol=5e-5))
        end
    end
end
