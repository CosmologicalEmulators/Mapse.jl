using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface
using DelimitedFiles, LinearAlgebra

@testset "HMCode 384-coordinate input-spectrum Mooncake" begin
    fixtures = joinpath(@__DIR__, "fixtures", "hmcode_mooncake_spectrum")
    x = vec(readdlm(joinpath(fixtures, "parameters.txt"), Float64))
    shifted = vec(readdlm(joinpath(fixtures, "perturbed_parameters.txt"), Float64))
    frozen_reference = vec(readdlm(joinpath(fixtures, "forwarddiff_gradient.txt"), Float64))
    @test length(x) == length(shifted) == length(frozen_reference) == 384
    k = collect(10.0 .^ range(-4, 2; length=64))
    z = [0.0, 0.5, 1.0]
    weights = reshape(collect(range(0.7, 1.3; length=192)), 64, 3)
    cosmo = Mapse.HMCodeCosmology(0.31, 0.049, 0.67, 0.965, 0.81, -0.93, 0.12, 0.0014, 0.01)
    f = function (p)
        mm = reshape(exp.(view(p, 1:192)), 64, 3)
        cb = reshape(exp.(view(p, 193:384)), 64, 3)
        prediction = Mapse.hmcode_pmm(cosmo, z, k, mm; pk_cb_z=cb,
            T_AGN=10.0^7.8, nM=8, threaded=false)
        return sum(weights .* log.(prediction)) / length(prediction)
    end
    mc, fd = AutoMooncake(), AutoForwardDiff()
    saved_x = copy(x)
    mp, fp = prepare_gradient(f, mc, x), prepare_gradient(f, fd, x)
    initial_value, initial_buffer = value_and_gradient(f, mp, mc, x)
    initial_gradient = copy(initial_buffer)
    @test all(isapprox.(initial_gradient, frozen_reference; rtol=1e-5, atol=1e-7))
    for point in (x, shifted, x)
        original = copy(point)
        value, g = value_and_gradient(f, mp, mc, point)
        reference = gradient(f, fp, fd, point)
        @test value ≈ f(point) rtol=1e-12 atol=1e-12
        @test all(isfinite, g)
        @test all(isapprox.(g, reference; rtol=1e-5, atol=1e-7))
        @test point == original
    end
    final_value, final_gradient = value_and_gradient(f, mp, mc, x)
    @test final_value == initial_value
    @test final_gradient == initial_gradient
    @test x == saved_x
    for i in 1:2
        direction = vec(readdlm(joinpath(fixtures, "direction_$i.txt"), Float64))
        @test length(direction) == length(x) && isapprox(norm(direction), 1.0)
        epsilon = 1e-5
        reference = (f(x .+ epsilon .* direction) - f(x .- epsilon .* direction)) / (2epsilon)
        @test dot(initial_gradient, direction) ≈ reference rtol=1e-4 atol=5e-7
    end
end
