using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface
if !isdefined(@__MODULE__, :_hmcode_public_mooncake_problem)
    include("hmcode_public_mooncake_helpers.jl")
end

@testset "Threaded public HMCode gradients and replay" begin
    for (fast, feedback, fixed_redshifts, fixed_temperature) in (
            (false, false, nothing, nothing), (false, true, nothing, nothing),
            (true, false, nothing, nothing), (true, true, nothing, nothing),
            (false, false, Int[0,1,2], nothing),
            (false, true, Int[0,1,2], nothing),
            (false, true, nothing, 10^8),
            (false, false, Float32[0,0.5,1], nothing))
        @testset "fast=$fast feedback=$feedback z=$(typeof(fixed_redshifts)) T=$(typeof(fixed_temperature)) threads=$(Threads.nthreads())" begin
            serial, x = _hmcode_public_mooncake_problem(;
                fast, feedback, fixed_redshifts, fixed_temperature)
            parallel, _ = _hmcode_public_mooncake_problem(;
                fast, feedback, fixed_redshifts, fixed_temperature, threaded=true)
            mc, fd = AutoMooncake(), AutoForwardDiff()
            sp = prepare_gradient(serial, mc, x)
            pp = prepare_gradient(parallel, mc, x)
            fp = prepare_gradient(serial, fd, x)
            first_value, first_gradient = value_and_gradient(parallel, pp, mc, x)
            saved = copy(first_gradient)
            for point in (x, x .+ 1e-4, x)
                old = copy(point)
                sv, sg = value_and_gradient(serial, sp, mc, point)
                reference = copy(sg)
                pv, pg = value_and_gradient(parallel, pp, mc, point)
                @test pv ≈ sv rtol=1e-12 atol=1e-12
                @test pv ≈ parallel(point) rtol=1e-12 atol=1e-12
                @test all(isfinite, pg)
                @test all(isapprox.(pg, reference; rtol=1e-8, atol=1e-9))
                @test all(isapprox.(pg, gradient(serial, fp, fd, point); rtol=1e-5, atol=1e-7))
                @test point == old
            end
            value, g = value_and_gradient(parallel, pp, mc, x)
            @test value == first_value
            @test g == saved
        end
    end
end
