using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface, Interpolations
import DataInterpolations
include("test_hmcode_one_halo_mooncake.jl")
include("test_hmcode_baryon_mooncake.jl")
include("test_hmcode_nfw_mooncake.jl")

function _inverse_growth_analytic_gradient(p)
    knots, values, query = p[1:5], p[6:10], p[11]
    i = clamp(searchsortedfirst(knots, query)-1, 1, 4)
    weight = (query-knots[i]) / (knots[i+1]-knots[i])
    slope = (values[i+1]-values[i]) / (knots[i+1]-knots[i])
    result = zeros(11)
    result[i], result[i+1] = -(1-weight)*slope, -weight*slope
    result[5+i], result[6+i], result[11] = 1-weight, weight, slope
    return result
end

function _check_hmcode_mooncake(f, points; rtol=1e-12, atol=1e-12, primal_rtol=0.0)
    mc, fd = AutoMooncake(), AutoForwardDiff()
    # Flush progress before potentially long compiler work in the full suite.
    println("MOONCAKE_PREPARE_BEGIN test=", Test.get_testset().description)
    flush(stdout)
    mp, fp = prepare_gradient(f, mc, first(points)), prepare_gradient(f, fd, first(points))
    println("MOONCAKE_PREPARE_END test=", Test.get_testset().description)
    flush(stdout)
    original_value, initial_gradient = value_and_gradient(f, mp, mc, first(points))
    original = copy(initial_gradient)
    for x in points
        saved = copy(x)
        value, grad = value_and_gradient(f, mp, mc, x)
        reference = gradient(f, fp, fd, x)
        @test isapprox(value, f(x); rtol=primal_rtol, atol=0.0)
        @test all(isapprox.(grad, reference; rtol=rtol, atol=atol))
        @test x == saved
    end
    replay_value, replay_gradient = value_and_gradient(f, mp, mc, first(points))
    @test replay_value == original_value
    @test replay_gradient == original
end

@testset "HMCode Mooncake root and quadrature rules" begin
    @test Base.get_extension(Mapse, :MapseMooncakeExt) !== nothing
    @testset "Inverse growth moving knots and one-sided derivatives" begin
        f = p -> Interpolations.linear_interpolation(view(p, 1:5), view(p, 6:10))(p[11])
        ref = p -> DataInterpolations.LinearInterpolation(view(p, 6:10), view(p, 1:5))(p[11])
        base = [0.1, 0.3, 0.55, 0.8, 1.0, 0.01, 0.12, 0.35, 0.7, 1.0, 0.41]
        mc, fd = AutoMooncake(), AutoForwardDiff()
        prep, fprep = prepare_gradient(f, mc, base), prepare_gradient(ref, fd, base)
        saved = copy(gradient(f, prep, mc, base))
        for q in (base[1], 0.11, 0.3, 0.41, 0.55, 0.55-1e-8, 0.55+1e-8, 0.99, base[5])
            p = copy(base)
            p[11] = q
            old = copy(p)
            value, grad = value_and_gradient(f, prep, mc, p)
            @test value ≈ ref(p) rtol=1e-13 atol=1e-14
            @test all(isapprox.(grad, _inverse_growth_analytic_gradient(p); rtol=1e-12, atol=1e-12))
            if q ∉ base[1:5]
                @test all(isapprox.(grad, gradient(ref, fprep, fd, p); rtol=1e-12, atol=1e-12))
            end
            @test p == old
        end
        changed = base .+ 1e-4
        @test all(isapprox.(gradient(f, prep, mc, changed), gradient(ref, fprep, fd, changed);
                            rtol=1e-12, atol=1e-12))
        @test gradient(f, prep, mc, base) == saved
        for q in (0.09, 1.01)
            p = copy(base)
            p[11] = q
            @test_throws BoundsError f(p)
        end
    end
    @testset "Mixed-mode growth with unchanged dense output" begin
        point = [0.31, 0.049, 0.67, 0.965, 0.81, -0.93, 0.12, 0.0014, 0.01]
        for lcdm in (false, true)
            f = function (x)
                g = Mapse.HMcode.get_growth_interpolator(Mapse.HMCodeCosmology(x...); LCDM=lcdm)
                return g(0.41) + 0.37g(0.73) - 0.21g(1.01) + 0.17g(1e-4)
            end
            _check_hmcode_mooncake(f, [point, point .+ 1e-4, point]; rtol=1e-5, atol=1e-7)
            mc = AutoMooncake()
            prep = prepare_gradient(f, mc, point)
            g = gradient(f, prep, mc, point)
            @test g[[2, 3, 4, 5, 8]] == zeros(5)
            lcdm && @test g[[6, 7, 9]] == zeros(3)
        end
    end
    @testset "Implicit radius and endpoint branches" begin
        f = x -> Mapse.HMcode.get_nonlinear_radius(
            0.1 + zero(x[1]), 10.0 + zero(x[1]), x[2], r -> x[1] / r)
        _check_hmcode_mooncake(f, [[2.0, 1.5], [2.1, 1.4], [2.0, 1.5]])
        shared = x -> Mapse.HMcode.get_nonlinear_radius(
            0.1 + zero(x[1]), 10.0, x[1] + 1.0, r -> x[1]^2 / r)
        _check_hmcode_mooncake(shared, [[2.0], [2.1], [2.0]])
        lower = x -> Mapse.HMcode.get_nonlinear_radius(x[1], 10.0, 100.0, r -> 1 / r)
        upper = x -> Mapse.HMcode.get_nonlinear_radius(0.1 + zero(x[1]), x[1], 0.001, r -> 1 / r)
        _check_hmcode_mooncake(lower, [[1.0], [1.1], [1.0]])
        _check_hmcode_mooncake(upper, [[10.0], [11.0], [10.0]])
        exact = x -> Mapse.HMcode.get_nonlinear_radius(x[1], x[2], x[3], r -> x[4] / r)
        _check_hmcode_mooncake(exact, [[1.0, 10.0, 2.0, 2.0], [1.0, 10.0, 0.2, 2.0]])
    end
    @testset "Native sigma functor with shared views" begin
        k = exp.(range(log(1e-4), log(10.0); length=33))
        logp = log.(@. 1e4 * (k / 0.1)^0.965 * exp(-k / 5))
        x = vcat(logp, [5.0, 0.1, 10.0])
        f = function (p)
            typed_k = eltype(p).(k)
            sigma = Mapse.HMCodeSigmaInterpolator(log.(typed_k),
                reshape(view(p, 1:33), :, 1), typed_k, [0.0])
            fun = Mapse.HMcode.SigmaREval(sigma, 0.0, 1)
            r1 = Mapse.HMcode.get_nonlinear_radius(p[35], p[36], p[34], fun)
            r2 = Mapse.HMcode.get_nonlinear_radius(p[35], p[36], 1.1p[34], fun)
            return 0.73r1 - 0.21r2
        end
        base_sigma = Mapse.HMCodeSigmaInterpolator(log.(k), reshape(logp, :, 1), k, [0.0])
        @test base_sigma(x[35], 1) > 1.1x[34] > x[34] > base_sigma(x[36], 1)
        _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-9, atol=1e-10)
    end
    @testset "Captured quadrature parameters and moving endpoint" begin
        f = x -> Mapse.HMcode.get_accumulated_growth(x[1], a -> x[2] * a^x[3])
        _check_hmcode_mooncake(f,
            [[0.6, 0.8, 1.1], [0.7, 0.9, 1.2], [1.0e-4, 0.8, 1.1]]; rtol=1e-5, atol=1e-7)
        shared = x -> -0.37Mapse.HMcode.get_accumulated_growth(x[1], a -> x[1] * a^x[2])
        _check_hmcode_mooncake(shared, [[0.6, 1.1], [0.7, 1.2]]; rtol=1e-5, atol=1e-7)
        exact = x -> x[2] * (x[1]^x[3] - 1e-4^x[3]) / x[3] + x[2] * 1e-4^x[3]
        x = [0.6, 0.8, 1.1]
        backend = AutoMooncake()
        prep = prepare_gradient(f, backend, x)
        @test all(isapprox.(gradient(f, prep, backend, x), ForwardDiff.gradient(exact, x);
                            rtol=1e-5, atol=1e-7))
    end
    @testset "Cubic spline array cotangents" begin
        grid = range(1.0e-4, 1.0; length=17)
        f = function (x)
            spline = extrapolate(scale(interpolate(view(x, 1:17),
                BSpline(Cubic(Line(OnGrid())))), grid), Line())
            return 0.73Mapse.HMcode.get_accumulated_growth(x[18], spline) -
                   0.21Mapse.HMcode.get_accumulated_growth(0.41 + zero(x[18]), spline)
        end
        x = vcat(0.8 .* grid .^ 1.1, 0.6)
        _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-5, atol=1e-7)
    end
    @testset "Finite displacement integral spectrum and endpoint cotangents" begin
        k = exp.(range(log(1e-4), log(10.0); length=33))
        logp = log.(@. 1e4 * (k/0.1)^0.965 * exp(-k/5))
        f = function (x)
            pk, _ = Mapse._hmcode_linear_interpolators([0.0], k,
                reshape(exp.(view(x, 4:36)), :, 1))
            return Mapse.HMcode.sigmaV(x[1], Mapse.HMcode.PkLinEval(pk, 0.0, 1);
                                       kmin=x[2], kmax=x[3])
        end
        x = vcat([0.0, 0.002, 8.1], logp)
        changed = vcat([0.17, 0.003, 8.3], logp .+ 1e-4)
        _check_hmcode_mooncake(f, [x, changed, x]; rtol=1e-5, atol=1e-7)
        simple = p -> Mapse.HMcode.sigmaV(0.0, k -> p[3]*exp(-p[4]*k);
                                         kmin=p[1], kmax=p[2])
        exact = p -> sqrt(p[3]*(exp(-p[4]*p[1])-exp(-p[4]*p[2])) / p[4] / (6pi^2))
        p = [0.002, 8.1, 2.0, 0.7]
        mc = AutoMooncake()
        prep = prepare_gradient(simple, mc, p)
        @test all(isapprox.(gradient(simple, prep, mc, p), ForwardDiff.gradient(exact, p);
                            rtol=1e-5, atol=1e-7))
        unbounded = p -> Mapse.HMcode.sigmaV(0.0, k -> p[1]*exp(-k))
        @test_throws ArgumentError prepare_gradient(unbounded, mc, [2.0])
    end
end

include("test_hmcode_public_mooncake.jl")
include("test_hmcode_spectrum_mooncake.jl")
include("test_hmcode_emulator_mooncake.jl")
include("test_hmcode_threaded_mooncake.jl")
include("test_hmcode_threaded_pipelines.jl")
