using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface

@testset "Baryonic analytical Mooncake pullback" begin
    H = Mapse.HMcode
    for (nm, nk) in ((3, 5), (5, 3), (1, 4), (4, 1)), views in (false, true)
        parentW = reshape(collect(range(-0.2, 1.3; length=4nm*nk)), 2nm, 2nk)
        W = views ? view(parentW, 1:2:2nm, 1:2:2nk) : copy(parentW[1:nm, 1:nk])
        parentM = exp.(range(log(1e10), log(1e14); length=2nm))
        M = views ? view(parentM, 1:2:2nm) : copy(parentM[1:nm])
        oldW, oldparentW, oldM = copy(W), copy(parentW), copy(parentM)
        p = [1e12, 0.01, 0.31, 0.2596, 0.049]
        seeds = reshape(collect(range(-0.3, 1.1; length=nm*nk)), nm, nk)
        args = Mooncake.zero_fcodual.((W, M, p...))
        out, back = Mooncake.rrule!!(Mooncake.zero_fcodual(H.apply_baryonic_transform!), args...)
        @test Mooncake.primal(out) === W
        reference = function (x)
            A = reshape(x[1:nm*nk], nm, nk)
            masses = x[nm*nk+1:nm*nk+nm]
            Mb, fs, Om, Oc, Ob = x[end-4:end]
            t = (masses ./ Mb).^2
            coefficient = @. Oc/Om + (Ob/Om-fs)*t/(1+t)
            return sum(seeds .* (A .* reshape(coefficient, :, 1) .+ fs))
        end
        x = vcat(vec(oldW), M, p)
        @test sum(seeds .* W) ≈ reference(x) rtol=1e-13 atol=1e-13
        Mooncake.arrayify(out)[2] .= seeds
        Mooncake.arrayify(args[2])[2] .= 1e-20
        bars = back(Mooncake.NoRData())
        expected = ForwardDiff.gradient(reference, x)
        got = vcat(vec(Mooncake.arrayify(out)[2]),
            Mooncake.arrayify(args[2])[2] .- 1e-20, collect(bars[4:end]))
        @test all(isapprox.(got, expected; rtol=1e-11, atol=1e-24))
        @test W == oldW
        @test parentW == oldparentW
        @test parentM == oldM
    end

    f = function (x)
        W = reshape(x[1:15], 3, 5)
        before = 0.23sum(abs2, W)
        H.apply_baryonic_transform!(W, exp.(x[16:18]), exp(x[19]),
            x[20], x[21], x[22], x[23])
        return before + sum(W .* reshape(collect(range(-0.3, 1.1; length=15)), 3, 5))
    end
    x = vcat(collect(range(0.7, 1.5; length=15)), log.([1e10, 1e12, 1e14]),
        log(1e12), 0.01, 0.31, 0.2596, 0.049)
    mc, fd = AutoMooncake(), AutoForwardDiff()
    mp, fp = prepare_gradient(f, mc, x), prepare_gradient(f, fd, x)
    original = copy(gradient(f, mp, mc, x))
    for point in (x, x .+ 1e-4, x)
        saved = copy(point)
        value, g = value_and_gradient(f, mp, mc, point)
        @test value ≈ f(point) rtol=1e-12 atol=1e-12
        @test g ≈ gradient(f, fp, fd, point) rtol=1e-10 atol=1e-11
        @test point == saved
    end
    @test gradient(f, mp, mc, x) == original

    W = ones(3, 5)
    args = Mooncake.zero_fcodual.((W, view(W, :, 1), 1.2, 0.01, 0.31, 0.2596, 0.049))
    @test_throws ArgumentError Mooncake.rrule!!(Mooncake.zero_fcodual(H.apply_baryonic_transform!), args...)
    @test W == ones(3, 5)
end
