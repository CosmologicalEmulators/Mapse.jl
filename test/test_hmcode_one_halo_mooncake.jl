using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface

@testset "One-halo analytical Mooncake pullback" begin
    H = Mapse.HMcode
    for (nm, nk) in ((3, 5), (5, 3), (1, 4), (4, 1)), views in (false, true)
        Wparent = reshape(collect(range(0.2, 1.8; length=4nm*nk)), 2nm, 2nk)
        W = views ? view(Wparent, 1:2:2nm, 1:2:2nk) : copy(Wparent[1:nm, 1:nk])
        wparent = collect(range(0.3, 1.1; length=2nm))
        w = views ? view(wparent, 1:2:2nm) : copy(wparent[1:nm])
        Iparent, W2parent = fill(-7.3, 2nk), fill(8.1, 2nm, 2nk)
        I = views ? view(Iparent, 1:2:2nk) : fill(-7.3, nk)
        W2 = views ? view(W2parent, 1:2:2nm, 1:2:2nk) : fill(8.1, nm, nk)
        rho = 1.7
        seedI = collect(range(-0.3; step=0.2, length=nk))
        seedW2 = reshape(collect(range(-0.7, 0.9; length=nm*nk)), nm, nk)
        args = Mooncake.zero_fcodual.((I, W2, W, w, rho))
        out, back = Mooncake.rrule!!(Mooncake.zero_fcodual(H._one_halo_integral!), args...)
        @test Mooncake.primal(out) === I
        @test W2 == W .^ 2
        @test I ≈ rho .* (transpose(W .^ 2) * w) rtol=1e-14
        Mooncake.arrayify(out)[2] .= seedI
        Mooncake.arrayify(args[2])[2] .= seedW2
        Mooncake.arrayify(args[3])[2] .= 0.13
        Mooncake.arrayify(args[4])[2] .= 0.17
        bars = back(Mooncake.NoRData())
        reference = function (p)
            A = reshape(p[1:nm*nk], nm, nk)
            b = p[nm*nk+1:nm*nk+nm]
            return sum(seedW2 .* A.^2) + p[end] * sum(seedI .* (transpose(A.^2)*b))
        end
        expected = ForwardDiff.gradient(reference, vcat(vec(W), w, rho))
        got = vcat(vec(Mooncake.arrayify(args[3])[2]) .- 0.13,
                   Mooncake.arrayify(args[4])[2] .- 0.17, bars[end])
        @test got ≈ expected rtol=1e-12 atol=1e-12
        @test all(iszero, Mooncake.arrayify(out)[2])
        @test all(iszero, Mooncake.arrayify(args[2])[2])
        @test all(==(-7.3), I)
        @test all(==(8.1), W2)
        @test all(==(-7.3), Iparent)
        @test all(==(8.1), W2parent)
    end

    # Read-only aliases and consumption of BOTH mutated outputs must compose.
    f = function (x)
        W = reshape(x[1:15], 3, 5)
        weights = view(W, :, 1)
        W2, I = similar(W), similar(x, 5)
        H._one_halo_integral!(I, W2, W, weights, x[16])
        return sum(I .* [0.7, 0.8, 0.9, 1.1, 1.3]) + 0.13sum(W2)
    end
    x = collect(range(0.2, 1.8; length=16))
    mc, fd = AutoMooncake(), AutoForwardDiff()
    mp, fp = prepare_gradient(f, mc, x), prepare_gradient(f, fd, x)
    original = copy(gradient(f, mp, mc, x))
    for point in (x, x .+ 0.01, x)
        saved = copy(point)
        value, grad = value_and_gradient(f, mp, mc, point)
        @test value == f(point)
        @test grad ≈ gradient(f, fp, fd, point) rtol=1e-12 atol=1e-12
        @test point == saved
    end
    @test gradient(f, mp, mc, x) == original

    # Reject output/read overlap before any primal mutation occurs.
    W = ones(3, 5)
    args = Mooncake.zero_fcodual.((view(W, 1, :), zeros(3, 5), W, ones(3), 1.2))
    @test_throws ArgumentError Mooncake.rrule!!(Mooncake.zero_fcodual(H._one_halo_integral!), args...)
    @test W == ones(3, 5)
end
