using Test, Mapse, Mooncake, ForwardDiff, DifferentiationInterface, Random

# Independent primal expressions with a branch selected from Float64 inputs.
# ForwardDiff comparisons can use tangent values to break ties at exactly 4/12;
# that would differentiate a different branch, not the one taken by the primal.
_nfw_reference_branch(x) = x <= 4.0 ? 1 : x*x <= 144.0 ? 2 : 3
function _nfw_reference_sici(x, branch)
    H = Mapse.HMcode
    if branch == 1
        return H.si_small(x), H.EULER_GAMMA + log(x) + H.ci_int_small(x)
    end
    u = inv(x*x)
    if branch == 2
        f = (evalpoly(u, H.P_F_RAT1) / evalpoly(u, H.Q_F_RAT1)) / x
        g = (evalpoly(u, H.R_G_RAT1) / evalpoly(u, H.S_G_RAT1)) * u
    else
        f = (1.0 - evalpoly(u, H.P_F_ASYM_NUM) * u / evalpoly(u, H.P_F_ASYM_DEN)) / x
        g = (1.0 - evalpoly(u, H.R_G_ASYM_NUM) * u / evalpoly(u, H.R_G_ASYM_DEN)) * u
    end
    sx, cx = sin(x), cos(x)
    return pi/2.0 - f*cx - g*sx, f*sx - g*cx
end

function _nfw_reference_window(p, plus_branch, minus_branch)
    x, c, l = p
    xp, xm = x * (1.0 + 1.0/c), x/c
    sp, cp = _nfw_reference_sici(xp, plus_branch)
    sm, cm = _nfw_reference_sici(xm, minus_branch)
    ds, dc = sp-sm, cp-cm
    return (ds*sin(xm) + dc*cos(xm) - sin(x)/xp) / (l-c/(1.0+c))
end

@testset "Analytical Si/Ci approximation derivatives" begin
    H = Mapse.HMcode
    ext = Base.get_extension(Mapse, :MapseMooncakeExt)
    fd = AutoForwardDiff()
    si = ntuple(b -> x -> _nfw_reference_sici(x, b)[1], 3)
    ci = ntuple(b -> x -> _nfw_reference_sici(x, b)[2], 3)
    ps = map(f -> prepare_derivative(f, fd, 1.0), si)
    pc = map(f -> prepare_derivative(f, fd, 1.0), ci)
    points = vcat(10.0 .^ range(-8, 5; length=81),
                  [prevfloat(4.0), 4.0, nextfloat(4.0),
                   prevfloat(12.0), 12.0, nextfloat(12.0)])
    for x in points
        s, c, ds, dc = ext._sici_with_derivatives(x)
        b = _nfw_reference_branch(x)
        @testset "argument=$x branch=$b" begin
            @test (s, c) == H.sici_fast(x) == _nfw_reference_sici(x, b)
            @test ds ≈ derivative(si[b], ps[b], fd, x) rtol=1e-10 atol=1e-12
            @test dc ≈ derivative(ci[b], pc[b], fd, x) rtol=1e-10 atol=1e-12
        end
    end
end

@testset "Analytical NFW scalar partials" begin
    H = Mapse.HMcode
    ext = Base.get_extension(Mapse, :MapseMooncakeExt)
    fd = AutoForwardDiff()
    refs = [p -> _nfw_reference_window(p, bp, bm) for bp in 1:3, bm in 1:3]
    points = [[x, c, log1p(c)] for x in 10.0 .^ range(-8, 4; length=25)
              for c in (0.1, 0.5, 1.0, 3.0, 10.0, 50.0)]
    rng = MersenneTwister(20261109)
    for _ in 1:128
        x, c = 10.0^(-5 + 9rand(rng)), 10.0^(-1 + 3rand(rng))
        push!(points, [x, c, log1p(c) + 0.01rand(rng)])
    end
    for cut in (4.0, 12.0), q in (prevfloat(cut), cut, nextfloat(cut)),
        c in (0.5, 3.0, 10.0), x in (q * c, q / (1 + 1/c))
        push!(points, [x, c, log1p(c)])
    end
    for x in (prevfloat(1e-3), 1e-3, nextfloat(1e-3)), c in (0.1, 1.0, 3.0)
        push!(points, [x, c, log1p(c)])
    end
    # Float64 ForwardDiff itself loses small-x digits (e.g. x=1e-8,c=0.1).
    # Use the same primal polynomials and fixed branches at higher precision,
    # not an exact-special-function substitute or a relaxed tolerance.
    setprecision(BigFloat, 256) do
        preps = map(f -> prepare_gradient(f, fd, BigFloat.([0.1, 3.0, log1p(3.0)])), refs)
        for p in points
            got = collect(ext._nfw_partials(p...))
            bp = _nfw_reference_branch(p[1] * (1.0 + 1.0/p[2]))
            bm = _nfw_reference_branch(p[1] / p[2])
            expected = Float64.(gradient(refs[bp, bm], preps[bp, bm], fd, BigFloat.(p)))
            @testset "x=$(p[1]) c=$(p[2]) l=$(p[3])" begin
                @test refs[bp, bm](p) == H.wnfw_fast(p...)
                @test all(isfinite, got)
                for i in 1:3
                    @testset "partial=$i" begin
                        @test got[i] ≈ expected[i] rtol=1e-8 atol=1e-9
                    end
                end
                if p[1] <= 1e-4
                    @test got[1] ≈ expected[1] rtol=1e-8 atol=1e-18
                end
            end
        end
    end
end

@testset "NFW mutation and view pullback contract" begin
    for use_views in (false, true)
        Wparent = fill(7.3, 5, 6)
        W = use_views ? view(Wparent, 2:4, 2:5) : fill(7.3, 3, 4)
        k = use_views ? view([99.0, 1e-4, 0.1, 4.0, 20.0, 99.0], 2:5) : [1e-4, 0.1, 4.0, 20.0]
        r, c = [0.2, 0.8, 1.5], [3.0, 5.0, 7.0]
        l = log1p.(c)
        seeds = reshape(collect(range(-0.7, 1.3; length=12)), 3, 4)
        inputs = Mooncake.zero_fcodual.((W, k, r, c, l))
        old = copy(W)
        out, back = Mooncake.rrule!!(Mooncake.zero_fcodual(Mapse.HMcode.win_NFW_fast!), inputs...)
        @test Mooncake.primal(out) === W
        _, dw = Mooncake.arrayify(out)
        dw .= seeds
        back(Mooncake.NoRData())
        @test W == old
        @test all(iszero, dw)
        @test all(==(7.3), Wparent)
        f = function (p)
            result = zeros(eltype(p), 3, 4)
            Mapse.HMcode.win_NFW_fast!(result, p[1:4], p[5:7], p[8:10], p[11:13])
            return sum(result .* seeds)
        end
        x = vcat(k, r, c, l)
        expected = ForwardDiff.gradient(f, x)
        got = vcat((Mooncake.arrayify(a)[2] for a in inputs[2:end])...)
        @test all(isapprox.(got, expected; rtol=1e-8, atol=1e-9))
    end
    shared = function (x)
        a = exp.(x)
        W = zeros(eltype(x), 3, 3)
        Mapse.HMcode.win_NFW_fast!(W, a, a, a, log1p.(a))
        return sum(W .* reshape(collect(1.0:9.0), 3, 3))
    end
    x = log.([0.7, 1.3, 2.1])
    mc, fd = AutoMooncake(), AutoForwardDiff()
    mp, fp = prepare_gradient(shared, mc, x), prepare_gradient(shared, fd, x)
    saved = copy(gradient(shared, mp, mc, x))
    for p in (x, x .+ 1e-4, x)
        old = copy(p)
        @test all(isapprox.(gradient(shared, mp, mc, p), gradient(shared, fp, fd, p); rtol=1e-8, atol=1e-9))
        @test p == old
    end
    @test gradient(shared, mp, mc, x) == saved
    W = ones(3, 3)
    inputs = Mooncake.zero_fcodual.((W, view(W, :, 1), ones(3), ones(3), ones(3)))
    @test_throws ArgumentError Mooncake.rrule!!(Mooncake.zero_fcodual(Mapse.HMcode.win_NFW_fast!), inputs...)
end

@testset "NFW nonzero cotangents and strided parents" begin
    H = Mapse.HMcode
    for (nm, nk) in ((3, 5), (5, 3), (1, 4), (4, 1))
        rows, columns = 1:2:2nm, 1:2:2nk
        Wparent = fill(7.3, 2nm, 2nk)
        W = view(Wparent, rows, columns)
        values = (10.0 .^ (-4 .+ 2 .* (0:nk-1)),
                  [0.2 + 0.3i for i in 1:nm],
                  [3.0 + 0.4i for i in 1:nm])
        values = (values..., log1p.(values[3]))
        parents = map(v -> fill(9.1, 2length(v)), values)
        vectors = map(p -> view(p, 1:2:length(p)), parents)
        for (v, x) in zip(vectors, values)
            copyto!(v, x)
        end
        saved_parents = copy.(parents)
        args = Mooncake.zero_fcodual.((W, vectors...))
        shadows = map(a -> Mooncake.arrayify(a)[2], args)
        foreach(d -> fill!(parent(d), 0.71), shadows)
        for (i, d) in enumerate(shadows[2:end])
            d .= 0.13i
        end
        expected_parents = copy.(parent.(shadows))
        seeds = reshape(collect(range(-0.7, 1.3; length=nm*nk)), nm, nk)
        out, back = Mooncake.rrule!!(Mooncake.zero_fcodual(H.win_NFW_fast!), args...)
        @test Mooncake.primal(out) === W
        shadows[1] .= seeds
        back(Mooncake.NoRData())
        reference = function (p)
            A = zeros(eltype(p), nm, nk)
            H.win_NFW_fast!(A, p[1:nk], p[nk+1:nk+nm],
                p[nk+nm+1:nk+2nm], p[nk+2nm+1:nk+3nm])
            return sum(A .* seeds)
        end
        expected = ForwardDiff.gradient(reference, vcat(values...))
        expected_parents[1][rows, columns] .= 0.0
        offset = 0
        for i in 1:4
            n = length(values[i])
            expected_parents[i+1][1:2:2n] .+= expected[offset+1:offset+n]
            offset += n
        end
        for (d, expected_parent) in zip(shadows, expected_parents)
            @test parent(d) ≈ expected_parent rtol=1e-8 atol=1e-9
        end
        @test all(==(0.71), parent(shadows[1])[2:2:2nm, :])
        @test all(==(0.71), parent(shadows[1])[:, 2:2:2nk])
        for d in shadows[2:end]
            @test all(==(0.71), parent(d)[2:2:end])
        end
        @test all(==(7.3), Wparent)
        @test parents == saved_parents
    end
end

@testset "NFW overlapping input views and overwritten live values" begin
    overlap = function (x)
        a = exp.(x)
        W = zeros(eltype(x), 3, 3)
        Mapse.HMcode.win_NFW_fast!(W, view(a, 1:3), view(a, 2:4),
            view(a, 3:5), view(a, 4:6))
        return 0.19sum(abs2, x) + sum(W .* reshape(collect(1.0:9.0), 3, 3))
    end
    overwritten = function (x)
        W = reshape(x[1:12], 3, 4)
        before = 0.19sum(abs2, W)
        c = exp.(x[20:22])
        Mapse.HMcode.win_NFW_fast!(W, exp.(x[13:16]), exp.(x[17:19]), c, log1p.(c))
        return before + sum(W .* reshape(collect(range(-0.7, 1.3; length=12)), 3, 4))
    end
    points = (log.([0.7, 1.1, 1.3, 1.7, 2.1, 2.6]),
        vcat(collect(range(0.2, 1.2; length=12)), log.([1e-4, 0.1, 4.0, 20.0]),
             log.([0.2, 0.8, 1.5]), log.([3.0, 5.0, 7.0])))
    for (f, x) in zip((overlap, overwritten), points)
        mc, fd = AutoMooncake(), AutoForwardDiff()
        mp, fp = prepare_gradient(f, mc, x), prepare_gradient(f, fd, x)
        baseline = copy(gradient(f, mp, mc, x))
        for p in (x, x .+ 1e-4, x)
            saved = copy(p)
            value, g = value_and_gradient(f, mp, mc, p)
            @test value ≈ f(p) rtol=1e-12 atol=1e-12
            @test g ≈ gradient(f, fp, fd, p) rtol=1e-8 atol=1e-9
            @test p == saved
        end
        @test gradient(f, mp, mc, x) == baseline
    end
end

@testset "NFW selected branches at Si/Ci approximation boundaries" begin
    f = function (p)
        W = zeros(eltype(p), 1, 1)
        Mapse.HMcode.win_NFW_fast!(W, [one(p[1])], [p[1]], [p[2]], [p[3]])
        return W[1, 1]
    end
    mc, fd = AutoMooncake(), AutoForwardDiff()
    initial = [0.1, 3.0, log1p(3.0)]
    mp, fp = prepare_gradient(f, mc, initial), prepare_gradient(f, fd, initial)
    baseline = copy(gradient(f, mp, mc, initial))
    # Check the selected approximation branch, not smoothness across a boundary
    # or agreement with a different, exact-special-function model.
    for cut in (4.0, 12.0), u in (prevfloat(cut), cut, nextfloat(cut)),
        x in (3u, u / (1 + 1/3))
        p = [x, 3.0, log1p(3.0)]
        value, g = value_and_gradient(f, mp, mc, p)
        @test value ≈ f(p) rtol=1e-12 atol=1e-12
        @test all(isfinite, g)
        @test g ≈ gradient(f, fp, fd, p) rtol=1e-8 atol=1e-9
    end
    @test gradient(f, mp, mc, initial) == baseline
end
