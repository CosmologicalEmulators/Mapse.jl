using Test
using Mapse
using DifferentiationInterface
using ForwardDiff
using FiniteDiff
using DelimitedFiles

@testset "HMCode native AD review regressions" begin
    k = collect(10.0 .^ range(-4, 2; length=64))
    p = 1e4 .* (k ./ 0.1).^0.96 ./ (1 .+ (k ./ 0.1).^2).^2
    cosmo = Mapse.HMCodeCosmology(0.3, 0.05, 0.6736, 0.96, 0.8,
        -1.0, 0.0, 0.001, 0.0)
    ad = AutoForwardDiff()
    fd = AutoFiniteDiff(; fdtype=Val(:central))

    @testset "Growth depends only on active background parameters" begin
        names = fieldnames(typeof(cosmo))
        scale_factors = [0.1, 0.5, 1.0]
        reference_table = readdlm(joinpath(@__DIR__, "data",
            "hmcode_growth_gradient_reference.txt"); comments=true)
        references = Dict((row[1] == 1, Symbol(row[2])) => Float64.(row[3:5])
            for row in eachrow(reference_table))
        @testset "$name LCDM=$lcdm" for lcdm in (false, true), name in names
            index = findfirst(==(name), names)
            function growth_loss(x)
                c = Mapse.HMCodeCosmology(ntuple(
                    i -> i == index ? x[1] : getfield(cosmo, i), length(names))...)
                growth = Mapse.HMcode.get_growth_interpolator(c; LCDM=lcdm)
                return sum(growth, scale_factors)
            end
            point = [getfield(cosmo, name)]
            prep = prepare_gradient(growth_loss, ad, point)
            value, g = value_and_gradient(growth_loss, prep, ad, point)
            active = name == :Omega_m || (!lcdm && name in (:Omega_k, :w0, :wa))
            if active
                # Default-step FD of the production-tolerance adaptive ODE
                # does not resolve the curvature derivative. Use the saved
                # converged FD pair and tight-ODE AD cross-check instead.
                expected = references[(lcdm, name)]
                @test all(isfinite, g)
                @test any(!iszero, g)
                @test all(isapprox.(expected, expected[1]; rtol=1e-6, atol=1e-8))
                @test all(isapprox.(g[1], expected; rtol=1e-4, atol=5e-5))
            else
                # Identical numerical problem, not merely the same continuum
                # equation: an unused dual field must not alter the ODE path.
                @test value == growth_loss(point)
                @test g == [0.0]
                changed = point .+ 0.01
                changed_value, changed_g = value_and_gradient(growth_loss, prep, ad, changed)
                @test changed_value == value
                @test changed_g == [0.0]
            end
        end
    end

    @testset "Moving integration limits have analytic derivatives" begin
        nodes = collect(exp.(range(log(0.1), log(4.0); length=16)))
        pk, _ = Mapse._hmcode_linear_interpolators(
            [0.0], nodes, reshape(nodes .^ 2, :, 1))
        evaluator = Mapse.HMcode.PkLinEval(pk, 0.0, 1)
        velocity(x) = Mapse.HMcode.sigmaV(0.0, evaluator;
            kmin=x[1], kmax=x[2])
        x = [0.2, 2.0]
        prep = prepare_gradient(velocity, ad, x)
        value, g = value_and_gradient(velocity, prep, ad, x)
        expected = sqrt((x[2]^3 - x[1]^3) / (18π^2))
        @test value ≈ expected rtol=1e-12
        @test g ≈ [-x[1]^2, x[2]^2] ./ (12π^2 * expected) rtol=1e-10 atol=1e-12

        # g(a)=c*a^2 gives G(a)=c*(a^2+a_init^2)/2. Both the
        # integrand parameter and the upper integration limit are active.
        accumulated(x) = Mapse.HMcode.get_accumulated_growth(
            x[1], a -> x[2] * a^2)
        point = [0.6, 1.3]
        ap = prepare_gradient(accumulated, ad, point)
        av, ag = value_and_gradient(accumulated, ap, ad, point)
        @test av ≈ point[2] * (point[1]^2 + 1e-8) / 2 rtol=1e-12
        @test ag ≈ [point[1] * point[2], (point[1]^2 + 1e-8) / 2] rtol=1e-12 atol=1e-12
    end

    @testset "Explicit columns survive dual queries" begin
        for zs in ([0.0, 0.0], [0, 1])
            pk, sig = Mapse._hmcode_linear_interpolators(zs, k, hcat(p, 4 .* p))
            for iz in 1:2
                for evaluator in (Mapse.HMcode.SigmaREval(sig, zs[iz], iz),
                                  Mapse.HMcode.PkLinEval(pk, zs[iz], iz))
                    @test ForwardDiff.value(evaluator(ForwardDiff.Dual(1.0, 1.0))) ≈ evaluator(1.0) rtol=1e-12
                    f(x) = evaluator(x[1])
                    # Avoid a log-k knot when comparing interpolation slopes.
                    x = [0.83]
                    ap = prepare_gradient(f, ad, x)
                    fp = prepare_gradient(f, fd, x)
                    @test gradient(f, ap, ad, x) ≈ gradient(f, fp, fd, x) rtol=1e-4 atol=5e-5
                end
            end
        end
    end

    @testset "Isolated parameters and prepared A/B/A reuse" begin
        for varying in (:temperature, :omega_m, :h), scalar in (false, true)
            function loss(x, support)
                c = if varying == :omega_m
                    Mapse.HMCodeCosmology(
                        x[1], cosmo.Omega_b, cosmo.h, cosmo.n_s, cosmo.sigma_8,
                        cosmo.w0, cosmo.wa, cosmo.Omega_nu, cosmo.Omega_k)
                elseif varying == :h
                    Mapse.HMCodeCosmology(
                        cosmo.Omega_m, cosmo.Omega_b, x[1], cosmo.n_s, cosmo.sigma_8,
                        cosmo.w0, cosmo.wa, cosmo.Omega_nu, cosmo.Omega_k)
                else
                    cosmo
                end
                temperature = varying == :temperature ? 10.0^x[1] : 10.0^7.8
                prediction = if scalar
                    Mapse.hmcode_pmm(c, 0.0, k, p;
                        k_support=support, T_AGN=temperature, nM=8, threaded=false)
                else
                    Mapse.hmcode_pmm(c, [0.0], k, reshape(p, :, 1);
                        k_support=support, T_AGN=temperature, nM=8, threaded=false)
                end
                return sum(log, prediction) / length(prediction)
            end
            implicit(x) = loss(x, nothing)
            explicit(x) = loss(x, k)
            a, b = if varying == :temperature
                ([7.8], [7.9])
            elseif varying == :omega_m
                ([0.3], [0.31])
            else
                ([0.6736], [0.6836])
            end
            ip = prepare_gradient(implicit, ad, a)
            ep = prepare_gradient(explicit, ad, a)
            fp = prepare_gradient(implicit, fd, a)
            # Reuse the same preparations at changed inputs and then return
            # to A. Do not rebuild the cache and call that a reuse test.
            for x in (a, b, a)
                value, g = value_and_gradient(implicit, ip, ad, x)
                supported_value, supported_g = value_and_gradient(explicit, ep, ad, x)
                finite_value, finite_g = value_and_gradient(implicit, fp, fd, x)
                @test isfinite(value) && all(isfinite, g)
                @test value ≈ supported_value rtol=1e-10
                @test value ≈ finite_value rtol=1e-10
                @test all(isapprox.(g, supported_g; rtol=1e-4, atol=5e-5))
                @test all(isapprox.(g, finite_g; rtol=1e-4, atol=5e-5))
            end
        end
    end

    @testset "Integer and duplicate redshift spectrum sensitivities" begin
        function spectrum_loss(x, zs)
            spectrum = hcat(p .* x[1], 4 .* p .* x[2])
            prediction = Mapse.hmcode_pmm(cosmo, zs, k, spectrum;
                T_AGN=nothing, nM=8, threaded=false)
            return sum(log, prediction) / length(prediction)
        end
        x = [1.0, 1.0]
        gradients = Vector{Float64}[]
        for zs in ([0, 1], [0.0, 1.0], [0.0, 0.0])
            f(x) = spectrum_loss(x, zs)
            ap = prepare_gradient(f, ad, x)
            fp = prepare_gradient(f, fd, x)
            av, ag = value_and_gradient(f, ap, ad, x)
            fv, fg = value_and_gradient(f, fp, fd, x)
            @test av ≈ fv rtol=1e-10
            @test all(isfinite, ag)
            @test all(isapprox.(ag, fg; rtol=1e-4, atol=5e-5))
            push!(gradients, copy(ag))
        end
        @test gradients[1] ≈ gradients[2] rtol=1e-12 atol=1e-12
    end

    @testset "Boost denominators preserve column identity" begin
        indices = 1:8:length(k)
        kout = k[indices]
        spectrum = hcat(p, 4 .* p)
        for zs in ([0, 1], [2, 1], [0.0, 0.0])
            boost = Mapse.hmcode_boost(cosmo, zs, kout, spectrum;
                k_support=k, T_AGN=nothing, nM=8, threaded=false)
            nonlinear = Mapse.hmcode_pmm(cosmo, zs, kout, spectrum;
                k_support=k, T_AGN=nothing, nM=8, threaded=false)
            @test boost ≈ nonlinear ./ spectrum[indices, :] rtol=1e-10
            float_boost = Mapse.hmcode_boost(cosmo, Float64.(zs), kout, spectrum;
                k_support=k, T_AGN=nothing, nM=8, threaded=false)
            @test boost ≈ float_boost rtol=1e-12
        end
        zs = collect(0:5)
        spectra = p .* reshape(1.0 ./ (1 .+ zs).^2, 1, :)
        boost = Mapse.hmcode_boost_fast(cosmo, zs, kout, spectra, 5;
            k_support=k, T_AGN=nothing, nM=8, threaded=false)
        nonlinear = Mapse.hmcode_pmm_fast(cosmo, zs, kout, spectra, 5;
            k_support=k, T_AGN=nothing, nM=8, threaded=false)
        @test boost ≈ nonlinear ./ spectra[indices, :] rtol=1e-10
    end
end
