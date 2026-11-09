using DifferentiationInterface
using FiniteDiff
using ForwardDiff

function _test_plain_julia_gradients(
    f, parameters;
    rtol=2.0e-5, atol=5.0e-5,
    value_rtol=1.0e-10, value_atol=1.0e-10,
)
    forward_backend = AutoForwardDiff()
    finite_backend = AutoFiniteDiff(; fdtype=Val(:central))
    forward_prep = prepare_gradient(f, forward_backend, parameters)
    finite_prep = prepare_gradient(f, finite_backend, parameters)
    forward_value, forward_gradient = value_and_gradient(
        f, forward_prep, forward_backend, parameters)
    finite_value, finite_gradient = value_and_gradient(
        f, finite_prep, finite_backend, parameters)

    @test isfinite(forward_value)
    @test isfinite(finite_value)
    @test all(isfinite, forward_gradient)
    @test all(isfinite, finite_gradient)
    @test forward_value ≈ finite_value rtol=value_rtol atol=value_atol
    # Compare every coordinate; a vector norm can hide a bad small derivative.
    @test all(isapprox.(forward_gradient, finite_gradient; rtol=rtol, atol=atol))
    return nothing
end

include("hmcode_pipeline_helpers.jl")

@testset "DifferentiationInterface plain Julia" begin
    parameters = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]

    @testset "Halofit" begin
        k = collect(exp.(range(log(1.0e-4), log(10.0); length=33)))
        z = [0.0, 0.5, 1.0]

        function halofit_loss(params)
            cosmo = Mapse.halofit_cosmology(params)
            pk = [
                1.0e4 * exp(params[1] - 3.044) *
                (k_value / 0.1)^params[2] * exp(-k_value / 5) /
                (1 + z_value)^2
                for k_value in k, z_value in z
            ]
            prediction = Mapse.halofit_Pmm(cosmo, z, k, pk)
            return sum(log, prediction) / length(prediction)
        end

        _test_plain_julia_gradients(halofit_loss, parameters)
    end

    emulator_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
    pmm_emulator = Mapse.load_emulator(joinpath(emulator_root, "Pk_lin_mm"))
    pcb_emulator = Mapse.load_emulator(joinpath(emulator_root, "Pk_lin_cb"))
    growth_emulator = Mapse.AbstractCosmologicalEmulators.trained_emulators[
        "ACE_mnuw0wacdm_ln10As_basis"]

    @testset "Linear emulators" begin
        z = collect(range(0.0, 1.0; length=5))

        function linear_emulator_loss(params)
            pmm, pcb = _di_native_linear_spectra(
                params, z, pmm_emulator, pcb_emulator, growth_emulator)
            return (sum(log, pmm) + sum(log, pcb)) /
                   (length(pmm) + length(pcb))
        end

        _test_plain_julia_gradients(linear_emulator_loss, parameters)
    end

    @testset "Baryonic coarse grid" begin
        baryonic_grid_loss(params) = sum(abs2,
            Mapse.build_baryonic_coarse_grid(
                params, [0.0, 3.0], [7.8], 24, 14)) / 24
        _test_plain_julia_gradients(
            baryonic_grid_loss, parameters; atol=5.0e-5)
    end

    @testset "Nonlinear radius implicit derivative" begin
        radius(x) = Mapse.HMcode.get_nonlinear_radius(
            0.1 + zero(x[1]), 10.0 + zero(x[1]), x[2], r -> x[1] / r)
        backend = AutoForwardDiff()
        point = [2.0, 1.5]
        prep = prepare_gradient(radius, backend, point)
        @test gradient(radius, prep, backend, point) ≈ [1 / point[2], -point[1] / point[2]^2] rtol=1e-12
        lower(x) = Mapse.HMcode.get_nonlinear_radius(x[1], 10.0, 100.0, r -> 1 / r)
        upper(x) = Mapse.HMcode.get_nonlinear_radius(zero(x[1]) + 0.1, x[1], 0.001, r -> 1 / r)
        for f in (lower, upper)
            x = [1.0]
            p = prepare_gradient(f, backend, x)
            @test gradient(f, p, backend, x) == [1.0]
        end
    end

    @testset "Native HMCode Float64 forward pass" begin
        fixture = readdlm(joinpath(
            @__DIR__, "data", "hmcode_forwarddiff_primal_reference.txt");
            comments=true)
        z = unique(fixture[:, 1])
        k = fixture[fixture[:, 1] .== first(z), 2]
        nk, nz = length(k), length(z)
        theta = vcat(parameters, 7.8)

        for (column, smart, feedback) in (
            (3, false, false),
            (4, true, false),
            (5, false, true),
            (6, true, true),
        )
            prediction = _di_native_hmcode_prediction(
                theta, z, k, pmm_emulator, pcb_emulator,
                growth_emulator, 0.8109118;
                smart=smart, feedback=feedback, nM=32)
            reference = reshape(fixture[:, column], nk, nz)
            # Local before/after development checks are bit-exact. Package
            # test environments may re-resolve SciML and emulator patch
            # versions, which can change the final one or two Float64 bits.
            @test prediction ≈ reference rtol=1.0e-13 atol=0
        end
    end

    @testset "Native HMCode direct feedback gradient" begin
        theta = vcat(parameters, 7.8)
        z = collect(range(0.0, 1.0; length=8))
        k = collect(10.0 .^ range(-3.0, 0.5; length=8)) .* (parameters[3] / 100)
        function direct_feedback_loss(values)
            prediction = _di_native_hmcode_prediction(
                values, z, k, pmm_emulator, pcb_emulator,
                growth_emulator, 0.8109118;
                smart=false, feedback=true, nM=8)
            return sum(log, prediction) / length(prediction)
        end

        _test_plain_julia_gradients(
            direct_feedback_loss, theta;
            rtol=1.0e-4, atol=5.0e-5,
            value_rtol=1.0e-10, value_atol=1.0e-10)
    end

    @testset "Native HMCode smart feedback gradient" begin
        theta = vcat(parameters, 7.8)
        z = collect(range(0.0, 1.0; length=32))
        k = collect(10.0 .^ range(-3.0, 0.5; length=8)) .* (parameters[3] / 100)
        function smart_feedback_loss(values)
            prediction = _di_native_hmcode_prediction(
                values, z, k, pmm_emulator, pcb_emulator,
                growth_emulator, 0.8109118;
                smart=true, feedback=true, nM=8)
            return sum(log, prediction) / length(prediction)
        end

        _test_plain_julia_gradients(
            smart_feedback_loss, theta;
            rtol=1.0e-4, atol=5.0e-5,
            value_rtol=1.0e-10, value_atol=1.0e-10)
    end

    @testset "Native HMCode DMO gradients" begin
        theta = vcat(parameters, 7.8)
        z = collect(range(0.0, 1.0; length=32))
        k = collect(10.0 .^ range(-3.0, 0.5; length=8)) .* (parameters[3] / 100)
        for smart in (false, true)
            loss(values) = sum(log, _di_native_hmcode_prediction(
                values, z, k, pmm_emulator, pcb_emulator,
                growth_emulator, 0.8109118;
                smart=smart, feedback=false, nM=8)) / (length(k) * length(z))
            _test_plain_julia_gradients(loss, theta; rtol=1.0e-4, atol=5.0e-5)
        end
    end

    @testset "Native HMCode spectrum-only gradients" begin
        z = [0.0, 0.5, 1.0]
        k_support = collect(Mapse.get_kgrid(pmm_emulator))
        k = k_support[1:40:end]
        pmm, pcb = _di_native_linear_spectra(
            parameters, z, pmm_emulator, pcb_emulator, growth_emulator)
        cosmo = _di_native_hmcode_cosmology(parameters, 0.8109118)
        # No dual-valued cosmology or k-grid is available to select buffers.
        loss(amplitudes) = sum(log, Mapse.hmcode_boost(
            cosmo, z, k, pmm .* amplitudes[1];
            pk_cb_z=pcb .* amplitudes[2], k_support=k_support,
            T_AGN=nothing, nM=8)) / (length(k) * length(z))
        _test_plain_julia_gradients(loss, [1.0, 1.0]; rtol=1.0e-4, atol=5.0e-5)
    end

    @testset "Native HMCode moving baryonic feature" begin
        z = collect(range(0.0, 3.0; length=32))
        k = collect(10.0 .^ range(-3.0, 0.5; length=8)) .* (parameters[3] / 100)
        function moving_feature_loss(values)
            prediction = _di_native_hmcode_prediction(
                values, z, k, pmm_emulator, pcb_emulator,
                growth_emulator, 0.8109118;
                smart=true, feedback=true, nM=8)
            return sum(log, prediction) / length(prediction)
        end
        for theta in (vcat(parameters, 7.8),
                      [3.05, 0.97, 68.0, 0.023, 0.125, 0.15, -0.9, 0.1, 8.0])
            feature = Mapse.predict_baryonic_discontinuity(theta[1:8]; T_AGN=10.0^theta[9])
            grid = Mapse.build_smart_coarse_grid(first(z), last(z), 24, feature)
            split = searchsortedlast(grid, feature)
            @test 5 <= split <= length(grid) - 4
            @test grid[split] == feature
            _test_plain_julia_gradients(moving_feature_loss, theta;
                rtol=5.0e-4, atol=5.0e-5)
        end
    end
end
