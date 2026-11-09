using JET

@testset "JET native Julia analysis" begin
    @testset "Interpolation and grid helpers" begin
        logx = log.([1.0, 2.0, 4.0])
        logy = log.([2.0, 4.0, 8.0])
        @test_call target_modules=(Mapse,) Mapse._hmcode_loglog_interp(
            logx, logy, 3.0)
        @test_opt target_modules=(Mapse,) Mapse._hmcode_loglog_interp(
            logx, logy, 3.0)

        @test_call target_modules=(Mapse,) Mapse.build_smart_coarse_grid(
            0.0, 3.0, 24, 1.2)
        @test_opt target_modules=(Mapse,) Mapse.build_smart_coarse_grid(
            0.0, 3.0, 24, 1.2)

        bounds = [0.0, 3.0, 1.2]
        @test_call target_modules=(Mapse,) Mapse.build_piecewise_coarse_grid(
            bounds, 24, 14)
        @test_opt target_modules=(Mapse,) Mapse.build_piecewise_coarse_grid(
            bounds, 24, 14)
    end

    @testset "Native Halofit" begin
        params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
        cosmo = Mapse.halofit_cosmology(params)
        k = collect(exp.(range(log(1.0e-4), log(10.0); length=33)))
        z = [0.0, 0.5, 1.0]
        pk = [
            1.0e4 * (k_value / 0.1)^0.96 * exp(-k_value / 5) /
            (1 + z_value)^2
            for k_value in k, z_value in z
        ]
        expansion2 = cosmo.Ωm0 .* (1 .+ z) .^ 3 .+ cosmo.ΩΛ0
        omega_m = cosmo.Ωm0 .* (1 .+ z) .^ 3 ./ expansion2
        omega_v = cosmo.ΩΛ0 ./ expansion2

        @test_call target_modules=(Mapse,) Mapse.halofit_Pmm(
            cosmo, z, k, pk, omega_m, omega_v)
        @test_opt target_modules=(Mapse,) Mapse.halofit_Pmm(
            cosmo, z, k, pk, omega_m, omega_v)
    end

    @testset "Native HMCode" begin
        h = 0.6736
        omega_b = 0.02237 / h^2
        omega_nu = 0.06 / (93.14 * h^2)
        omega_m = omega_b + 0.12 / h^2 + omega_nu
        cosmo = Mapse.HMCodeCosmology(
            omega_m, omega_b, h, 0.9649, 0.8109118,
            -1.0, 0.0, omega_nu, 0.0)
        k = collect(10.0 .^ range(-3.0, 1.0; length=24))
        z = [0.0, 0.5]
        pk = k .^ -2.8 .* reshape([1.0, 0.7], 1, :)

        @test_call target_modules=(Mapse, Mapse.HMcode) Mapse.hmcode_Pmm(
            cosmo, z, k, pk;
            pk_cb_z=pk, T_AGN=nothing, nM=8, threaded=false)

        z_coarse = collect(range(0.0, 1.0; length=6))
        z_fine = collect(range(0.0, 1.0; length=9))
        pk_coarse = k .^ -2.8 .* reshape(
            1.0 ./ (1 .+ z_coarse) .^ 2, 1, :)
        @test_call target_modules=(Mapse, Mapse.HMcode) Mapse.hmcode_pmm_fast(
            cosmo, z_coarse, z_fine, k, pk_coarse;
            pk_cb_coarse=pk_coarse, T_AGN=nothing,
            nM=8, threaded=false)
    end
end
