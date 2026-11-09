using Test
using Mapse

@testset "HMCode physical-unit public API" begin
    @test !isdefined(Mapse, :hmcode_pmm_physical)
    @test !isdefined(Mapse, :hmcode_pmm_fast_physical)
    z = collect(range(0.0, 1.0; length=6))
    z_fine = collect(range(0.0, 1.0; length=11))
    k = collect(10.0 .^ range(-4, 2; length=64))
    k_out = k[1:4:end]
    pmm = (1.0e4 .* (k ./ 0.1).^0.96 ./ (1 .+ (k ./ 0.1).^2).^2) ./
        reshape((1 .+ z).^2, 1, :)
    pcb = pmm .* 1.01

    # Hold the physical inputs fixed while changing h. The reference calls
    # the internal h-unit numerical kernel, independently of public wrappers.
    for h in (0.55, 0.6736, 0.88), temperature in (nothing, 10.0^7.8)
        cosmo = Mapse.HMCodeCosmology(0.3, 0.05, h, 0.96, 0.8,
            -1.0, 0.0, 0.001, 0.0)
        pk_interp, _ = Mapse._hmcode_linear_interpolators(z, k ./ h, pmm .* h^3)
        _, sigma_interp = Mapse._hmcode_linear_interpolators(z, k ./ h, pcb .* h^3)
        expected = Mapse.HMcode.hmcode_power(k_out ./ h, z,
            pk_interp, sigma_interp, cosmo;
            k_support=k ./ h, T_AGN=temperature, nM=8, threaded=false) ./ h^3

        prediction = Mapse.hmcode_pmm(cosmo, z, k_out, pmm;
            k_support=k, pk_cb_z=pcb, T_AGN=temperature, nM=8)
        @test prediction ≈ expected rtol=1.0e-12
        @test Mapse.hmcode_Pmm(cosmo, z, k_out, k, pmm;
            pk_cb_support_z=pcb, T_AGN=temperature, nM=8) ≈ expected rtol=1.0e-12
        @test Mapse.hmcode_pmm(cosmo, first(z), k_out, pmm[:, 1];
            k_support=k, pk_cb=pcb[:, 1], T_AGN=temperature, nM=8) ≈ expected[:, 1] rtol=1.0e-12
        @test Mapse.hmcode_boost(cosmo, z, k_out, pmm;
            k_support=k, pk_cb_z=pcb, T_AGN=temperature, nM=8) ≈
            expected ./ pmm[1:4:end, :] rtol=1.0e-12

        expected_fast = copy(transpose(Mapse.AbstractCosmologicalEmulators.akima_interpolation(
            copy(transpose(expected)), z, z_fine)))
        @test Mapse.hmcode_pmm_fast(cosmo, z, z_fine, k_out, pmm;
            k_support=k, pk_cb_support_coarse=pcb, T_AGN=temperature, nM=8) ≈ expected_fast rtol=1.0e-12
        @test Mapse.hmcode_pmm_fast(cosmo, z, z_fine, k_out, k, pmm;
            pk_cb_coarse=pcb, T_AGN=temperature, nM=8) ≈ expected_fast rtol=1.0e-12
    end
end
