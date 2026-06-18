using Test
using Reactant
using Mapse

@testset "Halofit Reactant support" begin
    Reactant.set_default_backend("cpu")

    ext = Base.get_extension(Mapse, :MapseReactantExt)
    @test !isnothing(ext)
    @test !occursin("@allowscalar", read(joinpath(pkgdir(Mapse), "ext", "MapseReactantExt.jl"), String))

    params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
    cpar = Mapse.halofit_cosmology(params)
    k = collect(exp.(range(log(1e-4), log(10.0), length=33)))
    z = collect(range(0.0, 2.0, length=100))
    @test length(z) == 100
    pk_lin = [1e4 * (kk / 0.1)^0.96 * exp(-kk / 5) / (1 + zz)^2
              for kk in k, zz in z]
    Ez2 = cpar.Ωm0 .* (1 .+ z) .^ 3 .+ cpar.ΩΛ0
    Ωm_z = cpar.Ωm0 .* (1 .+ z) .^ 3 ./ Ez2
    Ωv_z = cpar.ΩΛ0 ./ Ez2

    one_z_ref = Mapse._halofit_Pmm_one_z_unchecked(cpar, z[1], k,
                                                   view(pk_lin, :, 1),
                                                   Ωm_z[1], Ωv_z[1])
    all_z_ref = Mapse._halofit_Pmm_unchecked(cpar, z, k, pk_lin, Ωm_z, Ωv_z)

    kR = Reactant.to_rarray(k)
    pk1R = Reactant.to_rarray(pk_lin[:, 1])

    compiled_one_z = Reactant.@compile sync=true Mapse.halofit_Pmm(cpar, z[1], kR,
                                                                   pk1R, Ωm_z[1],
                                                                   Ωv_z[1])
    one_z_R = compiled_one_z(cpar, z[1], kR, pk1R, Ωm_z[1], Ωv_z[1])
    Reactant.synchronize(one_z_R)
    @test Array(one_z_R) ≈ one_z_ref rtol=1e-8 atol=1e-8

    zR = Reactant.to_rarray(z)
    pkR = Reactant.to_rarray(pk_lin)
    ΩmR = Reactant.to_rarray(Ωm_z)
    ΩvR = Reactant.to_rarray(Ωv_z)

    compiled_all_z = Reactant.@compile sync=true Mapse.halofit_Pmm(cpar, zR, kR,
                                                                   pkR, ΩmR, ΩvR)
    all_z_R = compiled_all_z(cpar, zR, kR, pkR, ΩmR, ΩvR)
    Reactant.synchronize(all_z_R)
    all_z = Array(all_z_R)
    @test size(all_z) == (length(k), length(z))
    @test all(isfinite, all_z)
    @test all_z ≈ all_z_ref rtol=1e-8 atol=1e-8
end
