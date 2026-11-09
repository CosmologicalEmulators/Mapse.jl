using Test
using DelimitedFiles
using Reactant
using Mapse

@testset "Curved HMCode Reactant fixture" begin
    Reactant.set_default_backend("cpu")
    h = 0.6736
    Ωb = 0.02237 / h^2
    Ων = 0.06 / (93.14 * h^2)
    Ωm = Ωb + 0.12 / h^2 + Ων
    cosmo = Mapse.HMCodeCosmology(Ωm, Ωb, h, 0.9649, 0.8109118,
        -0.9, 0.2, Ων, 0.01)
    reference = readdlm(joinpath(@__DIR__, "data", "hmcode_curved_parity_reference.txt"); comments=true)
    # The immutable fixture is in h-units; every public call uses Mpc units.
    reference[:, 2] .*= h
    reference[:, 3:6] ./= h^3
    z = unique(reference[:, 1])
    k = collect(reference[reference[:, 1] .== first(z), 2])
    pmm = reshape(reference[:, 3], length(k), length(z))
    pcb = reshape(reference[:, 4], length(k), length(z))
    zR, kR, pmmR, pcbR = Reactant.to_rarray.((collect(z), k, pmm, pcb))
    for (temperature, column) in ((nothing, 5), (10.0^7.8, 6))
        prediction = Mapse.hmcode_Pmm(cosmo, zR, kR, pmmR;
            pk_cb_z=pcbR, T_AGN=temperature, nM=32)
        Reactant.synchronize(prediction)
        expected = reshape(reference[:, column], length(k), length(z))
        # Preserve the existing native/adaptive versus compiled/fixed-grid gate.
        @test Array(prediction) ≈ expected rtol=1.5e-2
    end
end
