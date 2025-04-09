module Mapse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
import JSON.parsefile
import NPZ.npzread

abstract type AbstractPkEmulators end

"""
    LinearPkEmulator(TrainedEmulator::AbstractTrainedEmulators, kgrid::Array, zgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

This is the fundamental struct used to obtain the ``C_\\ell``'s from an emulator.
It contains:

- `TrainedEmulator::AbstractTrainedEmulators`, the trained emulator

- `kgrid::AbstractVector`, the ``k``-grid the emulator has been trained on.

- `zgrid::AbstractVector`, the ``z``-grid the emulator has been trained on.

- `InMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the input features

- `OutMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the output features

- `Postprocessing::Function`, the `Function` used for the postprocessing of the NN output
"""
@kwdef mutable struct LinearPkEmulator <: AbstractCℓEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::AbstractVector
    zgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure LinearPkEmulator

#"""
#    get_Cℓ(input_params, Cℓemu::AbstractCℓEmulators)
#Computes and returns the ``C_\\ell``'s on the ``\\ell``-grid the emulator has been trained on given input array `input_params`.

#"""
#function get_Pk(input_params, Cℓemu::AbstractCℓEmulators)
#    norm_input = maximin(input_params, Cℓemu.InMinMax)
#    output = Array(run_emulator(norm_input, Cℓemu.TrainedEmulator))
#    norm_output = inv_maximin(output, Cℓemu.OutMinMax)
#    return Cℓemu.Postprocessing(input_params, norm_output, Cℓemu)
#end

"""
    get_kgrid(PkEmulator::AbstractCℓEmulators)
Returns the ``k``-grid the emulator has been trained on.
"""
function get_kgrid(PkEmulator::AbstractPkEmulators)
    return PkEmulator.kgrid
end

"""
    get_zgrid(PkEmulator::AbstractPkEmulators)
Returns the ``z``-grid the emulator has been trained on.
"""
function get_zgrid(PkEmulator::AbstractPkEmulators)
    return PkEmulator.zgrid
end

"""
    get_emulator_description(PkEmulator::AbstractPkEmulators)
Print on screen the emulator description.
"""
function get_emulator_description(Pkemu::AbstractPkEmulators)
    if haskey(Pkemu.TrainedEmulator.Description, "emulator_description")
        get_emulator_description(Pkemu.TrainedEmulator)
    else
        @warn "No emulator description found!"
    end
    return nothing
end

"""
    load_emulator(path::String, emu_backend::AbstractTrainedEmulators)
Load the emulator with the files in the folder `path`, using the backend defined by `emu_backend`.
The following keyword arguments are used to specify the name of the files used to load the emulator:
- `k_file`, default `k.npy`
- `z_file`, default `z.npy`
- `weights_file`, default `weights.npy`
- `inminmax_file`, default `inminmax.npy`
- `outminmax_file`, default `outminmax.npy`
- `nn_setup_file`, default `nn_setup.json`
- `postprocessing_file`, default `postprocessing.jl`
If the corresponding file in the folder you are trying to load have different names,
 change the default values accordingly.
"""
function load_emulator(path::String; emu = SimpleChainsEmulator,
    k_file = "k.npy", z_file = "z.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl")
    NN_dict = parsefile(path*nn_setup_file)
    k = npzread(path*k_file)
    z = npzread(path*z_file)

    weights = npzread(path*weights_file)
    trained_emu = Capse.init_emulator(NN_dict, weights, emu)
    Pk_emu = Mapse.LinearPkEmulator(TrainedEmulator = trained_emu, kgrid = k, zgrid = z,
                             InMinMax = npzread(path*inminmax_file),
                             OutMinMax = npzread(path*outminmax_file),
                             Postprocessing = include(path*postprocessing_file))
    return Pk_emu
end

end # module
