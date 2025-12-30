abstract type AbstractPkEmulators end

"""
    LinearPkEmulator(TrainedEmulator::AbstractTrainedEmulators, kgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix)

This is the fundamental struct used to obtain the ``C_\\ell``'s from an emulator.
It contains:

- `TrainedEmulator::AbstractTrainedEmulators`, the trained emulator

- `kgrid::AbstractVector`, the ``k``-grid the emulator has been trained on.

- `InMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the input features

- `OutMinMax::AbstractMatrix`, the `Matrix` used for the MinMax normalization of the output features

- `Postprocessing::Function`, the `Function` used for the postprocessing of the NN output
"""
@kwdef mutable struct LinearPkEmulator <: AbstractPkEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure LinearPkEmulator


"""
    get_Pk(input_params, z, D, LinPkemu::LinearPkEmulator)
Computes and returns the linear power spectrum on the ``k``-grid the emulator has been trained on and the input ``z``, given input array `input_params`.

"""
function get_Pk(input_params, z::Number, D::Number, LinPkemu::LinearPkEmulator)
    input = vcat(input_params, z)
    norm_input = maximin(input, LinPkemu.InMinMax)
    output = Array(run_emulator(norm_input, LinPkemu.TrainedEmulator))
    norm_output = inv_maximin(output, LinPkemu.OutMinMax)
    return LinPkemu.Postprocessing(input_params, norm_output, D, LinPkemu)
end

function get_Pk(input_params, z::AbstractVector, D::AbstractVector, LinPkemu::LinearPkEmulator)
    input = reduce(hcat, [vcat(input_params, zi) for zi in z])
    norm_input = maximin(input, LinPkemu.InMinMax)
    output = Array(run_emulator(norm_input, LinPkemu.TrainedEmulator))
    norm_output = inv_maximin(output, LinPkemu.OutMinMax)
    return LinPkemu.Postprocessing(input_params, norm_output, D, LinPkemu)
end

"""
    get_kgrid(PkEmulator::AbstractCℓEmulators)
Returns the ``k``-grid the emulator has been trained on.
"""
function get_kgrid(PkEmulator::AbstractPkEmulators)
    return PkEmulator.kgrid
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
- `weights_file`, default `weights.npy`
- `inminmax_file`, default `inminmax.npy`
- `outminmax_file`, default `outminmax.npy`
- `nn_setup_file`, default `nn_setup.json`
- `postprocessing_file`, default `postprocessing.jl`
If the corresponding file in the folder you are trying to load have different names,
 change the default values accordingly.
"""
function load_emulator(path::String; emu = SimpleChainsEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    postprocessing_file = "postprocessing.jl")
    NN_dict = parsefile(path*nn_setup_file)
    k = npzread(path*k_file)

    weights = npzread(path*weights_file)
    trained_emu = Mapse.init_emulator(NN_dict, weights, emu)
    Pk_emu = Mapse.LinearPkEmulator(TrainedEmulator = trained_emu, kgrid = k,
                             InMinMax = npzread(path*inminmax_file),
                             OutMinMax = npzread(path*outminmax_file),
                             Postprocessing = include(path*postprocessing_file))
    return Pk_emu
end
