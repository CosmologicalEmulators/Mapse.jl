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

- `Preprocessing::Function`, the `Function` used for the preprocessing of the input features

- `Postprocessing::Function`, the `Function` used for the postprocessing of the NN output
"""
@kwdef mutable struct LinearPkEmulator <: AbstractPkEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Preprocessing::Function
    Postprocessing::Function
end

Adapt.@adapt_structure LinearPkEmulator

"""
    NonLinearBoostPkEmulator(TrainedEmulator::AbstractTrainedEmulators, kgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix, Postprocessing::Function)

Emulator for the non-linear boost factor.
"""
@kwdef mutable struct NonLinearBoostPkEmulator <: AbstractPkEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Postprocessing::Function
end

Adapt.@adapt_structure NonLinearBoostPkEmulator

"""
    get_Pk(input_params, z, D, LinPkemu::LinearPkEmulator)
Computes and returns the linear power spectrum on the ``k``-grid the emulator has been trained on and the input ``z``, given input array `input_params`.

"""
function get_Pk(input_params, z::Number, D::Number, LinPkemu::LinearPkEmulator)
    preprocessed_input = LinPkemu.Preprocessing(input_params)
    input = vcat(preprocessed_input, z)
    norm_input = maximin(input, LinPkemu.InMinMax)
    output = Array(run_emulator(norm_input, LinPkemu.TrainedEmulator))
    norm_output = inv_maximin(output, LinPkemu.OutMinMax)
    return LinPkemu.Postprocessing(input_params, norm_output, D, LinPkemu)
end

function get_Pk(input_params, z::AbstractVector, D::AbstractVector, LinPkemu::LinearPkEmulator)
    preprocessed_input = LinPkemu.Preprocessing(input_params)
    input = vcat(repeat(preprocessed_input, 1, length(z)), reshape(z, 1, :))
    norm_input = maximin(input, LinPkemu.InMinMax)
    output = Array(run_emulator(norm_input, LinPkemu.TrainedEmulator))
    norm_output = inv_maximin(output, LinPkemu.OutMinMax)
    return LinPkemu.Postprocessing(input_params, norm_output, D, LinPkemu)
end

"""
    get_Pk(input_params, z, BoostEmu::NonLinearBoostPkEmulator)
Computes and returns the non-linear boost on the ``k``-grid the emulator has been trained on and the input ``z``, given input array `input_params`.
"""
function get_Pk(input_params, z::Number, BoostEmu::NonLinearBoostPkEmulator)
    input = vcat(input_params, z)
    norm_input = maximin(input, BoostEmu.InMinMax)
    output = Array(run_emulator(norm_input, BoostEmu.TrainedEmulator))
    norm_output = inv_maximin(output, BoostEmu.OutMinMax)
    return BoostEmu.Postprocessing(input_params, norm_output, BoostEmu)
end

function get_Pk(input_params, z::AbstractVector, BoostEmu::NonLinearBoostPkEmulator)
    input = vcat(repeat(input_params, 1, length(z)), reshape(z, 1, :))
    norm_input = maximin(input, BoostEmu.InMinMax)
    output = Array(run_emulator(norm_input, BoostEmu.TrainedEmulator))
    norm_output = inv_maximin(output, BoostEmu.OutMinMax)
    return BoostEmu.Postprocessing(input_params, norm_output, BoostEmu)
end

"""
    PkEmulator(LinearPmm::LinearPkEmulator, LinearPkcb::LinearPkEmulator, Boost::NonLinearBoostPkEmulator)

Master emulator struct that combines linear matter, linear c+b, and non-linear boost emulators.
"""
@kwdef mutable struct PkEmulator <: AbstractPkEmulators
    LinearPmm::LinearPkEmulator
    LinearPkcb::LinearPkEmulator
    Boost::NonLinearBoostPkEmulator
end

Adapt.@adapt_structure PkEmulator

"""
    get_Pk(input_params, z, D, PkEmu::PkEmulator)
Computes the final non-linear matter power spectrum by combining the linear matter part and the boost factor.
Returns ``P_{mm, lin}(k, z) \\times Boost(k, z)``.
"""
function get_Pk(input_params, z, D, PkEmu::PkEmulator)
    lin_pmm = get_Pk(input_params, z, D, PkEmu.LinearPmm)
    boost = get_Pk(input_params, z, PkEmu.Boost)
    return lin_pmm .* boost
end

"""
    get_linear_Pmm(input_params, z, D, PkEmu::PkEmulator)
Returns only the linear matter power spectrum part of the PkEmulator.
"""
function get_linear_Pmm(input_params, z, D, PkEmu::PkEmulator)
    return get_Pk(input_params, z, D, PkEmu.LinearPmm)
end

"""
    get_linear_Pkcb(input_params, z, D, PkEmu::PkEmulator)
Returns only the linear c+b power spectrum part of the PkEmulator.
"""
function get_linear_Pkcb(input_params, z, D, PkEmu::PkEmulator)
    return get_Pk(input_params, z, D, PkEmu.LinearPkcb)
end

"""
    get_kgrid(PkEmulator::AbstractPkEmulators)
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
- `preprocessing_file`, default `preprocessing.jl`
- `postprocessing_file`, default `postprocessing.jl`
If the corresponding file in the folder you are trying to load have different names,
 change the default values accordingly.
"""
function load_emulator(path::String; emu = SimpleChainsEmulator,
    structure = LinearPkEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    preprocessing_file = "preprocessing.jl", postprocessing_file = "postprocessing.jl")
    NN_dict = parsefile(path*nn_setup_file)
    k = npzread(path*k_file)

    weights = npzread(path*weights_file)
    trained_emu = Mapse.init_emulator(NN_dict, weights, emu)

    if structure == LinearPkEmulator
        Pk_emu = Mapse.LinearPkEmulator(TrainedEmulator = trained_emu, kgrid = k,
                                 InMinMax = npzread(path*inminmax_file),
                                 OutMinMax = npzread(path*outminmax_file),
                                 Preprocessing = include(path*preprocessing_file),
                                 Postprocessing = include(path*postprocessing_file))
    elseif structure == NonLinearBoostPkEmulator
        Pk_emu = Mapse.NonLinearBoostPkEmulator(TrainedEmulator = trained_emu, kgrid = k,
                                 InMinMax = npzread(path*inminmax_file),
                                 OutMinMax = npzread(path*outminmax_file),
                                 Postprocessing = include(path*postprocessing_file))
    else
        throw(ArgumentError("Unknown emulator structure: $structure"))
    end
    return Pk_emu
end
