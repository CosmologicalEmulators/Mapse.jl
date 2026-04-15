abstract type AbstractPkEmulators end

abstract type AbstractCompression end
struct NoCompression <: AbstractCompression end
@kwdef struct PCACompression{T<:AbstractVector, M<:AbstractMatrix} <: AbstractCompression
    mean::T
    basis::M
end

Adapt.@adapt_structure PCACompression

reconstruct(y, ::NoCompression) = y
reconstruct(y, c::PCACompression) = c.mean .+ c.basis * y

"""
    MapseEmulator(TrainedEmulator::AbstractTrainedEmulators, kgrid::Array,
    InMinMax::Matrix, OutMinMax::Matrix, Preprocessing::Function, Postprocessing::Function,
    Compression::AbstractCompression)

Fundamental struct for Mapse emulators (Linear or Boost).
"""
@kwdef mutable struct MapseEmulator <: AbstractPkEmulators
    TrainedEmulator::AbstractTrainedEmulators
    kgrid::AbstractVector
    InMinMax::AbstractMatrix
    OutMinMax::AbstractMatrix
    Preprocessing::Function
    Postprocessing::Function
    Compression::AbstractCompression = NoCompression()
end

Adapt.@adapt_structure MapseEmulator

const NonLinearBoostPkEmulator = MapseEmulator

"""
    get_Pk(input_params, z, D, emu::MapseEmulator)
Computes and returns the power spectrum (or boost) given input parameters, redshift, and growth factor.
"""
function get_Pk(input_params, z::Number, D::Union{Number, Nothing}, emu::MapseEmulator)
    preprocessed_input = emu.Preprocessing(input_params)
    input = vcat(z, preprocessed_input)
    norm_input = maximin(input, emu.InMinMax)
    output = Array(run_emulator(norm_input, emu.TrainedEmulator))
    denorm_output = inv_maximin(output, emu.OutMinMax)
    reconstructed_output = reconstruct(denorm_output, emu.Compression)
    return emu.Postprocessing(input_params, reconstructed_output, D, emu)
end

function get_Pk(input_params, z::Number, emu::MapseEmulator)
    return get_Pk(input_params, z, nothing, emu)
end

function get_Pk(input_params, z::AbstractVector, D::AbstractVector, emu::MapseEmulator)
    preprocessed_input = emu.Preprocessing(input_params)
    input = vcat(reshape(z, 1, :), repeat(preprocessed_input, 1, length(z)))
    norm_input = maximin(input, emu.InMinMax)
    output = Array(run_emulator(norm_input, emu.TrainedEmulator))
    denorm_output = inv_maximin(output, emu.OutMinMax)
    reconstructed_output = reconstruct(denorm_output, emu.Compression)
    return emu.Postprocessing(input_params, reconstructed_output, D, emu)
end

function get_Pk(input_params, z::AbstractVector, emu::MapseEmulator)
    return get_Pk(input_params, z, fill(nothing, length(z)), emu)
end

"""
    PkEmulator(LinearPmm::MapseEmulator, LinearPcb::MapseEmulator, Boost::MapseEmulator)

Master emulator struct that combines linear matter, linear c+b, and non-linear boost emulators.
"""
@kwdef mutable struct PkEmulator <: AbstractPkEmulators
    LinearPmm::MapseEmulator
    LinearPcb::MapseEmulator
    Boost::MapseEmulator
end

Adapt.@adapt_structure PkEmulator

"""
    get_Pk(input_params, z, D, PkEmu::PkEmulator)
Computes the final non-linear matter power spectrum.
Returns ``P_{mm, lin}(k, z) \times Boost(k, z)``.
"""
function get_Pk(input_params, z, D, PkEmu::PkEmulator)
    lin_pmm = get_Pk(input_params, z, D, PkEmu.LinearPmm)
    boost = get_Pk(input_params, z, D, PkEmu.Boost)
    return lin_pmm .* boost
end

function get_linear_Pmm(input_params, z, D, PkEmu::PkEmulator)
    return get_Pk(input_params, z, D, PkEmu.LinearPmm)
end

function get_linear_Pkcb(input_params, z, D, PkEmu::PkEmulator)
    return get_Pk(input_params, z, D, PkEmu.LinearPcb)
end

function get_kgrid(Pkemu::AbstractPkEmulators)
    return Pkemu.kgrid
end

function get_emulator_description(Pkemu::AbstractPkEmulators)
    if haskey(Pkemu.TrainedEmulator.Description, "emulator_description")
        get_emulator_description(Pkemu.TrainedEmulator)
    else
        @warn "No emulator description found!"
    end
    return nothing
end

"""
    load_component_emulator(path::String; emu = LuxEmulator, ...)
Load a single MapseEmulator component from a directory.
"""
function load_component_emulator(path::String; emu = LuxEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    preprocessing_file = "preprocessing.jl", postprocessing_file = "postprocessing.jl",
    pca_mean_file = "pca_mean.npy", pca_basis_file = "pca_projection.npy")

    NN_dict = parsefile(path*nn_setup_file)
    k = npzread(path*k_file)
    weights = npzread(path*weights_file)
    trained_emu = Mapse.init_emulator(NN_dict, weights, emu)

    compression = if isfile(path*pca_mean_file) && isfile(path*pca_basis_file)
        PCACompression(
            mean = npzread(path*pca_mean_file),
            basis = npzread(path*pca_basis_file)
        )
    else
        NoCompression()
    end

    return Mapse.MapseEmulator(
        TrainedEmulator = trained_emu,
        kgrid = k,
        InMinMax = npzread(path*inminmax_file),
        OutMinMax = npzread(path*outminmax_file),
        Preprocessing = include(path*preprocessing_file),
        Postprocessing = include(path*postprocessing_file),
        Compression = compression
    )
end

"""
    load_emulator(path::String; emu = LuxEmulator, ...)
Load the master PkEmulator suite from a directory containing component subfolders.
"""
function load_emulator(path::String;
    emu = LuxEmulator,
    pmm_folder = "Pk_lin_mm/", pcb_folder = "Pk_lin_cb/", boost_folder = "Boost/")

    pmm = load_component_emulator(joinpath(path, pmm_folder); emu=emu)
    pcb = load_component_emulator(joinpath(path, pcb_folder); emu=emu)
    boost = load_component_emulator(joinpath(path, boost_folder); emu=emu)

    return PkEmulator(LinearPmm=pmm, LinearPcb=pcb, Boost=boost)
end

"""
    compute_pca(data::AbstractMatrix, n_components::Int)
Computes PCA on the training targets.
Returns: mean vector, basis matrix, and PCA coefficients.
"""
function compute_pca(data::AbstractMatrix, n_components::Int)
    μ = mean(data, dims=2)
    centered_data = data .- μ
    U, S, V = svd(centered_data)
    basis = U[:, 1:n_components]
    coefficients = basis' * centered_data
    return μ[:, 1], basis, coefficients
end

"""
    save_pca_metadata(path::String, mean::AbstractVector, basis::AbstractMatrix)
Saves PCA metadata needed for reconstruction.
"""
function save_pca_metadata(path::String, mean::AbstractVector, basis::AbstractMatrix)
    npzwrite(joinpath(path, "pca_mean.npy"), mean)
    npzwrite(joinpath(path, "pca_basis.npy"), basis)
end
