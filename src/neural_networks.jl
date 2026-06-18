abstract type AbstractPkEmulators end

abstract type AbstractCompression end
struct NoCompression <: AbstractCompression end
@kwdef struct PCACompression{T<:AbstractVector, M<:AbstractMatrix} <: AbstractCompression
    mean::T
    basis::M
end

Adapt.@adapt_structure PCACompression

reconstruct(y, ::NoCompression) = y
reconstruct(y::AbstractVector, c::PCACompression) = c.mean .+ c.basis * y
reconstruct(y::AbstractMatrix, c::PCACompression) = c.mean .+ c.basis * y

"""
    preprocessing_linear_pk_mnuw0wacdm(params)

Preprocessing used by ``mnuw0wacdm`` linear-power-spectrum MAPSE emulators.
The expected input order is
``[ln10As, ns, H0, ombh2, omch2, Mν, w0, wa]``; the linear
networks receive all parameters except the primordial amplitude and tilt.
"""
function preprocessing_linear_pk_mnuw0wacdm(params::AbstractVector)
    return params[3:end]
end

"""
    preprocessing_boost_mnuw0wacdm(params)

Preprocessing used by ``mnuw0wacdm`` nonlinear boost MAPSE emulators.
The boost networks receive the full cosmological parameter vector.
"""
function preprocessing_boost_mnuw0wacdm(params::AbstractVector)
    return params
end

"""
    postprocessing_linear_pk_mnuw0wacdm_sym_ratio(params, output, D, emu)

Postprocessing used by ``mnuw0wacdm`` sym-ratio linear-power-spectrum MAPSE
emulators. It applies the primordial spectrum, the growth factor squared, and
the analytic ``DIFF^2`` correction used by the matching jaxmapse artifacts.
"""
function postprocessing_linear_pk_mnuw0wacdm_sym_ratio(params::AbstractVector, output, D, emu)
    ln10As = params[1]
    ns = params[2]
    As = exp(ln10As) * 1e-10

    ωb = params[4]
    ωc = params[5]
    Mν = params[6]

    k = get_kgrid(emu)
    P_prim = primordial_Pk(As, ns, k)

    log10_k = log10.(k)
    ων = Mν / 93.14
    Δω = ωc + ων - ωb
    ωm = ωb + ωc + ων

    DIFF = exp.(0.4971733969600907 .+ (-24.849067935704547 .- log.((((((0.731102574104348 .^ log10_k) .+ Δω) ./ 0.17522861267519874) .^ log10_k) .+ ((63.65597287231169 .^ (log10_k .+ 0.0472474783701488)) .* ((0.9899093975978591 .^ (log10_k ./ (cos.(log10_k ./ ((1.1964213875807956 ^ -2.3661897652294015) ./ cos.(log10_k ./ -1.8173117588773222))) ./ 0.20037856443385513))) ./ (Δω ^ 0.7767030041348179)))) .+ (0.14823981687164764 * ωm))))

    D2 = D isa AbstractVector ? reshape(D .^ 2, 1, :) : D^2
    return (output .* DIFF .^ 2) .* D2 .* P_prim
end

"""
    postprocessing_boost_log10(params, output, D, emu)

Postprocessing used by MAPSE boost emulators trained in log10-space.
"""
function postprocessing_boost_log10(params::AbstractVector, output, D, emu)
    return 10.0 .^ output
end

const BUILTIN_PREPROCESSING = Dict{String, Function}(
    "identity" => identity,
    "linear_pk_mnuw0wacdm" => preprocessing_linear_pk_mnuw0wacdm,
    "boost_mnuw0wacdm" => preprocessing_boost_mnuw0wacdm,
)

const BUILTIN_POSTPROCESSING = Dict{String, Function}(
    "identity" => (input_params, output, D, emu) -> output,
    "linear_pk_mnuw0wacdm_sym_ratio" => postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
    "boost_log10" => postprocessing_boost_log10,
)

const LOAD_PRESETS = Dict{Symbol, NamedTuple}(
    :mnuw0wacdm_class => (
        linear_preprocessing_name = :linear_pk_mnuw0wacdm,
        linear_postprocessing_name = :linear_pk_mnuw0wacdm_sym_ratio,
        boost_preprocessing_name = :boost_mnuw0wacdm,
        boost_postprocessing_name = :boost_log10,
    ),
)

const ARTIFACTS_TOML = joinpath(dirname(@__DIR__), "Artifacts.toml")

const DEFAULT_EMULATOR_NAME = "mnuw0wacdm_class"

const DEFAULT_EMULATOR_ARTIFACT = "mnuw0wacdm_class"

const TRAINED_EMULATOR_ARTIFACTS = Dict{String, String}(
    DEFAULT_EMULATOR_NAME => DEFAULT_EMULATOR_ARTIFACT,
)

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
    _validate_compatible_pk_grids(PkEmu)
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

"""
    get_kgrid(Pkemu::AbstractPkEmulators)

Return the k-grid the emulator was trained on.
"""
function get_kgrid(Pkemu::AbstractPkEmulators)
    return Pkemu.kgrid
end

"""
    get_emulator_description(Pkemu::AbstractPkEmulators)

Print the metadata description stored in the trained emulator, when available.
"""
function get_emulator_description(Pkemu::AbstractPkEmulators)
    if haskey(Pkemu.TrainedEmulator.Description, "emulator_description")
        get_emulator_description(Pkemu.TrainedEmulator)
    else
        @warn "No emulator description found!"
    end
    return nothing
end

_function_name(name::Nothing) = nothing
_function_name(name::Symbol) = String(name)
_function_name(name::AbstractString) = String(name)

function _function_name(name)
    throw(ArgumentError("Function names must be strings or symbols; got $(typeof(name))."))
end

function _metadata_name(nn_dict::AbstractDict, key::String, explicit_name = nothing)
    if !isnothing(explicit_name)
        return _function_name(explicit_name)
    end

    name = get(nn_dict, key, nothing)
    if isnothing(name) && haskey(nn_dict, "emulator_description")
        name = get(nn_dict["emulator_description"], key, nothing)
    end
    return _function_name(name)
end

function _load_component_function(path::String, nn_dict::AbstractDict, key::String,
    file::String, registry::AbstractDict{String, Function}, role::String, explicit_name = nothing)

    name = _metadata_name(nn_dict, key, explicit_name)
    if !isnothing(name)
        if haskey(registry, name)
            return registry[name]
        end
        throw(ArgumentError(
            "$(role) function '$(name)' was requested in $(joinpath(path, "nn_setup.json")), " *
            "but it is not registered. Register it in the corresponding Mapse.BUILTIN_* dictionary, " *
            "or remove the metadata entry and provide $(file)."
        ))
    end

    function_path = joinpath(path, file)
    if !isfile(function_path)
        throw(ArgumentError("Missing $(role) function for MAPSE component: expected $(function_path)."))
    end

    return include(function_path)
end

function _load_compression(path::String, pca_mean_file::String, pca_basis_file::String)
    pca_mean_path = joinpath(path, pca_mean_file)
    pca_basis_path = joinpath(path, pca_basis_file)
    legacy_pca_basis_path = joinpath(path, "pca_basis.npy")

    if !isfile(pca_basis_path) && pca_basis_file == "pca_projection.npy" && isfile(legacy_pca_basis_path)
        @warn "Using legacy PCA basis filename pca_basis.npy; prefer pca_projection.npy for jaxmapse compatibility." path
        pca_basis_path = legacy_pca_basis_path
    end

    if isfile(pca_mean_path) && isfile(pca_basis_path)
        return PCACompression(
            mean = npzread(pca_mean_path),
            basis = npzread(pca_basis_path),
        )
    elseif isfile(pca_mean_path) || isfile(pca_basis_path)
        throw(ArgumentError("PCA metadata is incomplete in $(path): expected both $(pca_mean_file) and $(basename(pca_basis_path))."))
    else
        return NoCompression()
    end
end

function _validate_component_shapes(path::String, k, inminmax, outminmax,
    compression::AbstractCompression, nn_dict::AbstractDict)

    n_input = Int(nn_dict["n_input_features"])
    n_output = Int(nn_dict["n_output_features"])

    size(inminmax) == (n_input, 2) || throw(DimensionMismatch(
        "$(path): inminmax.npy has size $(size(inminmax)); expected ($(n_input), 2)."
    ))
    size(outminmax) == (n_output, 2) || throw(DimensionMismatch(
        "$(path): outminmax.npy has size $(size(outminmax)); expected ($(n_output), 2)."
    ))

    if compression isa NoCompression
        length(k) == n_output || throw(DimensionMismatch(
            "$(path): k-grid has length $(length(k)) but uncompressed NN output has $(n_output) features."
        ))
    elseif compression isa PCACompression
        length(compression.mean) == length(k) || throw(DimensionMismatch(
            "$(path): pca_mean.npy has length $(length(compression.mean)); expected k-grid length $(length(k))."
        ))
        size(compression.basis) == (length(k), n_output) || throw(DimensionMismatch(
            "$(path): PCA projection has size $(size(compression.basis)); expected ($(length(k)), $(n_output))."
        ))
    end

    return nothing
end

function _validate_compatible_pk_grids(PkEmu::PkEmulator)
    linear_k = get_kgrid(PkEmu.LinearPmm)
    boost_k = get_kgrid(PkEmu.Boost)

    if length(linear_k) != length(boost_k)
        throw(DimensionMismatch(
            "Cannot compute nonlinear P(k): LinearPmm grid has $(length(linear_k)) points " *
            "but Boost grid has $(length(boost_k)) points. Use get_linear_Pmm/get_linear_Pkcb, " *
            "or provide a boost emulator on the same k-grid."
        ))
    end

    if !all(linear_k .== boost_k)
        throw(ArgumentError(
            "Cannot compute nonlinear P(k): LinearPmm and Boost k-grids have the same length " *
            "but different values. Use matching component artifacts or interpolate explicitly before multiplying."
        ))
    end

    return nothing
end

function _load_preset(preset::Nothing)
    return NamedTuple()
end

function _load_preset(preset::Symbol)
    if haskey(LOAD_PRESETS, preset)
        return LOAD_PRESETS[preset]
    end
    throw(ArgumentError("Unknown MAPSE load preset :$(preset). Available presets are: $(join(sort!(String.(keys(LOAD_PRESETS))), ", "))."))
end

function _load_preset(preset::AbstractString)
    return _load_preset(Symbol(preset))
end

function _preset_value(preset::NamedTuple, key::Symbol, fallback = nothing)
    return haskey(preset, key) ? getfield(preset, key) : fallback
end

function _component_function_name(component_name, shared_name, preset::NamedTuple,
    component_key::Symbol, shared_key::Symbol)

    if !isnothing(component_name)
        return component_name
    elseif !isnothing(shared_name)
        return shared_name
    else
        return _preset_value(preset, component_key, _preset_value(preset, shared_key, nothing))
    end
end

"""
    load_component_emulator(path::String; emu = LuxEmulator, ...)
Load a single MapseEmulator component from a directory.
"""
function load_component_emulator(path::String; emu = LuxEmulator,
    k_file = "k.npy", weights_file = "weights.npy", inminmax_file = "inminmax.npy",
    outminmax_file = "outminmax.npy", nn_setup_file = "nn_setup.json",
    preprocessing_file = "preprocessing.jl", postprocessing_file = "postprocessing.jl",
    pca_mean_file = "pca_mean.npy", pca_basis_file = "pca_projection.npy",
    preprocessing_name = nothing, postprocessing_name = nothing)

    NN_dict = parsefile(joinpath(path, nn_setup_file))
    if !haskey(NN_dict, "emulator_description")
        NN_dict["emulator_description"] = Dict{String, Any}()
    end
    k = npzread(joinpath(path, k_file))
    weights = npzread(joinpath(path, weights_file))
    trained_emu = Mapse.init_emulator(NN_dict, weights, emu)

    inminmax = npzread(joinpath(path, inminmax_file))
    outminmax = npzread(joinpath(path, outminmax_file))
    compression = _load_compression(path, pca_mean_file, pca_basis_file)

    _validate_component_shapes(path, k, inminmax, outminmax, compression, NN_dict)

    return Mapse.MapseEmulator(
        TrainedEmulator = trained_emu,
        kgrid = k,
        InMinMax = inminmax,
        OutMinMax = outminmax,
        Preprocessing = _load_component_function(path, NN_dict, "preprocessing_name", preprocessing_file,
            BUILTIN_PREPROCESSING, "preprocessing", preprocessing_name),
        Postprocessing = _load_component_function(path, NN_dict, "postprocessing_name", postprocessing_file,
            BUILTIN_POSTPROCESSING, "postprocessing", postprocessing_name),
        Compression = compression
    )
end

"""
    load_emulator(path::String; emu = LuxEmulator, ...)
Load the master PkEmulator suite from a directory containing component subfolders.
"""
function load_emulator(path::String;
    emu = LuxEmulator,
    pmm_folder = "Pk_lin_mm/", pcb_folder = "Pk_lin_cb/", boost_folder = "Boost/",
    preset = nothing,
    linear_preprocessing_name = nothing, linear_postprocessing_name = nothing,
    pmm_preprocessing_name = nothing, pmm_postprocessing_name = nothing,
    pcb_preprocessing_name = nothing, pcb_postprocessing_name = nothing,
    boost_preprocessing_name = nothing, boost_postprocessing_name = nothing)

    load_preset = _load_preset(preset)

    pmm_preprocessing_name = _component_function_name(pmm_preprocessing_name, linear_preprocessing_name,
        load_preset, :pmm_preprocessing_name, :linear_preprocessing_name)
    pmm_postprocessing_name = _component_function_name(pmm_postprocessing_name, linear_postprocessing_name,
        load_preset, :pmm_postprocessing_name, :linear_postprocessing_name)
    pcb_preprocessing_name = _component_function_name(pcb_preprocessing_name, linear_preprocessing_name,
        load_preset, :pcb_preprocessing_name, :linear_preprocessing_name)
    pcb_postprocessing_name = _component_function_name(pcb_postprocessing_name, linear_postprocessing_name,
        load_preset, :pcb_postprocessing_name, :linear_postprocessing_name)
    boost_preprocessing_name = _component_function_name(boost_preprocessing_name, nothing,
        load_preset, :boost_preprocessing_name, :boost_preprocessing_name)
    boost_postprocessing_name = _component_function_name(boost_postprocessing_name, nothing,
        load_preset, :boost_postprocessing_name, :boost_postprocessing_name)

    pmm = load_component_emulator(joinpath(path, pmm_folder); emu=emu,
        preprocessing_name = pmm_preprocessing_name,
        postprocessing_name = pmm_postprocessing_name)
    pcb = load_component_emulator(joinpath(path, pcb_folder); emu=emu,
        preprocessing_name = pcb_preprocessing_name,
        postprocessing_name = pcb_postprocessing_name)
    boost = load_component_emulator(joinpath(path, boost_folder); emu=emu,
        preprocessing_name = boost_preprocessing_name,
        postprocessing_name = boost_postprocessing_name)

    return PkEmulator(LinearPmm=pmm, LinearPcb=pcb, Boost=boost)
end

function _artifact_emulator_root(path::AbstractString)
    if isdir(joinpath(path, "Pk_lin_mm")) && isdir(joinpath(path, "Pk_lin_cb")) && isdir(joinpath(path, "Boost"))
        return String(path)
    end

    subdirs = filter(isdir, readdir(path; join=true))
    if length(subdirs) == 1
        candidate = subdirs[1]
        if isdir(joinpath(candidate, "Pk_lin_mm")) && isdir(joinpath(candidate, "Pk_lin_cb")) && isdir(joinpath(candidate, "Boost"))
            return candidate
        end
    end

    return String(path)
end

"""
    artifact_path([artifact_name]; artifacts_toml = Mapse.ARTIFACTS_TOML)

Install, if necessary, and return the local path of a MAPSE trained-emulator
artifact declared in `Artifacts.toml`.
"""
function artifact_path(artifact_name::AbstractString = DEFAULT_EMULATOR_ARTIFACT;
    artifacts_toml::AbstractString = ARTIFACTS_TOML)

    installed = Pkg.Artifacts.ensure_artifact_installed(String(artifact_name), String(artifacts_toml))
    return installed isa AbstractString ? installed : Pkg.Artifacts.artifact_path(installed)
end

"""
    load_emulator_from_artifact([artifact_name]; kwargs...)

Load a `PkEmulator` from a trained-emulator artifact declared in
`Artifacts.toml`. Keyword arguments are forwarded to `load_emulator`.
"""
function load_emulator_from_artifact(artifact_name::AbstractString = DEFAULT_EMULATOR_ARTIFACT;
    artifacts_toml::AbstractString = ARTIFACTS_TOML, kwargs...)

    path = _artifact_emulator_root(artifact_path(artifact_name; artifacts_toml))
    return load_emulator(path; kwargs...)
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
    npzwrite(joinpath(path, "pca_projection.npy"), basis)
end
