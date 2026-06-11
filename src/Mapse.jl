module Mapse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
using DataInterpolations
import JSON.parsefile
import NPZ: npzread, npzwrite
using Statistics
using OrdinaryDiffEqTsit5
using Integrals
using FastGaussQuadrature
using LinearAlgebra
using SciMLSensitivity
import Pkg

const c_0 = 2.99792458e5

# Get the BackgroundCosmologyExt extension
const ext = Base.get_extension(AbstractCosmologicalEmulators, :BackgroundCosmologyExt)

# Import from extension if available
if !isnothing(ext)
    using .ext: AbstractCosmology, w0waCDMCosmology, D_z, D_f_z, f_z, E_z, E_a, d̃A_z, dM_z, dA_z, dL_z, r_z
    # Re-export background cosmology functions for user convenience
    export AbstractCosmology, w0waCDMCosmology, D_z, D_f_z, f_z, E_z, E_a, d̃A_z, dM_z, dA_z, dL_z, r_z
else
    @warn "BackgroundCosmologyExt extension not loaded. Background cosmology functions will not be available."
end


include("neural_networks.jl")
include("primordial.jl")

const trained_emulators = Dict{String, PkEmulator}()

function __init__()
    empty!(trained_emulators)
    for (emulator_name, artifact_name) in TRAINED_EMULATOR_ARTIFACTS
        trained_emulators[emulator_name] = load_emulator_from_artifact(artifact_name)
    end
end

export MapseEmulator, PkEmulator, load_component_emulator, load_emulator,
    load_emulator_from_artifact, artifact_path,
    get_Pk, get_linear_Pmm, get_linear_Pkcb, get_kgrid, get_emulator_description,
    compute_pca, save_pca_metadata, BUILTIN_PREPROCESSING, BUILTIN_POSTPROCESSING,
    LOAD_PRESETS, DEFAULT_EMULATOR_NAME, DEFAULT_EMULATOR_ARTIFACT,
    TRAINED_EMULATOR_ARTIFACTS, preprocessing_linear_pk_mnuw0wacdm,
    preprocessing_boost_mnuw0wacdm,
    postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
    postprocessing_boost_log10, trained_emulators

end # module
