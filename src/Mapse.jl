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
# Loaded to activate AbstractCosmologicalEmulators.BackgroundCosmologyExt.
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
include("halofit.jl")
include("hmcode/HMcode.jl")
include("hmcode.jl")

function __init__()
    _init_halofit_Fν_spline!()
end

export TransferFunctionEmulator, load_emulator,
    artifact_path,
    get_Pk, get_kgrid, get_emulator_description,
    compute_pca, save_pca_metadata, BUILTIN_PREPROCESSING, BUILTIN_POSTPROCESSING,
    LOAD_PRESETS, DEFAULT_EMULATOR_NAME, DEFAULT_EMULATOR_ARTIFACT,
    TRAINED_EMULATOR_ARTIFACTS, preprocessing_linear_pk_mnuw0wacdm,
    postprocessing_linear_pk_mnuw0wacdm_sym_ratio,
    HalofitCosmology, halofit_cosmology, halofit_background, halofit_Pmm,
    HMCodeCosmology, hmcode_Pmm, hmcode_boost

end # module
