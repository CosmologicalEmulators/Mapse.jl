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

const halofit_pmm = halofit_Pmm
const hmcode_pmm = hmcode_Pmm
const hmcode_Pmm_fast = hmcode_pmm_fast
const hmcode_Pmm_baryonic_smart = hmcode_pmm_baryonic_smart

export TransferFunctionEmulator, load_emulator,
    artifact_path,
    get_Pk, get_kgrid, get_emulator_description,
    compute_pca, save_pca_metadata, BUILTIN_PREPROCESSING, BUILTIN_POSTPROCESSING,
    LOAD_PRESETS, DEFAULT_EMULATOR_NAME, DEFAULT_EMULATOR_ARTIFACT,
    TRAINED_EMULATOR_ARTIFACTS, lcdm_transfer_function, preprocessing_drop_primordial_parameters,
    postprocessing_lcdm_transfer_ratio,
    HalofitCosmology, halofit_cosmology, halofit_background, halofit_Pmm, halofit_pmm,
    HMCodeCosmology, hmcode_Pmm, hmcode_pmm, hmcode_pmm_physical, hmcode_pmm_fast_physical, hmcode_boost, hmcode_pmm_fast, hmcode_pmm_fast_two_splines, hmcode_Pmm_fast, hmcode_boost_fast, validate_hmcode_fast_grids,
    predict_baryonic_discontinuity, build_smart_coarse_grid, build_piecewise_coarse_grid, build_baryonic_coarse_grid, hmcode_pmm_baryonic_smart, hmcode_Pmm_baryonic_smart, hmcode_pmm_dmo_smart

function growth_at(z, params, growth_emu)
    input = vcat(reshape(z, 1, :), repeat(reshape(params, :, 1), 1, length(z)))
    return vec(AbstractCosmologicalEmulators.run_emulator(input, growth_emu)[6:6, :])
end

function hmcode_pmm_dmo_smart(params::AbstractVector, z_fine::AbstractVector, z_limits::AbstractVector, k_out::AbstractVector,
    pmm_emu::TransferFunctionEmulator, pcb_emu::TransferFunctionEmulator, growth_emu,
    cosmo::HMCodeCosmology; N_coarse::Int=32, N_left::Int=16, nM::Int=128)

    zc = collect(range(z_limits[1], z_limits[2], length=N_coarse))
    
    growth_coarse = growth_at(zc, params, growth_emu)
    pmm_coarse = get_Pk(params, zc, growth_coarse, pmm_emu)
    pcb_coarse = get_Pk(params, zc, growth_coarse, pcb_emu)
    
    h = cosmo.h
    k_h = k_out ./ h
    
    k_support = get_kgrid(pmm_emu)
    @show size(pmm_coarse), length(k_h), length(k_support)
    result_h = hmcode_pmm_fast(cosmo, zc, z_fine, k_h, pmm_coarse .* h^3;
        pk_cb_coarse = pcb_coarse .* h^3,
        k_support = k_support ./ h,
        T_AGN = nothing,
        nM = nM)
        
    return result_h ./ h^3
end

end # module
