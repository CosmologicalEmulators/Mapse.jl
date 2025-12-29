module Mapse

using Base: @kwdef
using Adapt
using AbstractCosmologicalEmulators
import AbstractCosmologicalEmulators.get_emulator_description
using DataInterpolations
import JSON.parsefile
import NPZ.npzread
using OrdinaryDiffEqTsit5
using Integrals
using FastGaussQuadrature
using LinearAlgebra
using SciMLSensitivity

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

end # module
