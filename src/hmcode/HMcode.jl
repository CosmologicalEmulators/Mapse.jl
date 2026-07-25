module HMcode

using Base.Threads
using DataInterpolations
using Integrals
using Interpolations
using LinearAlgebra
using LoopVectorization
using OrdinaryDiffEqCore
using OrdinaryDiffEqTsit5
using Polyester
using Roots
using SpecialFunctions

export HMcodeCosmology, hmcode_power

include("cosmology.jl")
include("linear.jl")
include("profiles.jl")
include("feedback.jl")
include("power_spectrum.jl")

end # module
