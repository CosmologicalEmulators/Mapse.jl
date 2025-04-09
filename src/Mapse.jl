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

const c_0 = 2.99792458e5

function __init__()
    min_y = _get_y(0,0) #obvious, I knowadd OrdinaryDiffEqTsit5
    max_y = _get_y(1,10)
    y_grid = vcat(LinRange(min_y, 100, 100), LinRange(100.1, max_y, 1000))
    F_grid = [_F(y) for y in y_grid]
    global F_interpolant = AkimaInterpolation(F_grid, y_grid)
    y_grid = vcat(LinRange(min_y, 10., 10000), LinRange(10.1, max_y, 10000))
    dFdy_grid = [_dFdy(y) for y in y_grid]
    global dFdy_interpolant = AkimaInterpolation(dFdy_grid, y_grid)
end

include("background.jl")
include("neural_networks.jl")

end # module
