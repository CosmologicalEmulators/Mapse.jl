# Public-path regressions, not merely tests of the custom-rule boundaries.
# Match the accepted compute-node probe's spectrum, grid and active inputs.
# Compiled ordinary and AD evaluations can differ by a few rounding bits.
# Allow only roundoff here; primitive primal checks and A/B/A replay stay exact.
if !isdefined(@__MODULE__, :_hmcode_public_mooncake_problem)
    include("hmcode_public_mooncake_helpers.jl")
end

@testset "HMCode public direct DMO Mooncake" begin
    f, x = _hmcode_public_mooncake_problem()
    _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-5, atol=1e-7,
                          primal_rtol=8eps(Float64))
end

@testset "HMCode public fast DMO Mooncake" begin
    f, x = _hmcode_public_mooncake_problem(; fast=true)
    _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-5, atol=1e-7,
                          primal_rtol=8eps(Float64))
end

@testset "HMCode public direct feedback Mooncake" begin
    f, x = _hmcode_public_mooncake_problem(; feedback=true)
    _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-5, atol=1e-7,
                          primal_rtol=8eps(Float64))
end

@testset "HMCode public fast feedback Mooncake" begin
    f, x = _hmcode_public_mooncake_problem(; fast=true, feedback=true)
    _check_hmcode_mooncake(f, [x, x .+ 1e-4, x]; rtol=1e-5, atol=1e-7,
                          primal_rtol=8eps(Float64))
end
