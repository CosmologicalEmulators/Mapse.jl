#!/usr/bin/env julia
#
# Required Julia benchmark: standard vs fast HMCode paths.
#
# Benchmarks Mapse.hmcode_Pmm (standard) and Mapse.hmcode_pmm_fast (fast) in
# DMO and baryonic-feedback modes for both native Julia and Reactant backends.
#
# Grids:
#   128 log-spaced k-values: 1e-3 to 1e1 (output grid)
#   200 log-spaced k-support: 1e-4 to 50  (sigma(R) integration grid)
#   300 fine redshifts:      0.0 to 3.5 (linear)
#   50  coarse redshifts:    0.0 to 3.5 (linear)
#
# Run from the Mapse.jl test environment:
#   julia --project=test benchmark/bench_std_vs_fast.jl
#

using BenchmarkTools
using Reactant
using Mapse
using NPZ
using Printf
using LinearAlgebra
using Statistics

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
println("=" ^ 70)
println("Julia HMCode standard-vs-fast benchmark")
println("=" ^ 70)
println("Julia:   $(VERSION)")
println("Backend: CPU")
try
    println("CPU:     $(Sys.cpu_info()[1].model)")
    println("Cores:   $(length(Sys.cpu_info()))")
catch
end
println("Threads: $(Threads.nthreads())")
println("BLAS:    $(BLAS.get_num_threads()) threads")
println("Mapse:   $(pkgversion(Mapse))")
try
    println("Reactant: $(pkgversion(Reactant))")
catch
    println("Reactant: version unavailable")
end
println("BenchmarkTools: $(pkgversion(BenchmarkTools))")

# ---------------------------------------------------------------------------
# Load common input spectra
# ---------------------------------------------------------------------------
input_file = joinpath(@__DIR__, "..", "..", "benchmark_inputs.npz")
data = npzread(input_file)
k_np = data["k"]                         # (128,)
k_support_np = data["k_support"]         # (200,)
z_fine_np = data["z_fine"]               # (300,)
z_coarse_np = data["z_coarse"]           # (50,)
pk_mm_fine_np = data["pk_mm_fine"]       # (300, 128) JAX layout (nz, nk)
pk_cb_fine_np = data["pk_cb_fine"]
pk_mm_coarse_np = data["pk_mm_coarse"]   # (50, 128)
pk_cb_coarse_np = data["pk_cb_coarse"]
pk_mm_fine_sup_np = data["pk_mm_fine_support"]       # (300, 200)
pk_cb_fine_sup_np = data["pk_cb_fine_support"]
pk_mm_coarse_sup_np = data["pk_mm_coarse_support"]   # (50, 200)
pk_cb_coarse_sup_np = data["pk_cb_coarse_support"]

# Julia convention: (nk, nz) — transpose from JAX (nz, nk)
k = Vector{Float64}(k_np)
k_support = Vector{Float64}(k_support_np)
z_fine = Vector{Float64}(z_fine_np)
z_coarse = Vector{Float64}(z_coarse_np)
pk_mm_fine = Matrix{Float64}(permutedims(pk_mm_fine_np))          # (128, 300)
pk_cb_fine = Matrix{Float64}(permutedims(pk_cb_fine_np))          # (128, 300)
pk_mm_coarse = Matrix{Float64}(permutedims(pk_mm_coarse_np))      # (128, 50)
pk_cb_coarse = Matrix{Float64}(permutedims(pk_cb_coarse_np))      # (128, 50)
pk_mm_fine_sup = Matrix{Float64}(permutedims(pk_mm_fine_sup_np))  # (200, 300)
pk_cb_fine_sup = Matrix{Float64}(permutedims(pk_cb_fine_sup_np))  # (200, 300)
pk_mm_coarse_sup = Matrix{Float64}(permutedims(pk_mm_coarse_sup_np))  # (200, 50)
pk_cb_coarse_sup = Matrix{Float64}(permutedims(pk_cb_coarse_sup_np))  # (200, 50)

println("\nInput spectra: $(basename(input_file))")
println("  Source: CAMB z=0 linear P(k) scaled by Carroll et al. (1992) growth factor")
println("  k:         $(length(k)) log-spaced from $(k[1]) to $(k[end])")
println("  k_support: $(length(k_support)) log-spaced from $(k_support[1]) to $(k_support[end])")
println("  z_fine:    $(length(z_fine)) linear from $(z_fine[1]) to $(z_fine[end])")
println("  z_coarse:  $(length(z_coarse)) linear from $(z_coarse[1]) to $(z_coarse[end])")
println("  pk_mm_fine:           $(size(pk_mm_fine)) (nk, nz)")
println("  pk_mm_fine_support:   $(size(pk_mm_fine_sup)) (nk_support, nz)")
println("  pk_mm_coarse:         $(size(pk_mm_coarse)) (nk, nz)")
println("  pk_mm_coarse_support: $(size(pk_mm_coarse_sup)) (nk_support, nz)")

# ---------------------------------------------------------------------------
# Cosmology
# ---------------------------------------------------------------------------
h = 0.6736
Omega_b = 0.02237 / h^2
Omega_nu = 0.06 / (93.14 * h^2)
Omega_m = Omega_b + 0.12 / h^2 + Omega_nu
n_s = 0.9649
sigma8 = 0.8109118
cosmo = Mapse.HMCodeCosmology(Omega_m, Omega_b, h, n_s, sigma8, -1.0, 0.0, Omega_nu, 0.0)

T_AGN_FEEDBACK = 10.0^7.8
nM = 64

println("\nCosmology: Omega_m=$(Omega_m), Omega_b=$(Omega_b), h=$(h), sigma_8=$(sigma8), n_s=$(n_s)")
println("Baryonic model: T_AGN = 10^7.8 = $(T_AGN_FEEDBACK)")
println("nM (mass grid): $(nM)")

# ---------------------------------------------------------------------------
# Helper: extract benchmark stats
# ---------------------------------------------------------------------------
function bench_stats(b::BenchmarkTools.Trial)
    med = median(b.times) / 1e6  # ns -> ms
    mn = minimum(b.times) / 1e6
    mx = maximum(b.times) / 1e6
    iqr = (quantile(b.times, 0.75) - quantile(b.times, 0.25)) / 1e6
    return (median=med, min=mn, max=mx, iqr=iqr,
            samples=length(b.times),
            allocs=b.allocs, bytes=b.memory)
end

function print_stats(label, s)
    println("  $label:")
    @printf("    median = %.3f ms, min = %.3f ms, max = %.3f ms, iqr = %.3f ms\n",
            s.median, s.min, s.max, s.iqr)
    println("    samples = $(s.samples), allocs = $(s.allocs), bytes = $(s.bytes)")
end

# ---------------------------------------------------------------------------
# Native Julia benchmarks
# ---------------------------------------------------------------------------
println("\n" ^ 1)
println("=" ^ 70)
println("Native Julia benchmarks")
println("=" ^ 70)

# --- DMO standard ---
println("\n--- DMO standard (hmcode_Pmm, 300 z, k_support) ---")
out_std_dmo = Mapse.hmcode_Pmm(cosmo, z_fine, k, pk_mm_fine_sup;
                                k_support=k_support, pk_cb_z=pk_cb_fine_sup,
                                T_AGN=nothing, nM=nM, threaded=false)
b_std_dmo = @benchmark Mapse.hmcode_Pmm($cosmo, $z_fine, $k, $pk_mm_fine_sup;
                                        k_support=$k_support, pk_cb_z=$pk_cb_fine_sup,
                                        T_AGN=nothing, nM=$nM, threaded=false)
s_std_dmo = bench_stats(b_std_dmo)
print_stats("std_dmo", s_std_dmo)
println("  Output: shape=$(size(out_std_dmo)), eltype=$(eltype(out_std_dmo))")
@assert size(out_std_dmo) == (128, 300) "Expected (128, 300), got $(size(out_std_dmo))"
@assert all(isfinite, out_std_dmo) "Non-finite values in std_dmo output"

# --- DMO fast ---
println("\n--- DMO fast (hmcode_pmm_fast, 50 coarse z -> 300 fine z, k_support) ---")
out_fast_dmo = Mapse.hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse_sup;
                                     k_support=k_support, pk_cb_coarse=pk_cb_coarse_sup,
                                     T_AGN=nothing, nM=nM, threaded=false)
b_fast_dmo = @benchmark Mapse.hmcode_pmm_fast($cosmo, $z_coarse, $z_fine, $k, $pk_mm_coarse_sup;
                                              k_support=$k_support, pk_cb_coarse=$pk_cb_coarse_sup,
                                              T_AGN=nothing, nM=$nM, threaded=false)
s_fast_dmo = bench_stats(b_fast_dmo)
print_stats("fast_dmo", s_fast_dmo)
println("  Output: shape=$(size(out_fast_dmo)), eltype=$(eltype(out_fast_dmo))")
@assert size(out_fast_dmo) == (128, 300) "Expected (128, 300), got $(size(out_fast_dmo))"
@assert all(isfinite, out_fast_dmo) "Non-finite values in fast_dmo output"

# --- Baryonic standard ---
println("\n--- Baryonic standard (hmcode_Pmm, 300 z, k_support) ---")
out_std_bar = Mapse.hmcode_Pmm(cosmo, z_fine, k, pk_mm_fine_sup;
                                k_support=k_support, pk_cb_z=pk_cb_fine_sup,
                                T_AGN=T_AGN_FEEDBACK, nM=nM, threaded=false)
b_std_bar = @benchmark Mapse.hmcode_Pmm($cosmo, $z_fine, $k, $pk_mm_fine_sup;
                                        k_support=$k_support, pk_cb_z=$pk_cb_fine_sup,
                                        T_AGN=$T_AGN_FEEDBACK, nM=$nM, threaded=false)
s_std_bar = bench_stats(b_std_bar)
print_stats("std_bar", s_std_bar)
println("  Output: shape=$(size(out_std_bar)), eltype=$(eltype(out_std_bar))")
@assert size(out_std_bar) == (128, 300)
@assert all(isfinite, out_std_bar)

# --- Baryonic fast ---
println("\n--- Baryonic fast (hmcode_pmm_fast, 50 coarse z -> 300 fine z, k_support) ---")
out_fast_bar = Mapse.hmcode_pmm_fast(cosmo, z_coarse, z_fine, k, pk_mm_coarse_sup;
                                     k_support=k_support, pk_cb_coarse=pk_cb_coarse_sup,
                                     T_AGN=T_AGN_FEEDBACK, nM=nM, threaded=false)
b_fast_bar = @benchmark Mapse.hmcode_pmm_fast($cosmo, $z_coarse, $z_fine, $k, $pk_mm_coarse_sup;
                                              k_support=$k_support, pk_cb_coarse=$pk_cb_coarse_sup,
                                              T_AGN=$T_AGN_FEEDBACK, nM=$nM, threaded=false)
s_fast_bar = bench_stats(b_fast_bar)
print_stats("fast_bar", s_fast_bar)
println("  Output: shape=$(size(out_fast_bar)), eltype=$(eltype(out_fast_bar))")
@assert size(out_fast_bar) == (128, 300)
@assert all(isfinite, out_fast_bar)

# ---------------------------------------------------------------------------
# Native Julia speedup
# ---------------------------------------------------------------------------
println("\n--- Native Julia speedup ---")
@printf("  DMO:      std=%.3f ms, fast=%.3f ms, speedup=%.2fx\n",
        s_std_dmo.median, s_fast_dmo.median, s_std_dmo.median / s_fast_dmo.median)
@printf("  Baryonic: std=%.3f ms, fast=%.3f ms, speedup=%.2fx\n",
        s_std_bar.median, s_fast_bar.median, s_std_bar.median / s_fast_bar.median)

# ---------------------------------------------------------------------------
# Native Julia numerical agreement
# ---------------------------------------------------------------------------
println("\n--- Native Julia standard vs fast agreement ---")
for (mode, std_out, fast_out) in [("DMO", out_std_dmo, out_fast_dmo),
                                   ("Baryonic", out_std_bar, out_fast_bar)]
    diff_abs = abs.(std_out .- fast_out)
    denom = abs.(std_out)
    diff_rel = similar(diff_abs)
    for i in eachindex(diff_abs)
        diff_rel[i] = denom[i] > 1e-30 ? diff_abs[i] / denom[i] : zero(diff_abs[i])
    end
    idx_max_abs = argmax(diff_abs)
    idx_max_rel = argmax(diff_rel)
    ik_abs, iz_abs = Tuple(idx_max_abs)
    ik_rel, iz_rel = Tuple(idx_max_rel)
    println("  $mode:")
    @printf("    Max abs diff: %.6e at z=%.4f, k=%.6e\n",
            diff_abs[idx_max_abs], z_fine[iz_abs], k[ik_abs])
    @printf("    Max rel diff: %.6e at z=%.4f, k=%.6e\n",
            diff_rel[idx_max_rel], z_fine[iz_rel], k[ik_rel])
end

# ---------------------------------------------------------------------------
# Reactant benchmarks
# ---------------------------------------------------------------------------
println("\n" ^ 1)
println("=" ^ 70)
println("Reactant benchmarks")
println("=" ^ 70)

Reactant.set_default_backend("cpu")

# Convert inputs to Reactant arrays
z_fine_R = Reactant.to_rarray(z_fine)
z_coarse_R = Reactant.to_rarray(z_coarse)
k_R = Reactant.to_rarray(k)
k_support_R = Reactant.to_rarray(k_support)
pk_mm_fine_sup_R = Reactant.to_rarray(pk_mm_fine_sup)
pk_cb_fine_sup_R = Reactant.to_rarray(pk_cb_fine_sup)
pk_mm_coarse_sup_R = Reactant.to_rarray(pk_mm_coarse_sup)
pk_cb_coarse_sup_R = Reactant.to_rarray(pk_cb_coarse_sup)

# --- Compile all four signatures ---
println("\n--- Compiling Reactant signatures ---")

t_compile_std_dmo = @elapsed begin
    compiled_std_dmo = Reactant.@compile sync=true Mapse.hmcode_Pmm(
        cosmo, z_fine_R, k_R, k_support_R, pk_mm_fine_sup_R,
        pk_cb_fine_sup_R; T_AGN=nothing, nM=nM)
end
println("  std_dmo compiled: $(round(t_compile_std_dmo, digits=3)) s")

t_compile_fast_dmo = @elapsed begin
    compiled_fast_dmo = Reactant.@compile sync=true Mapse.hmcode_pmm_fast(
        cosmo, z_coarse_R, z_fine_R, k_R, k_support_R,
        pk_mm_coarse_sup_R, pk_cb_coarse_sup_R;
        T_AGN=nothing, nM=nM)
end
println("  fast_dmo compiled: $(round(t_compile_fast_dmo, digits=3)) s")

t_compile_std_bar = @elapsed begin
    compiled_std_bar = Reactant.@compile sync=true Mapse.hmcode_Pmm(
        cosmo, z_fine_R, k_R, k_support_R, pk_mm_fine_sup_R,
        pk_cb_fine_sup_R; T_AGN=T_AGN_FEEDBACK, nM=nM)
end
println("  std_bar compiled: $(round(t_compile_std_bar, digits=3)) s")

t_compile_fast_bar = @elapsed begin
    compiled_fast_bar = Reactant.@compile sync=true Mapse.hmcode_pmm_fast(
        cosmo, z_coarse_R, z_fine_R, k_R, k_support_R,
        pk_mm_coarse_sup_R, pk_cb_coarse_sup_R;
        T_AGN=T_AGN_FEEDBACK, nM=nM)
end
println("  fast_bar compiled: $(round(t_compile_fast_bar, digits=3)) s")

# --- Warm up and verify outputs ---
println("\n--- Reactant warm-up and output verification ---")
outR_std_dmo = compiled_std_dmo(cosmo, z_fine_R, k_R, k_support_R,
                                pk_mm_fine_sup_R, pk_cb_fine_sup_R)
Reactant.synchronize(outR_std_dmo)
outR_fast_dmo = compiled_fast_dmo(cosmo, z_coarse_R, z_fine_R, k_R,
                                  k_support_R, pk_mm_coarse_sup_R,
                                  pk_cb_coarse_sup_R)
Reactant.synchronize(outR_fast_dmo)
outR_std_bar = compiled_std_bar(cosmo, z_fine_R, k_R, k_support_R,
                                pk_mm_fine_sup_R, pk_cb_fine_sup_R)
Reactant.synchronize(outR_std_bar)
outR_fast_bar = compiled_fast_bar(cosmo, z_coarse_R, z_fine_R, k_R,
                                  k_support_R, pk_mm_coarse_sup_R,
                                  pk_cb_coarse_sup_R)
Reactant.synchronize(outR_fast_bar)

# Materialize for shape/type checks
outR_std_dmo_mat = Array(outR_std_dmo)
outR_fast_dmo_mat = Array(outR_fast_dmo)
outR_std_bar_mat = Array(outR_std_bar)
outR_fast_bar_mat = Array(outR_fast_bar)

for (label, out) in [("std_dmo", outR_std_dmo_mat), ("fast_dmo", outR_fast_dmo_mat),
                      ("std_bar", outR_std_bar_mat), ("fast_bar", outR_fast_bar_mat)]
    @assert size(out) == (128, 300) "$label: expected (128, 300), got $(size(out))"
    @assert all(isfinite, out) "$label: non-finite values"
    println("  $label: shape=$(size(out)), eltype=$(eltype(out))  [OK]")
end

# --- Reactant benchmarks ---
println("\n--- Reactant timing (post-compilation) ---")

println("\n  DMO standard:")
bR_std_dmo = @benchmark begin
    out = $compiled_std_dmo($cosmo, $z_fine_R, $k_R, $k_support_R,
                             $pk_mm_fine_sup_R, $pk_cb_fine_sup_R)
    Reactant.synchronize(out)
end
sR_std_dmo = bench_stats(bR_std_dmo)
print_stats("std_dmo", sR_std_dmo)

println("\n  DMO fast:")
bR_fast_dmo = @benchmark begin
    out = $compiled_fast_dmo($cosmo, $z_coarse_R, $z_fine_R, $k_R,
                              $k_support_R, $pk_mm_coarse_sup_R,
                              $pk_cb_coarse_sup_R)
    Reactant.synchronize(out)
end
sR_fast_dmo = bench_stats(bR_fast_dmo)
print_stats("fast_dmo", sR_fast_dmo)

println("\n  Baryonic standard:")
bR_std_bar = @benchmark begin
    out = $compiled_std_bar($cosmo, $z_fine_R, $k_R, $k_support_R,
                             $pk_mm_fine_sup_R, $pk_cb_fine_sup_R)
    Reactant.synchronize(out)
end
sR_std_bar = bench_stats(bR_std_bar)
print_stats("std_bar", sR_std_bar)

println("\n  Baryonic fast:")
bR_fast_bar = @benchmark begin
    out = $compiled_fast_bar($cosmo, $z_coarse_R, $z_fine_R, $k_R,
                              $k_support_R, $pk_mm_coarse_sup_R,
                              $pk_cb_coarse_sup_R)
    Reactant.synchronize(out)
end
sR_fast_bar = bench_stats(bR_fast_bar)
print_stats("fast_bar", sR_fast_bar)

# ---------------------------------------------------------------------------
# Reactant speedup
# ---------------------------------------------------------------------------
println("\n--- Reactant speedup ---")
@printf("  DMO:      std=%.3f ms, fast=%.3f ms, speedup=%.2fx\n",
        sR_std_dmo.median, sR_fast_dmo.median, sR_std_dmo.median / sR_fast_dmo.median)
@printf("  Baryonic: std=%.3f ms, fast=%.3f ms, speedup=%.2fx\n",
        sR_std_bar.median, sR_fast_bar.median, sR_std_bar.median / sR_fast_bar.median)

# ---------------------------------------------------------------------------
# Reactant numerical agreement
# ---------------------------------------------------------------------------
println("\n--- Reactant standard vs fast agreement ---")
for (mode, std_out, fast_out) in [("DMO", outR_std_dmo_mat, outR_fast_dmo_mat),
                                   ("Baryonic", outR_std_bar_mat, outR_fast_bar_mat)]
    diff_abs = abs.(std_out .- fast_out)
    denom = abs.(std_out)
    diff_rel = similar(diff_abs)
    for i in eachindex(diff_abs)
        diff_rel[i] = denom[i] > 1e-30 ? diff_abs[i] / denom[i] : zero(diff_abs[i])
    end
    idx_max_abs = argmax(diff_abs)
    idx_max_rel = argmax(diff_rel)
    ik_abs, iz_abs = Tuple(idx_max_abs)
    ik_rel, iz_rel = Tuple(idx_max_rel)
    println("  $mode:")
    @printf("    Max abs diff: %.6e at z=%.4f, k=%.6e\n",
            diff_abs[idx_max_abs], z_fine[iz_abs], k[ik_abs])
    @printf("    Max rel diff: %.6e at z=%.4f, k=%.6e\n",
            diff_rel[idx_max_rel], z_fine[iz_rel], k[ik_rel])
end

# ---------------------------------------------------------------------------
# Compilation times summary
# ---------------------------------------------------------------------------
println("\n--- Compilation times ---")
@printf("  std_dmo:  %.3f s\n", t_compile_std_dmo)
@printf("  fast_dmo: %.3f s\n", t_compile_fast_dmo)
@printf("  std_bar:  %.3f s\n", t_compile_std_bar)
@printf("  fast_bar: %.3f s\n", t_compile_fast_bar)

println("\n" ^ 1)
println("=" ^ 70)
println("Benchmark complete.")
println("=" ^ 70)
