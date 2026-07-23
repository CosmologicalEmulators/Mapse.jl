#!/usr/bin/env julia
# Performance benchmarking script for native Julia and Reactant pipelines
using Mapse
using Reactant
using AbstractCosmologicalEmulators
using BenchmarkTools
using DelimitedFiles
using Printf
using Statistics

const ROOT = @__DIR__
const H = 0.6736
const ΩB = 0.02237 / H^2
const Ων = 0.06 / (93.14 * H^2)
const ΩM = ΩB + 0.12 / H^2 + Ων
const NS = 0.9649
const SIGMA8 = 0.8109118
const T_AGN = 10.0^7.8
const NM = 128
const PARAMS = [3.044, NS, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
const COSMO = Mapse.HMCodeCosmology(ΩM, ΩB, H, NS, SIGMA8, -1.0, 0.0, Ων, 0.0)
const K_LABEL = collect(10.0 .^ range(-3.0, 1.0; length=128)) # requested h Mpc⁻¹ grid
const K_OUT = K_LABEL .* H # physical Mpc⁻¹ passed to the public API
const Z_FINE = collect(range(0.0, 3.5; length=150))
const N_COARSE = (24, 32, 40)
const LOGT_AGN = log10(T_AGN)

println("Initializing execution environments...")
println("grid: nk=$(length(K_OUT)), nz=$(length(Z_FINE)), coarse=$(N_COARSE)")

println("Loading emulators...")
artifact_root = Mapse.artifact_path(Mapse.DEFAULT_EMULATOR_ARTIFACT)
pmm_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_mm"))
pcb_emu = Mapse.load_emulator(joinpath(artifact_root, "Pk_lin_cb"))
growth_emu = AbstractCosmologicalEmulators.trained_emulators["ACE_mnuw0wacdm_ln10As_basis"]
K_SUPPORT = collect(Mapse.get_kgrid(pmm_emu)) # physical Mpc⁻¹

function growth_at(z, params, growth_emu)
    input = vcat(reshape(z, 1, :), repeat(reshape(params, :, 1), 1, length(z)))
    return vec(AbstractCosmologicalEmulators.run_emulator(input, growth_emu)[6:6, :])
end

function baryonic_smart_physical(cosmo, z_fine, k_out, k_support, pk_mm, pk_cb; N_coarse, T_AGN, nM)
    h = cosmo.h
    result_h = Mapse.hmcode_pmm_baryonic_smart(
        cosmo, z_fine, k_out ./ h, pk_mm .* h^3;
        pk_cb_coarse=pk_cb .* h^3, k_support=k_support ./ h,
        N_coarse=N_coarse, T_AGN=T_AGN, nM=nM)
    return result_h ./ h^3
end

function direct_pipeline(params, cosmo, z, k_out, k_support, pmm_emu, pcb_emu, growth_emu; T_AGN=nothing, nM=NM)
    growth = growth_at(z, params, growth_emu)
    pmm = Mapse.get_Pk(params, z, growth, pmm_emu)
    pcb = Mapse.get_Pk(params, z, growth, pcb_emu)
    return Mapse.hmcode_pmm_physical(cosmo, z, k_out, pmm;
        pk_cb_z=pcb, k_support=k_support, T_AGN=T_AGN, nM=nM)
end

function smart_pipeline(params, cosmo, z_fine, k_out, k_support, pmm_emu, pcb_emu, growth_emu; N_coarse, T_AGN, nM=NM)
    z_feature = Mapse.predict_baryonic_discontinuity(cosmo; T_AGN=T_AGN)
    zc = Mapse.build_smart_coarse_grid(first(z_fine), last(z_fine), N_coarse, z_feature)
    growth = growth_at(zc, params, growth_emu)
    pmm = Mapse.get_Pk(params, zc, growth, pmm_emu)
    pcb = Mapse.get_Pk(params, zc, growth, pcb_emu)
    return baryonic_smart_physical(cosmo, z_fine, k_out, k_support, pmm, pcb; N_coarse=N_coarse, T_AGN=T_AGN, nM=nM)
end

function smart_dmo_pipeline(params, cosmo, z_fine, k_out, k_support, pmm_emu, pcb_emu, growth_emu; N_coarse, nM=NM)
    z_limits = [first(z_fine), last(z_fine)]
    return Mapse.hmcode_pmm_dmo_smart(params, z_fine, z_limits, k_out, pmm_emu, pcb_emu, growth_emu, cosmo; N_coarse=N_coarse, N_left=N_coarse ÷ 2, nM=nM)
end

function trial_stats(t)
    (median=median(t.times) / 1e6, minimum=minimum(t.times) / 1e6,
     iqr=(quantile(t.times, .75) - quantile(t.times, .25)) / 1e6,
     allocs=t.allocs, bytes=t.memory)
end

function report(label, s)
    @printf("%-27s median %9.3f ms  min %9.3f ms  IQR %8.3f ms  allocs %d\n",
            label, s.median, s.minimum, s.iqr, s.allocs)
end

# -----------------------------------------------------------------------------
# Warmup to ensure JIT compile time isn't measured
# -----------------------------------------------------------------------------
println("\nWarming up native Julia compiler...")
direct_dmo = direct_pipeline(PARAMS, COSMO, Z_FINE, K_OUT, K_SUPPORT, pmm_emu, pcb_emu, growth_emu; T_AGN=nothing)
direct_fb = direct_pipeline(PARAMS, COSMO, Z_FINE, K_OUT, K_SUPPORT, pmm_emu, pcb_emu, growth_emu; T_AGN=T_AGN)

for n in N_COARSE
    smart_dmo_pipeline(PARAMS, COSMO, Z_FINE, K_OUT, K_SUPPORT, pmm_emu, pcb_emu, growth_emu; N_coarse=n)
    smart_pipeline(PARAMS, COSMO, Z_FINE, K_OUT, K_SUPPORT, pmm_emu, pcb_emu, growth_emu; N_coarse=n, T_AGN=T_AGN)
end

# -----------------------------------------------------------------------------
# Benchmarks
# -----------------------------------------------------------------------------
println("\n[1/2] Native Julia Benchmark Suite")
println("-"^60)
t_dmo = @benchmark direct_pipeline($PARAMS, $COSMO, $Z_FINE, $K_OUT, $K_SUPPORT, $pmm_emu, $pcb_emu, $growth_emu; T_AGN=nothing)
t_fb = @benchmark direct_pipeline($PARAMS, $COSMO, $Z_FINE, $K_OUT, $K_SUPPORT, $pmm_emu, $pcb_emu, $growth_emu; T_AGN=$T_AGN)
s_dmo, s_fb = trial_stats(t_dmo), trial_stats(t_fb)

report("direct DMO", s_dmo)
report("direct feedback", s_fb)

for n in N_COARSE
    t_dmo = @benchmark smart_dmo_pipeline($PARAMS, $COSMO, $Z_FINE, $K_OUT, $K_SUPPORT, $pmm_emu, $pcb_emu, $growth_emu; N_coarse=$n)
    report("smart DMO[$n]", trial_stats(t_dmo))
    @printf("  speedup[%d] = %.2fx\n", n, s_dmo.median / trial_stats(t_dmo).median)
    
    t = @benchmark smart_pipeline($PARAMS, $COSMO, $Z_FINE, $K_OUT, $K_SUPPORT, $pmm_emu, $pcb_emu, $growth_emu; N_coarse=$n, T_AGN=$T_AGN)
    s = trial_stats(t)
    report("smart feedback[$n]", s)
    @printf("  speedup[%d] = %.2fx\n", n, s_fb.median / s.median)
end

println("\n[2/2] Reactant JIT Benchmark Suite")
println("-"^60)
try
    Reactant.set_default_backend("cpu")
    params_R = Reactant.to_rarray(PARAMS)
    z_fine_R = Reactant.to_rarray(Z_FINE)
    z_limits_R = Reactant.to_rarray([first(Z_FINE), last(Z_FINE)])
    k_out_R = Reactant.to_rarray(K_OUT)
    logT_R = Reactant.to_rarray([LOGT_AGN])
    pmm_emu_R = AbstractCosmologicalEmulators.to_reactant(pmm_emu)
    pcb_emu_R = AbstractCosmologicalEmulators.to_reactant(pcb_emu)
    growth_emu_R = AbstractCosmologicalEmulators.to_reactant(growth_emu)
    
    for n in N_COARSE
        println("Compiling and warming up Reactant smart DMO (N=$n)...")
        compiled_dmo = Reactant.@compile sync=true Mapse.hmcode_pmm_dmo_smart(
            params_R, z_fine_R, z_limits_R, k_out_R, pmm_emu_R, pcb_emu_R, growth_emu_R, COSMO; N_coarse=n, N_left=n ÷ 2, nM=NM)
        warm_dmo = compiled_dmo(params_R, z_fine_R, z_limits_R, k_out_R, pmm_emu_R, pcb_emu_R, growth_emu_R, COSMO)
        Reactant.synchronize(warm_dmo)
        t_dmo = @benchmark begin
            result = $compiled_dmo($params_R, $z_fine_R, $z_limits_R, $k_out_R, $pmm_emu_R, $pcb_emu_R, $growth_emu_R, $COSMO)
            Reactant.synchronize(result)
        end
        report("Reactant smart DMO[$n]", trial_stats(t_dmo))

        println("Compiling and warming up Reactant smart feedback (N=$n)...")
        compiled = Reactant.@compile sync=true Mapse.hmcode_pmm_baryonic_smart(
            params_R, z_fine_R, z_limits_R, k_out_R, logT_R, pmm_emu_R, pcb_emu_R,
            growth_emu_R, COSMO; N_coarse=n, N_left=n ÷ 2, nM=NM)
        warm = compiled(params_R, z_fine_R, z_limits_R, k_out_R, logT_R,
                        pmm_emu_R, pcb_emu_R, growth_emu_R, COSMO)
        Reactant.synchronize(warm)
        
        t = @benchmark begin
            result = $compiled($params_R, $z_fine_R, $z_limits_R, $k_out_R, $logT_R,
                               $pmm_emu_R, $pcb_emu_R, $growth_emu_R, $COSMO)
            Reactant.synchronize(result)
        end
        report("Reactant feedback[$n]", trial_stats(t))
    end
catch err
    @warn "Reactant full smart path unavailable; native results remain valid" exception=(err, catch_backtrace())
end

println("\nBenchmark complete.")
