# CAMB HMCode Benchmarks

This directory contains standalone scripts to regenerate exact precision benchmarks using the `CAMB` package and to evaluate execution timings of the `Mapse.jl` package native and `Reactant` pipelines.

## Contents

1. **`generate_fixtures.py`**: A Python script utilizing the `camb` library to produce the baseline reference grids for the Dark Matter Only (DMO) and Baryonic Feedback non-linear pipelines.
2. **`benchmark.jl`**: A pure Julia benchmark suite focusing solely on pipeline execution speed, comparing the native evaluation with the XLA JIT-compiled `Reactant` evaluation.

## 1. Generating CAMB Fixtures

The reference grids evaluate exactly 150 redshift points between $z \in [0.0, 3.5]$ and 128 physical wavenumber points for $k \in [10^{-3}, 10^{1}]$ $h$/Mpc.

To regenerate these `.txt` fixtures, ensure `camb` is installed in your Python environment and run:

```bash
python generate_fixtures.py
```

This will output `camb_pk_hmcode_dmo.txt` and `camb_pk_hmcode_fb.txt`. These specific files are utilized within the `Mapse.jl` unit test suite to enforce maximum relative error boundaries.

## 2. Running the Benchmarks

The benchmark script evaluates the complete linear and non-linear power spectrum emulator pathways, including `Lux` neural network evaluations, HMCode integration, and redshift interpolation. JIT compilation/warmup times are deliberately excluded from the timing outputs.

To execute the suite, run it within an environment that has `Mapse.jl`, `Reactant.jl`, and `AbstractCosmologicalEmulators.jl` available (e.g., the `bench_env` environment):

```bash
julia --project=../bench_env benchmark.jl
```

### Expected Timings

Based on baseline runs, you should expect the following median execution timescales for a full $150 \times 128$ power spectrum evaluation volume:

#### Native Julia Pathway
- **Direct DMO**: ~260 ms
- **Direct Feedback**: ~500 ms
- **Smart DMO (N=24)**: ~43 ms (6.0x speedup)
- **Smart Feedback (N=24)**: ~82 ms (6.1x speedup)
- **Smart Feedback (N=32)**: ~107 ms (4.6x speedup)
- **Smart Feedback (N=40)**: ~134 ms (3.7x speedup)

#### Reactant Compiled Pathway (CPU Backend)
The `Reactant` benchmarks construct the fully traced XLA evaluation graph for the emulators and HMCode integrations.
- **Smart Feedback (N=24)**: ~28 ms 
- **Smart Feedback (N=32)**: ~42 ms
- **Smart Feedback (N=40)**: ~52 ms

*Note: The first pass of the `Reactant` block in the benchmark script can take several minutes to successfully JIT-compile and trace the emulator networks. Subsequent traced executions drop entirely to the ~30-50 ms timescale above.*
