# Mapse.jl

`Mapse.jl` is a Julia package designed to emulate the computation of the Linear and Nonlinear Matter Power Spectrum, with a speedup of several orders of magnitude compared to standard codes such as `CAMB` or `CLASS`. The core functionalities of `Mapse.jl` are inherithed by the upstream library [`AbstractCosmologicalEmulators.jl`](https://github.com/CosmologicalEmulators/AbstractCosmologicalEmulators.jl).

## Installation

In order to install  `Mapse.jl`, run on the `Julia` REPL

```julia
using Pkg, Pkg.add(url="https://github.com/CosmologicalEmulators/Mapse.jl")
```

## Usage

In order to be able to use `Mapse.jl`, there are two major steps that need to be performed:

- Instantiating the emulators, e.g. initializing the Neural Network, its weights and biases, and the quantities employed in pre and post-processing
- Use the instantiated emulators to retrieve the spectra

In the reminder of this section we are showing how to do this.

#### Instantiation

The recommended workflow loads each component explicitly using `load_emulator` after resolving the artifact path:

```julia
# Resolve the artifact directory path
root = Mapse.artifact_path("mnuw0wacdm_class")

# Load linear total-matter and linear cold+baryon components explicitly
pmm = Mapse.load_emulator(joinpath(root, "Pk_lin_mm"))
pcb = Mapse.load_emulator(joinpath(root, "Pk_lin_cb"))
```

To load a single component emulator from a local directory containing `nn_setup.json`, use `load_emulator`:

```julia
LinearPmm = Mapse.load_emulator(weights_folder)
```

It is possible to pass an additional argument to `load_emulator`, which is used to choose between the two NN backends now available:

- [SimpleChains](https://github.com/PumasAI/SimpleChains.jl), which is tailored for small NNs running on a CPU (default)
- [Lux](https://github.com/LuxDL/Lux.jl), which can run both on CPUs and GPUs

`SimpleChains.jl` is faster especially for small NNs on the CPU. If you want to run on a GPU, you should use `Lux.jl` by passing the `Mapse.LuxEmulator` backend:

```julia
LinearPmm = Mapse.load_emulator(weights_folder, emu = Mapse.LuxEmulator)
```

Each trained emulator should be shipped with a description within the JSON file. In order to print the description, just run:

```julia
Mapse.get_emulator_description(pmm)
```

!!! warning

    Cosmological parameters must be fed to `Mapse.jl` with **arrays**. It is the user's
    responsibility to check the right ordering, by reading the output of the
    `get_emulator_description` method.

### Usage

Use the loaded component emulators to retrieve the linear power spectrum:

```julia
params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
z = 0.0
D = 1.0

# Evaluate linear matter power spectrum on its k-grid
pk_lin = Mapse.get_Pk(params, z, D, pmm)
k_lin  = Mapse.get_kgrid(pmm)
```

### Nonlinear modeling with Halofit/HMCode

Once you have evaluated the linear power spectrum, you can obtain the nonlinear power spectrum using `Mapse`'s built-in Halofit or HMCode:

```julia
# Compute nonlinear Pmm using Halofit
pk_halofit = Mapse.halofit_pmm(params, z, k_lin, pk_lin)
```

For HMCode, you can load both the total-matter and cold+baryon emulators to get the respective linear spectra, then pass them to HMCode:

```julia
# Evaluate linear cold+baryon spectrum
pk_cb_lin = Mapse.get_Pk(params, z, D, pcb)

# Set up background HMCode cosmology
cosmo = Mapse.HMCodeCosmology(
    0.3,   # Ωm
    0.05,  # Ωb
    0.7,   # h
    0.96,  # ns
    0.8,   # σ8
    -1.0,  # w0
    0.0,   # wa
    0.0,   # Ων
    0.0,   # Ωk
)

# Compute nonlinear Pmm using HMCode2020
pk_hmcode = Mapse.hmcode_pmm(cosmo, [z], k_lin, reshape(pk_lin, :, 1); pk_cb_z=reshape(pk_cb_lin, :, 1))
```

### Background cosmology helpers

The linear growth factor $D(z)$ must be normalized to 1 at $z=0$. You can compute it using the background cosmology functions exported by `Mapse`:

```julia
# Set up background cosmology parameters
H0 = 67.36
h = H0 / 100.0
ωb = 0.02237
ωc = 0.12
Mν = 0.06
w0 = -1.0
wa = 0.0

Ωcb0 = (ωb + ωc) / h^2

# Compute D(z) at redshift z
D = Mapse.D_z(z, Ωcb0, h; mν=Mν, w0=w0, wa=wa)
```

### Halofit and Reactant

`Mapse.halofit_pmm` can compute a Halofit nonlinear total-matter spectrum from a
linear `Pmm` grid. For Reactant/XLA workflows the background calculation must be
performed outside the compiled Halofit kernel and passed explicitly:

```julia
pk_nl = Mapse.halofit_pmm(cpar, z, k, pk_lin_mm_z, Ωm_z, Ωv_z)
```

Here `Ωm_z` and `Ωv_z` are the matter and dark-energy density fractions at the
redshifts in `z`. Loading `Reactant` activates `MapseReactantExt`, which provides
Reactant dispatch for this explicit-background API. The convenience helper
`Mapse.halofit_background` remains a host-side CLASS-parity background helper and
should not be called inside `Reactant.@compile`.

### HMCode2020

`Mapse` also includes an embedded HMCode2020 implementation. Given a sampled
linear matter spectrum, it can return either the nonlinear HMCode spectrum or the
nonlinear boost:

```julia
k = exp.(range(log(1e-4), log(50.0), length=100))
z = [0.0]
pk_mm_z = reshape(pk_mm_lin, length(k), length(z))
pk_cb_z = reshape(pk_cb_lin, length(k), length(z))

cosmo = Mapse.HMCodeCosmology(
    0.3,   # Ωm
    0.05,  # Ωb
    0.7,   # h
    0.96,  # ns
    0.8,   # σ8
    -1.0,  # w0
    0.0,   # wa
    0.0,   # Ων
    0.0,   # Ωk
)

pk_nl = Mapse.hmcode_pmm(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z)
boost = Mapse.hmcode_boost(cosmo, z, k, pk_mm_z; pk_cb_z=pk_cb_z)

# Or use a wide support grid for HMCode's σ(R) calculations and a smaller
# output grid for the returned nonlinear spectrum.
k_support = exp.(range(log(1e-4), log(1e3), length=220))
k_out = exp.(range(log(1e-4), log(10.0), length=128))
pk_nl_out = Mapse.hmcode_pmm(cosmo, z, k_out, k_support, pk_mm_support_z;
                             pk_cb_support_z=pk_cb_support_z)
```

As for `halofit_Pmm`, spectra have shape `(length(k), length(z))`. `pk_mm_z` is
the total-matter linear spectrum and sets the returned nonlinear `Pmm`/boost. For
massive-neutrino cosmologies, pass `pk_cb_z` as the cold+baryon linear spectrum;
HMCode2020 uses cold+baryon σ(R,z) internally for halo collapse and transition
parameters. If `pk_cb_z` is omitted, the total spectrum is used for both roles.
The same input `k` grid is used internally to compute the σ(R,z) integrals, so it
should cover the linear spectrum over a sufficiently broad range. The default baryonic feedback parameter is `T_AGN = 10^7.8`; pass
`T_AGN = nothing` to compute the gravity-only HMCode spectrum. Use `nM` to set
the mass-integration grid directly; the default is `nM=128`.

When `k_support` is supplied, the linear spectra are interpreted on `k_support`,
but the nonlinear output is returned on `k_out`. This is useful when σ(R,z) needs
a broad high-k support grid while the downstream application only needs a smaller
or coarser output grid. `hmcode_boost` divides by the total-matter linear spectrum
interpolated onto `k_out` in this mode.

With `Reactant` loaded, `hmcode_Pmm` and `hmcode_boost` also dispatch on
Reactant arrays. For compiled support-grid calls, pass the cold+baryon spectrum
as a positional argument so it is traced as part of the numerical payload:

```julia
using Reactant

zR = Reactant.to_rarray(z)
koutR = Reactant.to_rarray(k_out)
ksupportR = Reactant.to_rarray(k_support)
pmmR = Reactant.to_rarray(pk_mm_support_z)
pcbR = Reactant.to_rarray(pk_cb_support_z)

compiled = Reactant.@compile sync=true Mapse.hmcode_pmm(
    cosmo, zR, koutR, ksupportR, pmmR, pcbR; nM=128
)
pk_nlR = compiled(cosmo, zR, koutR, ksupportR, pmmR, pcbR)
```

`SimpleChains.jl` and `Lux.jl` have almost the same performance and they give the same result up to floating point precision.

These benchmarks have been performed locally, with a 13th Gen Intel® Core™ i7-13700H, using a single core.

Considering that a high-precision settings calculation performed with [`CAMB`](https://github.com/cmbant/CAMB) on the same machine requires around 60 seconds, `Mapse.jl` is 5-6 order of magnitudes faster.

!!! warning

    Currently, there is a performance issue when using `Lux.jl` in a multi-threaded scenario. This is
    something known (see discussion [here](https://github.com/LuxDL/Lux.jl/issues/847)).
    In case you want to launch multiple chains locally, the suggested (working) strategy with `Lux.jl`
    is to use distributed computing.

### Authors

- Marco Bonici, PostDoctoral researcher at Waterloo Center for Astrophysics

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you
would like to change.

Please make sure to update tests as appropriate.

### License

`Mapse.jl` is licensed under the MIT "Expat" license; see
[LICENSE](https://github.com/CosmologicalEmulators/Mapse.jl/blob/main/LICENSE) for the full
license text.
