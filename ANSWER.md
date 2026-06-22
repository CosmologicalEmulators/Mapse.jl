# Response to REVIEW.md for PR #22

Thanks for the review. I went through the blocking points and fixed the issues that should be addressed before merging. Below is the point-by-point response.

## Blocking / high-priority issues

### 1. Exported API methods fail on the new `PkEmulator` type

Fixed.

I added explicit methods for the top-level `PkEmulator`:

```julia
Mapse.get_kgrid(PkEmu::PkEmulator)
Mapse.get_emulator_description(PkEmu::PkEmulator)
```

The official artifact has different component grids: the linear emulators live on a 300-point grid, while the nonlinear boost lives on a 98-point grid. Because of that, `get_kgrid(PkEmu)` returns the top-level nonlinear output grid, i.e. the boost grid.

I also fixed `get_Pk(input_params, z, D, PkEmu::PkEmulator)` so it no longer assumes matching component grids. It now computes the linear `Pmm`, computes the boost, interpolates the linear spectrum onto the boost grid, and returns the nonlinear spectrum on the top-level output grid.

Tests were added for:

- `get_kgrid(::PkEmulator)` on synthetic and official artifact-backed emulators,
- `get_emulator_description(::PkEmulator)`,
- the public artifact workflow,
- finite output from `get_Pk(params, z, D, official_emu)`.

### 2. Documented usage is stale for the new `get_Pk` signatures

Fixed.

The docs no longer show the stale two-argument call:

```julia
Mapse.get_Pk(x, Pk_emu)
```

They now show the current API:

```julia
params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
z = 0.0
D = 1.0
k = Mapse.get_kgrid(Pk_emu)
pk_nonlinear = Mapse.get_Pk(params, z, D, Pk_emu)
```

The docs also state the official `mnuw0wacdm_class` parameter order and explain that `D(z)` must be supplied.

Tracked docs source was checked for stale references such as `Capse`, `LinearPkEmulator`, `get_zgrid`, and old `get_Pk(x, Pk_emu)` usage.

### 3. Halofit tests need an external scientific reference

Fixed.

I added a small CLASS-generated reference table:

```text
test/data/halofit_class_reference.txt
```

The table contains:

```text
z, k [1/Mpc], CLASS linear P(k), CLASS Ωm(z), CLASS Ωv(z), CLASS nonlinear Halofit P(k)
```

It was generated with `classy`/CLASS using:

```text
output = mPk
non linear = halofit
h = 0.6736
omega_b = 0.02237
omega_cdm = 0.12
ln10As = 3.044
n_s = 0.9649
N_ur = 2.033
N_ncdm = 1
m_ncdm = 0.06
```

The Julia test now reads that table and checks `Mapse.halofit_Pmm` against the CLASS nonlinear Halofit values using the saved CLASS linear spectrum and saved CLASS background fractions. This is no longer just a self-consistency test.

### 4. Importing `Mapse` eagerly installs and loads the trained emulator artifact

Kept intentionally.

The package currently follows the Effort.jl-style pattern where official trained emulators are available immediately after import through:

```julia
Mapse.trained_emulators[Mapse.DEFAULT_EMULATOR_NAME]
```

We considered lazy loading, but decided not to change that behavior in this PR. The artifact-backed official emulator is meant to be part of the package-level user experience, and the project preference is that it is loaded when importing the package.

So this point is acknowledged but intentionally not changed.

### 5. Reactant extension tests are opt-in but CI does not enable them

Fixed.

The Reactant tests are now part of the regular `Pkg.test()` suite. They are no longer gated behind an environment variable.

The regular CI test job runs on the full supported Julia matrix:

```yaml
1.10
1.11
1.12
```

This means the Reactant extension is always covered in PR checks, not just in local optional testing.

### 6. Test-only dependencies should be declared in the root `Project.toml`

Fixed.

I moved test-only dependencies into the root package metadata following the Effort.jl-style pattern:

```toml
[extras]
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"
Reactant = "3c362404-f566-11ee-1572-e11a4b42c853"
SimpleChains = "de6bee2f-e2f4-4ec7-b6ed-219cc6f6e9e5"
Static = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[targets]
test = ["DelimitedFiles", "Reactant", "SimpleChains", "Static", "Test"]
```

The sidecar `test/Project.toml` was removed. `Pkg.test()` now constructs its temporary test environment from the root `[extras]` / `[targets]` metadata.

### 7. Manifest is stale relative to `Project.toml`

Handled.

I ran:

```julia
using Pkg; Pkg.resolve()
```

No packages were added to or removed from the root `Manifest.toml`. After moving test-only dependencies into root `[extras]` / `[targets]`, both default and Reactant-enabled `Pkg.test()` runs resolved and passed from the root package metadata.

I also added an explicit Julia compat entry:

```toml
julia = "1.10"
```

## Smaller comments

### `julia` compat

Fixed with:

```toml
julia = "1.10"
```

### `FastGaussQuadrature`

Kept intentionally.

It is loaded to activate `AbstractCosmologicalEmulators.BackgroundCosmologyExt`. I added a source comment so it does not look like a random unused import.

### Old docs references

Fixed in tracked docs source. The old `Capse` examples in `docs/src/index.md` were removed/replaced.

### Public artifact workflow tests

Added.

The tests now load the official emulator through `Mapse.trained_emulators`, call `get_kgrid`, call `get_emulator_description`, compute `get_Pk`, and verify finite output with the expected top-level grid length.

## Local validation

The following passed locally:

```bash
julia --project=. --startup-file=no -e 'using Pkg; Pkg.test(; coverage=false)'
```

Result:

```text
Mapse tests | 77 passed
```

Because Reactant tests are unconditional, the same `Pkg.test()` command also runs the Reactant extension testset. Result:

```text
Mapse tests              | 77 passed
Halofit Reactant support | 7 passed
```

A direct public workflow smoke test is covered by the test suite and also passed manually:

```julia
using Mapse
emu = Mapse.trained_emulators[Mapse.DEFAULT_EMULATOR_NAME]
params = [3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1.0, 0.0]
pk = Mapse.get_Pk(params, 0.0, 1.0, emu)
```

with finite output on the 98-point top-level nonlinear grid.
