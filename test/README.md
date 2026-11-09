# HMCode validation

Run the complete package suite with `using Pkg; Pkg.test()`.
The focused units/native-AD subset is available through
`Pkg.test(; test_args=["hmcode"])`. The default suite includes this subset
and the Reactant forward tests unconditionally.
The curved Reactant fixture can be checked alone with
`Pkg.test(; test_args=["hmcode-curved"])`; the same file is included in the
normal suite, with the same reference and tolerance.
Review regressions for isolated parameters, explicit column selection, and
prepared A/B/A reuse run with `Pkg.test(; test_args=["hmcode-review"])` and
are also included unconditionally in the complete suite.

## Units and references

All public HMCode calls take k in Mpc⁻¹ and P in Mpc³. Some historical CAMB
and patched-CLASS input fixtures are stored in h-units, as recorded in their
headers. Convert those inputs when loading them; do not regenerate or relabel
the stored reference numbers to accommodate an API change.

`test_hmcode_units.jl` compares public direct/scalar/fast/boost calls with the
internal h-unit numerical calculation at several values of h. The
`hmcode_forwarddiff_primal_reference.txt` fixture preserves the native Float64
direct/smart DMO/feedback outputs from before the ForwardDiff generalization.

## Differentiation

`test_differentiation_interface.jl` uses prepared DifferentiationInterface
ForwardDiff and central FiniteDiff gradients. It compares individual
coordinates, not just vector norms, and covers linear emulators, Halofit,
DMO and feedback HMCode, spectrum-only inputs, and moving baryonic features
inside the interpolation interval.

The growth-parameter regressions require exact independence from inactive
parameters. Active derivatives are compared against
`data/hmcode_growth_gradient_reference.txt`: central finite differences at two
step sizes, with the same equation/initial conditions/spline and a tighter
ODE tolerance of 1e-13, cross-checked against tight-solve ForwardDiff.
Default-step finite differences of the production-tolerance adaptive solve
are not a reliable curvature-derivative reference. The production solver and
gradient acceptance tolerances are unchanged.

To reproduce the convergence tables from the repository root, using the
normal resolved test environment without overwriting the fixture:

```julia
using Pkg
root = pwd()
Pkg.test(; test_args=["hmcode-review"], test_fn=() -> begin
    project = dirname(Base.active_project())
    script = joinpath(root, "test/reference/growth_sensitivity.jl")
    run(`$(Base.julia_cmd()) --startup-file=no --check-bounds=yes --depwarn=yes --project=$project $script`)
end)
```

The ForwardDiff tests above do not establish Mooncake support or second
derivatives at interpolation/capping kinks.
Reactant package tests cover forward compilation, numerical agreement, and
reuse of compiled executables with changed numerical inputs. Enzyme's
compiled differentiation implementation is not a Mapse unit-test target.

## Native Mooncake

`Pkg.test(; test_args=["hmcode-mooncake"])` runs the native Mooncake rules and
public-path regressions, including the NFW, one-halo and baryonic mutation
contracts. These tests also run unconditionally in the complete suite and the
`hmcode` subset. `hmcode-pipelines` selects the independent 384-coordinate
Pmm/Pcb and live-emulator regressions alone; their frozen text fixtures live in
`fixtures/hmcode_mooncake_spectrum` and `fixtures/hmcode_mooncake_emulators`.

Coverage is first-order native `Float64`, with serial/threaded assembly and
`use_fast_specials=true`. Tests compare prepared gradients componentwise with
ForwardDiff, exercise nonzero cotangents, views and shared inputs, and require
input restoration and exact A/B/A value/gradient replay. Public-path values
allow only `8eps(Float64)` relative roundoff against ordinary evaluation, with
zero absolute tolerance; existing exact primitive checks remain exact. This is separate
from the unchanged gradient and scientific-reference tolerances.

NFW derivatives are analytical derivatives of the actual piecewise Si/Ci
approximation. Growth/root sensitivities and the existing emulator pullback
retain local ForwardDiff components, not a whole-HMCode forward-mode Jacobian.
The `hmcode-threaded` subset covers private worker state, nonzero cotangents,
restoration, supported identity aliases, rejected overlapping storage headers,
output shape and independent-caller concurrency. The
`hmcode-threaded-pipelines` subset covers direct/fast DMO and feedback, passive
integer/Float32 redshift grids, integer temperature, and prepared MC/FD A/B/A.
Both files also run in the normal suite. Concurrent tests use fixed method
definitions; do not load methods or clear Mooncake caches during evaluation.
The assembly boundary also exercises `use_fast_specials=false` against serial
reverse mode, including aliases and nonzero cotangents. End-to-end validation
of that slow path and second derivatives remain outside the certified scope;
see the native differentiation section in the package docs.
