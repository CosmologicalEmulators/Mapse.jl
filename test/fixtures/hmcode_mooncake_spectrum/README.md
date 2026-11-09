# HMCode input-spectrum gradient reference

Generated on 2026-11-08 in Narval compute job 8789807, campaign
`20261108_mooncake/rules-v8`, using Julia 1.12.5, Mapse 0.1.0 with the
uncommitted `nonlinear_develop` changes, AbstractCosmologicalEmulators 0.11.0,
Mooncake 0.5.67 and DifferentiationInterface 0.7.21. The campaign's resolved
environment and source snapshot retain the complete provenance.

The 384 inputs are column-major log Pmm followed by log Pcb, each on 64
log-spaced physical k values from 1e-4 to 1e2 Mpc^-1 and z = [0, 0.5, 1].
The initial spectra are
`Pmm(k,z) = 1e4*(k/0.1)^0.96/(1+(k/0.1)^2)^2/(1+z)^2`, with
`Pcb = 1.01*Pmm`. The test specifies the fixed curved, non-Lambda cosmology,
feedback temperature, mass grid and nonuniform output weights explicitly.

`forwarddiff_gradient.txt` is the independent reference, not the Mooncake
result. `parameters.txt` and `perturbed_parameters.txt` retain both inputs;
the perturbation is a reproducible local-box draw, not a cosmological prior.
`direction_1.txt` and `direction_2.txt` are normalized finite-difference
directions. The accepted Mooncake/reference maximum absolute discrepancy
was 6.938893903907228e-17. Fresh ForwardDiff and directional finite differences
are also checked by the test; these fixtures do not replace live validation.
