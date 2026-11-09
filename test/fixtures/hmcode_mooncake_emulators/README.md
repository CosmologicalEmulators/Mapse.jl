# Live-emulator gradient references

Compute-node references from 2026-11-08, campaign
`20261108_mooncake/rules-v8`, Julia 1.12.5, AbstractCosmologicalEmulators 0.11.0,
Mooncake 0.5.67 and DifferentiationInterface 0.7.21. Mapse is the uncommitted
`nonlinear_develop` snapshot, version 0.1.0. Full source/dependency provenance
is retained in that campaign.

* `linear`: job 8789806, eight cosmological inputs through live growth and
  linear Pmm/Pcb emulators. Maximum Mooncake/ForwardDiff difference 5.33e-15.
* `direct`: job 8789967, the same live inputs plus log10(T_AGN), through direct
  HMCode with feedback. Maximum difference 3.31e-8.
* `smart`: job 8790027, nine live inputs through the moving-feature smart
  feedback path, with 32 fine redshifts over z=0–3 and 24 coarse nodes.
  Maximum difference 2.95e-9. Both baseline and perturbed inputs retain enough
  nodes on each side of the moving feature to exercise the two-spline path.

Pmm/Pcb use Mapse's `mnuw0wacdm_class` artifact; growth uses
`ACE_mnuw0wacdm_ln10As_basis`. Inputs are ln(10^10 As), ns, H0, physical baryon
and cold-dark-matter densities, neutrino mass, w0, wa, then log10(T_AGN) where
applicable. All emulators execute inside the differentiated function.

Each directory preserves baseline and locally perturbed inputs, ForwardDiff
gradients, and central finite-difference gradients. The perturbations are local
box draws, not a declared cosmological prior. CI checks fresh derivatives as
well as the frozen ForwardDiff reference. Numerical settings are explicit in
`test_hmcode_emulator_mooncake.jl` and the shared pipeline helpers.
