import numpy as np
import camb

# Exact cosmology matching the Mapse benchmark reference
# ln10As=3.044, ns=0.9649, H0=67.36, omega_b=0.02237, omega_cdm=0.12, mnu=0.06
H0 = 67.36
h = H0 / 100.0
ombh2 = 0.02237
omch2 = 0.12
omk = 0.0
ns = 0.9649
As = np.exp(3.044) * 1e-10
mnu = 0.06

# Standard target grid in physical units
k_out_h = np.logspace(-3, 1, 128)
k_out_phys = k_out_h * h
z_fine = np.linspace(0.0, 3.5, 150)

pars = camb.CAMBparams()
pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
pars.InitPower.set_params(As=As, ns=ns)
# Make sure CAMB computes fine enough grid internally to support high-k and high-z interpolation
pars.set_matter_power(redshifts=[0.0], kmax=50.0, nonlinear=True)

print("Generating CAMB DMO baseline...")
# 1. DMO baseline
pars.NonLinearModel.set_params(halofit_version='mead2020')
PK_interp_dmo = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4.0, zs=z_fine)
pk_dmo = PK_interp_dmo.P(z_fine, k_out_phys)
np.savetxt("camb_pk_hmcode_dmo.txt", pk_dmo)

print("Generating CAMB Baryonic Feedback baseline...")
# 2. Baryonic feedback
pars.NonLinearModel.set_params(halofit_version='mead2020_feedback', HMCode_logT_AGN=7.8)
PK_interp_fb = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4.0, zs=z_fine)
pk_fb = PK_interp_fb.P(z_fine, k_out_phys)
np.savetxt("camb_pk_hmcode_fb.txt", pk_fb)

print("CAMB text fixtures generated successfully (camb_pk_hmcode_dmo.txt, camb_pk_hmcode_fb.txt).")
