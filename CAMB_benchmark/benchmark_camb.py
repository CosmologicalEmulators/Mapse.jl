import numpy as np
import camb
import timeit

H0 = 67.36
h = H0 / 100.0
ombh2 = 0.02237
omch2 = 0.12
omk = 0.0
ns = 0.9649
As = np.exp(3.044) * 1e-10
mnu = 0.06

k_out_h = np.logspace(-3, 1, 128)
k_out_phys = k_out_h * h
z_fine = np.linspace(0.0, 3.5, 150)

def run_camb_dmo():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=50.0, nonlinear=True)
    pars.NonLinearModel.set_params(halofit_version='mead2020')
    PK_interp_dmo = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4.0, zs=z_fine)
    return PK_interp_dmo.P(z_fine, k_out_phys)

def run_camb_fb():
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=omk)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_matter_power(redshifts=[0.0], kmax=50.0, nonlinear=True)
    pars.NonLinearModel.set_params(halofit_version='mead2020_feedback', HMCode_logT_AGN=7.8)
    PK_interp_fb = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=50.0, zmax=4.0, zs=z_fine)
    return PK_interp_fb.P(z_fine, k_out_phys)

if __name__ == "__main__":
    print("Warming up CAMB instances...")
    run_camb_dmo()
    run_camb_fb()

    print("-" * 50)
    print("Benchmarking CAMB DMO (mean over 5 runs)...")
    t_dmo = timeit.timeit("run_camb_dmo()", globals=globals(), number=5) / 5.0
    print(f"CAMB DMO: {t_dmo * 1000:.2f} ms")

    print("-" * 50)
    print("Benchmarking CAMB Baryonic Feedback (mean over 5 runs)...")
    t_fb = timeit.timeit("run_camb_fb()", globals=globals(), number=5) / 5.0
    print(f"CAMB Baryonic Feedback: {t_fb * 1000:.2f} ms")
