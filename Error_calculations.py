import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates, center_of_mass
import os
from scipy.integrate import cumulative_trapezoid as cumtrapz
from matplotlib import rcParams

# -------------------------------
# General Uncertainty Functions
# -------------------------------

def mean_and_std(values):
    """Return the mean and sample standard deviation of a list/array."""
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values, ddof=1)  # sample std
    return mean, std


def typeA_uncertainty(std, n):
    """Return Type A standard uncertainty of the mean."""
    return std / np.sqrt(n)


def combine_uncertainties_quadrature(*uncs):
    """Combine multiple independent uncertainties in quadrature."""
    return np.sqrt(np.sum(np.array(uncs)**2))


def propagated_unc_subtraction(u1, u2):
    """Return propagated uncertainty for a subtraction (x2 - x1)."""
    return np.sqrt(u1**2 + u2**2)


def uncertainty_of_mean(values, u_single=None):
    """
    Return uncertainty of the mean for a list of repeated measurements.

    If u_single is given, include it as a Type B contribution.
    """
    mean, std = mean_and_std(values)
    n = len(values)
    uA = typeA_uncertainty(std, n)
    if u_single is not None:
        uB = u_single / np.sqrt(n)
        u_combined = combine_uncertainties_quadrature(uA, uB)
    else:
        u_combined = uA
    return mean, u_combined


# -------------------------------
# Example: Using the functions
# -------------------------------

# Example dataset: 3 pairs (off, on)
measurements = np.array([
    (0.224, 1.35),
    (7.87,  9.11),
    (0.222, 1.47)
])

# Known reading uncertainty (±0.01 µW)
u_read = 0.01

# 1. Compute differences and their uncertainty
diffs = measurements[:, 1] - measurements[:, 0]
u_diff_single = propagated_unc_subtraction(u_read, u_read)  # per difference

# 2. Compute mean and combined uncertainty
mean_P, u_P = uncertainty_of_mean(diffs, u_diff_single)

# 3. Display result
print(f"Measured differences: {diffs}")
print(f"Mean value: {mean_P:.3f} µW")
print(f"Combined uncertainty: {u_P:.3f} µW")
print(f"Final result: P = {mean_P:.3f} ± {u_P:.3f} µW")

import numpy as np

Gamma = 1.472e8
u_Gamma = 0.007e8

# Constants
h = 6.62607015e-34       # J·s
c = 2.99792458e8          # m/s
lam = 328.1629601e-9      # m
u_lam = 0.0000022e-9      # m
tau = 6.79e-9             # s
u_tau = 0.03e-9           # s

# Derived quantities
Gamma = 1 / tau
u_Gamma = Gamma * (u_tau / tau)   # propagate inverse

# Compute I_sat
Isat = (np.pi * h * c * Gamma) / (3 * lam**3)

# Relative uncertainties
rel_lam = 3 * (u_lam / lam)
rel_gamma = u_Gamma / Gamma

# Total relative and absolute uncertainty
rel_I = np.sqrt(rel_lam**2 + rel_gamma**2)
u_Isat = Isat * rel_I

print(f"Gamma = {Gamma:.3e} ± {u_Gamma:.3e} s⁻¹")
print(f"I_sat = {Isat:.3f} ± {u_Isat:.3f} W/m² ({rel_I*100:.3f}%)")


Rb_gamma = 6.065e6
Rb_lam = 780.2413272e-9
IsatRb = (np.pi * h * c * Rb_gamma) / (3 * Rb_lam**3)
print(IsatRb)