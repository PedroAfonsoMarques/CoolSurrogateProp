# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:36:19 2026

@author: pmarques
"""

#%% Import packages

import jax.numpy as jnp
from flax.core import FrozenDict

#%% Stainless steel 316
# https://trc.nist.gov/cryogenics/materials/316Stainless/316Stainless_rev.htm

# Thermal conductivity [W/mK] (1-300 K)
_SS316_Kappa1 = jnp.array([
    -1.4087, 1.3982, 0.2543, -0.6260, 0.2334, 
    0.4256, -0.4658, 0.1650, -0.0199,
])

# Specific heat [J/kg/K] (4-50 K)
_SS316_C1 = jnp.array([
    12.2486, -80.6422, 218.743, -308.854, 239.5296, 
    -89.9982, 3.15315, 8.44996, -1.91368,
])

# Specific heat [J/kg/K] (50-300 K)
_SS316_C2 = jnp.array([
    -1879.464, 3643.198, 76.70125, -6176.028, 7437.6247,
    -4305.7217, 1382.4627, -237.22704, 17.05262,
])

# Compile material coefficients
DATA = FrozenDict({
    "Rho":    7954.0,
    "Kappa1": _SS316_Kappa1,
    "C1":     _SS316_C1,
    "C2":     _SS316_C2,
    "T_C12":  50.0
})