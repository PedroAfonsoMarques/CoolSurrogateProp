# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:37:06 2026

@author: pmarques
"""

#%% Import packages

import jax.numpy as jnp
from flax.core import FrozenDict

#%% Stainless steel 304
# https://trc.nist.gov/cryogenics/materials/304Stainless/304Stainless_rev.htm

# Thermal conductivity [W/m/K] (1-300 K)
_SS304_Kappa1 = jnp.array([
    -1.4087, 1.3982, 0.2543, -0.6260, 0.2334,
     0.4256, -0.4658, 0.1650, -0.0199,
])

# Specific heat [J/kg/K] (1-300 K)
_SS304_C1 = jnp.array([
    22.0061, -127.5528, 303.647, -381.0098, 274.0328,
    -112.9212, 24.7593, -2.239153, 0.0,
])

# Compile material coefficients
DATA = FrozenDict({
    "Rho":    7930.0,
    "Kappa1": _SS304_Kappa1,
    "C1":     _SS304_C1,
    "C2":     _SS304_C1,
    "T_C12":  300.0,
})