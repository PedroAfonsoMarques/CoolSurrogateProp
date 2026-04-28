# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 14:35:08 2026

@author: pmarques
"""

#%% Import packages

import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from CoolSurrogateProp.SolidProperties import MATERIALS as _DEFAULT_MATERIALS

#%% Core NIST polynomial evaluator

@jax.jit
def _NIST_eval(T: float, coeffs: jnp.ndarray) -> float:
    """
    Evaluate a NIST cryogenic log-log polynomial in Horner form.

    Model:  log10(y) = sum_{k=0}^{8} coeffs[k] * log10(T)^k
    """
    log10T = jnp.log10(T)
    log10y = coeffs[-1]
    for c in coeffs[-2::-1]:        # Horner's method: stable + minimal FLOPs
        log10y = log10y * log10T + c
    return 10.0 ** log10y

#%% EquationOfSolid class

class EquationOfSolid:

    def __init__(self, materials: dict[str, FrozenDict] | None = None):
        self._registry = materials or _DEFAULT_MATERIALS
        self._index    = {n: i for i, n in enumerate(self._registry)}

        # Build branches once at construction — closes over JAX arrays only
        branches = [self._make_branch(fd) for fd in self._registry.values()]
        self._jit_props = jax.jit(
            lambda T, idx: jax.lax.switch(idx, branches, T)
        )

    def get_material_id(self, name: str) -> int:
        """Return the integer index for a material name."""
        if name not in self._index:
            raise KeyError(f"Unknown material '{name}'. Available: {list(self._index)}")
        return self._index[name]

    @staticmethod
    def _make_branch(mat: FrozenDict):
        """Return a JIT-traceable branch closing over one material's FrozenDict."""
        def branch(T):
            Kappa = _NIST_eval(T, mat["Kappa1"])
            C = jax.lax.cond(
                T < mat["T_C12"],
                lambda T: _NIST_eval(T, mat["C1"]),
                lambda T: _NIST_eval(T, mat["C2"]),
                T,
            )
            return {"Rho": mat["Rho"], "Kappa": Kappa, "C": C}
        return branch

    def properties(self, material: int, T: float) -> dict:
        """Evaluate all solid properties at temperature T [K]."""
        idx = self.get_material_id(material) if isinstance(material, str) else material
        return self._jit_props(T, idx)

    def Kappa(self, material: int, T: float) -> float:
        """Thermal conductivity [W/m/K] at temperature T [K]."""
        return self.properties(material, T)["Kappa"]

    def C(self, material: int, T: float) -> float:
        """Specific heat [J/kg/K] at temperature T [K]."""
        return self.properties(material, T)["C"]

    def Rho(self, material: int) -> float:
        """Density [kg/m³] (temperature-independent)."""
        return self.properties(material, 1.0)["Rho"]

    def __repr__(self) -> str:
        return f"SolidMaterials([{', '.join(self._index)}])"