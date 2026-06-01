#%% Imports
import jax
import jax.numpy as jnp
from flax.core import FrozenDict
from .SolidProperties import MATERIALS

#%% Core NIST polynomial evaluator

@jax.jit
def _NIST_eval(T: jnp.ndarray, coeffs: jnp.ndarray) -> jnp.ndarray:
    "Horner via dot product: maps cleanly to a single BLAS/XLA kernel."
    log10T  = jnp.log10(T)
    powers  = log10T ** jnp.arange(len(coeffs))
    return 10.0 ** jnp.dot(coeffs, powers)

#%% Branch builder

def _make_branch(mat: FrozenDict):
    """
    Return a traceable scalar branch for one material.
    T must be a scalar — use SpecializedEoS for array inputs.
    """
    def branch(T):
        Kappa = _NIST_eval(T, mat["Kappa1"])
        C = jnp.where(
            T < mat["T_C12"],
            _NIST_eval(T, mat["C1"]),
            _NIST_eval(T, mat["C2"]),
        )
        return {"Rho": mat["Rho"], "Kappa": Kappa, "C": C}
    return branch

#%% SpecializedEoS — fixed material, array T

class SpecializedEoS:
    """
    Single-material EoS compiled for a fixed material.
    Accepts scalar or array T — no runtime dispatch overhead.
    """

    def __init__(self, name: str, mat: FrozenDict):
        self.name    = name
        self.Rho     = float(mat["Rho"])   # temperature-independent: just a float

        branch = _make_branch(mat)
        self._eval_scalar = jax.jit(branch)
        self._eval_array  = jax.jit(jax.vmap(branch))

    def _eval(self, T: jnp.ndarray) -> dict:
        T = jnp.asarray(T)
        return self._eval_array(T) if T.ndim > 0 else self._eval_scalar(T)

    def Kappa(self, T: jnp.ndarray) -> jnp.ndarray:
        """Thermal conductivity [W/m/K]."""
        return self._eval(T)["Kappa"]

    def C(self, T: jnp.ndarray) -> jnp.ndarray:
        """Specific heat [J/kg/K]."""
        return self._eval(T)["C"]

    def __repr__(self) -> str:
        return f"SpecializedEoS(material='{self.name}', Rho={self.Rho} kg/m³)"

#%% EquationOfSolid — registry + specialization

class EquationOfSolid:
    """
    Registry of solid materials. Use .specialize() to get a
    fast, material-fixed evaluator for use inside JAX pipelines.
    """

    def __init__(self, materials: dict[str, FrozenDict] | None = None):
        self._registry = materials or MATERIALS
        self._index    = {n: i for i, n in enumerate(self._registry)}

    def specialize(self, material: str | int) -> SpecializedEoS:
        """
        Return a SpecializedEoS for a fixed material.
        Call once at component construction, not inside eval().
        """
        if isinstance(material, int):
            name = list(self._registry)[material]
        else:
            name = material
        if name not in self._registry:
            raise KeyError(f"Unknown material '{name}'. Available: {list(self._registry)}")
        return SpecializedEoS(name, self._registry[name])

    def __repr__(self) -> str:
        return f"EquationOfSolid([{', '.join(self._registry)}])"
    
#%% Testing

if __name__ == "__main__":

    import jax
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    from CoolSurrogateProp.SolidEoS.SolidEoS import EquationOfSolid

    jax.config.update("jax_enable_x64", True)  # for better precision in plots

    for material in ["SS304", "SS316"]:

        solid_ID = EquationOfSolid()._index[material]

        EOS = EquationOfSolid().specialize(solid_ID)

        T = jnp.linspace(20,300,200)

        Rho_w   = EOS.Rho
        C_w     = EOS.C(T)
        Kappa_w = EOS.Kappa(T)

        fig, ax = plt.subplots(nrows=2, sharex=True)
        fig.suptitle(f"{material} properties")
        ax[0].plot(T, C_w,     label="C")
        ax[1].plot(T, Kappa_w, label="Kappa")
        ax[0].set_ylabel("C [J/kg/K]")
        ax[1].set_ylabel("Kappa [W/m/K]")
        ax[1].set_xlabel("T [K]")
        ax[0].minorticks_on(); ax[0].grid(axis="both", which="both", alpha=0.3)
        ax[1].minorticks_on(); ax[1].grid(axis="both", which="both", alpha=0.3)
