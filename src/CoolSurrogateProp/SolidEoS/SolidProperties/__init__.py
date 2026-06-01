import importlib
import pkgutil
import jax.numpy as jnp
from flax.core import FrozenDict

"Convert a raw material dict to a FrozenDict with JAX arrays."
def _freeze(raw: dict) -> FrozenDict:
    # Setup frozen dictionary
    out = {
        "Rho":      float(raw["Rho"]),
        "Kappa1":   jnp.array(raw["Kappa1"]),
        "C1":       jnp.array(raw["C1"]),
        "single_C": bool(raw["single_C"]),
    }
    if raw["single_C"]:
        out["C2"]    = out["C1"]       # alias — no extra allocation
        out["T_C12"] = jnp.inf         # branch never taken
    else:
        out["C2"]    = jnp.array(raw["C2"])
        out["T_C12"] = float(raw["T_C12"])
    return FrozenDict(out)

MATERIALS: dict[str, FrozenDict] = {}

for _info in pkgutil.iter_modules(__path__):
    _mod = importlib.import_module(f"{__name__}.{_info.name}")
    if hasattr(_mod, "DATA") and isinstance(_mod.DATA, dict):
        MATERIALS[_info.name] = _freeze(_mod.DATA)