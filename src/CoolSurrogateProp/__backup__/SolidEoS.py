#%% Import packages

import jax
import jax.numpy as jnp
from jax import tree_util
from flax.core import FrozenDict  # You can also just use Python dict if you prefer.

#%% Material property data — parameterize the models

# Copper parameters
EoS_Copper = FrozenDict({
    "kappa_coeff": jnp.array([1.39794306e-6, -1.02635276e-3, 2.75691180e-1, -3.21885722e1, 1.78460157e3]),
    "cp_coeffs":   jnp.array([2.6996e-11, -2.6280e-8, 9.9675e-6, -1.8348e-3, 1.5848e-1, -2.8821, 14.009]),
    "rho":         8940.0,
    "emissivity":  0.05 # Emissivity of copper https://www.thermoworks.com/emissivity-table/
})

# Quartz parameters
EoS_Quartz = FrozenDict({
    "kappa_coeff": jnp.array([0.0, 0.0, -8.79162873e-6, 6.00875972e-3, 6.67442814e-2]),
    "cp_coeffs":   jnp.array([0.0, 0.0, 1.077e-7, -8.3701e-5, 1.8631e-2, 1.6128, -21.692]),
    "rho":         2650.0,
    "emissivity":  0.92 # Emissivity of quartz https://www.thermoworks.com/emissivity-table/
})

# Aluminum parameters https://asm.matweb.com/search/specificmaterial.asp?bassnum=ma6061t6
EoS_Aluminum = FrozenDict({
    "kappa_coeff": jnp.array([0.0, 0.0, 0.0, 0.0, 167.0]),
    "cp_coeffs":   jnp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 896.0]),
    "rho":         2700.0,
    "emissivity":  0.77 # Emissivity of anodised aluminum https://www.thermoworks.com/emissivity-table/
})

#%% Evaluation functions

def kappa_eval(eos, T):
    p = eos["kappa_coeff"]
    return p[0]*T**4 + p[1]*T**3 + p[2]*T**2 + p[3]*T + p[4]

def cp_eval(eos, T):
    p = eos["cp_coeffs"]
    # Horner's method for polynomial evaluation
    return p[0]*T**6 + p[1]*T**5 + p[2]*T**4 + p[3]*T**3 + p[4]*T**2 + p[5]*T + p[6]

# Tds = du + p dv (assume dv = 0 for solids) => ds = du/dT
# Thus, entropy can be computed by integrating c_w/T dT
def entropy_eval(eos, T):
    p = eos["cp_coeffs"]
    return p[0]*(T/6)**6 + p[1]*(T/5)**5 + p[2]*(T/4)**4 + p[3]*(T/3)**3 + p[4]*(T/2)**2 + p[5]*T + p[6]*jnp.log(jnp.abs(T)+1e-10)

def rho_eval(eos, T=None):
    return eos["rho"]

def emissivity_eval(eos, T=None):
    return eos["emissivity"]

#%% Solid material class

"Equation-of-state for solid materials, evaluating thermal conductivity, specific heat capacity, and density."
class EquationOfSolid:
    def __init__(self):
        self.models = [
            lambda: EoS_Copper, 
            lambda: EoS_Quartz,
            lambda: EoS_Aluminum,
            ]
    "Select material based on its ID."
    def select_material(self, material_id):
        return jax.lax.switch(material_id, self.models)
    "Evaluate thermal conductivity"
    def kappa(self, eos, T):
        return kappa_eval(eos, T)
    "Evaluate specific heat capacity"
    def cp(self, eos, T):
        return cp_eval(eos, T)
    "Evaluate density"
    def rho(self, eos, T=None):
        return rho_eval(eos, T)
    "Evaluate density"
    def entropy(self, eos, T):
        return entropy_eval(eos, T)
    "Evaluate emissivity"
    def emissivity(self, eos, T=None):
        return emissivity_eval(eos, T)
    
    "Return material ID based on input string"
    def get_solid_id(self,solid):
        if solid == "Copper":
            index = 0
        elif solid == "Quartz":
            index = 1
        else:
            index = None
            raise ValueError(f"Material '{solid}' not recognized in EoSw_mapping.")
        # Return index
        return index

#%% Test the EquationOfSolid class

if __name__ == "__main__":

    EoW = EquationOfSolid()
    T = 77.0

    # Copper
    copper_eos = EoW.select_material(0)
    print("Copper:")
    print("  kappa:",   EoW.kappa(copper_eos, T))
    print("  cp:",      EoW.cp(copper_eos, T))
    print("  rho:",     EoW.rho(copper_eos))
    print("  entropy:", EoW.entropy(copper_eos, T))

    # Quartz
    quartz_eos = EoW.select_material(1)
    print("\nQuartz:")
    print("  kappa:",   EoW.kappa(quartz_eos, T))
    print("  cp:",      EoW.cp(quartz_eos, T))
    print("  rho:",     EoW.rho(quartz_eos))
    print("  entropy:", EoW.entropy(quartz_eos, T))

    # Aluminum
    aluminum_eos = EoW.select_material(2)
    print("\nAluminum:")
    print("  kappa:",   EoW.kappa(aluminum_eos, T))
    print("  cp:",      EoW.cp(aluminum_eos, T))
    print("  rho:",     EoW.rho(aluminum_eos))
    print("  entropy:", EoW.entropy(aluminum_eos, T))
    
    # Test with JIT
    @jax.jit
    def test_jit(material_id, T):
        eos = EoW.select_material(material_id)
        return {
            "kappa": EoW.kappa(eos, T),
            "cp":    EoW.cp(eos, T),
            "rho":   EoW.rho(eos)
        }
    
    # Test JIT for Copper
    result_copper = test_jit(0, T=jnp.linspace(100.0, 300.0, 10))
    print("\nJIT Copper Result:", result_copper) 

    # Test vmap for multiple materials
    material_ids = jnp.array([0, 1, 2])  # Copper, Quartz, Aluminum
    T_values = jnp.stack((jnp.linspace(100.0, 300.0, 10),
                          jnp.linspace(100.0, 300.0, 10),
                          jnp.linspace(100.0, 300.0, 10)))  # Temperature range
    
    test_vmap = jax.vmap(test_jit, in_axes=(0, 0))

    results_vmap = test_vmap(material_ids, T_values)
    print("\nJIT vmap Results:", results_vmap)

# %%
