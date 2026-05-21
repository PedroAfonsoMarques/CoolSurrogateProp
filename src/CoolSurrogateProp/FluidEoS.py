#%% Import packages

import jax
import jax.numpy as jnp

from pathlib import Path
from CoolSurrogateProp.ModelEoS import SurrogateEoS

#%% Define class

"Equation-of-state template"
class EquationOfState:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent   
        
        # 0) Define available fluids and their surrogate model paths
        fluids = [
            ("Parahydrogen", base_dir / "Surrogate" / "Parahydrogen"),
            ("Hydrogen",     base_dir / "Surrogate" / "Hydrogen"),
            ("Nitrogen",     base_dir / "Surrogate" / "Nitrogen"),
            ("Oxygen",       base_dir / "Surrogate" / "Oxygen"),
            ("Methane",      base_dir / "Surrogate" / "Methane"),
        ]
        
        print("========================================================")
        print("CoolSurrogateProp fluid database")
        # 1) Recursively check all surrogate folders exist
        for name, folder in fluids:
            print("%s | %s" %(name, folder))
            if not folder.exists():
                raise FileNotFoundError(f"Missing surrogate folder for fluid '{name}' at: {folder.resolve()}")
        print("========================================================")

        # 1) Instantiate all CoolSurrogateProp fluids once (outside jit regions)
        self._props = [SurrogateEoS(path, JIT=True) for (name, path) in fluids]

        # 2) Map fluid name -> id (order defines id)
        self.fluid_id = {name: i for i, (name, _) in enumerate(fluids)}

        # 3) Auto-discover all property names from the first surrogate
        PROPERTIES = [
            # Loop and assign all properties generically, so that we don't have to hardcode them here
            name for name in vars(self._props[0])
            # Skip private Python attributes (anything starting with '_')
            if not name.startswith('_')  
        ]

        # 4) Build branch tables generically
        for name in PROPERTIES:
            # Generate table of all surrogate methods
            table = tuple(getattr(p, name) for p in self._props)
            
            # # Create method to evaluate properties
            # def make_method(t):
            #     return lambda self, fid, x: jax.lax.switch(fid, t, x)
            # # Set method according to the property name
            # setattr(self.__class__, name, make_method(table))
            
            # Create method to evaluate properties
            def make_method(t, is_sat):
                if is_sat:
                    return lambda self, fid, *args: jax.lax.switch(fid, t, self.SaturatedInput(*args))
                else:
                    return lambda self, fid, *args: jax.lax.switch(fid, t, self.SinglePhaseInput(*args))
            # Set method according to the property name
            # setattr(self.__class__, name, make_method(table, is_sat="_sat" in name))
            setattr(self.__class__, name, make_method(table, is_sat=any(s in name.lower() for s in ("sat", "sigma"))))


    "Extract single-fluid ID by fluid name"
    def get_fluid_id(self, fluid_name):
        return self.fluid_id[fluid_name]

    "Evaluate single-phase property"
    def SinglePhaseInput(self, prop1, prop2):
        return jnp.stack((jnp.array(prop1) , jnp.array(prop2))).T

    "Evaluate saturation property"
    def SaturatedInput(self, prop1):
        return jnp.array(prop1).reshape(-1,1)