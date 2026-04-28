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
            ("Nitrogen", base_dir / "Surrogate" / "Nitrogen"),
            ("Hydrogen", base_dir / "Surrogate" / "Hydrogen"),
            # ("HFE-7000", base_dir / "Surrogate" / "RE347MCC"),
            # ("Oxygen",   base_dir / "Surrogate" / "Oxygen"),
        ]
        
        print("========================================================")
        print("CoolSurrogateProp fluid database")
        # 1) Recursively check all surrogate folders exist
        for name, folder in fluids:
            print("%s | %s" %(name, folder))
            if not folder.exists():
                raise FileNotFoundError(f"Missing surrogate folder for fluid '{name}' at: {folder.resolve()}")
        print("========================================================")
        # 1) Instantiate all CoolSurrogateProp once (outside jit regions)
        self._props = [SurrogateEoS(path, JIT=True) for (name, path) in fluids]

        # 2) Map fluid name -> id (order defines id)
        self.fluid_id = {name: i for i, (name, _) in enumerate(fluids)}

        # 3) Define all branch-tables you want:  <ATTR ON SELF> -> <METHOD ON PROP>
        BRANCH_MAP = {
            # ============================================
            # Vapor: Y(P, T)
            # ============================================
            "VPT_RHO":   "vPT_Rho",
            "VPT_CP":    "vPT_Cp",
            "VPT_CV":    "vPT_Cv",
            "VPT_H":     "vPT_H",
            "VPT_S":     "vPT_S",
            "VPT_KAPPA": "vPT_Kappa",
            "VPT_MU":    "vPT_Mu",
            # Vapor partial derivatives
            "VPT_dRho_dP_T": "vPT_dRhodP_T",
            "VPT_dRho_dT_P": "vPT_dRhodT_P",
            "VPT_dP_dRho_T": "vPT_dPdRho_T",
            "VPT_dP_dT_Rho": "vPT_dPdT_Rho",
            "VPT_dT_dP_H":   "vPT_dTdP_H", # NEW
            "VPT_dT_dH_P":   "vPT_dTdH_P", # NEW
            "VPT_dU_dRho_T": "vPT_dUdRho_T",
            "VPT_dU_dT_Rho": "vPT_dUdT_Rho",
            "VPT_dU_dP_T":   "vPT_dUdP_T",
            "VPT_dU_dT_P":   "vPT_dUdT_P",
            # Entropy
            "VPT_dS_dRho_T": "vPT_dSdRho_T",
            "VPT_dS_dT_Rho": "vPT_dSdT_Rho",
            "VPT_dS_dP_T":   "vPT_dSdP_T",
            "VPT_dS_dT_P":   "vPT_dSdT_P",
            # ============================================
            # Liquid: Y(P, T)
            # ============================================
            "LPT_RHO":   "lPT_Rho",
            "LPT_CP":    "lPT_Cp",
            "LPT_CV":    "lPT_Cv",
            "LPT_H":     "lPT_H",
            "LPT_S":     "lPT_S",
            "LPT_KAPPA": "lPT_Kappa",
            "LPT_MU":    "lPT_Mu",
            # Liquid partial derivatives
            "LPT_dRho_dP_T": "lPT_dRhodP_T",
            "LPT_dRho_dT_P": "lPT_dRhodT_P",
            "LPT_dP_dRho_T": "lPT_dPdRho_T",
            "LPT_dP_dT_Rho": "lPT_dPdT_Rho",
            "LPT_dT_dP_H":   "lPT_dTdP_H", # NEW
            "LPT_dT_dH_P":   "lPT_dTdH_P", # NEW
            "LPT_dU_dRho_T": "lPT_dUdRho_T",
            "LPT_dU_dT_Rho": "lPT_dUdT_Rho",
            "LPT_dU_dP_T":   "lPT_dUdP_T",
            "LPT_dU_dT_P":   "lPT_dUdT_P",
            # Entropy
            "LPT_dS_dRho_T": "lPT_dSdRho_T",
            "LPT_dS_dT_Rho": "lPT_dSdT_Rho",
            "LPT_dS_dP_T":   "lPT_dSdP_T",
            "LPT_dS_dT_P":   "lPT_dSdT_P",
            # ============================================
            # Fluid: Y(P, H)
            # ============================================
            "VPH_T": "vPH_T",
            "LPH_T": "lPH_T",
            # ============================================
            # Saturation relationships
            # ============================================
            "T_SAT": "T_sat",
            "P_SAT": "P_sat",
            # Vapor(P)
            "VP_RHO_SAT":   "vP_Rho_sat",
            "VP_H_SAT":     "vP_H_sat",
            "VP_S_SAT":     "vP_S_sat",
            "VP_KAPPA_SAT": "vP_Kappa_sat",
            "VP_CP_SAT":    "vP_Cp_sat",
            "VP_MU_SAT":    "vP_Mu_sat",
            # Vapor(T)
            "VT_RHO_SAT":   "vT_Rho_sat",
            "VT_H_SAT":     "vT_H_sat",
            "VT_S_SAT":     "vT_S_sat",
            "VT_KAPPA_SAT": "vT_Kappa_sat",
            "VT_CP_SAT":    "vT_Cp_sat",
            "VT_MU_SAT":    "vT_Mu_sat",
            # Liquid(P)
            "LP_RHO_SAT":   "lP_Rho_sat",
            "LP_H_SAT":     "lP_H_sat",
            "LP_S_SAT":     "lP_S_sat",
            "LP_KAPPA_SAT": "lP_Kappa_sat",
            "LP_CP_SAT":    "lP_Cp_sat",
            "LP_MU_SAT":    "lP_Mu_sat",
            # Liquid(T)
            "LT_RHO_SAT":   "lT_Rho_sat",
            "LT_H_SAT":     "lT_H_sat",
            "LT_S_SAT":     "lT_S_sat",
            "LT_KAPPA_SAT": "lT_Kappa_sat",
            "LT_CP_SAT":    "lT_Cp_sat",
            "LT_MU_SAT":    "lT_Mu_sat",
            # Surface tension
            "LP_SIGMA":   "lP_Sigma",
            "LT_SIGMA":   "lT_Sigma",
        }

        # 4) Build branch tables generically
        for attr, method in BRANCH_MAP.items():
            table = tuple(getattr(p, method) for p in self._props)
            # Optional: validate all methods exist
            if any(m is None for m in table):
                raise AttributeError(f"Method '{method}' missing on one of the props")
            setattr(self, attr, table)
        return
    
    "Extract single-fluid ID by fluid name"
    def get_fluid_id(self, fluid_name):
        return self.fluid_id[fluid_name]

    """
    Fluid thermophysical properties for the selected fluid ID
    Each method uses jax.lax.switch to select the correct surrogate model
    based on the fluid ID provided.
    JIT-compilable.
    """
    # =====================================================================
    # Y_vapor = Y(P, T)
    # =====================================================================
    # Vapor Y(P, T)
    vPT_Rho      = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_RHO,      x)
    vPT_Cp       = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_CP,       x)
    vPT_Cv       = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_CV,       x)
    vPT_H        = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_H,        x)
    vPT_S        = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_S,        x)
    vPT_Kappa    = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_KAPPA,    x)
    vPT_Mu       = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_MU,       x)
    # Vapor partials (P, T)
    vPT_dRhodP_T = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dRho_dP_T, x)
    vPT_dRhodT_P = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dRho_dT_P, x)
    vPT_dPdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dP_dRho_T, x)
    vPT_dPdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dP_dT_Rho, x)
    vPT_dTdP_H   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dT_dP_H,   x) # NEW
    vPT_dTdH_P   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dT_dH_P,   x) # NEW
    vPT_dUdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dU_dRho_T, x)
    vPT_dUdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dU_dT_Rho, x)
    vPT_dUdP_T   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dU_dP_T,   x)
    vPT_dUdT_P   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dU_dT_P,   x)
    # Vapor entropy partials
    vPT_dSdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dS_dRho_T, x)
    vPT_dSdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dS_dT_Rho, x)
    vPT_dSdP_T   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dS_dP_T,   x)
    vPT_dSdT_P   = lambda self, fid, x:  jax.lax.switch(fid, self.VPT_dS_dT_P,   x)
    # =====================================================================

    # =====================================================================
    # Y_liquid = Y(P, T)
    # =====================================================================
    # Liquid Y(P, T)
    lPT_Rho      = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_RHO,      x)
    lPT_Cp       = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_CP,       x)
    lPT_Cv       = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_CV,       x)
    lPT_H        = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_H,        x)
    lPT_S        = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_S,        x)
    lPT_Kappa    = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_KAPPA,    x)
    lPT_Mu       = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_MU,       x)
    # Liquid partials (P, T)
    lPT_dRhodP_T = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dRho_dP_T, x)
    lPT_dRhodT_P = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dRho_dT_P, x)
    lPT_dPdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dP_dRho_T, x)
    lPT_dPdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dP_dT_Rho, x)
    lPT_dTdP_H   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dT_dP_H,   x) # NEW
    lPT_dTdH_P   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dT_dH_P,   x) # NEW
    lPT_dUdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dU_dRho_T, x)
    lPT_dUdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dU_dT_Rho, x)
    lPT_dUdP_T   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dU_dP_T,   x)
    lPT_dUdT_P   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dU_dT_P,   x)
    # Liquid entropy partials
    lPT_dSdRho_T = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dS_dRho_T, x)
    lPT_dSdT_Rho = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dS_dT_Rho, x)
    lPT_dSdP_T   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dS_dP_T,   x)
    lPT_dSdT_P   = lambda self, fid, x:  jax.lax.switch(fid, self.LPT_dS_dT_P,   x)
    # =====================================================================

    # =====================================================================
    # T_fluid = T(P, HMASS)
    # =====================================================================
    vPH_T = lambda self, fid, x:  jax.lax.switch(fid, self.VPH_T, x)
    lPH_T = lambda self, fid, x:  jax.lax.switch(fid, self.LPH_T, x)
    # =====================================================================
    # Saturation T(P) and P(T)
    T_sat        = lambda self, fid, P:  jax.lax.switch(fid, self.T_SAT,        P)
    P_sat        = lambda self, fid, T:  jax.lax.switch(fid, self.P_SAT,        T)
    # Saturated vapor (P)
    vP_Rho_sat   = lambda self, fid, P:  jax.lax.switch(fid, self.VP_RHO_SAT,   P)
    vP_H_sat     = lambda self, fid, P:  jax.lax.switch(fid, self.VP_H_SAT,     P)
    vP_S_sat     = lambda self, fid, P:  jax.lax.switch(fid, self.VP_S_SAT,     P)
    vP_Kappa_sat = lambda self, fid, P:  jax.lax.switch(fid, self.VP_KAPPA_SAT, P)
    vP_Cp_sat    = lambda self, fid, P:  jax.lax.switch(fid, self.VP_CP_SAT,    P)
    vP_Mu_sat    = lambda self, fid, P:  jax.lax.switch(fid, self.VP_MU_SAT,    P)
    # Saturated vapor (T)
    vT_Rho_sat   = lambda self, fid, T:  jax.lax.switch(fid, self.VT_RHO_SAT,   T)
    vT_H_sat     = lambda self, fid, T:  jax.lax.switch(fid, self.VT_H_SAT,     T)
    vT_S_sat     = lambda self, fid, T:  jax.lax.switch(fid, self.VT_S_SAT,     T)
    vT_Kappa_sat = lambda self, fid, T:  jax.lax.switch(fid, self.VT_KAPPA_SAT, T)
    vT_Cp_sat    = lambda self, fid, T:  jax.lax.switch(fid, self.VT_CP_SAT,    T)
    vT_Mu_sat    = lambda self, fid, T:  jax.lax.switch(fid, self.VT_MU_SAT,    T)
    # Saturated liquid (P)
    lP_Rho_sat   = lambda self, fid, P:  jax.lax.switch(fid, self.LP_RHO_SAT,   P)
    lP_H_sat     = lambda self, fid, P:  jax.lax.switch(fid, self.LP_H_SAT,     P)
    lP_S_sat     = lambda self, fid, P:  jax.lax.switch(fid, self.LP_S_SAT,     P)
    lP_Kappa_sat = lambda self, fid, P:  jax.lax.switch(fid, self.LP_KAPPA_SAT, P)
    lP_Cp_sat    = lambda self, fid, P:  jax.lax.switch(fid, self.LP_CP_SAT,    P)
    lP_Mu_sat    = lambda self, fid, P:  jax.lax.switch(fid, self.LP_MU_SAT,    P)
    # Saturated liquid (T)
    lT_Rho_sat   = lambda self, fid, T:  jax.lax.switch(fid, self.LT_RHO_SAT,   T)
    lT_H_sat     = lambda self, fid, T:  jax.lax.switch(fid, self.LT_H_SAT,     T)
    lT_S_sat     = lambda self, fid, T:  jax.lax.switch(fid, self.LT_S_SAT,     T)
    lT_Kappa_sat = lambda self, fid, T:  jax.lax.switch(fid, self.LT_KAPPA_SAT, T)
    lT_Cp_sat    = lambda self, fid, T:  jax.lax.switch(fid, self.LT_CP_SAT,    T)
    lT_Mu_sat    = lambda self, fid, T:  jax.lax.switch(fid, self.LT_MU_SAT,    T)
    # Surface tension
    lP_Sigma     = lambda self, fid, P:  jax.lax.switch(fid, self.LP_SIGMA,     P)
    lT_Sigma     = lambda self, fid, T:  jax.lax.switch(fid, self.LT_SIGMA,     T)

    def PropI(self, prop1, prop2):
        return jnp.stack((jnp.array(prop1) , jnp.array(prop2))).T
    
    def SatI(self, prop1):
        return jnp.array(prop1).reshape(-1,1)