# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:42:57 2025

@author: pedro
"""

#%% Import packages

import jax
from CoolSurrogateProp.OptimizeEoS import PropNN
from CoolSurrogateProp.SaveAndLoad_EoS import load_jax_model, str_to_afns

#%% Define Thermopropax class

class SurrogateEoS:
    
    def __init__(self, FOL_IN, JIT=False):
        # =====================================================================
        # Y_vapor = Y(P, T)
        # =====================================================================
        # Properties
        self.vPT_Rho   = self.load_surrogate(FOL_IN,"gas_DMASS(P,T)", JIT)
        self.vPT_Cp    = self.load_surrogate(FOL_IN,"gas_CPMASS(P,T)", JIT)
        self.vPT_Cv    = self.load_surrogate(FOL_IN,"gas_CVMASS(P,T)", JIT)
        self.vPT_H     = self.load_surrogate(FOL_IN,"gas_HMASS(P,T)", JIT)
        self.vPT_S     = self.load_surrogate(FOL_IN,"gas_SMASS(P,T)", JIT)
        self.vPT_Kappa = self.load_surrogate(FOL_IN,"gas_CONDUCTIVITY(P,T)", JIT)
        self.vPT_Mu    = self.load_surrogate(FOL_IN,"gas_VISCOSITY(P,T)", JIT)
        # Partial derivatives
        self.vPT_dRhodP_T = self.load_surrogate(FOL_IN,"gas_d(DMASS)-d(P)_T(P,T)", JIT)
        self.vPT_dRhodT_P = self.load_surrogate(FOL_IN,"gas_d(DMASS)-d(T)_P(P,T)", JIT)
        self.vPT_dPdRho_T = self.load_surrogate(FOL_IN,"gas_d(P)-d(DMASS)_T(P,T)", JIT)
        self.vPT_dPdT_Rho = self.load_surrogate(FOL_IN,"gas_d(P)-d(T)_DMASS(P,T)", JIT)
        self.vPT_dTdP_H   = self.load_surrogate(FOL_IN,"gas_d(T)-d(P)_HMASS(P,T)", JIT) # NEW
        self.vPT_dTdH_P   = self.load_surrogate(FOL_IN,"gas_d(T)-d(HMASS)_P(P,T)", JIT) # NEW
        self.vPT_dUdRho_T = self.load_surrogate(FOL_IN,"gas_d(UMASS)-d(DMASS)_T(P,T)", JIT)
        self.vPT_dUdT_Rho = self.load_surrogate(FOL_IN,"gas_d(UMASS)-d(T)_DMASS(P,T)", JIT)
        self.vPT_dUdP_T   = self.load_surrogate(FOL_IN,"gas_d(UMASS)-d(P)_T(P,T)", JIT)
        self.vPT_dUdT_P   = self.load_surrogate(FOL_IN,"gas_d(UMASS)-d(T)_P(P,T)", JIT)
        # Entropy
        self.vPT_dSdRho_T = self.load_surrogate(FOL_IN,"gas_d(SMASS)-d(DMASS)_T(P,T)", JIT)
        self.vPT_dSdT_Rho = self.load_surrogate(FOL_IN,"gas_d(SMASS)-d(T)_DMASS(P,T)", JIT)
        self.vPT_dSdP_T   = self.load_surrogate(FOL_IN,"gas_d(SMASS)-d(P)_T(P,T)", JIT)
        self.vPT_dSdT_P   = self.load_surrogate(FOL_IN,"gas_d(SMASS)-d(T)_P(P,T)", JIT)
        
        # =====================================================================
        # Y_liquid = Y(P, T)
        # =====================================================================
        self.lPT_Rho   = self.load_surrogate(FOL_IN,"liquid_DMASS(P,T)", JIT)
        self.lPT_Cp    = self.load_surrogate(FOL_IN,"liquid_CPMASS(P,T)", JIT)
        self.lPT_Cv    = self.load_surrogate(FOL_IN,"liquid_CVMASS(P,T)", JIT)
        self.lPT_H     = self.load_surrogate(FOL_IN,"liquid_HMASS(P,T)", JIT)
        self.lPT_S     = self.load_surrogate(FOL_IN,"liquid_SMASS(P,T)", JIT)
        self.lPT_Kappa = self.load_surrogate(FOL_IN,"liquid_CONDUCTIVITY(P,T)", JIT)
        self.lPT_Mu    = self.load_surrogate(FOL_IN,"liquid_VISCOSITY(P,T)", JIT)
        # Partial derivatives
        self.lPT_dRhodP_T = self.load_surrogate(FOL_IN,"liquid_d(DMASS)-d(P)_T(P,T)", JIT)
        self.lPT_dRhodT_P = self.load_surrogate(FOL_IN,"liquid_d(DMASS)-d(T)_P(P,T)", JIT)
        self.lPT_dPdRho_T = self.load_surrogate(FOL_IN,"liquid_d(P)-d(DMASS)_T(P,T)", JIT)
        self.lPT_dPdT_Rho = self.load_surrogate(FOL_IN,"liquid_d(P)-d(T)_DMASS(P,T)", JIT)
        self.lPT_dTdP_H   = self.load_surrogate(FOL_IN,"liquid_d(T)-d(P)_HMASS(P,T)", JIT) # NEW
        self.lPT_dTdH_P   = self.load_surrogate(FOL_IN,"liquid_d(T)-d(HMASS)_P(P,T)", JIT) # NEW
        self.lPT_dUdRho_T = self.load_surrogate(FOL_IN,"liquid_d(UMASS)-d(DMASS)_T(P,T)", JIT)
        self.lPT_dUdT_Rho = self.load_surrogate(FOL_IN,"liquid_d(UMASS)-d(T)_DMASS(P,T)", JIT)
        self.lPT_dUdP_T   = self.load_surrogate(FOL_IN,"liquid_d(UMASS)-d(P)_T(P,T)", JIT)
        self.lPT_dUdT_P   = self.load_surrogate(FOL_IN,"liquid_d(UMASS)-d(T)_P(P,T)", JIT)
        # Entropy
        self.lPT_dSdRho_T = self.load_surrogate(FOL_IN,"liquid_d(SMASS)-d(DMASS)_T(P,T)", JIT)
        self.lPT_dSdT_Rho = self.load_surrogate(FOL_IN,"liquid_d(SMASS)-d(T)_DMASS(P,T)", JIT)
        self.lPT_dSdP_T   = self.load_surrogate(FOL_IN,"liquid_d(SMASS)-d(P)_T(P,T)", JIT)
        self.lPT_dSdT_P   = self.load_surrogate(FOL_IN,"liquid_d(SMASS)-d(T)_P(P,T)", JIT)
        
        # =====================================================================
        # Y_fluid = Y(P, HMASS)
        # =====================================================================
        self.vPH_T = self.load_surrogate(FOL_IN, "gas_T(P,HMASS)",    JIT)
        self.lPH_T = self.load_surrogate(FOL_IN, "liquid_T(P,HMASS)", JIT)
        
        # =====================================================================
        # Saturation temperature and pressure
        # =====================================================================
        self.T_sat = self.load_surrogate(FOL_IN,"saturated_vapor_T(P)", JIT)
        self.P_sat = self.load_surrogate(FOL_IN,"saturated_vapor_P(T)", JIT)
        # =====================================================================
        # Y_sat_vapor = Y(P) or Y(T)
        # =====================================================================
        # Vapor(P)
        self.vP_Rho_sat   = self.load_surrogate(FOL_IN,"saturated_vapor_DMASS(P)", JIT)
        self.vP_H_sat     = self.load_surrogate(FOL_IN,"saturated_vapor_HMASS(P)", JIT)
        self.vP_S_sat     = self.load_surrogate(FOL_IN,"saturated_vapor_SMASS(P)", JIT)
        self.vP_Kappa_sat = self.load_surrogate(FOL_IN,"saturated_vapor_CONDUCTIVITY(P)", JIT)
        self.vP_Cp_sat    = self.load_surrogate(FOL_IN,"saturated_vapor_CPMASS(P)",       JIT)
        self.vP_Mu_sat    = self.load_surrogate(FOL_IN,"saturated_vapor_VISCOSITY(P)",    JIT)
        # Vapor(T)
        self.vT_Rho_sat   = self.load_surrogate(FOL_IN,"saturated_vapor_DMASS(T)", JIT)
        self.vT_H_sat     = self.load_surrogate(FOL_IN,"saturated_vapor_HMASS(T)", JIT)
        self.vT_S_sat     = self.load_surrogate(FOL_IN,"saturated_vapor_SMASS(T)", JIT)
        self.vT_Kappa_sat = self.load_surrogate(FOL_IN,"saturated_vapor_CONDUCTIVITY(T)", JIT)
        self.vT_Cp_sat    = self.load_surrogate(FOL_IN,"saturated_vapor_CPMASS(T)",       JIT)
        self.vT_Mu_sat    = self.load_surrogate(FOL_IN,"saturated_vapor_VISCOSITY(T)",    JIT)
        # =====================================================================
        # Y_sat_liquid = Y(P) or Y(T)
        # =====================================================================
        # Liquid(P)
        self.lP_Rho_sat   = self.load_surrogate(FOL_IN,"saturated_liquid_DMASS(P)", JIT)
        self.lP_H_sat     = self.load_surrogate(FOL_IN,"saturated_liquid_HMASS(P)", JIT)
        self.lP_S_sat     = self.load_surrogate(FOL_IN,"saturated_liquid_SMASS(P)", JIT)
        self.lP_Kappa_sat = self.load_surrogate(FOL_IN,"saturated_liquid_CONDUCTIVITY(P)", JIT)
        self.lP_Cp_sat    = self.load_surrogate(FOL_IN,"saturated_liquid_CPMASS(P)",       JIT)
        self.lP_Mu_sat    = self.load_surrogate(FOL_IN,"saturated_liquid_VISCOSITY(P)",    JIT)
        # Liquid(T)
        self.lT_Rho_sat   = self.load_surrogate(FOL_IN,"saturated_liquid_DMASS(T)", JIT)
        self.lT_H_sat     = self.load_surrogate(FOL_IN,"saturated_liquid_HMASS(T)", JIT)
        self.lT_S_sat     = self.load_surrogate(FOL_IN,"saturated_liquid_SMASS(T)", JIT)
        self.lT_Kappa_sat = self.load_surrogate(FOL_IN,"saturated_liquid_CONDUCTIVITY(T)", JIT)
        self.lT_Cp_sat    = self.load_surrogate(FOL_IN,"saturated_liquid_CPMASS(T)",       JIT)
        self.lT_Mu_sat    = self.load_surrogate(FOL_IN,"saturated_liquid_VISCOSITY(T)",    JIT)
        # Surface tension
        self.lP_Sigma     = self.load_surrogate(FOL_IN,"saturated_liquid_SURFACE_TENSION(P)", JIT)
        self.lT_Sigma     = self.load_surrogate(FOL_IN,"saturated_liquid_SURFACE_TENSION(T)", JIT)
        # Print-out
        # print("========================================================")
        return
    
    "Load CoolSurrogateProp"
    def load_surrogate(self,FOL_IN, FILE_IN, JIT, verbose=False):
        # Input path
        PATH_IN = '%s/%s' %(FOL_IN, FILE_IN) 
        # Print property path
        print("- %s" %PATH_IN) if verbose else None
        # Read surrogate parameters
        surrogate_params = load_jax_model('%s' %PATH_IN)
        # with open('%s' %PATH_IN, 'rb') as f:
        #     surrogate_params = dill.load(f)
        # Setup model
        model = PropNN(
            n_layers = surrogate_params["n_layers"],
            hidden_dim = surrogate_params["hidden_dim"],
            activation_fns = str_to_afns(surrogate_params["activation_fns"])
            )
        # Assign optimal weights and scaling parameters
        model.surrogate_params(surrogate_params["weights"], 
                               surrogate_params["x_hat_params"], 
                               surrogate_params["y_hat_params"])
        if JIT:
            model_out = jax.jit(model.compute)
        else:
            model_out = model.compute
        # Output model
        return model_out

