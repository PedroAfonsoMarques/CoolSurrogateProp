#%% Import packages

import os
import jax
import optax
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from jax import nn
from CoolSurrogateProp.OptimizeEoS import CalibrateAndSave, Settings

# Use 64-bit precision
jax.config.update("jax_enable_x64", True)

#%%% Universal parameters

# Disable interactive plots
plt.ioff()

# Working fluid
Fluids = ["Parahydrogen", "Hydrogen", "Nitrogen", "Oxygen", "Methane"] 
REFPROP = False # True/False to use REFPROP or CoolProp

N_layers          = 4 # Number of hidden layers in the surrogate model
N_epochs          = int(1e4) # Number of training epochs
N_data_points     = int(np.power(2, 9)) # Number of training data points (per property)
N_data_points_sat = int(np.power(2,12)) # Number of training data points for saturation properties (per property)

#%% Loop through fluid list

for Fluid in Fluids:
    print("========================================================")
    print("Fluid: %s" %Fluid)
    print("========================================================")
    
    # Output folder
    FOL_OUT   = "Surrogate/%s" %Fluid
    FOL_OUT_I = "%s/Figures"   %FOL_OUT
    # Create output folder if it does not exist
    if not os.path.exists(FOL_OUT): os.makedirs(FOL_OUT)
    if not os.path.exists(FOL_OUT_I): os.makedirs(FOL_OUT_I)
    
    #%%% EoS
    
    P_max = 10.0e5 # Max. pressure [Pa]
    T_max = 300.15 # Max. temperature [K]
    P_min = None   # Min. pressure [Pa] (if None, set to triple-point pressure)
    T_min = None   # Min. temperature [K] (if None, set to triple-poin5t temperature)
    
    #%% Full-range (Enthalpy)

    # Initialize training settings
    settings = Settings(
        n_layers = N_layers, 
        hidden_dim = [24]*N_layers,
        activation_fns = [nn.silu]*N_layers,
        eta0 = optax.schedules.exponential_decay(
            init_value=2e-3, transition_begin=200, transition_steps=200, decay_rate=0.95
            ),
        deriv_penalty = 1e-4, # 1e-2
        batch_frac    = 0.20,
        N_epochs      = N_epochs,
        N_data_points = N_data_points
        )

    # List thermodynamic properties
    Y     = ["P", "HMASS", "T"]
    X_IN  = ["DMASS", "UMASS"]
    PHASE = ["universal"]
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )

    # List thermodynamic properties
    Y     = ["P", "UMASS", "T"]
    X_IN  = ["DMASS", "HMASS"]
    PHASE = ["universal"]
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )

    #%% Single-phase properties

    # Initialize training settings
    settings = Settings(
        n_layers = N_layers, 
        hidden_dim = [24]*N_layers,
        activation_fns = [nn.silu]*N_layers,
        eta0 = optax.schedules.exponential_decay(
            init_value=2e-3, transition_begin=200, transition_steps=200, decay_rate=0.90
            ),
        deriv_penalty = 1e-4, # 1e-2
        batch_frac    = 0.20,
        N_epochs      = N_epochs,
        N_data_points = N_data_points
        )

    # List thermodynamic properties
    Y = [
        # Thermodynamic properties
        "DMASS", "CPMASS", "CVMASS", "HMASS", "SMASS",
        # Thermodynamic partial derivatives
        "d(P)/d(T)|DMASS",     "d(P)/d(DMASS)|T",
        "d(P)/d(HMASS)|DMASS", "d(P)/d(DMASS)|HMASS",
        "d(T)/d(P)|HMASS",     "d(T)/d(HMASS)|P",
        "d(DMASS)/d(T)|P",     "d(DMASS)/d(P)|T", 
        "d(DMASS)/d(HMASS)|P", "d(DMASS)/d(P)|HMASS",
        "d(UMASS)/d(T)|P",     "d(UMASS)/d(P)|T",
        "d(UMASS)/d(DMASS)|T", "d(UMASS)/d(T)|DMASS",
        "d(SMASS)/d(T)|P",     "d(SMASS)/d(P)|T",
        "d(SMASS)/d(DMASS)|T", "d(SMASS)/d(T)|DMASS",
        # Transport properties
        "CONDUCTIVITY", "VISCOSITY"
        ]
    # Target fluid phase
    PHASE = ["gas","liquid"]
    
    # Input arguments
    X_IN = ["P","T"]
    
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )

    #%% Additional properties
    
    # List thermodynamic properties
    Y = ["T",]
    # Target fluid phase
    PHASE = ["liquid","gas"]
    # Input arguments
    X_IN = ["P","HMASS"]
    
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )
    
    #%%% Saturation
    
    # Initialize training settings
    settings = Settings(
        n_layers = N_layers, 
        hidden_dim = [24]*N_layers, 
        activation_fns = [nn.silu]*N_layers,
        eta0 = optax.schedules.exponential_decay(
            init_value=5e-3, transition_begin=200, transition_steps=200, decay_rate=0.90
            ),
        N_epochs = N_epochs,
        batch_frac    = 0.20,
        N_data_points = N_data_points_sat,
        deriv_penalty = 1e-3,
        )
    
    # Target fluid phase
    PHASE = ["saturated_vapor","saturated_liquid"]
    # # ==========================
    # # Y = f(T)
    # # ==========================
    Ya = [
        "DMASS", "HMASS", "SMASS", "P", "SURFACE_TENSION", "CONDUCTIVITY", "CPMASS", "VISCOSITY"
        ]
    Xa = "T"
    # ==========================
    # Y = f(P)
    # ==========================
    Yb = [
        # Thermodynamic properties
        "DMASS", "HMASS", "SMASS", "T", "SURFACE_TENSION", "CONDUCTIVITY", "CPMASS", "VISCOSITY",
        # Thermodynamic partial derivatives
        "d(P)/d(T)|DMASS",     "d(P)/d(DMASS)|T",
        "d(P)/d(HMASS)|DMASS", "d(P)/d(DMASS)|HMASS",
        "d(T)/d(P)|HMASS",     "d(T)/d(HMASS)|P",
        "d(DMASS)/d(T)|P",     "d(DMASS)/d(P)|T", 
        "d(DMASS)/d(HMASS)|P", "d(DMASS)/d(P)|HMASS",
        "d(UMASS)/d(T)|P",     "d(UMASS)/d(P)|T",
        "d(UMASS)/d(DMASS)|T", "d(UMASS)/d(T)|DMASS",
        "d(SMASS)/d(T)|P",     "d(SMASS)/d(P)|T",
        "d(SMASS)/d(DMASS)|T", "d(SMASS)/d(T)|DMASS",
        ]
    Xb = "P"
    
    # Train Ysat(T)
    CalibrateAndSave(
        Ya, Xa, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP)
    # Train Ysat(P)
    CalibrateAndSave(Yb, Xb, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
                     P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
                     REFPROP=REFPROP)
    
    # Compress
    shutil.make_archive(FOL_OUT_I, "zip", FOL_OUT_I)
    
    plt.close("all")
    # time.sleep(10)  # or 2 if Dropbox is involved
    # # Delete the original folder
    # shutil.rmtree(FOL_OUT_I)
    
#%% Run above ^^