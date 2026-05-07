#%% Import packages

import os
import jax
import optax
import shutil
import time
import matplotlib.pyplot as plt
from jax import nn
from CoolSurrogateProp.OptimizeEoS import CalibrateAndSave, Settings

# Use 64-bit precision
jax.config.update("jax_enable_x64", True)

#%%% Universal parameters

# Disable interactive plots
plt.ioff()

# Working fluid
Fluids = ['Parahydrogen', 'Hydrogen', 'Nitrogen', 'Oxygen', 'Methane' , 'Water'] 
REFPROP = False # True/False to use REFPROP or CoolProp


#%% Loop through fluid list

for Fluid in Fluids:
    
    # Output folder
    FOL_OUT   = 'Surrogate_test/%s' %Fluid
    FOL_OUT_I = '%s/Figures'   %FOL_OUT
    # Create output folder if it does not exist
    if not os.path.exists(FOL_OUT): os.makedirs(FOL_OUT)
    if not os.path.exists(FOL_OUT_I): os.makedirs(FOL_OUT_I)
    
    #%%% EoS
    
    N = 4
    
    P_max = 10.0e5, # Max. pressure [Pa]
    T_max = 300.15, # Max. temperature [K]
    P_min = None    # Min. pressure [Pa] (if None, set to triple-point pressure)
    T_min = None    # Min. temperature [K] (if None, set to triple-poin5t temperature)
    
    # Initialize training settings
    settings = Settings(n_layers = N, 
                        hidden_dim = [24]*N,
                        activation_fns = [nn.silu]*N,
                        eta0 = optax.schedules.exponential_decay(
                            init_value=2e-3, transition_begin=200, transition_steps=200, decay_rate=0.90
                            ),
                        deriv_penalty = 1e-4, # 1e-2
                        batch_frac    = 0.20,
                        N_epochs      = int(5e3),
                        N_data_points = int(64*6))
    # List thermodynamic properties
    Y = [
        # Thermodynamic properties
        'DMASS', 'CPMASS', 'CVMASS', 'HMASS', 'SMASS',
        # Thermodynamic partial derivatives
        'd(P)/d(T)|DMASS', 'd(P)/d(DMASS)|T',
        'd(T)/d(P)|HMASS', 'd(T)/d(HMASS)|P',
        'd(DMASS)/d(T)|P', 'd(DMASS)/d(P)|T', 
        'd(UMASS)/d(T)|P', 'd(UMASS)/d(P)|T',
        'd(UMASS)/d(DMASS)|T', 'd(UMASS)/d(T)|DMASS',
        'd(SMASS)/d(T)|P', 'd(SMASS)/d(P)|T',
        'd(SMASS)/d(DMASS)|T', 'd(SMASS)/d(T)|DMASS',
        # Transport properties
        'CONDUCTIVITY', 'VISCOSITY'
        ]
    # Target fluid phase
    PHASE = ['gas','liquid']
    
    # Input arguments
    X_IN = ['P','T']
    
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )
    
    #%% Additional properties
    
    # List thermodynamic properties
    Y = ['T',]
    # Target fluid phase
    PHASE = ['liquid','gas']
    # Input arguments
    X_IN = ['P','HMASS']
    
    # Train equation-of-state Y = f(X0, X1)
    CalibrateAndSave(
        Y, X_IN, PHASE, Fluid, FOL_OUT, FOL_OUT_I, settings, 
        P_max=P_max, T_max=T_max, P_min=P_min, T_min=T_min,
        REFPROP=REFPROP
        )
    
    #%%% Saturation
    
    # Initialize training settings
    settings = Settings(
        n_layers = N, 
        hidden_dim = [24]*N, 
        activation_fns = [nn.silu]*N,
        eta0 = optax.schedules.exponential_decay(
            init_value=5e-3, transition_begin=200, transition_steps=200, decay_rate=0.90
            ),
        N_epochs = int(5e3),
        batch_frac    = 0.20,
        N_data_points = int(2048*4),
        deriv_penalty = 1e-2,
        )
    
    # Target fluid phase
    PHASE = ['saturated_vapor','saturated_liquid']
    # Y = f(T)
    Ya = ['DMASS', 'HMASS', 'SMASS', 'P', 'SURFACE_TENSION', 'CONDUCTIVITY', 'CPMASS', 'VISCOSITY']; Xa = 'T'
    # Y = f(P)
    Yb = ['DMASS', 'HMASS', 'SMASS', 'T', 'SURFACE_TENSION', 'CONDUCTIVITY', 'CPMASS', 'VISCOSITY']; Xb = 'P'
    
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
    shutil.make_archive(FOL_OUT_I, 'zip', FOL_OUT_I)
    plt.close('all')
    # time.sleep(10)  # or 2 if Dropbox is involved
    # # Delete the original folder
    # shutil.rmtree(FOL_OUT_I)
    
#%% Run above ^^