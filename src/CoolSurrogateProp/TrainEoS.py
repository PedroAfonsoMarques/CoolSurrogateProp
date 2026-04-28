"""
The efforts to make the partial derivatives more accurate have failed!
Especially on the non-saturated properties!
I think there is a strong mismatch on the values that I obtain!
"""

#%% Import packages

import os
import jax
import optax
import matplotlib.pyplot as plt
from jax import nn
from CoolSurrogateProp.OptimizeEoS import CalibrateAndSave, Settings

#%%% Universal parameters

# Disable interactive plots
plt.ioff()

# Working fluid
Fluid = "Nitrogen" # 'Hydrogen' 'Nitrogen' 'RE347MCC'
REFPROP = False # True/False to use REFPROP or CoolProp

# Output folder
FOL_OUT   = 'Surrogate_test/%s' %Fluid
FOL_OUT_I = '%s/Figures'   %FOL_OUT
# Create output folder if it does not exist
if not os.path.exists(FOL_OUT): os.makedirs(FOL_OUT)
if not os.path.exists(FOL_OUT_I): os.makedirs(FOL_OUT_I)

# Use 64-bit precision
jax.config.update("jax_enable_x64", True)

#%%% EoS

P_max = 10.0e5, # Max. pressure [Pa]
T_max = 373.15, # Max. temperature [K]
P_min = None # Min. pressure [Pa] (if None, set to triple-point pressure)
T_min = None # Min. temperature [K] (if None, set to triple-poin5t temperature)

# Initialize training settings
settings = Settings(n_layers = 3, 
                    hidden_dim = [16,16,16],
                    activation_fns = [nn.silu]*8, # [nn.silu, nn.silu, nn.silu, nn.silu],
                    eta0 = optax.schedules.exponential_decay(
                        init_value=5e-3, transition_begin=500, transition_steps=500, decay_rate=0.90
                        ),
                    N_epochs = int(2e4),
                    N_data_points = int(64*4))
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
    n_layers = 3, 
    hidden_dim = [16]*3, 
    activation_fns = [nn.silu]*3,
    eta0 = optax.schedules.exponential_decay(
        init_value=5e-3, transition_begin=500, transition_steps=500, decay_rate=0.95
        ),
    N_epochs = int(6e3),
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

#%% Run above ^^