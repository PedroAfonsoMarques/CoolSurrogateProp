#%% Import packages

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI
from CoolSurrogateProp.FluidEoS import EquationOfState
from CoolSurrogateProp.MixtureProps import PropTP, RhoTP

#%% User inputs

# Working fluid
fluids    = ['Parahydrogen', 'Hydrogen', 'Nitrogen', 'Oxygen', 'Methane' ]
EOS      = EquationOfState()

# Pressure/temperature sweep
P = jnp.linspace(1.0e5, 5.0e5, 100)

#%% Saturation properties

for fluid in fluids:
    # Fluid ID
    fluid_ID = EOS.get_fluid_id(fluid)

    # Temperature
    T_sat = [
        PropsSI("T","P",np.array(P),"Q",0.0,fluid),
        EOS.T_sat(fluid_ID, P)
        ]
    # Surface tension
    sigma = [
        PropsSI("SURFACE_TENSION","P",np.array(P),"Q",0.0,fluid),
        EOS.lP_Sigma(fluid_ID, P)
        ]

    # =============================================================================
    # Density
    # =============================================================================
    # Vapor
    Rho_v_sat = [
        PropsSI("DMASS","P",np.array(P),"Q",1.0,fluid),
        EOS.vP_Rho_sat(fluid_ID, P)
        ]
    # Liquid
    Rho_l_sat = [
        PropsSI("DMASS","P",np.array(P),"Q",0.0,fluid),
        EOS.lP_Rho_sat(fluid_ID, P)
        ]

    # =============================================================================
    # Enthalpy
    # =============================================================================
    # Vapor
    H_v_sat = [
        PropsSI("HMASS","P",np.array(P),"Q",1.0,fluid),
        EOS.vP_H_sat(fluid_ID, P)
        ]
    # Liquid
    H_l_sat = [
        PropsSI("HMASS","P",np.array(P),"Q",0.0,fluid),
        EOS.lP_H_sat(fluid_ID, P)
        ]

    # =============================================================================
    # Plot
    # =============================================================================
    fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(8,6),tight_layout=True)
    # Flatten axes
    ax = ax.flatten()
    # Temperature
    ax[0].plot(P/1e5, T_sat[0], color="k", linestyle="dashed", label="CoolProp")
    ax[0].plot(P/1e5, T_sat[1], color="b", linestyle="solid",  label="Surrogate")
    # Surface tension
    ax[1].plot(P/1e5, sigma[0], color="k", linestyle="dashed", label="CoolProp")
    ax[1].plot(P/1e5, sigma[1], color="b", linestyle="solid",  label="Surrogate")
    # Density
    ax[2].plot(P/1e5, Rho_v_sat[0], color="k", linestyle="dashed", label="CoolProp")
    ax[2].plot(P/1e5, Rho_v_sat[1], color="r", linestyle="solid",  label="Surrogate")
    ax[3].plot(P/1e5, Rho_l_sat[0], color="k", linestyle="dashed", label="CoolProp")
    ax[3].plot(P/1e5, Rho_l_sat[1], color="b", linestyle="solid",  label="Surrogate")
    # Enthalpy
    ax[4].plot(P/1e5, H_v_sat[0], color="k", linestyle="dashed", label="CoolProp")
    ax[4].plot(P/1e5, H_v_sat[1], color="r", linestyle="solid",  label="Surrogate")
    ax[5].plot(P/1e5, H_l_sat[0], color="k", linestyle="dashed", label="CoolProp")
    ax[5].plot(P/1e5, H_l_sat[1], color="b", linestyle="solid",  label="Surrogate")
    # Customization
    for a in ax:
        a.legend()
        a.set_xlabel(r"Pressure [bara]")
        a.minorticks_on()
        a.grid(which="both",axis="both",alpha=0.3)
    # Y-labels
    ax[0].set_ylabel(r"$T$ [K]")
    ax[1].set_ylabel(r"$\sigma$ [N/m]")
    ax[2].set_ylabel(r"$\rho_v$ [kg/m$^3$]"); ax[3].set_ylabel(r"$\rho_l$ [kg/m$^3$]")
    ax[4].set_ylabel(r"$H_v$ [J/kg]"); ax[5].set_ylabel(r"$H_l$ [J/kg]")
    # Title
    fig.suptitle(r"%s | Saturation properties | CoolProp vs Surrogate" %(fluid))
    fig.show()

    # Fixed temperature, pressure sweep

    T_list = [ T_sat[0][0] - 1.0, 293.15 ]

    for T_eval in T_list:      
        
        # For CoolSurrogateProp, the pressure & temperature inputs must 
        # have the same shapes, i.e., P.shape == T_jeval.shape.
        T_jval = T_eval*jnp.ones_like(P)
        
        # Detect phase according to P,T input
        H = PropsSI("HMASS","P",np.array(P),"T",T_eval,fluid)
        H_v_sat = PropsSI("HMASS","P",np.array(P),"Q",1.0,fluid)
        H_l_sat = PropsSI("HMASS","P",np.array(P),"Q",0.0,fluid)
        
        # Thermodynamic equilibrium quality
        Xe = (H - H_l_sat) / (H_v_sat - H_l_sat)
        
        # =============================================================================
        # Density
        # =============================================================================
        Rho = [
            PropsSI("DMASS","P",np.array(P),"T",T_eval,fluid),
            RhoTP(
                Xe, 
                EOS.lPT_Rho(fluid_ID, P, T_jval), 
                EOS.vPT_Rho(fluid_ID, P, T_jval), 
                EOS.lP_Rho_sat(fluid_ID, P), 
                EOS.vP_Rho_sat(fluid_ID, P)
                )
            ]
        
        # =============================================================================
        # Enthalpy
        # =============================================================================
        H = [
            PropsSI("HMASS","P",np.array(P),"T",T_eval,fluid),
            PropTP(
                Xe, 
                EOS.lPT_H(fluid_ID, P, T_jval), 
                EOS.vPT_H(fluid_ID, P, T_jval), 
                EOS.lP_H_sat(fluid_ID, P), 
                EOS.vP_H_sat(fluid_ID, P)
                )
            ]
        
        # =============================================================================
        # Conductivity
        # =============================================================================
        Kappa = [
            PropsSI("CONDUCTIVITY","P",np.array(P),"T",T_eval,fluid),
            PropTP(
                Xe, 
                EOS.lPT_Kappa(fluid_ID, P, T_jval), 
                EOS.vPT_Kappa(fluid_ID, P, T_jval), 
                EOS.lP_Kappa_sat(fluid_ID, P), 
                EOS.vP_Kappa_sat(fluid_ID, P)
                )
            ]
        Mu = [
            PropsSI("VISCOSITY","P",np.array(P),"T",T_eval,fluid),
            PropTP(
                Xe, 
                EOS.lPT_Mu(fluid_ID, P, T_jval), 
                EOS.vPT_Mu(fluid_ID, P, T_jval), 
                EOS.lP_Mu_sat(fluid_ID, P), 
                EOS.vP_Mu_sat(fluid_ID, P)
                )
            ]
        
        # =============================================================================
        # Specific heats
        # =============================================================================
        Cp = [
            PropsSI("CPMASS","P",np.array(P),"T",T_eval,fluid),
            PropTP(
                Xe, 
                EOS.lPT_Cp(fluid_ID, P, T_jval), 
                EOS.vPT_Cp(fluid_ID, P, T_jval), 
                EOS.lP_Cp_sat(fluid_ID, P), 
                EOS.vP_Cp_sat(fluid_ID, P)
                )
            ]
        Cv = [
            PropsSI("CVMASS","P",np.array(P),"T",T_eval,fluid),
            PropTP(
                Xe, 
                EOS.lPT_Cv(fluid_ID, P, T_jval), 
                EOS.vPT_Cv(fluid_ID, P, T_jval), 
                EOS.lPT_Cv(fluid_ID, P, T_jval), 
                EOS.vPT_Cv(fluid_ID, P, T_jval), 
                )
            ]
        
        # =============================================================================
        # Plot
        # =============================================================================
        fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(8,6),tight_layout=True)
        # Flatten axes
        ax = ax.flatten()
        # Density
        ax[0].plot(P/1e5, Rho[0], color="k", linestyle="dashed", label="CoolProp")
        ax[0].plot(P/1e5, Rho[1], color="b", linestyle="solid",  label="Surrogate")
        # Enthalpy
        ax[1].plot(P/1e5, H[0], color="k", linestyle="dashed", label="CoolProp")
        ax[1].plot(P/1e5, H[1], color="b", linestyle="solid",  label="Surrogate")
        # Conductivity
        ax[2].plot(P/1e5, Kappa[0], color="k", linestyle="dashed", label="CoolProp")
        ax[2].plot(P/1e5, Kappa[1], color="b", linestyle="solid",  label="Surrogate")
        # Dynamic viscosity
        ax[3].plot(P/1e5, Mu[0], color="k", linestyle="dashed", label="CoolProp")
        ax[3].plot(P/1e5, Mu[1], color="b", linestyle="solid",  label="Surrogate")
        # Cp
        ax[4].plot(P/1e5, Cp[0], color="k", linestyle="dashed", label="CoolProp")
        ax[4].plot(P/1e5, Cp[1], color="b", linestyle="solid",  label="Surrogate")
        # Cv
        ax[5].plot(P/1e5, Cv[0], color="k", linestyle="dashed", label="CoolProp")
        ax[5].plot(P/1e5, Cv[1], color="b", linestyle="solid",  label="Surrogate")
        # Customization
        for a in ax:
            a.legend()
            a.set_xlabel(r"Pressure [bara]")
            a.minorticks_on()
            a.grid(which="both",axis="both",alpha=0.3)
        # Y-labels
        ax[0].set_ylabel(r"$\rho$ [kg/m$^3$]")
        ax[1].set_ylabel(r"$H$ [J/kg]")
        ax[2].set_ylabel(r"$\kappa$ [W/(mK)]")
        ax[3].set_ylabel(r"$\mu$ [Pa$\cdot$s]")
        ax[4].set_ylabel(r"$C_p$ [J/kg]")
        ax[5].set_ylabel(r"$C_v$ [J/kg]")
        # Title
        fig.suptitle(r"%s | $T=%.2f$ K | CoolProp vs Surrogate" %(fluid, T_eval))
        fig.show()

#%% Run above ^^

