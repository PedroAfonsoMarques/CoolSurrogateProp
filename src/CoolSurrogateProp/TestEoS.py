#%% Import packages

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI
from CoolSurrogateProp.FluidEoS import EquationOfState
from CoolSurrogateProp.MixtureProps import PropTPS as PropTP
from CoolSurrogateProp.MixtureProps import RhoTPS as RhoTP

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

#%% Fixed pressure, temperature sweep

for fluid in fluids:
    # Fluid ID
    fluid_ID = EOS.get_fluid_id(fluid)

    T = jnp.linspace(T_sat[0][0] - 5.0, 293.15, 100)
    P_list = [P[0], P[-1]]

    for P_eval in P_list:      
        
        # For CoolSurrogateProp, the pressure & temperature inputs must 
        # have the same shapes, i.e., P.shape == T_jeval.shape.
        P_jval = P_eval*jnp.ones_like(P)
        
        # Detect phase according to P,T input
        H = PropsSI("HMASS","P",P_eval,"T",np.array(T),fluid)
        H_v_sat = PropsSI("HMASS","P",P_eval,"Q",1.0,fluid)
        H_l_sat = PropsSI("HMASS","P",P_eval,"Q",0.0,fluid)
        
        # Thermodynamic equilibrium quality
        Xe = (H - H_l_sat) / (H_v_sat - H_l_sat)
        
        # =============================================================================
        # Density
        # =============================================================================
        Rho = [
            PropsSI("DMASS","P",P_eval,"T",np.array(T),fluid),
            RhoTP(
                Xe, 
                EOS.lPT_Rho(fluid_ID, P_jval, T), 
                EOS.vPT_Rho(fluid_ID, P_jval, T), 
                EOS.lP_Rho_sat(fluid_ID, P_jval), 
                EOS.vP_Rho_sat(fluid_ID, P_jval)
                )
            ]
        
        # =============================================================================
        # Enthalpy
        # =============================================================================
        H = [
            PropsSI("HMASS","P",P_eval,"T",np.array(T),fluid),
            PropTP(
                Xe, 
                EOS.lPT_H(fluid_ID, P_jval, T), 
                EOS.vPT_H(fluid_ID, P_jval, T), 
                EOS.lP_H_sat(fluid_ID, P_jval), 
                EOS.vP_H_sat(fluid_ID, P_jval)
                )
            ]
        
        # =============================================================================
        # Conductivity
        # =============================================================================
        Kappa = [
            PropsSI("CONDUCTIVITY","P",P_eval,"T",np.array(T),fluid),
            PropTP(
                Xe, 
                EOS.lPT_Kappa(fluid_ID, P_jval, T), 
                EOS.vPT_Kappa(fluid_ID, P_jval, T), 
                EOS.lP_Kappa_sat(fluid_ID, P_jval), 
                EOS.vP_Kappa_sat(fluid_ID, P_jval)
                )
            ]
        Mu = [
            PropsSI("VISCOSITY","P",P_eval,"T",np.array(T),fluid),
            PropTP(
                Xe, 
                EOS.lPT_Mu(fluid_ID, P_jval, T), 
                EOS.vPT_Mu(fluid_ID, P_jval, T), 
                EOS.lP_Mu_sat(fluid_ID, P_jval), 
                EOS.vP_Mu_sat(fluid_ID, P_jval)
                )
            ]
        
        # =============================================================================
        # Specific heats
        # =============================================================================
        Cp = [
            PropsSI("CPMASS","P",P_eval,"T",np.array(T),fluid),
            PropTP(
                Xe, 
                EOS.lPT_Cp(fluid_ID, P_jval, T), 
                EOS.vPT_Cp(fluid_ID, P_jval, T), 
                EOS.lP_Cp_sat(fluid_ID, P_jval), 
                EOS.vP_Cp_sat(fluid_ID, P_jval)
                )
            ]
        Cv = [
            PropsSI("CVMASS","P",P_eval,"T",np.array(T),fluid),
            PropTP(
                Xe, 
                EOS.lPT_Cv(fluid_ID, P_jval, T), 
                EOS.vPT_Cv(fluid_ID, P_jval, T), 
                EOS.lPT_Cv(fluid_ID, P_jval, T), 
                EOS.vPT_Cv(fluid_ID, P_jval, T), 
                )
            ]
        
        # =============================================================================
        # Plot
        # =============================================================================
        fig,ax=plt.subplots(nrows=3,ncols=2,figsize=(8,6),tight_layout=True)
        # Flatten axes
        ax = ax.flatten()
        # Density
        ax[0].plot(T, Rho[0], color="k", linestyle="dashed", label="CoolProp")
        ax[0].plot(T, Rho[1], color="b", linestyle="solid",  label="Surrogate")
        # Enthalpy
        ax[1].plot(T, H[0], color="k", linestyle="dashed", label="CoolProp")
        ax[1].plot(T, H[1], color="b", linestyle="solid",  label="Surrogate")
        # Conductivity
        ax[2].plot(T, Kappa[0], color="k", linestyle="dashed", label="CoolProp")
        ax[2].plot(T, Kappa[1], color="b", linestyle="solid",  label="Surrogate")
        # Dynamic viscosity
        ax[3].plot(T, Mu[0], color="k", linestyle="dashed", label="CoolProp")
        ax[3].plot(T, Mu[1], color="b", linestyle="solid",  label="Surrogate")
        # Cp
        ax[4].plot(T, Cp[0], color="k", linestyle="dashed", label="CoolProp")
        ax[4].plot(T, Cp[1], color="b", linestyle="solid",  label="Surrogate")
        # Cv
        ax[5].plot(T, Cv[0], color="k", linestyle="dashed", label="CoolProp")
        ax[5].plot(T, Cv[1], color="b", linestyle="solid",  label="Surrogate")
        # Customization
        for a in ax:
            a.legend()
            a.set_xlabel(r"Temperature [K]")
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
        fig.suptitle(r"%s | $P=%.2e$ bar | CoolProp vs Surrogate" %(fluid, P_eval))
        fig.show()

#%% Reconstruct pressure from density and enthalpy

fluids2 = fluids # ["Parahydrogen","Hydrogen"]

for fluid in fluids2:
    # Fluid ID
    fluid_ID = EOS.get_fluid_id(fluid)

    N = 1000

    P = jnp.linspace(1.0e5, 5.0e5, N)
    T = jnp.linspace(PropsSI("T","P",P[0],"Q",0,fluid)-2.0, 300.0, N)

    # Density
    Rho   = PropsSI("DMASS","P",np.array(P),"T",np.array(T),fluid)
    Umass = PropsSI("UMASS","P",np.array(P),"T",np.array(T),fluid)
    Hmass = PropsSI("HMASS","P",np.array(P),"T",np.array(T),fluid)

    # =============================================================================
    # Enthalpy
    # =============================================================================
    H_test = EOS.uDU_H(fluid_ID, Rho, Umass)
    U_test = EOS.uDH_U(fluid_ID, Rho, H_test)
    Rho_test = Rho*(H_test - U_test)

    P_test = [
        PropsSI("T","HMASS",Hmass,"DMASS",Rho,fluid),
        EOS.uDH_T(fluid_ID, Rho, H_test)
        ]

    fig,ax = plt.subplots(nrows=2,tight_layout=True)
    # f(Rho)
    ax[0].plot(Rho, P_test[0], color="k")
    ax[0].plot(Rho, P_test[1], color="b")
    # f(H)
    ax[1].plot(Hmass, P_test[0], color="k")
    ax[1].plot(Hmass, P_test[1], color="b")
    # Axes
    ax[0].set_xscale("log")
    

#%% Run above ^^

