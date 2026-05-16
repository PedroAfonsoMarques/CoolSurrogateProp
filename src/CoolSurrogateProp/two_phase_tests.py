#%% Import packages

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI
from CoolSurrogateProp.FluidEoS import EquationOfState

jax.config.update("jax_enable_x64", True)

#%% Define functions

def smoothstep(x):
    # Standard GLSL smoothstep: 3x^2 - 2x^3
    x = jnp.clip(x, 0, 1)
    return x * x * (3 - 2 * x)

def rho_smoothstep(Xe, rho_l, rho_v, rho_tp, eps=0.05):
    # Transition near 0
    w0 = smoothstep(Xe / eps)
    rho_0 = (1 - w0) * rho_l + w0 * rho_tp
    
    # Transition near 1
    w1 = smoothstep((Xe - (1 - eps)) / eps)
    rho_est = (1 - w1) * rho_0 + w1 * rho_v
    
    return rho_est

def rho_estimate(P, h, EOS, fid):
    # Ensure scalar inputs
    P = jnp.squeeze(P)
    h = jnp.squeeze(h)

    # Saturation boundaries
    hl   = jnp.squeeze(EOS.lP_H_sat(fid,   EOS.SatI(P)))
    hv   = jnp.squeeze(EOS.vP_H_sat(fid,   EOS.SatI(P)))
    rhol = jnp.squeeze(EOS.lP_Rho_sat(fid, EOS.SatI(P)))
    rhov = jnp.squeeze(EOS.vP_Rho_sat(fid, EOS.SatI(P)))

    # Quality
    Xe = (h - hl) / (hv - hl)
    
    # Two-phase estimate
    rho_tp = 1.0 / (Xe/rhov + (1.0-Xe)/rhol)
    
    # Single-phase estimates
    T_l   = jnp.squeeze(EOS.lPH_T(fid,   EOS.PropI(P, h)))
    rho_l = jnp.squeeze(EOS.lPT_Rho(fid, EOS.PropI(P, T_l)))
    
    T_v   = jnp.squeeze(EOS.vPH_T(fid,   EOS.PropI(P, h)))
    rho_v = jnp.squeeze(EOS.vPT_Rho(fid, EOS.PropI(P, T_v)))
    
    # # Select based on Xe
    # rho_est = jnp.where(
    #     Xe < 0, rho_l,
    #     jnp.where(
    #         Xe > 1, rho_v,
    #         rho_tp
    #         )
    #     )
    
    rho_est = rho_smoothstep(Xe, rho_l, rho_v, rho_tp)
    return rho_est

"Scalar Newton solver to estimate pressure from density and enthalpy"
@jax.jit(static_argnames=["EOS", "fid"])
def estimate_pressure_scalar(P0, rho, h, N_iter, EOS, fid):
    
    # Evaluate residual (pressure only input)
    def f(P):
        return rho_estimate(P, h, EOS, fid) - rho
    # Compute gradient of the residual w.r.t. pressure
    df_dP = jax.grad(f)

    # Newton iteration loop
    def body_fun(_, P, eps=1e-12):
        # Evaluate residual
        f_val  = f(P)
        # Evaluate residual derivative
        df_val = df_dP(P)

        # Damped Newton step
        dP = -f_val / (df_val + eps)
        # Update pressure guess
        P_new = P + dP
        # Return updated pressure
        return P_new
    
    # Run Newton iterations
    P_final = jax.lax.fori_loop(0, N_iter, body_fun, P0)
    # Return final pressure estimate
    return P_final

"Vectorize pressure estimator over 1D input densities and enthalpies"
@jax.jit(static_argnames=["EOS", "fid"])
def estimate_pressure_vectorized(P0, rho, h, N_iter, EOS, fid):
    # Vectorized version of the Newton solver using vmap
    vectorized_pressure_solver = jax.jit(
        jax.vmap(
            estimate_pressure_scalar,
            in_axes=(0, 0, 0, None, None, None)
        ),   static_argnames=["EOS", "fid"]
    )
    # Run vectorized Newton solver
    P_sol = vectorized_pressure_solver(P0, rho, h, N_iter, EOS, fid)
    # Return pressure
    return P_sol

#%% Setup

fluid    = "Parahydrogen"
EOS      = EquationOfState()
fluid_ID = EOS.get_fluid_id(fluid)

# Pressure/temperature sweep
P2,T2 = jnp.meshgrid(
    jnp.linspace(1.0e5, 5.0e5, 500),
    jnp.linspace(18.0,  300.0, 500)
    )

# Flatten to evaluate
P_flat = P2.flatten()
T_flat = T2.flatten()

# Evaluate enthalpy and density
H_flat   = jnp.array( PropsSI("HMASS","P",np.array(P_flat),"T",np.array(T_flat),fluid) )
Rho_flat = jnp.array( PropsSI("DMASS","P",np.array(P_flat),"T",np.array(T_flat),fluid) )

#%% Evaluate residuals

P_guess = jnp.ones_like(P_flat) * 1.0e5

P_estimate = estimate_pressure_vectorized(P_guess, Rho_flat, H_flat, N_iter=10, EOS=EOS, fid=fluid_ID)

plt.figure(figsize=(6,5))
plt.scatter(P_flat, P_estimate, alpha=0.5)
plt.plot(P_flat, P_flat, 'k--')
plt.xlabel("True Pressure [Pa]")
plt.ylabel("Estimated Pressure [Pa]")
plt.title("Pressure Estimation from Density and Enthalpy")
# plt.xscale("log")
# plt.yscale("log")

#%% Run above ^^

