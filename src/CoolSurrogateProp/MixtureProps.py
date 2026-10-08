#%% Import packages

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

#%% Alternative for JAXPROP

"Mixture properties based on thermodynamic quality"
def PropJTP(Xe, Ysp, Yl_sat, Yv_sat):
    return jnp.where(
        Xe < 0, Ysp, 
        jnp.where(
            Xe > 1, Ysp, 
            Yl_sat + Xe * (Yv_sat - Yl_sat)
            )
        )
"Mixture density based on inversion of the specific volume"
def RhoJTP(Xe, rho_sp, rho_l_sat, rho_v_sat):
    # Convert to specific volumes
    v_sp = 1.0 / rho_sp
    vl_sat = 1.0 / rho_l_sat
    vv_sat = 1.0 / rho_v_sat
    # Evaluate mixture specific volume
    vs = PropJTP(Xe, v_sp, vl_sat, vv_sat)
    # Return mixture density
    return 1/vs

#%% Hard-switch functions

"Mixture properties based on thermodynamic quality"
def PropTP(Xe, Yl, Yv, Yl_sat, Yv_sat):
    return jnp.where(
        Xe < 0, Yl, 
        jnp.where(
            Xe > 1, Yv, 
            Yl_sat + Xe * (Yv_sat - Yl_sat)
            )
        )
"Mixture density based on inversion of the specific volume"
def RhoTP(Xe, rho_l, rho_v, rho_l_sat, rho_v_sat):
    # Convert to specific volumes
    vl = 1.0 / rho_l
    vv = 1.0 / rho_v
    vl_sat = 1.0 / rho_l_sat
    vv_sat = 1.0 / rho_v_sat
    # Evaluate mixture specific volume
    vs = PropTP(Xe, vl, vv, vl_sat, vv_sat)
    # Return mixture density
    return 1/vs

#%% Smooth-switch functions

"Define sigmoid smoothing factor according to quality (Xe) sensitivity"
def target_k_sigmoid(Xe_t= 0.01, sigma_t=0.99):
    # For sigmoid-based gates
    # Once x crosses Xe_t, we want to have f(x) = sigma_t
    k_t = -jnp.log( (1/sigma_t) - 1 ) / Xe_t
    # Return target k
    return k_t

"Define sigmoid smoothing factor according to quality (Xe) sensitivity"
def target_k_poly(Xe_t=0.01):
    # Return target k
    return 1.0 / Xe_t

# Define a target k for smooth switching
K_T = target_k_poly(Xe_t = 0.05)

"Smooth gate function (smooth version of jnp.clip)"
def compact_gate(x):
    # Clip quality to [0, 1]
    xc = jnp.clip(x, 0.0, 1.0)
    # Continuous gate function that goes from 0 to 1 as v goes from 0 to 1
    # Derivative is zero at xc = 0 and xc = 1
    gate = (xc**3) * ((6.0*xc - 15.0)*xc + 10.0)
    # Return the gate value
    return gate

"Sigmoid-bsed gates for smooth switching"
def gates(Xe, k):
    g0 = compact_gate(k * Xe)         # 0 -> 1 as Xe crosses 0
    g1 = compact_gate(k * (Xe - 1.0)) # 0 -> 1 as Xe crosses 1
    return g0, g1

"Soft-clip rather than the non-continuous hard-clip (jnp.clip)"
def softclip(x, xmin, xmax, k=K_T):
    # Make sure the input is an array
    x = jnp.asarray(x)
    # Clip on the left
    x_min = jnp.asarray(xmin)
    x_min = jnp.broadcast_to(x_min, x.shape)
    x = (1.0 / k) * logsumexp(jnp.stack([k * x, k * x_min], axis=0), axis=0)
    # Clip on the right
    x_max = jnp.asarray(xmax)
    x_max = jnp.broadcast_to(x_max, x.shape)
    x = (-1.0 / k) * logsumexp(jnp.stack([-k * x, -k * x_max], axis=0), axis=0)
    # Return clipped boi
    return x

#%% JAXPROP

"Mixture properties based on thermodynamic quality (smooth-ish)"
def PropJTPS(Xe, Y_sp, Yl_sat, Yv_sat, k=K_T):
    # Smooth-switch gates
    g0, g1 = gates(Xe, k)
    # Clip quality
    Xe_c = jnp.clip(Xe, 0.0, 1.0) # softclip(Xe, xmin=0.0, xmax=1.0) 
    # Mixture property in the two-phase region
    Ym = Yl_sat + Xe_c * (Yv_sat - Yl_sat)
    # Define a single gating factor that evaluates to 1.0 inside the dome (0 < Xe < 1)
    # and 0.0 outside the dome (Xe <= 0 or Xe >= 1)
    inside_dome = g0 * (1.0 - g1)
    # Smoothly blend the three regions together
    Y_s = (1 - inside_dome) * Y_sp  +  inside_dome * Ym 
    # Return mixture property
    return Y_s

"Mixture density based on thermodynamic quality (smooth-ish)"
def RhoJTPS(Xe, rho_sp, rho_l_sat, rho_v_sat, k=K_T, eps=1e-6):
    # Convert to specific volumes
    v_sp   = 1.0 / rho_sp
    vl_sat = 1.0 / rho_l_sat
    vv_sat = 1.0 / rho_v_sat
    # Evaluate mixture specific volume
    vs = PropJTPS(Xe, v_sp, vl_sat, vv_sat, k)
    # Return mixture density
    return 1/(vs + eps)

#%% Surrogate

"Mixture properties based on thermodynamic quality (smooth-ish)"
def PropTPS(Xe, Yl, Yv, Yl_sat, Yv_sat, k=K_T):
    # Smooth-switch gates
    g0, g1 = gates(Xe, k)
    # Clip quality
    Xe_c = jnp.clip(Xe, 0.0, 1.0) # softclip(Xe, xmin=0.0, xmax=1.0) 
    # Mixture property in the two-phase region
    Ym = Yl_sat + Xe_c * (Yv_sat - Yl_sat)
    # Smoothly blend the three regions together
    Y_s = (1 - g0) * Yl  +  g0 * (1 - g1) * Ym  +  g1 * Yv
    # Return mixture property
    return Y_s

"Mixture density based on thermodynamic quality (smooth-ish)"
def RhoTPS(Xe, rho_l, rho_v, rho_l_sat, rho_v_sat, k=K_T, eps=1e-6):
    # Convert to specific volumes
    vl = 1.0 / rho_l
    vv = 1.0 / rho_v
    vl_sat = 1.0 / rho_l_sat
    vv_sat = 1.0 / rho_v_sat
    # Evaluate mixture specific volume
    vs = PropTPS(Xe, vl, vv, vl_sat, vv_sat, k)
    # Return mixture density
    return 1/(vs + eps)

#%% Test

# Test the function here
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    print("Testing mixture_property functions")
    
    X_arr = jnp.linspace(-0.1, 1.1, 1000)
    
    # Vapor
    Yv     = 10.0
    Yv_sat = 12.0
    # Liquid
    Yl     = 40.1
    Yl_sat = 20.0

    # =============================================
    # Properties
    # =============================================
    # Hard-switch
    Y_mp, dYdX_mp = jax.jit(
            jax.vmap(
                jax.value_and_grad(PropTP, argnums=0),
                in_axes=(0, None, None, None, None)   # map over Xe, broadcast scalars
            )
        )(X_arr, Yl, Yv, Yl_sat, Yv_sat)
    # Smooth-switch
    Y_ms, dYdX_ms = jax.jit(
            jax.vmap(
                jax.value_and_grad(PropTPS, argnums=0),
                in_axes=(0, None, None, None, None)   # map over Xe, broadcast scalars
            )
        )(X_arr, Yl, Yv, Yl_sat, Yv_sat)

    # =============================================
    # Densities
    # =============================================
    # Hard-switch
    R_mp, dRdX_mp = jax.jit(
            jax.vmap(
                jax.value_and_grad(RhoTP, argnums=0),
                in_axes=(0, None, None, None, None)   # map over Xe, broadcast scalars
            )
        )(X_arr, Yl, Yv, Yl_sat, Yv_sat)
    # Smooth-switch
    R_ms, dRdX_ms = jax.jit(
            jax.vmap(
                jax.value_and_grad(RhoTPS, argnums=0),
                in_axes=(0, None, None, None, None)   # map over Xe, broadcast scalars
            )
        )(X_arr, Yl, Yv, Yl_sat, Yv_sat)

    fig,ax=plt.subplots(ncols=2,nrows=2,tight_layout=True)
    # =============================================
    # Properties
    # =============================================
    # Value
    ax[0,0].plot(X_arr, Y_mp, label="Hard-switch")
    ax[0,0].plot(X_arr, Y_ms, label="Smooth-switch")
    # Gradient
    ax[1,0].plot(X_arr, dYdX_mp)
    ax[1,0].plot(X_arr, dYdX_ms)
    # =============================================
    # Densities
    # =============================================
    # Value
    ax[0,1].plot(X_arr, R_mp)
    ax[0,1].plot(X_arr, R_ms)
    # Gradient
    ax[1,1].plot(X_arr, dRdX_mp)
    ax[1,1].plot(X_arr, dRdX_ms)
    # Customization
    ax[0,0].legend()
    ax[0,0].set_title("NOT Density")
    ax[0,1].set_title("Density")
    ax = ax.flatten()
    for a in ax:
        a.set_xlabel(r"$X_e$ [-]")
        a.minorticks_on()
        a.grid(axis="both", which="both", alpha=0.3)

#%% Run above ^^
