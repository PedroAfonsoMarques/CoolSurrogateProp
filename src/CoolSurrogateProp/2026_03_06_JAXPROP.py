#%% Import packages
import jax
import numpy as np
import jax.numpy as jnp
from CoolProp.CoolProp import PropsSI

jax.config.update("jax_enable_x64", True)

#%% Wrapper for JAX-JIT coolprop

def _safe_inputs(x0,x1):
    x0_arr = np.atleast_1d(np.array(x0))
    x1_arr = np.atleast_1d(np.array(x1))
    # Broadcast arrays to have matching types
    x0_brc, x1_brc = np.broadcast_arrays(x0_arr, x1_arr)
    # Flatten broadcasted arrays
    x0_ = np.atleast_1d(x0_brc).ravel()
    x1_ = np.atleast_1d(x1_brc).ravel()
    # Return flattened arrays
    return x0_, x1_

def _broadcast_inputs(x0, x1):
    """Broadcast inputs to common shape, return (x0_, x1_, orig_shape)"""
    x0_arr = np.array(x0)
    x1_arr = np.array(x1)
    orig_shape = np.broadcast_shapes(x0_arr.shape, x1_arr.shape)
    x0_ = np.broadcast_to(x0_arr, orig_shape).ravel()
    x1_ = np.broadcast_to(x1_arr, orig_shape).ravel()
    return x0_, x1_, orig_shape

def _reshape_result(result, orig_shape):
    """Reshape result back to orig_shape, handling scalar case"""
    return np.array(result).reshape(orig_shape) if orig_shape else np.array(result).reshape(())

def PropsJI(output, input0, input1, fluid):
    """
    Creates a JAX-differentiable CoolProp property function.
    f(x0, x1) -> PropsSI(output, input0, x0, input1, x1, fluid)
    with exact thermodynamic derivatives.
    """
    # Selects finite-differences or analytical derivatives
    deriv_numeric =(
        output.upper() in {"CONDUCTIVITY", "VISCOSITY", "SURFACE_TENSION"}
        or input0.upper() == "Q"
        or input1.upper() == "Q"
    )
    
    "Function evaluation: f(x)"
    def _call_batched(x0, x1):
        # Convert JAX inputs to numpy
        x0_, x1_, orig_shape = _broadcast_inputs(x0, x1)
        # Evaluate PropsSI call
        result = PropsSI(output, input0, x0_, input1, x1_, fluid)
        # Restore original input shape
        return _reshape_result(result, orig_shape)

    # dfdx0 > Derivative wrt input0
    def _dfdx0(x0, x1):
        # Convert JAX inputs to numpy
        x0_, x1_, orig_shape = _broadcast_inputs(x0, x1)
        # Numerical derivative
        if deriv_numeric:
            eps = 1e-6 * np.maximum(np.abs(x0_), 1.0)
            # Check for quality
            if input0.upper() == "Q":
                dx0_pos = np.minimum(x0_ + eps, 1.0)
                dx0_neg = np.maximum(x0_ - eps, 0.0)
            else:
                dx0_pos = x0_ + eps
                dx0_neg = x0_ - eps
            # Evaluate perturbed values
            fp  = PropsSI(output, input0, dx0_pos, input1, x1_, fluid)
            fm  = PropsSI(output, input0, dx0_neg, input1, x1_, fluid)
            result = (fp - fm) / (dx0_pos - dx0_neg)
        # Analytical partial derivative
        else:
            result = PropsSI(f"d({output})/d({input0})|{input1}",input0, x0_, input1, x1_, fluid)
        # Return partial derivative with original shape
        return _reshape_result(result, orig_shape)
    
    # dfdx1 > Derivative wrt input1
    def _dfdx1(x0, x1):
        # Convert JAX inputs to numpy
        x0_, x1_, orig_shape = _broadcast_inputs(x0, x1)
        # Numerical derivative
        if deriv_numeric:
            eps = 1e-6 * np.maximum(np.abs(x1_), 1.0)
            # Check for quality
            if input1.upper() == "Q":
                dx1_pos = np.minimum(x1_ + eps, 1.0)
                dx1_neg = np.maximum(x1_ - eps, 0.0)
            else:
                dx1_pos = x1_ + eps
                dx1_neg = x1_ - eps
            fp  = PropsSI(output, input0, x0_, input1, dx1_pos, fluid)
            fm  = PropsSI(output, input0, x0_, input1, dx1_neg, fluid)
            result = (fp - fm) / (dx1_pos - dx1_neg)
        # Analytical partial derivative
        else:
            result = PropsSI(f"d({output})/d({input1})|{input0}", input0, x0_, input1, x1_, fluid)
        # Return partial derivative with original shape
        return _reshape_result(result, orig_shape)

    @jax.custom_jvp
    def f(x0, x1):
        # Ensure we are dealing with JAX-arrays
        x0 = jnp.array(x0)
        x1 = jnp.array(x1)
        out_shape = jnp.broadcast_shapes(x0.shape, x1.shape)
        # Callback
        return jax.pure_callback(_call_batched, jax.ShapeDtypeStruct(out_shape, x0.dtype), x0, x1, vmap_method="legacy_vectorized")

    @f.defjvp
    def f_jvp(primals, tangents):
        # Unpack primals, tangents
        x0, x1   = primals
        dx0, dx1 = tangents
        # Ensure we are dealing with JAX-arrays
        x0, x1 = jnp.array(x0), jnp.array(x1)
        dx0, dx1 = jnp.array(dx0), jnp.array(dx1)
        # Output
        out_shape = jnp.broadcast_shapes(x0.shape, x1.shape)
        out_struct = jax.ShapeDtypeStruct(out_shape, x0.dtype)
        # Functional evaluation
        primal_out = f(x0, x1)
        # Chain rule: df = (df/dx0)*dx0 + (df/dx1)*dx1
        grad0 = jax.pure_callback(_dfdx0, out_struct, x0, x1, vmap_method="legacy_vectorized")
        grad1 = jax.pure_callback(_dfdx1, out_struct, x0, x1, vmap_method="legacy_vectorized")
        # Derivative terms
        tangent_out = grad0 * dx0 + grad1 * dx1
        # Value and derivatives
        return primal_out, tangent_out
    # Return function value
    return f

#%% Testing

if __name__ == "__main__":

    # Instantiate property functions: inputs are (rho [kg/m3], u [J/kg])
    fluid   = "Nitrogen"
    DU_P = PropsJI("P", "DMASS", "UMASS", fluid) # P(rho, u)
    DU_T = PropsJI("T", "DMASS", "UMASS", fluid) # T(rho, u)

    V   = 1.0        # [m3]  box volume
    m   = 10.0       # [kg]  fixed mass
    rho = m / V      # [kg/m3] fixed density
    
    # Initial conditions: T0=100K, P0 from CoolProp
    T0  = 100.0
    P0  = float(PropsSI("P", "D", rho, "T", T0, fluid))
    u0  = float(PropsSI("U", "D", rho, "T", T0, fluid))

    P0_test = DU_P(jnp.array(rho), jnp.array(u0))
    T0_test = DU_T(jnp.array(rho), jnp.array(u0))

    P0_test, dPdX_test = jax.value_and_grad(DU_P, argnums=(0,1))(jnp.array(rho), jnp.array(u0))
    # JaxProp
    dPdRho_test = dPdX_test[0]
    dPdUms_test = dPdX_test[1]
    # CoolProp
    dPdRho = PropsSI("d(P)/d(DMASS)|UMASS", "DMASS", rho, "UMASS", u0, fluid)
    dPdUms = PropsSI("d(P)/d(UMASS)|DMASS", "DMASS", rho, "UMASS", u0, fluid)

    def composed_fn(P):
        P = jnp.array(P)
        T = jnp.array(T0)
        return PropsJI("DMASS", "P", "T", fluid)(10*P, T)
    
    rho_c_test  = composed_fn(P0)
    drhodP_test = jax.grad(composed_fn)(P0)
    drhodP_01 = PropsSI("d(DMASS)/d(P)|T", "P", P0, "T", T0, fluid)
    drhodP_10 = PropsSI("d(DMASS)/d(P)|T", "P", 10*P0, "T", T0, fluid) * 10

    PT_Rho = PropsJI("DMASS", "P", "T", fluid)
    P_arr = jnp.linspace(1.0e5, 10.0e5, 100)
    T_arr = jnp.linspace(100.0,  200.0, 100)
    Rho_arr = PT_Rho(P_arr, T_arr)

    import time

    PT_H = PropsJI("H", "P", "T", "Nitrogen")   # H(P, T)
    PT_S = PropsJI("S", "P", "T", "Nitrogen")   # S(P, T)
    DT_P = PropsJI("P", "D", "T", "Nitrogen")   # P(rho, T)
    
    # ── Test arrays ───────────────────────────────────────────────────────────────
    
    N   = 50
    P_arr = jnp.linspace(1e5, 50e5, N)   # [Pa]
    T_arr = jnp.linspace(100.0, 400.0, N) # [K]
    
    # ── Scalar grad functions ─────────────────────────────────────────────────────
    
    grad_PT_H = jax.grad(PT_H, argnums=(0, 1))   # (dH/dP, dH/dT)
    grad_PT_S = jax.grad(PT_S, argnums=(0, 1))   # (dS/dP, dS/dT)
    grad_DT_P = jax.grad(DT_P, argnums=(0, 1))   # (dP/drho, dP/dT)

    def timeit(fn, *args, n=3, label=""):
        # Warmup
        fn(*args)
        t0 = time.perf_counter()
        for _ in range(n):
            jax.block_until_ready(fn(*args))
        elapsed = (time.perf_counter() - t0) / n * 1000
        print(f"  {label:<45s}  {elapsed:8.2f} ms")
        return fn(*args)
    
    def section(title):
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 1) SCALAR — value + grad
    # ─────────────────────────────────────────────────────────────────────────────
    section("1) Scalar — value + grad")
    
    P0, T0 = P_arr[0], T_arr[0]
    
    timeit(PT_H,      P0, T0,  label="PT_H(P, T)                  value")
    timeit(grad_PT_H, P0, T0,  label="grad(PT_H)(P, T)            grad")
    timeit(PT_S,      P0, T0,  label="PT_S(P, T)                  value")
    timeit(grad_PT_S, P0, T0,  label="grad(PT_S)(P, T)            grad")
    timeit(DT_P,      P0, T0,  label="DT_P(rho, T)                value")
    timeit(grad_DT_P, P0, T0,  label="grad(DT_P)(rho, T)          grad")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 2) VMAP (no JIT) — batched value + grad
    # ─────────────────────────────────────────────────────────────────────────────
    section("2) vmap (no JIT) — batched over N=%d points" % N)
    
    vmap_PT_H      = jax.vmap(PT_H,      in_axes=(0, 0))
    vmap_PT_S      = jax.vmap(PT_S,      in_axes=(0, 0))
    vmap_DT_P      = jax.vmap(DT_P,      in_axes=(0, 0))
    vmap_grad_PT_H = jax.vmap(grad_PT_H, in_axes=(0, 0))
    vmap_grad_PT_S = jax.vmap(grad_PT_S, in_axes=(0, 0))
    vmap_grad_DT_P = jax.vmap(grad_DT_P, in_axes=(0, 0))
    
    timeit(vmap_PT_H,      P_arr, T_arr, label="vmap(PT_H)                  value")
    timeit(vmap_grad_PT_H, P_arr, T_arr, label="vmap(grad(PT_H))            grad")
    timeit(vmap_PT_S,      P_arr, T_arr, label="vmap(PT_S)                  value")
    timeit(vmap_grad_PT_S, P_arr, T_arr, label="vmap(grad(PT_S))            grad")
    timeit(vmap_DT_P,      P_arr, T_arr, label="vmap(DT_P)                  value")
    timeit(vmap_grad_DT_P, P_arr, T_arr, label="vmap(grad(DT_P))            grad")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 3) JIT (no vmap) — scalar but compiled
    # ─────────────────────────────────────────────────────────────────────────────
    section("3) jit (no vmap) — scalar")
    
    jit_PT_H      = jax.jit(PT_H)
    jit_grad_PT_H = jax.jit(grad_PT_H)
    jit_PT_S      = jax.jit(PT_S)
    jit_grad_PT_S = jax.jit(grad_PT_S)
    jit_DT_P      = jax.jit(DT_P)
    jit_grad_DT_P = jax.jit(grad_DT_P)
    
    timeit(jit_PT_H,      P0, T0, label="jit(PT_H)                   value")
    timeit(jit_grad_PT_H, P0, T0, label="jit(grad(PT_H))             grad")
    timeit(jit_PT_S,      P0, T0, label="jit(PT_S)                   value")
    timeit(jit_grad_PT_S, P0, T0, label="jit(grad(PT_S))             grad")
    timeit(jit_DT_P,      P0, T0, label="jit(DT_P)                   value")
    timeit(jit_grad_DT_P, P0, T0, label="jit(grad(DT_P))             grad")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 4) JIT + VMAP — batched and compiled
    # ─────────────────────────────────────────────────────────────────────────────
    section("4) jit(vmap) — batched + compiled over N=%d points" % N)
    
    jit_vmap_PT_H      = jax.jit(vmap_PT_H)
    jit_vmap_grad_PT_H = jax.jit(vmap_grad_PT_H)
    jit_vmap_PT_S      = jax.jit(vmap_PT_S)
    jit_vmap_grad_PT_S = jax.jit(vmap_grad_PT_S)
    jit_vmap_DT_P      = jax.jit(vmap_DT_P)
    jit_vmap_grad_DT_P = jax.jit(vmap_grad_DT_P)
    
    timeit(jit_vmap_PT_H,      P_arr, T_arr, label="jit(vmap(PT_H))             value")
    timeit(jit_vmap_grad_PT_H, P_arr, T_arr, label="jit(vmap(grad(PT_H)))       grad")
    timeit(jit_vmap_PT_S,      P_arr, T_arr, label="jit(vmap(PT_S))             value")
    timeit(jit_vmap_grad_PT_S, P_arr, T_arr, label="jit(vmap(grad(PT_S)))       grad")
    timeit(jit_vmap_DT_P,      P_arr, T_arr, label="jit(vmap(DT_P))             value")
    timeit(jit_vmap_grad_DT_P, P_arr, T_arr, label="jit(vmap(grad(DT_P)))       grad")
    
    # ─────────────────────────────────────────────────────────────────────────────
    # 5) Correctness check — AD vs CoolProp analytical
    # ─────────────────────────────────────────────────────────────────────────────
    section("5) Correctness check — AD vs CoolProp analytical")
    
    rho0 = PropsJI("DMASS","P","T",fluid)(jnp.array(P0),jnp.array(T0))

    dHdP_ad, dHdT_ad   = grad_PT_H(P0,   T0)
    dSdP_ad, dSdT_ad   = grad_PT_S(P0,   T0)
    dPdrho_ad, dPdT_ad = grad_DT_P(rho0, T0)
    
    checks = [
        ("dH/dP|T",   float(dHdP_ad),   PropsSI("d(H)/d(P)|T",   "P", float(P0), "T", float(T0), "Nitrogen")),
        ("dH/dT|P",   float(dHdT_ad),   PropsSI("d(H)/d(T)|P",   "P", float(P0), "T", float(T0), "Nitrogen")),
        ("dS/dP|T",   float(dSdP_ad),   PropsSI("d(S)/d(P)|T",   "P", float(P0), "T", float(T0), "Nitrogen")),
        ("dS/dT|P",   float(dSdT_ad),   PropsSI("d(S)/d(T)|P",   "P", float(P0), "T", float(T0), "Nitrogen")),
        ("dP/dD|T",   float(dPdrho_ad), PropsSI("d(P)/d(D)|T",   "P", float(P0), "T", float(T0), "Nitrogen")),
        ("dP/dT|D",   float(dPdT_ad),   PropsSI("d(P)/d(T)|D",   "P", float(P0), "T", float(T0), "Nitrogen")),
    ]
    
    print(f"\n  {'Derivative':<12}  {'AD':>14}  {'CoolProp':>14}  {'rel error':>10}  {'✓/✗':>4}")
    print("  " + "-"*58)
    for name, ad, ref in checks:
        err = abs(ad - ref) / abs(ref) if ref != 0 else abs(ad - ref)
        ok  = "✓" if err < 1e-6 else "✗"
        print(f"  {name:<12}  {ad:>14.6e}  {ref:>14.6e}  {err:>10.2e}  {ok:>4}")

    #%% Test partial derivatives

    PT_Beta = PropsJI("d(DMASS)/d(T)|P", "P", "T", fluid)
    P_sat_fn = PropsJI("P", "T", "Q", fluid)

    Beta, dBetadX = jax.value_and_grad(PT_Beta, argnums=(0,1))(jnp.array(1e5), jnp.array(77.0))

    P_sat = P_sat_fn(jnp.array(77.0), jnp.zeros(1))
    dPdX_sat = jax.grad(P_sat_fn, argnums=(0,1))(jnp.array(77.0), jnp.array(0.0))

    Rho_sat = PropsJI("DMASS","P","Q",fluid)(P_sat, jnp.array(0.5))

    #%% Test 2nd order derivatives

    

#%% Run above ^^
