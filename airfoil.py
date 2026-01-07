### Establish Coefficients for Given Airfoil
import jax
from jax import numpy as jnp
from jax import lax
import matplotlib.pyplot as plt
from airflow import calc_phi

## ---Parametric Coefficients---
# (good for approximate analysis, use LUT for more accuracy)

def parametric_coeffs(V_ia, params, debug=False):
    # Return parametric estimations of lift, drag, and moment coefficients based on angle of attack
    # Calculate effective angle of attack for the airfoil
    phi = calc_phi(V_ia, params)
    alpha = params.beta[:,None]-phi
    # Coefficient parameters
    alphaabs = jnp.abs(alpha)  # Use absolute value of alpha for symmetry
    # Set piecewise function between 20 and 160 degrees
    stall_low  = jnp.abs(alphaabs) >= jnp.abs(jnp.radians(20))
    stall_high = jnp.abs(alphaabs) >  jnp.radians(180) # Don't model high stall angle, inverse flow is complex
    stall      = stall_low & ~stall_high   # 20°–160°
    
    C_La = 5.7     # Lift curve slope per rad (ARAD10)
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.035    # Zero lift drag coefficient (ARAD10)

    # Coefficients in non-stall region
    C_L_unstall = C_La * alphaabs                                                          # Lift Coefficient
    C_T_unstall = C_D0 * jnp.cos(alphaabs)                                                 # Tangential Force Coefficient
    C_N_unstall = (C_L_unstall + C_T_unstall*jnp.sin(alphaabs)) / (jnp.cos(alphaabs))         # Normal Force Coefficient
    C_D_unstall = C_N_unstall * jnp.sin(alphaabs) + C_T_unstall* jnp.cos(alphaabs)            # Drag Coefficient
    C_M_unstall = -C_N_unstall*(0.25-0.175*(1-2*alphaabs/jnp.pi))                          # Moment Coefficient
    
    # Coefficients in stall region based on flat plate model
    C_N_stall = C_D90 * (jnp.sin(alphaabs)) / (0.56+0.44*jnp.sin(alphaabs))          # Normal Force Coefficient
    C_T_stall = 0.5*C_D0 *jnp.cos(alphaabs)                                                # Tangential Force Coefficient
    C_L_stall = C_N_stall*jnp.cos(alphaabs)-C_T_stall*jnp.sin(alphaabs)                       # Lift Coefficient
    C_D_stall = C_N_stall*jnp.sin(alphaabs)+C_T_stall*jnp.cos(alphaabs)                       # Drag Coefficient
    C_M_stall = -C_N_stall*(0.25-0.175*(1-2*alphaabs/jnp.pi))                              # Moment Coefficient

    # Piecewise coefficients (stall for 20°–160°)
    C_L = jnp.sign(alpha)*jnp.where(stall, C_L_stall, C_L_unstall)
    C_D = jnp.where(stall, C_D_stall, C_D_unstall)
    C_M = jnp.sign(alpha)*jnp.where(stall, C_M_stall, C_M_unstall)

    if debug:
        # Debug plot of parametric coefficients
        plt.plot(jnp.degrees(alpha), C_L)
        plt.plot(jnp.degrees(alpha), C_D)
        plt.plot(jnp.degrees(alpha), C_M)
        plt.legend(["C_L", "C_D"])
        plt.title("Parametric Lift, Drag and Moment Coefficient vs Angle of Attack")
        plt.xlabel("Angle of Attack (degrees)")
        plt.ylabel("Lift Coefficient C_L")
        plt.show()
    
    return C_L, C_D, C_M

def lookup_coeffs(V_ia, params):
    # Placeholder for future lookup table implementation, use parametric for now to allow compile
    C_L, C_D, C_M  = parametric_coeffs(V_ia, params)
    # Look up/estimate CL/CD from CFD data tables
    # Input parameters of alpha, Re, Ma, airfoil type. Return suitable coefficients that minimise chord.
    return C_L, C_D, C_M

def aero_coeffs(V_ia, params):
    # Select Method to Establish Aerodynamic Coefficients
    C_L, C_D, C_M = lax.switch(params.coeff_method,
                    [parametric_coeffs, lookup_coeffs],
                    V_ia, params)
    return C_L, C_D, C_M
