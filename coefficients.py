### Establish Coefficients for Given Airflow and Geometry
from jax import numpy as jnp
import matplotlib.pyplot as plt

## ---Parametric Coefficients---
# (good for approximate analysis, use LUT for more accuracy)

def parametric_coeffs(alphain, debug=False):
    # Return parametric estimations of lift, drag, and moment coefficients based on angle of attack
    # Coefficient parameters
    alpha = jnp.abs(alphain)  # Use absolute value of alpha for symmetry
    # Set piecewise function between 20 and 160 degrees
    stall_low  = jnp.abs(alpha) >= jnp.abs(jnp.radians(20))
    stall_high = jnp.abs(alpha) >  jnp.radians(180) # Don't model high stall angle, inverse flow is complex
    stall      = stall_low & ~stall_high   # 20°–160°
    
    C_La = 5.7     # Lift curve slope per rad (ARAD10)
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.035    # Zero lift drag coefficient (ARAD10)

    # Coefficients in non-stall region
    C_L_unstall = C_La * alpha                                                          # Lift Coefficient
    C_T_unstall = C_D0 * jnp.cos(alpha)                                                 # Tangential Force Coefficient
    C_N_unstall = (C_L_unstall + C_T_unstall*jnp.sin(alpha)) / (jnp.cos(alpha))         # Normal Force Coefficient
    C_D_unstall = C_N_unstall * jnp.sin(alpha) + C_T_unstall* jnp.cos(alpha)            # Drag Coefficient
    C_M_unstall = -C_N_unstall*(0.25-0.175*(1-2*alpha/jnp.pi))                          # Moment Coefficient
    
    # Coefficients in stall region based on flat plate model
    C_N_stall = C_D90 * (jnp.sin(alpha)) / (0.56+0.44*jnp.sin(alpha))          # Normal Force Coefficient
    C_T_stall = 0.5*C_D0 *jnp.cos(alpha)                                                # Tangential Force Coefficient
    C_L_stall = C_N_stall*jnp.cos(alpha)-C_T_stall*jnp.sin(alpha)                       # Lift Coefficient
    C_D_stall = C_N_stall*jnp.sin(alpha)+C_T_stall*jnp.cos(alpha)                       # Drag Coefficient
    C_M_stall = -C_N_stall*(0.25-0.175*(1-2*alpha/jnp.pi))                              # Moment Coefficient

    # Piecewise coefficients (stall for 20°–160°)
    C_L = jnp.sign(alphain)*jnp.where(stall, C_L_stall, C_L_unstall)
    C_D = jnp.where(stall, C_D_stall, C_D_unstall)
    C_M = jnp.sign(alphain)*jnp.where(stall, C_M_stall, C_M_unstall)





    if debug:
        # Debug plot of parametric coefficients
        plt.plot(jnp.degrees(alphain), C_L)
        plt.plot(jnp.degrees(alphain), C_D)
        plt.plot(jnp.degrees(alphain), C_M)
        plt.legend(["C_L", "C_D"])
        plt.title("Parametric Lift, Drag and Moment Coefficient vs Angle of Attack")
        plt.xlabel("Angle of Attack (degrees)")
        plt.ylabel("Lift Coefficient C_L")
        plt.show()
    
    return C_L, C_D, C_M

def lookup_coeffs(alphain):
    # Placeholder for future lookup table implementation
    raise NotImplementedError("Lookup table coefficients not yet implemented.")
    # Look up/estimate CL/CD from CFD data tables
    # Input parameters of alpha, Re, Ma, airfoil type. Return suitable coefficients that minimise chord.
