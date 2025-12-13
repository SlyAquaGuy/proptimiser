## Define All Aerodynamic Properties for Given Propeller Geometry

import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax

import matplotlib.pyplot as plt




# Air properties
rho = 1.225 # Density of air in kg/m^3



def radial_stops(R, r_hub, n):
    # Return radial node centers and dr for integration
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * N), R - (R - r_hub) / (2 * N), N)
    dr = (R - r_hub) / N
    return r, dr

### ---Parametric Coefficients---
# (good for approximate analysis, use LUT for more accuracy)

alpha = jnp.radians(jnp.linspace(0, 360, 100))  # Angle of attack in radians

def parametric_coeffs(alpha):
    # Return parametric estimations of lift, drag, and moment coefficients based on angle of attack
    # Coefficient parameters


    # Set piecewise function between 20 and 160 degrees
    stall_low  = alpha >= jnp.radians(20)
    stall_high = alpha >  jnp.radians(160)
    stall      = stall_low & ~stall_high   # 20°–160°
    
    C_La = 0.11     # Lift curve slope per radian (ARAD10)
    C_D90 = 1.98    # Drag coefficient at 90 degrees angle of attack (flat plate)
    C_D0 = 0.018    # Zero lift drag coefficient (ARAD10)

    print(C_La, C_D0)

    # Coefficients in non-stall region
    C_L_unstall = C_La * alpha                                                          # Lift Coefficient
    C_T_unstall = C_D0 * jnp.cos(alpha)                                                 # Tangential Force Coefficient
    C_N_unstall = (C_L_unstall + C_T_unstall*jnp.sin(alpha)) / (jnp.cos(alpha))         # Normal Force Coefficient
    C_D_unstall = C_N_unstall * jnp.sin(alpha) + C_T_unstall* jnp.cos(alpha)            # Drag Coefficient
    C_M_unstall = -C_N_unstall*(0.25-0.175*(1-2*alpha/jnp.pi))                          # Moment Coefficient
    
    # Coefficients in stall region based on flat plate model
    C_N_stall = C_D90 * (jnp.sin(alpha)) / (0.56+0.44*jnp.sin(alpha))                   # Normal Force Coefficient
    C_T_stall = 0.5*C_D0 * jnp.cos(alpha)                                               # Tangential Force Coefficient
    C_L_stall = C_N_stall*jnp.cos(alpha)-C_T_stall*jnp.sin(alpha)                       # Lift Coefficient
    C_D_stall = C_N_stall*jnp.sin(alpha)+C_T_stall*jnp.cos(alpha)                       # Drag Coefficient
    C_M_stall = -C_N_unstall*(0.25-0.175*(1-2*alpha/jnp.pi))                            # Moment Coefficient

    # Piecewise coefficients
    C_L = jnp.where(stall, C_L_stall,
           jnp.where(stall_high, C_L_stall, C_L_unstall))

    C_D = jnp.where(stall, C_D_stall,
           jnp.where(stall_high, C_D_stall, C_D_unstall))                              
    return C_L, C_D

C_L, C_D = parametric_coeffs(alpha)

# Debug plot of parametric coefficients
plt.plot(jnp.degrees(alpha), C_L)
plt.plot(jnp.degrees(alpha), C_D)
plt.legend(["C_L", "C_D"])
plt.title("Parametric Lift and Drag Coefficient vs Angle of Attack")
plt.xlabel("Angle of Attack (degrees)")
plt.ylabel("Lift Coefficient C_L")
plt.show()

# def bemt_annulus(w, T, D):
    # Solve for differential torque and thrust of annulus elements
    










# Calculate Torque, Thrust, and Power Coefficients in Propeller Frame




if __name__ == "__main__":
    # Example usage
    R = 0.5  # Propeller radius in meters
    r_hub = 0.15  # Hub radius in meters
    n = 10  # Number of radial stations

