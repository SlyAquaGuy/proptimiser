## Define All Aerodynamic Properties for Given Propeller Geometry

# External Libraries
import jax
import jax.numpy as jnp
from jax import jit, vmap, grad, lax

import matplotlib.pyplot as plt
from airflow import AirflowParams

# Internal Libraries
from init import inputs
from coefficients import parametric_coeffs
from solvers import NewtonSolver


### Integrand Functions

def radial_stops(R, r_hub, n):
    # Return radial node centers and dr for integration
    r = jnp.linspace(r_hub + (R - r_hub) / (2 * n), R - (R - r_hub) / (2 * n), n)
    dr = (R - r_hub) / n
    return r, dr

def angular_stops(n_phi):
    # Return angular node centers and dphi for integration
    phi = jnp.linspace((jnp.pi / n_phi), (2 * jnp.pi) - (jnp.pi / n_phi), n_phi)
    dphi = (2 * jnp.pi) / n_phi
    return phi, dphi

def thrust_residual(V_ia_0, tr_params):
    """
    Calculate thrust residual for an annular element at radius r using AirflowParams dataclass.
    """
    # Unpack parameters
    N, c, phi, r, dr, psi, dpsi = tr_params.N, tr_params.c, tr_params.phi, tr_params.r, tr_params.dr, tr_params.psi, tr_params.dphi, 
    V_x, V_yz, omega = tr_params.V_x, tr_params.V_yz, tr_params.omega
    rho, C_L, C_D = tr_params.rho, tr_params.C_L, tr_params.C_D

    # Calculate Flow Properties
    xi = jnp.arctan(V_yz / (V_x + V_ia_0))    # wake skew angle
    V_yz_perp = V_yz * jnp.sin(psi)           # Perpendicular component of velocity in yz plane

    V_rel = jnp.sqrt((V_x + V_ia) ** 2 + (omega * r + V_yz_perp) ** 2)  # relative velocity

    # Inflow Model for Skewed Rotor (Pitt & Peters 1981)
    V_ia = V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / tr_params.R) * jnp.cos(phi))  

    # Implement DTU model here to improve accuracy?


    # Momentum Theory about annular element 
    dT_momentum = (2 * rho * r * V_ia * jnp.sqrt((V_x + V_ia) ** 2 + V_yz ** 2) * dpsi * dr)  
    # Blade Element Theory about annular element 
    dT_blade =( 
        (N / (4 * jnp.pi)) * rho * c * V_rel ** 2 
        * (C_L * jnp.cos(phi) + C_D * jnp.sin(phi))* dpsi* dr
        # tip and root loss factors
        * f_tip * f_root
        )

    return jnp.sum(dT_momentum - dT_blade)


def induced_velocity(params):
    # Unpack Parameters
    inflow_model = inputs.inflow_model
    rho = 
    # Minimise Thrust Residual to Solve for Induced Velocity
    solve_annulus = lambda 
    thrust_residual_fun = lambda V_ia_0, tr_params: thrust_residual(V_ia_0, tr_params)
    solver = NewtonSolver(thrust_residual_fun)
    # Test Case Single Annulus
    V_ia0_guess = 0.1 * params.V_x  # Initial guess for induced velocity
    V_ia = solver.solve(V_ia0_guess, tr_params)
    



    # Solve for induced velocity using momentum theory and blade element theory
    # First Calculate 
    return

    



    



  



# Calculate Torque, Thrust, and Power Coefficients in Propeller Frame

if __name__ == "__main__":
    # Air Properties
    rho = 1.225  # Air density in kg/m^3


    # Example/Test Case Parameters
    R = 0.2  # Propeller radius in meters
    r_hub = 0.05  # Hub radius in meters
    n_r = 10  # Number of radial stations
    n_phi = 20  # Number of angular stations

    # Define integrands for Internal Solver use
    alpha = jnp.radians(jnp.linspace(-180, 180, 100))  # Angle of attack in radians
    r, dr = radial_stops(R, r_hub, n_r)  # Radial stations from hub to tip
    phi, dphi = angular_stops(n_phi)  # Angular stations around the propeller

    # Solve Induced Velocity using BEMT
    C_L, C_D, C_M = parametric_coeffs(alpha, debug_plots=False)  # Get parametric coefficients




