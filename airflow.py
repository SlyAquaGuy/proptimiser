# Airflow Definitions, Boundary Conditions for Momentum Theory
from dataclasses import replace
import jax.numpy as jnp
from init import Inputs
from jax import lax

## Dimensionless Constants
def mach(V, a):
    # Calculate Mach number
    return V / a
def reynolds(rho, V, c, mu):
    # Calculate Reynolds number
    return (rho * V * c) / mu

## Inflow Model Definitions and Selection
# 0 - Simple uniform inflow
# 1 - Pitt & Peters (1981)
# 2 - DTU model

def simple(params):
    # Simple uniform inflow model
    V_ia = jnp.full((params.r.size, params.psi.size), jnp.average(params.V_ia_0))
    return V_ia

def pitt_peters(params):
    # Unpack parameters
    V_ia_0 = params.V_ia_0[:,None]
    V_x, V_yz = params.V_x, params.V_yz
    r,psi,R = params.r[:,None], params.psi[None,:], params.R

    xi = jnp.arctan(V_yz / (V_x + V_ia_0))    # wake skew angle
    # Inflow model for skewed rotor (Pitt & Peters 1981)
    V_ia = V_ia_0 * (1 + ((15 * jnp.pi) / 32) * jnp.tan(xi / 2) * (r / R) * jnp.cos(psi))
    return V_ia

def dtu_model(params):
    V_ia = jnp.full((params.r.size, params.psi.size), jnp.average(params.V_ia_0))
    # Implement DTU model here to improve accuracy? /doi.org/10.5194/wes-5-1-2020
    return V_ia  # Placeholder, return initial guess for now

def inflow_model(params):
    # Calculate inflow velocity based on selected model
    return lax.switch(params.inflow_model,
                    [simple, pitt_peters, dtu_model],
                    params
                    )


## Tip and Root Loss Model Definitions and Selection
# 0 - No loss
# 1 - Prandtl Loss

def no_loss(params):
    # No loss factor
    return jnp.ones_like(params.r)

def prandtl_tip_loss(params):
    # Calculate Prandtl tip loss factor
    return jnp.ones_like(params.r)  # Placeholder, implement actual model later

def prandtl_root_loss(params):
    # Calculate Prandtl root loss factor
    return jnp.ones_like(params.r)  # Placeholder, implement actual model later

def tip_root_loss_factors(params):
    # Calculate tip and root loss factors based on selected models
    tip_loss = lax.switch(params.tip_loss_model,
                      [no_loss, prandtl_tip_loss],
                      params)
    
    root_loss = lax.switch(params.root_loss_model,
                      [no_loss, prandtl_root_loss],
                        params)

    return tip_loss, root_loss 





