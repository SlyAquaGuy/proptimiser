import jax
import jax.numpy as jnp
import optax
import functools
import matplotlib.pyplot as plt

# Internal Libraries
from init import Params, Inputs
import propmodel

def run_optimization():
    jax.config.update("jax_enable_x64", True)

    # --- 1. Setup Base Geometry ---
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(0.5, 0.05, Inputs.n_r)
    V_ia_0 = jnp.full_like(r, 10.0)

    params_base = Params(
        N=2.0, R=jnp.array([0.15]),
        c=jnp.full_like(r, 0.01)[:, None],
        beta=jnp.full_like(r, jnp.radians(20.0))[:, None],
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([10.0]), V_yz=jnp.array([2.0]),
        omega=jnp.array([6000.0]),
        V_ia_0=V_ia_0[:, None]
    )

    # Helper: Convert optimizer's raw values to physical parameters (c > 0)
    def get_physical_params(opt_params, static_params: Params):
        # softplus ensures chord is always positive: c = log(1 + exp(c_raw))
        return static_params.replace(
            c=jax.nn.softplus(opt_params["c_raw"]),
            beta=opt_params["beta"]
        )

    # Initial Guess for Optimizer
    # We use inverse softplus so that the initial physical chord is 0.01m
    def inv_softplus(y): return jnp.log(jnp.exp(y) - 1.0)
    init_opt_params = {
        "c_raw": inv_softplus(params_base.c),
        "beta": params_base.beta
    }

    # --- 2. Define the Augmented Lagrangian Function ---
    def lagrangian_fn(opt_params, static_params, lmbda, mu):
        p = get_physical_params(opt_params, static_params)
        V_ia = propmodel.induced_velocity(p)
        
        # Objective: Minimize Power
        power = jnp.squeeze(propmodel.power(V_ia, p))
        
        # Constraint: Thrust - Target = 0
        thrust_error = jnp.squeeze(propmodel.thrust(V_ia, p) - Inputs.thrust)
        
        # ALM Formula: Objective + (Multiplier * Error) + (Penalty/2 * Error^2)
        return power + lmbda * thrust_error + 0.5 * mu * (thrust_error**2)

    # --- 3. Optimizer Setup ---
    optimiser = optax.lbfgs(
        learning_rate=None, # LBFGS uses linesearch
        memory_size=10,
        linesearch=optax.scale_by_zoom_linesearch(max_linesearch_steps=20)
    )

    @jax.jit
    def inner_step(params, state, lmbda, mu):
        # Partial function for the linesearch to see the current lambda/mu
        v_fn = lambda p: lagrangian_fn(p, params_base, lmbda, mu)
        
        loss_val, grads = jax.value_and_grad(lagrangian_fn)(params, params_base, lmbda, mu)
        
        updates, next_state = optimiser.update(
            grads, state, params, 
            value=loss_val, grad=grads, value_fn=v_fn
        )
        next_params = optax.apply_updates(params, updates)

        gnorm = optax.global_norm(grads)

        return next_params, next_state, loss_val, gnorm

    # --- 4. Outer Loop (Augmented Lagrangian Iterations) ---
    opt_params = init_opt_params
    opt_state = optimiser.init(opt_params)
    
    # ALM multipliers
    lmbda = 0.0   # The "Force" pushing the constraint to zero
    mu = 10.0     # The "Stiffness" of the penalty
    
    print(f"{'Iter':<6} | {'Loss':<10} | {'Thrust Err':<12} | {'Lambda':<10}")
    print("-" * 50)

    for outer_i in range(20):
        # Solve the Lagrangian for the current multipliers
        for inner_i in range(Inputs.opt_max_iter):
            opt_params, opt_state, loss, gnorm = inner_step(opt_params, opt_state, lmbda, mu)
            if jnp.abs(gnorm) < 1e-5:
                print("Inner Iterations Satisfied")
                break
        
        # Calculate current physical violation
        phys = get_physical_params(opt_params, params_base)
        V_ia = propmodel.induced_velocity(phys)
        thrust_err = jnp.squeeze(propmodel.thrust(V_ia, phys) - Inputs.thrust)

        # Update Rule: Shift the "target" based on the error
        lmbda += mu * thrust_err
        
        # Increase penalty (mu) to sharpen the constraint
        mu = min(mu * 1.5, 1e6) 

        print(f"{outer_i:<6} | {loss:<10.4f} | {thrust_err:<12.4e} | {lmbda:<10.4f}")

        if abs(thrust_err) < 1e-5:
            print("\nConstraint Rigorously Satisfied.")
            break

    

    # --- 5. Final Results ---
    final_params = get_physical_params(opt_params, params_base)
    print(f"\nFinal Chord: {jnp.squeeze(final_params.c)} m")
    print(f"\nFinal Beta: {jnp.squeeze(final_params.beta)} degree")
    print(f"Final Power: {jnp.squeeze(propmodel.power(V_ia, final_params)):.4f} W")
    print(f"Final Thrust: {jnp.squeeze(propmodel.thrust(V_ia, final_params)):4f} N")

    plt.plot(jnp.squeeze(final_params.r), jnp.squeeze(final_params.c))
    plt.xlabel("r")
    plt.ylabel("chord")
    plt.show()

    return final_params

if __name__ == "__main__":
    run_optimization()