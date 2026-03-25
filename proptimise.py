import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt

from init import Params, Inputs
import propmodel

def curvature_penalty(x):
    '''
    Discrete Second-Derivative Penalty (Approximation)
    '''
    d2 = x[2:]-2.0*x[1:-1]+x[:-2]

    return jnp.sum(d2**2)


def run_optimization():
    '''
    WIP. Code currently doesn't solve, and a re-implementation is required.
    '''
    # 64 Bit Precision, otherwise thrust residuals don't converge
    jax.config.update("jax_enable_x64", True)
    # --- 1. Setup ---
    psi, dpsi = propmodel.angular_stops(Inputs.n_psi)
    r, dr = propmodel.radial_stops(0.5, 0.05, Inputs.n_r)
    V_ia_0 = jnp.full_like(r, 10.0)

    # Initial Guesses
    init_c = jnp.full((len(r), 1), 0.01)
    init_beta = jnp.full((len(r), 1), jnp.radians(10.0))
    init_vars = (init_c, init_beta)

    # Initial static parameters
    params_static = Params(
        N=2.0, R=jnp.array([0.15]),
        c=init_c,
        beta=init_beta,
        r=r[:, None], dr=dr,
        psi=psi[None, :], dpsi=dpsi,
        V_x=jnp.array([10.0]), V_yz=jnp.array([2.0]),
        omega=jnp.array([400.0]),
        V_ia_0=V_ia_0[:, None]
    )

    # --- 2. Define Bounds ---
    # Chord: 1mm to 50mm | Beta: 2 deg to 70 deg
    lower_bounds = (jnp.full_like(init_c, 0.001), jnp.full_like(init_beta, jnp.radians(2.0)))
    upper_bounds = (jnp.full_like(init_c, 0.050), jnp.full_like(init_beta, jnp.radians(70.0)))

    # --- 3. Objective Function (with Penalty) ---
    def objective_fn(opt_vars, lmbda, mu):
        c, beta = opt_vars
        if Inputs.verbose:
            jax.debug.print("Current Mean Chord: {x}", x=jnp.mean(c))
        
        # Build the param object for the model
        params = params_static.replace(c=c, beta=beta)
        
        # Solve for induced velocity
        V_ia = propmodel.induced_velocity(params)
        
        # Calculate performance
        power = jnp.squeeze(propmodel.power(V_ia, params))
        thrust = jnp.squeeze(propmodel.thrust(V_ia, params))
        
        # Penalty Function for Thrust
        thrust_err = thrust - Inputs.thrust

        # Smoothness/Continuity Penalty 
        smooth_beta = curvature_penalty(beta)
        smooth_chord = curvature_penalty(c)

        smooth_penalty = (
            Inputs.lambda_beta_smooth*smooth_beta+
            Inputs.lambda_chord_smooth*smooth_chord
        )
        
        # Augmented Lagrangian: Obj + (L * err) + (mu/2 * err^2)
        # Using a heavy penalty (mu) ensures the thrust constraint is met
        return power + lmbda * thrust_err + 0.5 * mu * (thrust_err**2) + smooth_penalty

    # --- 4. Optimization Loop ---
    # (L-BFGS-B) for outer loop optimisation allows implementing constraints, 
    solver = jaxopt.ScipyBoundedMinimize(fun=objective_fn, method="L-BFGS-B", tol=Inputs.opt_tol)
    
    '''
    Janky initial solver loop just to get a propeller converging.
    '''
    
    curr_vars = init_vars
    lmbda = 0.0
    mu = 10.0 # Initial stiffness for thrust

    print(f"{'Iter':<6} | {'Total Loss':<12} | {'Thrust Err':<12} | {'Mu':<10}")
    print("-" * 55)

    for i in range(50):
        # Solve the sub-problem
        sol = solver.run(curr_vars, bounds=(lower_bounds, upper_bounds), lmbda=lmbda, mu=mu)
        curr_vars = sol.params
        
        # Check physical error
        c_final, beta_final = curr_vars
        params_final = params_static.replace(c=c_final, beta=beta_final)
        V_ia_final = propmodel.induced_velocity(params_final)
        t_final = jnp.squeeze(propmodel.thrust(V_ia_final, params_final))
        thrust_err = t_final - Inputs.thrust
        
        print(f"{i:<6} | {sol.state.fun_val:<12.4f} | {thrust_err:<12.4e} | {mu:<10.1f}")

        # Update Multipliers
        lmbda += mu * thrust_err
        mu *= 2.0  # Aggressively increase penalty stiffness
        
        if abs(thrust_err) < 1e-4:
            break

    return params_final

if __name__ == "__main__":
    params_opt = run_optimization()

    # Unpack Parameters
    r = params_opt.r
    c, beta = params_opt.c, params_opt.beta
    # --- 5. Visualization ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(r, c)
    plt.title("Optimized Chord (m)")
    plt.subplot(1, 2, 2)
    plt.plot(r, jnp.degrees(beta))
    plt.title("Optimized Beta (deg)")
    plt.show()