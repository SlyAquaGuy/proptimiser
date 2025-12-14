class inputs:

    # Optimisation / design settings

    # Outer-loop optimisation (Optax / similar)
    opt_max_iter: int = 200
    opt_tol: float = 1e-6

    # Thrust constraint
    thrust = 10 # Newtons
    

    ## Discretisation settings

    # Radial discretisation
    n_r: int = 5

    # Azimuthal discretisation (for skewed inflow)
    n_phi: int = 5

    ## Physical / modelling flags
    use_prandtl_tip_loss: bool = False
    use_prandtl_root_loss: bool = False
    use_skew_inflow_model: bool = True

    # Diagnostics / debugging

    verbose: bool = False
    record_convergence_history: bool = False

