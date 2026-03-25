# 1 Implementation

## 1.1 Algorithm/Workflow

- [ ] Improve use of JAX to mulithread/gpu accellerate solving.

### 1.1.1 Problem Specification

### BEMT Core
![alt text](image.png)
- [ ] Implement induced velocity logging between iterations (improve initial guess for $V_{ai}$, reduce convergence time)

#### Blade Element Theory Improvements

- [ ] Improve/Evaluate Power Consumption Code.
- [ ] Implement Variable flat-plate estimation as fallback for stall conditions

#### Momentum Theory Improvements

- [ ] Downwash/induced wake implementation [@lengComparisonsDifferentPropeller2019]

### 1.1.3 Iterative Solving

#### ADAM Optimiser

- [ ] Implement Minimisation Function for each radial stop with Target of Minimum Power Consumption
- [ ] Impose a strong [Lagrangian Constraint](https://math.libretexts.org/Bookshelves/Calculus/Vector_Calculus_(Corral)/02%3A_Functions_of_Several_Variables/2.07%3A_Constrained_Optimization_-_Lagrange_Multipliers) on Thrust.
- [ ] Impose a weak [Lagrangian Constraint]([Lagrangian%20Constraint](https://math.libretexts.org/Bookshelves/Calculus/Vector_Calculus_(Corral)/02%3A_Functions_of_Several_Variables/2.07%3A_Constrained_Optimization_-_Lagrange_Multipliers)) on Curvature.

#### Line Search Optimiser (Broyden/L-BFGS-B)

- [ ] Optimise a single annuli with single core. Hard curvature limit implemented in code (b-spline energy related?) which sets .jaxopt broyden bounds.  *For an alternative approach we could set a harsh/exponential curvature penalty*
- [ ] Research ways to 

### 1.1.4 Performance Evaluation+Plotting

- [ ] Calculate Induced Loss and Efficiency Compared to actuator disk theory.
- [ ] Visualise solution space/solver iterations

## UI/UX

- [ ] Fix trame Refresh Rate with matplotlib plots (refresh on interaction not release of slider)
- [ ] Windows+linux/universal installer (.pkg/.exe)
- [ ] Implement regedit to run the application from weblinks
- [ ] Running application opens website in browser automatically
- [ ] Important/export propeller geometry via standardised file format(s)?