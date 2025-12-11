## Propeller Geometry Tools and Functions

import jax.numpy as jnp
import numpy as np

def generate_le_te(r,beta,chord):
    # Calculate Chord Offset
    chord_offset = 0.2*max(chord)

    # Generate LE Points in xyz
    le_points = [r,]

    # Generate TE Points in xyz
    te_points = 

    return le_points, te_points