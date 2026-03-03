## Propeller Geometry Tools and Functions

import jax.numpy as jnp
import numpy as np


def params_to_le_te(r, beta, chord):
    # Calculate Chord Offset
    chord_offset = 0.2 * np.max(chord)

    # 1. Calculate the components
    x = r.ravel()
    
    y_le = ((chord_offset + chord) * np.sin(beta)).ravel()
    z_le = ((chord_offset + chord) * np.cos(beta)).ravel()
    
    y_te = ((-chord_offset + chord) * np.sin(beta)).ravel()
    z_te = ((-chord_offset + chord) * np.cos(beta)).ravel()

    # 2. Stack them into (N, 3) arrays
    # We cast to np.array to ensure PyVista can read the memory
    le_points = np.column_stack([x, y_le, z_le])
    te_points = np.column_stack([x, y_te, z_te])

    return le_points, te_points
