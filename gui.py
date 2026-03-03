from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify, matplotlib
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import geometry
import propmodel

# Server Setup
server = get_server(client_type = "vue2")
state, ctrl = server.state, server.controller
# Activate PyVista Plotter
plotter = pv.Plotter(off_screen=True)
# -----------------------------
# Solver Stub (replace with BEMT)
# -----------------------------


def generate_propeller_mesh(r, c, beta, N, airfoil_pts=20):
    """
    r: radial stations
    c: chord at stations
    beta: twist in radians
    N: number of blades
    """
    # 1. Create a simple airfoil profile (NACA 0012 approximation or flat plate)
    # x_local goes from -0.25 (leading edge) to 0.75 (trailing edge) if rotating about 1/4 chord
    s = np.linspace(0, 1, airfoil_pts)
    x_local = (s - 0.25) 
    y_local = 0.05 * 0.12 * (0.2969*np.sqrt(s) - 0.1260*s - 0.3516*s**2 + 0.2843*s**3 - 0.1015*s**4) # Thickness

    # 2. Build the structured grid for one blade
    # Grid shape: (num_radial_stations, airfoil_pts)
    rs, xs = np.meshgrid(r, x_local, indexing='ij')
    cs, ys = np.meshgrid(c, y_local, indexing='ij')
    bs, _  = np.meshgrid(beta, x_local, indexing='ij')

    def get_blade_coords(azimuth_offset):
        # Apply twist rotation and chord scaling
        # We rotate the local airfoil coordinates by beta
        x_rot = (xs * cs) * np.cos(bs) - (ys * cs) * np.sin(bs)
        z_rot = (xs * cs) * np.sin(bs) + (ys * cs) * np.cos(bs)
        
        # Transform to 3D Global Coordinates
        X = rs * np.cos(azimuth_offset) - x_rot * np.sin(azimuth_offset)
        Y = rs * np.sin(azimuth_offset) + x_rot * np.cos(azimuth_offset)
        Z = z_rot
        
        return np.stack([X, Y, Z], axis=-1)

    # 3. Create PyVista MultiBlock to hold all blades
    propeller = pv.MultiBlock()
    
    for i in range(int(N)):
        angle = 2 * np.pi * i / N
        coords = get_blade_coords(angle)
        # Reshape for PyVista: (points, 3)
        grid = pv.StructuredGrid(coords[:,:,0], coords[:,:,1], coords[:,:,2])
        propeller.append(grid)
        
    return propeller

def make_velocity_heightmap(params, V_ia, scale=0.05):
    """Creates a 3D surface where height = induced velocity."""
    # Ensure we are using numpy arrays for PyVista
    r = np.array(params.r).ravel()
    psi = np.array(params.psi).ravel()
    v_z = np.array(V_ia) # Shape (Nr, Npsi)

    # 1. Create a 2D meshgrid of polar coordinates
    # indexing='ij' ensures the grid matches your (Nr, Npsi) V_ia shape
    R, PSI = np.meshgrid(r, psi, indexing='ij')

    # 2. Convert to Cartesian X, Y
    X = R * np.cos(PSI)
    Y = R * np.sin(PSI)
    
    # 3. Use V_ia as the Z coordinate (scaled so it's not a skyscraper)
    Z = v_z * scale

    # 4. Create a StructuredGrid
    grid = pv.StructuredGrid(X, Y, Z)
    
    # 5. Add the velocity data to the grid for coloring
    grid["Induced Velocity [m/s]"] = v_z.ravel(order='F') 
    
    return grid

# -----------------------------
# 2. Logic to Prepare Data
# -----------------------------
def update_visuals():
    params, V_ia, T = propmodel.flatblade()
    
    # --- A. Generate Propeller Mesh ---
    r, c, beta, N = params.r, params.c, params.beta, params.N
    prop_mesh = generate_propeller_mesh(r, c, beta, N)
    
    # --- B. Generate Velocity Heightmap ---
    # Adjust scale based on your typical V_ia values
    vel_mesh = make_velocity_heightmap(params, V_ia, scale=0.01)

    # --- C. Update 3D Plotter ---
    plotter.clear()
    
    # Add the propeller (solid color)
    plotter.add_mesh(prop_mesh, color="grey", label="Propeller")
    
    # Add the velocity heightmap (colored by data)
    plotter.add_mesh(
        vel_mesh, 
        cmap="viridis", 
        scalars="Induced Velocity [m/s]", 
        opacity=0.8,
        show_scalar_bar=True
    )
    plotter.add_legend()
    plotter.render()
    if ctrl.view_update:
        ctrl.view_update()

# 3. GUI Layout
with SinglePageLayout(server) as layout:
    # Use a reactive state variable for the title
    layout.title.set_text("{{ header_title }}")
    state.header_title = "Propeller Design Dashboard"
    
    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            # VtkRemoteView now takes up the whole space
            view = vtk.VtkRemoteView(plotter.ren_win, style="width: 100%; height: 100%;")
            ctrl.view_update = view.update

    # Interaction Bar
    with layout.footer:
        vuetify.VSpacer()
        # You can add a slider here later for parameters
        vuetify.VBtn("Solve & Visualize", click=update_visuals, color="success", dark=True)

if __name__ == "__main__":
    # Initial run to populate the screen
    update_visuals()
    server.start()    