from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify, matplotlib
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import propmodel
from init import Params, Inputs

server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# --- Initial State ---
state.header_title = "Proptimiser: Blade Flow Visualisation"
state.N = 2
state.R = 1.0
state.chord_root = 0.2
state.chord_tip = 0.05
state.beta_root = 45
state.beta_tip = 10

# Activate PyVista Plotter
plotter = pv.Plotter(off_screen=True)

# -----------------------------------------------------------------------------
# Logic: Geometry and Plots
# -----------------------------------------------------------------------------


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

def get_distributions():
    """Calculates r and c based on UI sliders"""
    # Placeholder for twist: 45 deg at root to 10 deg at tip
    beta = np.linspace(state.beta_root, state.beta_tip, Inputs.n_r)
    # Linear chord distribution for preview
    c = np.linspace(state.chord_root, state.chord_tip, Inputs.n_r)
    return c, beta

def update_2d_preview():
    """Generates the Matplotlib figure with separate subplots for Chord and Twist"""
    c, beta = get_distributions()
    # Get radial distribution
    r, dr = propmodel.radial_stops(state.R, 0.05, Inputs.n_r) 

    # Create two stacked subplots sharing the same X-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)

    # --- Subplot 1: Chord Distribution ---
    ax1.plot(r, c, color='tab:blue', marker='o', markersize=4)
    ax1.set_ylabel("Chord (m)")
    ax1.set_ylim(0, 0.5)
    ax1.set_title("Blade Planform Design")
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Twist Distribution ---
    ax2.plot(r, beta, color='tab:red', marker='s', markersize=4)
    ax2.set_ylabel("Twist (deg)")
    ax2.set_xlabel("Radius (m)")
    ax2.set_ylim(0, 90)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    ctrl.view_update_plt(fig)
    plt.close(fig)


def update_visuals():
    """The 'Solve' button logic: Updates the 3D Plotter"""
    c, beta = get_distributions()

    # Update Parameters
    params, V_ia, T = propmodel.flatblade(state.R, c, beta, state.N)
    
    # Generate Mesh (using your existing function)
    prop_mesh = generate_propeller_mesh(params.r, params.c, params.beta, state.N)
    # Translate up a small offset to prevent intersection with plot
    prop_mesh.translate((0,0,0.5), inplace=True)

    # Generate Velocity Heightmap
    vel_mesh = make_velocity_heightmap(params, V_ia, scale=0.01)
    
    # Update Plotter
    plotter.clear()
    # Add Propeller
    plotter.add_mesh(prop_mesh, color="silver", show_edges=True, )
    # Add the velocity heightmap (colored by data)
    plotter.add_mesh(
        vel_mesh, 
        cmap="viridis", 
        scalars="Induced Velocity [m/s]", 
        opacity=0.8,
        show_scalar_bar=True,
        lighting=False
    )
    plotter.reset_camera()
    plotter.render()
    
    if ctrl.view_update:
        ctrl.view_update()
    
    # Also update the 2D plot to ensure consistency
    update_2d_preview()

# -----------------------------------------------------------------------------
# GUI Layout
# -----------------------------------------------------------------------------

with SinglePageLayout(server) as layout:
    layout.title.set_text("{{ header_title }}")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-2 fill-height"):
            with vuetify.VRow(classes="fill-height"):
                
                # --- LEFT COLUMN: DESIGN PLANE ---
                with vuetify.VCol(cols="4", classes="d-flex flex-column"):
                    with vuetify.VCard(classes="pa-4 mb-4", elevation=2):
                        vuetify.VCardTitle("Geometry Parameters")
                        
                        # Sliders
                        vuetify.VSlider(label="Blades (N)", v_model=("N", 3), min=2, max=8, step=1, thumb_label=True)
                        vuetify.VSlider(label="Radius (R)", v_model=("R", 1.0), min=0.1, max=5.0, step=0.1, thumb_label=True)
                        vuetify.VSlider(label="Chord Root", v_model=("chord_root", 0.2), min=0.01, max=0.5, step=0.01, thumb_label=True)
                        vuetify.VSlider(label="Chord Tip", v_model=("chord_tip", 0.05), min=0.01, max=0.5, step=0.01, thumb_label=True)
                        vuetify.VSlider(label="Beta Root", v_model=("beta_root", 45), min=0.01, max=90, step=0.01, thumb_label=True)
                        vuetify.VSlider(label="Beta Tip", v_model=("beta_tip", 10), min=0.01, max=90, step=0.01, thumb_label=True)



                        vuetify.VBtn("Update Preview", click=update_2d_preview, color="primary", block=True)

                    with vuetify.VCard(classes="pa-2 flex-grow-1", elevation=2):
                        # Matplotlib Widget
                        view_plt = matplotlib.Figure(style="width: 100%")
                        ctrl.view_update_plt = view_plt.update

                # --- RIGHT COLUMN: OUTPUT PLANE ---
                with vuetify.VCol(cols="8"):
                    with vuetify.VCard(classes="fill-height pa-0", elevation=4):
                        view = vtk.VtkLocalView(plotter.ren_win, style="width: 100%; height: 100%;")
                        ctrl.view_update = view.update

    # Interaction Bar
    with layout.footer:
        vuetify.VSpacer()
        vuetify.VBtn("Run Full 3D Solve", click=update_visuals, color="success", dark=True)

if __name__ == "__main__":
    update_visuals()
    server.start()