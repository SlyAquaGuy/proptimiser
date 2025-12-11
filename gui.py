from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify, matplotlib
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np

server = get_server(client_type = "vue2")
state, ctrl = server.state, server.controller

# -----------------------------
# Solver Stub (replace with BEMT)
# -----------------------------
def solve_prop(rpm, pitch):
    r = np.linspace(0.1, 1.0, 30)
    loss = 0.1 * r * rpm / 5000
    return r, loss

def make_prop_mesh():
    return pv.Cone()

mesh = make_prop_mesh()

plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(mesh)
plotter.view_isometric()
vtk_view = plotter.ren_win

# -----------------------------
# Matplotlib Plot
# -----------------------------
fig, ax = plt.subplots()
r, loss = solve_prop(4000, 20)
(line,) = ax.plot(r, loss)

# -----------------------------
# UI
# -----------------------------
with SinglePageLayout(server) as layout:
    layout.title.set_text("Proptimiser Web Viewer")

    with layout.content:
        with vuetify.VContainer(fluid=True):
            with vuetify.VRow():
                with vuetify.VCol(cols=6):
                    vtk.VtkLocalView(vtk_view)

                with vuetify.VCol(cols=6):
                    matplotlib.Figure(fig)

            vuetify.VSlider(v_model=("rpm", 4000), min=1000, max=10000)
            vuetify.VSlider(v_model=("pitch", 20), min=5, max=40)

# -----------------------------
# Live Update Hook
# -----------------------------
@state.change("rpm", "pitch")
def update_solver(rpm, pitch, **kwargs):
    r, loss = solve_prop(rpm, pitch)
    line.set_ydata(loss)
    ctrl.update_figure()

# -----------------------------
server.start()