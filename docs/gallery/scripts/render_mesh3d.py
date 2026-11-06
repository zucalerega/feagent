"""Plain 3D picture of a gmsh mesh (surface + element edges), CPU off-screen.

usage: render_mesh3d.py OUT_DIR NAME=MESH.msh [...]
"""
import sys, numpy as np, pyvista as pv
pv.OFF_SCREEN = True
SOLID = [10, 24, 12, 25, 29]
for arg in sys.argv[2:]:
    name, path = arg.split("=", 1)
    g = pv.read(path)
    g = g.extract_cells(np.where(np.isin(g.celltypes, SOLID))[0]).linear_copy()
    surf = g.extract_surface(algorithm="dataset_surface")
    pl = pv.Plotter(off_screen=True, window_size=(2400, 2000))
    pl.set_background("white")
    pl.add_mesh(surf, color="#d9dde3", show_edges=True, edge_color="#2b2f36", line_width=0.6,
                lighting=True, ambient=0.25, diffuse=0.75, specular=0.1)
    pl.camera_position = "xy"; pl.camera.Azimuth(-35); pl.camera.Elevation(25); pl.reset_camera(); pl.camera.zoom(0.95)
    pl.enable_anti_aliasing("ssaa")
    pl.screenshot(f"{sys.argv[1]}/{name}.png"); pl.close()
    print(name, g.n_points, g.n_cells)
