"""Render a USD stage offscreen with PyVista/VTK; report render time. usage: render_usd.py stage.usdc out.png [reps]"""
import sys, time
import numpy as np
import pyvista as pv
from pxr import Usd, UsdGeom

pv.OFF_SCREEN = True


def stage_meshes(path):
    stage = Usd.Stage.Open(path)
    out = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0)).T
        p = np.array(m.GetPointsAttr().Get(), float)
        p = p @ xf[:3, :3].T + xf[:3, 3]
        t = np.array(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
        c = np.array(UsdGeom.PrimvarsAPI(m).GetPrimvar("displayColor").Get())
        pd = pv.PolyData.from_regular_faces(p, t)
        pd.point_data["rgb"] = (c * 255).astype(np.uint8)
        out.append(pd)
    return out


def render(meshes, png, reps=20, size=(1600, 900)):
    pl = pv.Plotter(off_screen=True, window_size=size)
    for m in meshes:
        pl.add_mesh(m, scalars="rgb", rgb=True, smooth_shading=True)
    pl.camera_position = "iso"
    pl.set_background("#1b1d22")
    pl.show(auto_close=False)
    t0 = time.time()
    for _ in range(reps):
        pl.render()
    pl.screenshot(png)  # forces GPU sync
    dt = (time.time() - t0) / (reps + 1)
    gl = pl.render_window.ReportCapabilities().split("\n")
    info = [l for l in gl if "OpenGL" in l and ("renderer" in l or "vendor" in l)]
    ntri = sum(m.n_faces_strict for m in meshes)
    print(f"win={type(pl.render_window).__name__} tris={ntri} {dt*1000:.1f} ms/frame {1/dt:.1f} fps {info}")
    pl.close()


if __name__ == "__main__":
    render(stage_meshes(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 20)
