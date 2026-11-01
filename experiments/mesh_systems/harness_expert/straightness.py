"""Report how many tet10 elements have a mid-edge node off the straight edge midpoint (curved edges), and by how much
relative to the edge length. usage: straightness.py mesh.msh"""
import sys
import gmsh
import numpy as np

gmsh.initialize(["-v", "0"]); gmsh.open(sys.argv[1])
tags, nodes, _ = gmsh.model.mesh.getNodes()
xyz = dict(zip(tags.tolist(), nodes.reshape(-1, 3)))
_, et, en = gmsh.model.mesh.getElements(3)
c = np.array([xyz[i] for i in en[0]]).reshape(-1, 10, 3)
# gmsh tet10 order: 0-3 corners, then edges (0,1)(1,2)(2,0)(3,0)(3,2)(3,1)
edges = [(0, 1), (1, 2), (2, 0), (3, 0), (3, 2), (3, 1)]
en0 = en[0].reshape(-1, 10)
dim = {int(t): gmsh.model.mesh.getNode(int(t))[2] for t in np.unique(en0[:, 4:])}
interior = np.array([[dim[int(t)] == 3 for t in row[4:]] for row in en0])
d6 = np.stack([np.linalg.norm(c[:, 4 + k] - 0.5 * (c[:, a] + c[:, b]), axis=1) / np.linalg.norm(c[:, a] - c[:, b], axis=1)
                for k, (a, b) in enumerate(edges)], axis=1)
dev = (d6 * interior).max(axis=1)  # interior-edge nodes only (boundary edges may curve with the CAD)
print(f"{len(dev)} tet10: elements with a curved INTERIOR edge {np.sum(dev > 1e-6)} ({100*np.mean(dev > 1e-6):.2f}%), max dev/edge {dev.max():.3f}")
