# fig4_error_vs_nodes: "Post-trained meshes: 2.0-2.3 % off the real test, automesher 7.8 %"

**Caption.** With the learned test conditions frozen, the mesh decides how far the simulation is off the real
test. gmsh's default mesh is 7.8 % off and the swarm's pre-trained mesh
4.6 %. Post-training on the real test moves the swarm's meshes to about 2 %: the most accurate
is 1.97 % off at 8,753 nodes (g1-abc_00000065_compression-a2; the same mesh in
g0-abc_00000065_compression-a2, g1-abc_00000065_compression-a1, g1-abc_00000065_compression-a2), and the best-scoring 2.30 % at 5,612 nodes
(g7-abc_00000065_compression-a1). The last generation's four meshes are all
2.2-2.3 % off at 5,650-5,981 nodes.

| reference mesh | nodes | off the real test |
|---|---|---|
| automesher (gmsh default) | 2,937 | 7.84 % |
| pre-trained swarm mesh | 6,430 | 4.57 % |
| uniform 4 mm | 9,506 | 1.99 % |
| uniform 2.5 mm | 24,194 | 1.20 % |
| uniform 2 mm | 41,186 | 1.08 % |
| post-trained, most accurate (g1-abc_00000065_compression-a2) | 8,753 | 1.97 % |
| post-trained, best lean (< 6k nodes, g8-abc_00000065_compression-a1) | 5,981 | 2.21 % |
| post-trained, best-scoring (g7-abc_00000065_compression-a1) | 5,612 | 2.30 % |

The most accurate post-trained mesh is 4.0x more accurate than the
automesher; the best-scoring one 3.4x at 1.9x its nodes.

**Data.** ace_runs/nl_ft/gen*/runs/*.json (checks.nonlinear: mean_rel_err, nodes);
results/real_data_finetune/baselines/baselines.json and */curve.csv.

**Caveats.**
- Uniform 4 mm (9,506 nodes, 1.99 %) is as accurate as the most accurate
  post-trained mesh with 8 % more nodes. The lean post-trained meshes are slightly less accurate (2.2-2.3 %) with
  37-41 % fewer nodes (5.6-6.0k vs 9.5k): the gain over uniform refinement is in nodes, not accuracy. The finer
  uniform meshes are more accurate (1.1-1.2 %) at 2.8-4.7x the nodes of the most accurate post-trained mesh.
- Below about 1 % the remaining difference is model error (material, conditions), not mesh error; the seating gap
  and compliance were fitted on this specimen.
- Several agents resubmitted the same mesh, so some dots overlap.
