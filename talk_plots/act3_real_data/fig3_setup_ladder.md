# fig3_setup_ladder: "From 33 % to 2.0 % off the real test: test conditions, then post-training"

**Caption.** Each step on top of the previous one. Before post-training the simulation is 33 % off the
real test. Learning the test conditions from the experiment (sticking platens, a seating gap, load-train
compliance; then frozen) brings gmsh's default mesh to 7.8 %. The swarm's pre-trained mesh (trained on a
different load case) gets 4.6 %, and the mesh from post-training on the real test 2.0 %.

| step | mesh | nodes | off the real test |
|---|---|---|---|
| before post-training (frictionless platens, no seating/compliance) | automesher | 2,937 | 32.65 % |
| + learned test conditions | automesher (curved P2) | 2,937 | 7.84 % |
| + pre-trained swarm mesh | swarm bc0 mesh | 6,430 | 4.57 % |
| + post-trained swarm mesh (most accurate, g1-abc_00000065_compression-a2) | post-trained | 8,753 | 1.97 % |

**Data.** talk_plots/act3_real_data/videos/before_run/curve.csv; results/real_data_finetune/baselines/baselines.json (auto, swarm_bc0);
ace_runs/nl_ft/gen1/work/abc_00000065_compression-a2/nonlinear/curve.csv. Every error is recomputed from its curve and matches the recorded value.

**Caveats.**
- The automesher's 7.8 % uses its curved P2 mesh with only the midside nodes CalculiX rejects straightened. An
  earlier, higher automesher figure straightened every midside node (facetting the curved faces) and was wrong.
- The seating gap and compliance were fitted once on the finest uniform mesh against this same test, so the last
  steps are calibrated on the specimen they are scored against; there is one specimen.
- The last bar is the most accurate post-trained mesh, which uses more nodes than the others; the best-scoring
  one is 2.3 % off at
  5,612 nodes (fig4).
- Uniform 4 mm (9,506 nodes) is 2.0 % off: about as accurate as the most
  accurate post-trained mesh with 8 % more nodes.
