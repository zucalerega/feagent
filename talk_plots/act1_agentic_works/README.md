# Act 1: agentic meshing works, and training works

Regenerate the data and figures:

```
.venv/bin/python talk_plots/act1_agentic_works/automesher_sweep.py   # ~1 min, solver only, no API
.venv/bin/python talk_plots/act1_agentic_works/make_figures.py
```

## What "about 4x the automesher" really is (Cook's membrane, deep1)

| statement | number | source |
|---|---|---|
| best agent's base score / the automesher's 0.25 | **4.50x** (1.126) | g10-cooks_membrane-a7 |
| last-generation mean score / 0.25 | 4.38x (1.095) | deep1 gen 10, 4 agents |
| accuracy gain g = e_auto / e, best run | **4.76x** (1.07% vs 5.11%), at **2.11x the DOFs** (710 vs 336) | checks.gain |
| accuracy gain, last-generation mean | 4.68x, at 2.14x the DOFs | |
| error at the **same DOFs** vs the automesher refined uniformly | **3.4x lower** (1.07% vs 3.69% at 710 DOFs) | automesher_sweep.json |
| DOFs the uniformly refined automesher needs for the **same error** | **10.8x more** (7,690 vs 710) | automesher_sweep.json |

So "4x" is true as the score ratio (4.5x) and as the error ratio (4.8x). Both compare against the *unrefined* automesher, while the agent spends 2.1x its DOFs. The cost-fair statements are "3.4x lower error at equal DOFs" and "the same accuracy for 1/11 of the DOFs".

The same numbers on the other problems (best run per problem):

| problem | score ratio | lower error at equal DOFs | fewer DOFs at equal error |
|---|---|---|---|
| LE1 | 4.25x | 2.3x | 2.7x |
| LE11 axisym. | 3.52x | 2.4x | 5.0x |
| LE11 3D | 1.57x | 1.06x | 1.11x (no real gain) |

## Figures

| file | takeaway title |
|---|---|
| fig1_learning_curves | Training works: the mean agent climbs 1.3x → 4.4x the automesher (Cook) |
| fig1b_learning_curves_all_runs | The deep loop learns steadily; the basic loop plateaus or drifts (backup) |
| fig2_error_vs_dofs | 2D: the best agents beat the automesher's whole cost curve; 3D: a tie |
| fig3_headline_scores | The best agent scores 1.6-4.5x the automesher on every benchmark |
| fig3b_equal_cost_equal_accuracy | Matching the best agent costs the automesher up to 11x the DOFs |
| fig4_mesh_before_after | Same part, same solver: the trained agent grades the mesh into the corners |

Each `<stem>.md` has the caption, the key numbers, the sources and the caveats.

Data files:
- `act1_runs.csv`: all 442 agent runs used.
- `act1_numbers.json`: every quoted number, plus what each figure plots.
- `automesher_sweep.json`: the uniform-refinement baseline.

## Runs used

These are the field-mode runs, scored against the gmsh automesher:

| run | setup | cost |
|---|---|---|
| ace_runs/deep1 | Cook + LE11 axisym., 8 agents (4 per task), Opus teacher, 11 generations | $46.37 |
| ace_runs/cook_auto | Cook, 6 agents, basic loop, 20 generations | $2.09 |
| ace_runs/split1 | Cook + LE1, 3 agents per task, 20 generations | $3.98 |
| ace_runs/le11 | LE11 3D + axisym., 3 agents per task, 19 generations | $5.21 |

Not used:
- `train1` and `cook_field`: they use the older point-mode score (relative to the cheapest in-tolerance run), so they carry no automesher comparison.
- Lamé: never run in field mode. It appears only in the point-mode smoke1 run.

## Caveats and surprises

- **Sustained learning shows only with the deep loop (Opus teacher).**
  - In the basic loop, Cook improves early (mean 0.449 -> 0.700 in cook_auto, 0.392 -> 0.711 in split1), then plateaus at about 0.71 (stuck on one recipe).
  - LE1 has no trend: the last generation's mean is 0.487, below gen 0's 0.782.
  - LE11 3D is flat.
  - LE11 axisym. keeps falling back into a temperature-field setup bug (33 of 57 runs).
- deep1's LE11-axisym. jump at gen 0 -> 1 is that setup bug being fixed (all 4 gen-0 agents had it), not better meshing. The mesh learning there is gen 1 -> 10: best agent 0.706 -> 0.880, mean 0.678 -> 0.845. On Cook, the gain from gen 0 to gen 10 comes entirely from meshing.
- Only one run per configuration; no seeds or confidence intervals.
- The uniform-refinement baseline was computed for this talk, not by the runs. Its factor-1 point reproduces the cached automesher exactly on all 4 problems.
- LE1's highest-gain run (g = 6.40) hits the scorer's error floor of 0.005: its true error is 0.36%, but the gain is computed as 0.032 / 0.005.
