# FEAgent: parallel mesh systems → synthesis (a Sculptured team)

A [Sculptured](https://usesculptured.com) team that shows how FEAgent works. Many agents run in parallel, each on a
slightly different mesh system but under one contract. Each writes its own analysis. A verifier gates the evidence,
and a synthesis agent builds the consensus and compares it with the physical test. The team mirrors the mesh-system
study of 2026-11-01 on branch `nonlinear-setup` (`experiments/mesh_systems/`, `experiments/physical/CRUSH_RUNS.md`).

```
                              Study lead (contract)
   ┌──────────┬──────────┬──────────┬────┴─────┬──────────┬──────────┐
harness_auto harness_expert ace_sizing gold_nested netgen harness_hex harness_neck   (parallel)
   └──────────┴──────────┴──────────┴────┬─────┴──────────┴──────────┘
                              Verifier (gate)
                                    │
                               Synthesis (brief)
                 (every agent also reports to the Study lead)
```

| Agent | Does | Writes |
|---|---|---|
| Study lead (Opus) | the contract: parts, axes, groups `bottom` / `top` / `domain`, Alpha Fast Resin (E 1879.8 MPa, UTS 50.48 MPa, failure PEEQ 0.043), loads and F_full, a 40k-node budget, the neck crush recipe, the machine slots. Tracks status | `RUN/plan/` |
| 7 mesh-system agents | one system each (its `build.py`): meshes, gates (`report.json`), linear compression (feacore/skfem), the CalculiX crush (`crush.py --platen bonded --failure peeq_tension --compression-switch 0.5:5` on a buckling-imperfect mesh), videos, `analysis.json` (peak force, travel at peak, crack site and height, refinement map, caveats). Sculptured checks the analysis keys after each turn | `RUN/systems/<id>/` |
| Verifier | the gate: equilibrium, gates disclosed, node budget, recipe in `run.json`, the run reached its peak, numbers recomputed from the CSV | `RUN/verify/` |
| Synthesis (Opus) | median and spread per quantity, what is trustworthy and what is mesh-sensitive, outliers (harness_hex: 54k nodes, linear C3D8), the comparison with the printed clamp that broke at the arm neck, refinement maps, photo comparison, 4 x 2 grid video, the brief with a recommendation | `RUN/synthesis/` |

| system | what is different | 40k knob |
|---|---|---|
| harness_auto | zero-config `feagent automesh` | `--global` 3.05 mm |
| harness_expert | digest + hazards → local refinement, thin-wall layers | `--global` 12 mm |
| ace_sizing | global h + hotspot refinement, ACE loop | `--h` 3.5, `--node-cap` |
| gold_nested | one base mesh, uniform nested split | `--h0-frac` 0.324 |
| netgen | independent mesher (Netgen on OCC) | `--maxh` 2.10 mm |
| harness_hex | all-hex C3D8 for the crush (unmatched: 54k nodes) | `--thin-factor` 12.9 |
| harness_neck | harness_expert + refinement of the arms' neck wall | `--neck-size` 3.83 mm |

The study's numbers are in the prompts as the expected outputs. With the neck recipe, all seven crack at the arm necks
(height fraction 0.60-0.63, like the photo), at a peak of 5.79-6.00 kN (spread 3.6 %, tets only 1.3 %).

`RUN` is `runs/multisystem-<date>/` in the `nonlinear-setup` checkout the agents work in (git-ignored).
[`starter/`](starter/) holds the synthesis scripts (`refine_maps.py`, `neck_hf.py`, `make_figs.py`) and reference
figures from the study.

## Install and view

Needs a running Sculptured (port 4410), Claude Code signed in, a checkout of `nonlinear-setup` (default
`../FEAgent-setup`) and the harness on `PATH` (`uv tool install --editable <checkout>`).

```bash
FEAGENT_SPARK=<user>@<spark-host> uv run --no-sync python teams/feagent-multisystem/build_team.py --install \
    [--worktree ../FEAgent-setup] [--photo PHOTO.png]
```

Then open http://127.0.0.1:4410/p/feagent-parallel-mesh-systems-synthesis/board. **Start team** runs ten real Claude
Code agents on your plan (the lead and synthesis on Opus, the rest on Sonnet). The crushes take hours.

`feagent-multisystem.json` is generated with this machine's absolute paths and the Spark host, so it is git-ignored;
rerun `build_team.py` after changing a prompt, or on another machine.

## Rules every agent gets

The laptop GPU is never used (`CUDA_VISIBLE_DEVICES=""`, CPU rendering through llvmpipe). GPU work goes to the DGX
Spark only, over ssh, under its `~/AGENTS.md`: no sudo, one large GPU job at a time, memory caps, never the
`ridgeline-isaac` container, no systemd units or cron. Python runs as `uv run --no-sync`, with no `uv sync` in the
shared venv. Every run goes into a new run-id dir and no output is overwritten. Nothing committed holds absolute paths
or the Spark's address.
