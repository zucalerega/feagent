# Gallery: mesh systems, crush to snap, neck break, multi-agent results

Stills and figures from the multi-system mesh study on the two physical-test parts (abc_00000003 tube, abc_00000065
clamp), the CalculiX crush-to-snap runs, the clamp neck-break study and the wingbox twin. Videos are not in git;
every still here is a frame of one. Numbers: `experiments/mesh_systems/RESULTS.md`, `experiments/physical/CRUSH_RUNS.md`.

## Video: abc_00000065 crushed on six mesh systems (Omniverse RTX)

![crush grid](videos/all_systems_abc_00000065_crush_rtx.gif)

[MP4](videos/all_systems_abc_00000065_crush_rtx.mp4) (16 s).

## Multi-agent results

- `fig_ma1_swarm_vs_single`: the team's ACE swarm runs (main): mean of each round's best mesh vs team size, against one agent given the same solves, turns and time. One run per configuration; on the tube the single agent ties from round 3.
- `fig_ma2_crosscheck`: six independent mesh agents, each result against the median of the six. Stiffness agrees within 1.3 %; crush force spreads 20-31 %, single-point peak stress 42-88 %.
- `fig_ma3_neck_recipe`: abc_00000065 with the neck-study recipe (tension-only erosion, buckling imperfection, bonded tips) on 7 meshes at about 40k nodes: snap load 5.79-6.00 kN (1.3 % across the tets), crack at the arm necks (height 0.60-0.63) instead of the tips (0.89).
- `mesh_study_comparison`: the comparison table of the six systems.

| ![fig_ma1_swarm_vs_single.png](multiagent/fig_ma1_swarm_vs_single.png) | ![fig_ma2_crosscheck.png](multiagent/fig_ma2_crosscheck.png) |
|---|---|
| `fig_ma1_swarm_vs_single` | `fig_ma2_crosscheck` |
| ![fig_ma3_neck_recipe.png](multiagent/fig_ma3_neck_recipe.png) | ![mesh_study_comparison.png](multiagent/mesh_study_comparison.png) |
| `fig_ma3_neck_recipe` | `mesh_study_comparison` |

## abc_00000065: the full 3D meshes

The equal-budget (about 40k nodes; hex 54k) nonlinear meshes, surface edges only.

| ![gold_nested.jpg](meshes_3d_p65/gold_nested.jpg) | ![harness_auto.jpg](meshes_3d_p65/harness_auto.jpg) | ![harness_expert.jpg](meshes_3d_p65/harness_expert.jpg) |
|---|---|---|
| `gold_nested` | `harness_auto` | `harness_expert` |
| ![harness_hex.jpg](meshes_3d_p65/harness_hex.jpg) | ![netgen.jpg](meshes_3d_p65/netgen.jpg) |
| `harness_hex` | `netgen` |

## abc_00000065: where each system refines

Three mesh sets (matched 40k, the study's nonlinear meshes, the linear meshes). `*_size_maps`: element size on the mid-thickness cut; `*_vs_consensus`: size against the median of the tet systems at the same point (blue = finer, red = coarser; hex left out); `*_region_density`: elements per volume in neck, tip pocket, base and rest. At equal budget every system spends its elements on the tip pockets (2-2.8x the part average) and none on the neck (0.3-0.6x).

| ![matched_40k_region_density.png](refinement_p65/matched_40k_region_density.png) | ![matched_40k_size_maps.png](refinement_p65/matched_40k_size_maps.png) | ![matched_40k_vs_consensus.png](refinement_p65/matched_40k_vs_consensus.png) |
|---|---|---|
| `matched_40k_region_density` | `matched_40k_size_maps` | `matched_40k_vs_consensus` |
| ![study_linear_region_density.png](refinement_p65/study_linear_region_density.png) | ![study_linear_size_maps.png](refinement_p65/study_linear_size_maps.png) | ![study_linear_vs_consensus.png](refinement_p65/study_linear_vs_consensus.png) |
| `study_linear_region_density` | `study_linear_size_maps` | `study_linear_vs_consensus` |
| ![study_nonlinear_region_density.png](refinement_p65/study_nonlinear_region_density.png) | ![study_nonlinear_size_maps.png](refinement_p65/study_nonlinear_size_maps.png) | ![study_nonlinear_vs_consensus.png](refinement_p65/study_nonlinear_vs_consensus.png) |
| `study_nonlinear_region_density` | `study_nonlinear_size_maps` | `study_nonlinear_vs_consensus` |

## abc_00000065: neck break vs the printed part

The user's crushed Alpha Fast clamp next to the simulation (harness_neck + 0.1 mm imperfection, bonded tips, tension-only erosion, 4.84 mm), and the montage of the matched 40k meshes with a zoom on the right arm.

| ![n40k_mesh_montage_p65.jpg](neck_break/n40k_mesh_montage_p65.jpg) | ![photo_vs_simulation_p65.jpg](neck_break/photo_vs_simulation_p65.jpg) |
|---|---|
| `n40k_mesh_montage_p65` | `photo_vs_simulation_p65` |

## Crush to snap (CalculiX J2 + NLGEOM + erosion)

A late frame (72 % of the clip) of every crush video: `<system>_<part>_crush` in the original 3-panel look, `_crush_rtx` the Omniverse (Isaac Sim RTX, DGX Spark) render, `all_systems_*` the 3x2 grids, `*_snap_compare` the side-by-side, and the neck-break run.

| ![abc_00000003_snap_compare.jpg](crush/abc_00000003_snap_compare.jpg) | ![abc_00000065_snap_compare.jpg](crush/abc_00000065_snap_compare.jpg) | ![ace_sizing_abc_00000003_crush.jpg](crush/ace_sizing_abc_00000003_crush.jpg) |
|---|---|---|
| `abc_00000003_snap_compare` | `abc_00000065_snap_compare` | `ace_sizing_abc_00000003_crush` |
| ![ace_sizing_abc_00000003_crush_rtx.jpg](crush/ace_sizing_abc_00000003_crush_rtx.jpg) | ![ace_sizing_abc_00000065_crush.jpg](crush/ace_sizing_abc_00000065_crush.jpg) | ![ace_sizing_abc_00000065_crush_rtx.jpg](crush/ace_sizing_abc_00000065_crush_rtx.jpg) |
| `ace_sizing_abc_00000003_crush_rtx` | `ace_sizing_abc_00000065_crush` | `ace_sizing_abc_00000065_crush_rtx` |
| ![all_systems_abc_00000003_crush.jpg](crush/all_systems_abc_00000003_crush.jpg) | ![all_systems_abc_00000003_crush_rtx.jpg](crush/all_systems_abc_00000003_crush_rtx.jpg) | ![all_systems_abc_00000065_crush.jpg](crush/all_systems_abc_00000065_crush.jpg) |
| `all_systems_abc_00000003_crush` | `all_systems_abc_00000003_crush_rtx` | `all_systems_abc_00000065_crush` |
| ![all_systems_abc_00000065_crush_rtx.jpg](crush/all_systems_abc_00000065_crush_rtx.jpg) | ![gold_nested_abc_00000003_crush.jpg](crush/gold_nested_abc_00000003_crush.jpg) | ![gold_nested_abc_00000003_crush_rtx.jpg](crush/gold_nested_abc_00000003_crush_rtx.jpg) |
| `all_systems_abc_00000065_crush_rtx` | `gold_nested_abc_00000003_crush` | `gold_nested_abc_00000003_crush_rtx` |
| ![gold_nested_abc_00000065_crush.jpg](crush/gold_nested_abc_00000065_crush.jpg) | ![gold_nested_abc_00000065_crush_rtx.jpg](crush/gold_nested_abc_00000065_crush_rtx.jpg) | ![harness_auto_abc_00000003_crush.jpg](crush/harness_auto_abc_00000003_crush.jpg) |
| `gold_nested_abc_00000065_crush` | `gold_nested_abc_00000065_crush_rtx` | `harness_auto_abc_00000003_crush` |
| ![harness_auto_abc_00000003_crush_rtx.jpg](crush/harness_auto_abc_00000003_crush_rtx.jpg) | ![harness_auto_abc_00000065_crush.jpg](crush/harness_auto_abc_00000065_crush.jpg) | ![harness_auto_abc_00000065_crush_rtx.jpg](crush/harness_auto_abc_00000065_crush_rtx.jpg) |
| `harness_auto_abc_00000003_crush_rtx` | `harness_auto_abc_00000065_crush` | `harness_auto_abc_00000065_crush_rtx` |
| ![harness_expert_abc_00000003_crush.jpg](crush/harness_expert_abc_00000003_crush.jpg) | ![harness_expert_abc_00000003_crush_rtx.jpg](crush/harness_expert_abc_00000003_crush_rtx.jpg) | ![harness_expert_abc_00000065_crush.jpg](crush/harness_expert_abc_00000065_crush.jpg) |
| `harness_expert_abc_00000003_crush` | `harness_expert_abc_00000003_crush_rtx` | `harness_expert_abc_00000065_crush` |
| ![harness_expert_abc_00000065_crush_rtx.jpg](crush/harness_expert_abc_00000065_crush_rtx.jpg) | ![harness_hex_abc_00000003_crush.jpg](crush/harness_hex_abc_00000003_crush.jpg) | ![harness_hex_abc_00000003_crush_rtx.jpg](crush/harness_hex_abc_00000003_crush_rtx.jpg) |
| `harness_expert_abc_00000065_crush_rtx` | `harness_hex_abc_00000003_crush` | `harness_hex_abc_00000003_crush_rtx` |
| ![harness_hex_abc_00000065_crush.jpg](crush/harness_hex_abc_00000065_crush.jpg) | ![harness_hex_abc_00000065_crush_rtx.jpg](crush/harness_hex_abc_00000065_crush_rtx.jpg) | ![harness_neck_imp_abc_00000065_neckbreak_crush.jpg](crush/harness_neck_imp_abc_00000065_neckbreak_crush.jpg) |
| `harness_hex_abc_00000065_crush` | `harness_hex_abc_00000065_crush_rtx` | `harness_neck_imp_abc_00000065_neckbreak_crush` |
| ![netgen_abc_00000003_crush.jpg](crush/netgen_abc_00000003_crush.jpg) | ![netgen_abc_00000003_crush_rtx.jpg](crush/netgen_abc_00000003_crush_rtx.jpg) | ![netgen_abc_00000065_crush.jpg](crush/netgen_abc_00000065_crush.jpg) |
| `netgen_abc_00000003_crush` | `netgen_abc_00000003_crush_rtx` | `netgen_abc_00000065_crush` |
| ![netgen_abc_00000065_crush_rtx.jpg](crush/netgen_abc_00000065_crush_rtx.jpg) |
| `netgen_abc_00000065_crush_rtx` |

## Linear compression ramp (feacore/skfem)

A frame at 70 % of each linear-ramp video (0 to F_full, outside / cut von Mises / cut displacement).

| ![ace_sizing_abc_00000003_linear.jpg](linear/ace_sizing_abc_00000003_linear.jpg) | ![ace_sizing_abc_00000065_linear.jpg](linear/ace_sizing_abc_00000065_linear.jpg) | ![all_systems_abc_00000003_linear.jpg](linear/all_systems_abc_00000003_linear.jpg) |
|---|---|---|
| `ace_sizing_abc_00000003_linear` | `ace_sizing_abc_00000065_linear` | `all_systems_abc_00000003_linear` |
| ![all_systems_abc_00000065_linear.jpg](linear/all_systems_abc_00000065_linear.jpg) | ![gold_nested_abc_00000003_linear.jpg](linear/gold_nested_abc_00000003_linear.jpg) | ![gold_nested_abc_00000065_linear.jpg](linear/gold_nested_abc_00000065_linear.jpg) |
| `all_systems_abc_00000065_linear` | `gold_nested_abc_00000003_linear` | `gold_nested_abc_00000065_linear` |
| ![harness_auto_abc_00000003_linear.jpg](linear/harness_auto_abc_00000003_linear.jpg) | ![harness_auto_abc_00000065_linear.jpg](linear/harness_auto_abc_00000065_linear.jpg) | ![harness_expert_abc_00000003_linear.jpg](linear/harness_expert_abc_00000003_linear.jpg) |
| `harness_auto_abc_00000003_linear` | `harness_auto_abc_00000065_linear` | `harness_expert_abc_00000003_linear` |
| ![harness_expert_abc_00000065_linear.jpg](linear/harness_expert_abc_00000065_linear.jpg) | ![harness_hex_abc_00000003_linear.jpg](linear/harness_hex_abc_00000003_linear.jpg) | ![harness_hex_abc_00000065_linear.jpg](linear/harness_hex_abc_00000065_linear.jpg) |
| `harness_expert_abc_00000065_linear` | `harness_hex_abc_00000003_linear` | `harness_hex_abc_00000065_linear` |
| ![netgen_abc_00000003_linear.jpg](linear/netgen_abc_00000003_linear.jpg) | ![netgen_abc_00000065_linear.jpg](linear/netgen_abc_00000065_linear.jpg) |
| `netgen_abc_00000003_linear` | `netgen_abc_00000065_linear` |

## Wingbox virtual twin (Isaac Sim RTX on the DGX Spark)

Stills of the 43 s wingbox twin video: 1.15M DOFs solved with cuDSS on the GB10 in 24 s.

| ![wingbox_twin_t01.5s.jpg](wingbox_rtx/wingbox_twin_t01.5s.jpg) | ![wingbox_twin_t10.8s.jpg](wingbox_rtx/wingbox_twin_t10.8s.jpg) | ![wingbox_twin_t12.5s.jpg](wingbox_rtx/wingbox_twin_t12.5s.jpg) | ![wingbox_twin_t16.0s.jpg](wingbox_rtx/wingbox_twin_t16.0s.jpg) |
|---|---|---|---|
| `wingbox_twin_t01.5s` | `wingbox_twin_t10.8s` | `wingbox_twin_t12.5s` | `wingbox_twin_t16.0s` |
| ![wingbox_twin_t19.0s.jpg](wingbox_rtx/wingbox_twin_t19.0s.jpg) | ![wingbox_twin_t25.0s.jpg](wingbox_rtx/wingbox_twin_t25.0s.jpg) | ![wingbox_twin_t29.0s.jpg](wingbox_rtx/wingbox_twin_t29.0s.jpg) | ![wingbox_twin_t42.0s.jpg](wingbox_rtx/wingbox_twin_t42.0s.jpg) |
| `wingbox_twin_t19.0s` | `wingbox_twin_t25.0s` | `wingbox_twin_t29.0s` | `wingbox_twin_t42.0s` |

## Mesh quality and face groups, every system x part x purpose

`feagent render` views of all 24 meshes of the study: `_quality` (worst elements marked where visible) and `_groups` (bottom / top / domain), plus extra views.

| ![ace_sizing_abc_00000003_linear_groups.jpg](mesh_quality/ace_sizing_abc_00000003_linear_groups.jpg) | ![ace_sizing_abc_00000003_linear_quality.jpg](mesh_quality/ace_sizing_abc_00000003_linear_quality.jpg) | ![ace_sizing_abc_00000003_nonlinear_groups.jpg](mesh_quality/ace_sizing_abc_00000003_nonlinear_groups.jpg) | ![ace_sizing_abc_00000003_nonlinear_quality.jpg](mesh_quality/ace_sizing_abc_00000003_nonlinear_quality.jpg) |
|---|---|---|---|
| `ace_sizing_abc_00000003_linear_groups` | `ace_sizing_abc_00000003_linear_quality` | `ace_sizing_abc_00000003_nonlinear_groups` | `ace_sizing_abc_00000003_nonlinear_quality` |
| ![ace_sizing_abc_00000065_linear_groups.jpg](mesh_quality/ace_sizing_abc_00000065_linear_groups.jpg) | ![ace_sizing_abc_00000065_linear_quality.jpg](mesh_quality/ace_sizing_abc_00000065_linear_quality.jpg) | ![ace_sizing_abc_00000065_nonlinear_groups.jpg](mesh_quality/ace_sizing_abc_00000065_nonlinear_groups.jpg) | ![ace_sizing_abc_00000065_nonlinear_quality.jpg](mesh_quality/ace_sizing_abc_00000065_nonlinear_quality.jpg) |
| `ace_sizing_abc_00000065_linear_groups` | `ace_sizing_abc_00000065_linear_quality` | `ace_sizing_abc_00000065_nonlinear_groups` | `ace_sizing_abc_00000065_nonlinear_quality` |
| ![gold_nested_abc_00000003_linear_groups.jpg](mesh_quality/gold_nested_abc_00000003_linear_groups.jpg) | ![gold_nested_abc_00000003_linear_levels_L0_groups.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L0_groups.jpg) | ![gold_nested_abc_00000003_linear_levels_L0_quality.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L0_quality.jpg) | ![gold_nested_abc_00000003_linear_levels_L1_groups.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L1_groups.jpg) |
| `gold_nested_abc_00000003_linear_groups` | `gold_nested_abc_00000003_linear_levels_L0_groups` | `gold_nested_abc_00000003_linear_levels_L0_quality` | `gold_nested_abc_00000003_linear_levels_L1_groups` |
| ![gold_nested_abc_00000003_linear_levels_L1_quality.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L1_quality.jpg) | ![gold_nested_abc_00000003_linear_levels_L2_groups.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L2_groups.jpg) | ![gold_nested_abc_00000003_linear_levels_L2_quality.jpg](mesh_quality/gold_nested_abc_00000003_linear_levels_L2_quality.jpg) | ![gold_nested_abc_00000003_linear_quality.jpg](mesh_quality/gold_nested_abc_00000003_linear_quality.jpg) |
| `gold_nested_abc_00000003_linear_levels_L1_quality` | `gold_nested_abc_00000003_linear_levels_L2_groups` | `gold_nested_abc_00000003_linear_levels_L2_quality` | `gold_nested_abc_00000003_linear_quality` |
| ![gold_nested_abc_00000003_nonlinear_groups.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_groups.jpg) | ![gold_nested_abc_00000003_nonlinear_levels_L0_groups.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_levels_L0_groups.jpg) | ![gold_nested_abc_00000003_nonlinear_levels_L0_quality.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_levels_L0_quality.jpg) | ![gold_nested_abc_00000003_nonlinear_levels_L1_groups.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_levels_L1_groups.jpg) |
| `gold_nested_abc_00000003_nonlinear_groups` | `gold_nested_abc_00000003_nonlinear_levels_L0_groups` | `gold_nested_abc_00000003_nonlinear_levels_L0_quality` | `gold_nested_abc_00000003_nonlinear_levels_L1_groups` |
| ![gold_nested_abc_00000003_nonlinear_levels_L1_quality.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_levels_L1_quality.jpg) | ![gold_nested_abc_00000003_nonlinear_quality.jpg](mesh_quality/gold_nested_abc_00000003_nonlinear_quality.jpg) | ![gold_nested_abc_00000065_linear_groups.jpg](mesh_quality/gold_nested_abc_00000065_linear_groups.jpg) | ![gold_nested_abc_00000065_linear_levels_L0_groups.jpg](mesh_quality/gold_nested_abc_00000065_linear_levels_L0_groups.jpg) |
| `gold_nested_abc_00000003_nonlinear_levels_L1_quality` | `gold_nested_abc_00000003_nonlinear_quality` | `gold_nested_abc_00000065_linear_groups` | `gold_nested_abc_00000065_linear_levels_L0_groups` |
| ![gold_nested_abc_00000065_linear_levels_L0_quality.jpg](mesh_quality/gold_nested_abc_00000065_linear_levels_L0_quality.jpg) | ![gold_nested_abc_00000065_linear_levels_L1_groups.jpg](mesh_quality/gold_nested_abc_00000065_linear_levels_L1_groups.jpg) | ![gold_nested_abc_00000065_linear_levels_L1_quality.jpg](mesh_quality/gold_nested_abc_00000065_linear_levels_L1_quality.jpg) | ![gold_nested_abc_00000065_linear_quality.jpg](mesh_quality/gold_nested_abc_00000065_linear_quality.jpg) |
| `gold_nested_abc_00000065_linear_levels_L0_quality` | `gold_nested_abc_00000065_linear_levels_L1_groups` | `gold_nested_abc_00000065_linear_levels_L1_quality` | `gold_nested_abc_00000065_linear_quality` |
| ![gold_nested_abc_00000065_nonlinear_groups.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_groups.jpg) | ![gold_nested_abc_00000065_nonlinear_levels_L0_groups.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_levels_L0_groups.jpg) | ![gold_nested_abc_00000065_nonlinear_levels_L0_quality.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_levels_L0_quality.jpg) | ![gold_nested_abc_00000065_nonlinear_levels_L1_groups.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_levels_L1_groups.jpg) |
| `gold_nested_abc_00000065_nonlinear_groups` | `gold_nested_abc_00000065_nonlinear_levels_L0_groups` | `gold_nested_abc_00000065_nonlinear_levels_L0_quality` | `gold_nested_abc_00000065_nonlinear_levels_L1_groups` |
| ![gold_nested_abc_00000065_nonlinear_levels_L1_quality.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_levels_L1_quality.jpg) | ![gold_nested_abc_00000065_nonlinear_quality.jpg](mesh_quality/gold_nested_abc_00000065_nonlinear_quality.jpg) | ![harness_auto_abc_00000003_linear_groups.jpg](mesh_quality/harness_auto_abc_00000003_linear_groups.jpg) | ![harness_auto_abc_00000003_linear_harness_out_groups.jpg](mesh_quality/harness_auto_abc_00000003_linear_harness_out_groups.jpg) |
| `gold_nested_abc_00000065_nonlinear_levels_L1_quality` | `gold_nested_abc_00000065_nonlinear_quality` | `harness_auto_abc_00000003_linear_groups` | `harness_auto_abc_00000003_linear_harness_out_groups` |
| ![harness_auto_abc_00000003_linear_harness_out_quality.jpg](mesh_quality/harness_auto_abc_00000003_linear_harness_out_quality.jpg) | ![harness_auto_abc_00000003_linear_quality.jpg](mesh_quality/harness_auto_abc_00000003_linear_quality.jpg) | ![harness_auto_abc_00000003_nonlinear_groups.jpg](mesh_quality/harness_auto_abc_00000003_nonlinear_groups.jpg) | ![harness_auto_abc_00000003_nonlinear_harness_out_groups.jpg](mesh_quality/harness_auto_abc_00000003_nonlinear_harness_out_groups.jpg) |
| `harness_auto_abc_00000003_linear_harness_out_quality` | `harness_auto_abc_00000003_linear_quality` | `harness_auto_abc_00000003_nonlinear_groups` | `harness_auto_abc_00000003_nonlinear_harness_out_groups` |
| ![harness_auto_abc_00000003_nonlinear_harness_out_quality.jpg](mesh_quality/harness_auto_abc_00000003_nonlinear_harness_out_quality.jpg) | ![harness_auto_abc_00000003_nonlinear_quality.jpg](mesh_quality/harness_auto_abc_00000003_nonlinear_quality.jpg) | ![harness_auto_abc_00000065_linear_groups.jpg](mesh_quality/harness_auto_abc_00000065_linear_groups.jpg) | ![harness_auto_abc_00000065_linear_harness_out_groups.jpg](mesh_quality/harness_auto_abc_00000065_linear_harness_out_groups.jpg) |
| `harness_auto_abc_00000003_nonlinear_harness_out_quality` | `harness_auto_abc_00000003_nonlinear_quality` | `harness_auto_abc_00000065_linear_groups` | `harness_auto_abc_00000065_linear_harness_out_groups` |
| ![harness_auto_abc_00000065_linear_harness_out_quality.jpg](mesh_quality/harness_auto_abc_00000065_linear_harness_out_quality.jpg) | ![harness_auto_abc_00000065_linear_quality.jpg](mesh_quality/harness_auto_abc_00000065_linear_quality.jpg) | ![harness_auto_abc_00000065_nonlinear_groups.jpg](mesh_quality/harness_auto_abc_00000065_nonlinear_groups.jpg) | ![harness_auto_abc_00000065_nonlinear_harness_out_groups.jpg](mesh_quality/harness_auto_abc_00000065_nonlinear_harness_out_groups.jpg) |
| `harness_auto_abc_00000065_linear_harness_out_quality` | `harness_auto_abc_00000065_linear_quality` | `harness_auto_abc_00000065_nonlinear_groups` | `harness_auto_abc_00000065_nonlinear_harness_out_groups` |
| ![harness_auto_abc_00000065_nonlinear_harness_out_quality.jpg](mesh_quality/harness_auto_abc_00000065_nonlinear_harness_out_quality.jpg) | ![harness_auto_abc_00000065_nonlinear_quality.jpg](mesh_quality/harness_auto_abc_00000065_nonlinear_quality.jpg) | ![harness_expert_abc_00000003_linear_groups.jpg](mesh_quality/harness_expert_abc_00000003_linear_groups.jpg) | ![harness_expert_abc_00000003_linear_quality.jpg](mesh_quality/harness_expert_abc_00000003_linear_quality.jpg) |
| `harness_auto_abc_00000065_nonlinear_harness_out_quality` | `harness_auto_abc_00000065_nonlinear_quality` | `harness_expert_abc_00000003_linear_groups` | `harness_expert_abc_00000003_linear_quality` |
| ![harness_expert_abc_00000003_nonlinear_groups.jpg](mesh_quality/harness_expert_abc_00000003_nonlinear_groups.jpg) | ![harness_expert_abc_00000003_nonlinear_quality.jpg](mesh_quality/harness_expert_abc_00000003_nonlinear_quality.jpg) | ![harness_expert_abc_00000065_linear_groups.jpg](mesh_quality/harness_expert_abc_00000065_linear_groups.jpg) | ![harness_expert_abc_00000065_linear_quality.jpg](mesh_quality/harness_expert_abc_00000065_linear_quality.jpg) |
| `harness_expert_abc_00000003_nonlinear_groups` | `harness_expert_abc_00000003_nonlinear_quality` | `harness_expert_abc_00000065_linear_groups` | `harness_expert_abc_00000065_linear_quality` |
| ![harness_expert_abc_00000065_nonlinear_groups.jpg](mesh_quality/harness_expert_abc_00000065_nonlinear_groups.jpg) | ![harness_expert_abc_00000065_nonlinear_quality.jpg](mesh_quality/harness_expert_abc_00000065_nonlinear_quality.jpg) | ![harness_hex_abc_00000003_linear_groups.jpg](mesh_quality/harness_hex_abc_00000003_linear_groups.jpg) | ![harness_hex_abc_00000003_linear_out_groups.jpg](mesh_quality/harness_hex_abc_00000003_linear_out_groups.jpg) |
| `harness_expert_abc_00000065_nonlinear_groups` | `harness_expert_abc_00000065_nonlinear_quality` | `harness_hex_abc_00000003_linear_groups` | `harness_hex_abc_00000003_linear_out_groups` |
| ![harness_hex_abc_00000003_linear_out_quality.jpg](mesh_quality/harness_hex_abc_00000003_linear_out_quality.jpg) | ![harness_hex_abc_00000003_linear_quality.jpg](mesh_quality/harness_hex_abc_00000003_linear_quality.jpg) | ![harness_hex_abc_00000003_nonlinear_out_quality.jpg](mesh_quality/harness_hex_abc_00000003_nonlinear_out_quality.jpg) | ![harness_hex_abc_00000003_nonlinear_quality.jpg](mesh_quality/harness_hex_abc_00000003_nonlinear_quality.jpg) |
| `harness_hex_abc_00000003_linear_out_quality` | `harness_hex_abc_00000003_linear_quality` | `harness_hex_abc_00000003_nonlinear_out_quality` | `harness_hex_abc_00000003_nonlinear_quality` |
| ![harness_hex_abc_00000065_linear_groups.jpg](mesh_quality/harness_hex_abc_00000065_linear_groups.jpg) | ![harness_hex_abc_00000065_linear_out_groups.jpg](mesh_quality/harness_hex_abc_00000065_linear_out_groups.jpg) | ![harness_hex_abc_00000065_linear_out_quality.jpg](mesh_quality/harness_hex_abc_00000065_linear_out_quality.jpg) | ![harness_hex_abc_00000065_linear_quality.jpg](mesh_quality/harness_hex_abc_00000065_linear_quality.jpg) |
| `harness_hex_abc_00000065_linear_groups` | `harness_hex_abc_00000065_linear_out_groups` | `harness_hex_abc_00000065_linear_out_quality` | `harness_hex_abc_00000065_linear_quality` |
| ![harness_hex_abc_00000065_nonlinear_out_quality.jpg](mesh_quality/harness_hex_abc_00000065_nonlinear_out_quality.jpg) | ![harness_hex_abc_00000065_nonlinear_quality.jpg](mesh_quality/harness_hex_abc_00000065_nonlinear_quality.jpg) | ![netgen_abc_00000003_linear_groups.jpg](mesh_quality/netgen_abc_00000003_linear_groups.jpg) | ![netgen_abc_00000003_linear_quality.jpg](mesh_quality/netgen_abc_00000003_linear_quality.jpg) |
| `harness_hex_abc_00000065_nonlinear_out_quality` | `harness_hex_abc_00000065_nonlinear_quality` | `netgen_abc_00000003_linear_groups` | `netgen_abc_00000003_linear_quality` |
| ![netgen_abc_00000003_nonlinear_groups.jpg](mesh_quality/netgen_abc_00000003_nonlinear_groups.jpg) | ![netgen_abc_00000003_nonlinear_quality.jpg](mesh_quality/netgen_abc_00000003_nonlinear_quality.jpg) | ![netgen_abc_00000065_linear_groups.jpg](mesh_quality/netgen_abc_00000065_linear_groups.jpg) | ![netgen_abc_00000065_linear_quality.jpg](mesh_quality/netgen_abc_00000065_linear_quality.jpg) |
| `netgen_abc_00000003_nonlinear_groups` | `netgen_abc_00000003_nonlinear_quality` | `netgen_abc_00000065_linear_groups` | `netgen_abc_00000065_linear_quality` |
| ![netgen_abc_00000065_nonlinear_groups.jpg](mesh_quality/netgen_abc_00000065_nonlinear_groups.jpg) | ![netgen_abc_00000065_nonlinear_quality.jpg](mesh_quality/netgen_abc_00000065_nonlinear_quality.jpg) |
| `netgen_abc_00000065_nonlinear_groups` | `netgen_abc_00000065_nonlinear_quality` |

