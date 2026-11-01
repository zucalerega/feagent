# case.json schema (feagent-tap25d-case-v1)

case.json is ASCII JSON. Every number is an IEEE double written in its shortest round-trip form, so floorplan values carry the generator's own floats (for example 8.999999999999999e-05). Use them as given. Units are SI: m, W, W/(m K), W/(m^2 K), K. Thermal resistivity rho (m K/W) is kept as configured, and k = 1/rho.

A string KEY:N, or KEY:N-M, cites line N (to M) of source KEY in "sources". Lists use commas, as in KEY:154,160-161. "GEN:<case>/..." stands for both GEN:A/... and GEN:B/....

## Top level

| key | meaning |
|---|---|
| schema, title, prepared | identity of this file |
| is, is_not, status_statement | what the case is and is not |
| units | unit of each quantity |
| commits | the public TAP-2.5D and HotSpot repositories and the commits read |
| sources | KEY -> record (below) |
| paper | citation, URL, PDF sha256, and the locations used (section, page, figure) |
| verification | what was checked before release |
| coordinate_system | origin at the interposer's lower-left corner on the substrate bottom; +x, +y as in the floorplans; +z toward the heat sink |
| stack | the 8 layers, bottom to top (below) |
| model_top_z_m | z of the convective face |
| materials | each material's rho, k, where it is used, and the composite rule |
| boundary_conditions | ambient, convective face, adiabatic faces with areas, interfaces, inactive HotSpot options |
| hotspot_reference_settings | the paper's solver settings (grid, centre mapping, output rounding) |
| heat_source | where the power goes, in-plane and through the thickness (primary and variant S) |
| metric | the paper's statistic, the FE observables and the z planes |
| cases | "A" and "B" (below) |
| ambiguities | A1-A11: id, topic, configuration_value, treatment (A1 also what_hotspot_does and scale) |
| known_differences_hotspot6_vs_continuum | why a compact model and a continuum FE model can differ |
| report_back | what to return |

## sources

Each record has "kind" plus:
- "paper": citation, url, pdf_sha256, role.
- "public repository file" (TAP:<path>, HS6:<path>): repo, commit, path, url (pinned to the commit), sha256 and git_blob of the file bytes, bytes, license, role.
- "generated input" (GEN:A/<file>, GEN:B/<file>): the generator and its inputs, normalization, sha256, bytes, role.
- "derived from the paper's figures by the SentiLab team" (SL:decode): what, method, assumption, uncertainty, consistency_check.

## stack[i]

index (0-7), name, z_bottom_m, z_top_m, thickness_m, extent_m {x0, y0, x1, y1}, has_power, src.
- Layers 0-5 also carry default_rho_mK_per_W and default_k_W_per_mK (the layer value, used by blocks without their own material), k_values_present_W_per_mK (the k values that actually occur in the layer), k_note, lateral_conduction and blocks (the path of the block list).
- The spreader (6) and sink (7) carry default_k_W_per_mK only.

## A case: cases.A, cases.B

| key | meaning |
|---|---|
| id, title | layout identity |
| paper_target_C, paper_location, target_source | the published value and where it is printed |
| layout | identity, decode_uncertainty_mm_max (per coordinate), note |
| tap_system_config | the TAP-2.5D config used, and the intp_size override |
| total_power_W, total_power_src | total die power |
| chiplets | the dies (below) |
| layers | "L0".."L5": every floorplan block of that layer (below) |
| tiling_check | for L3 and L4: block count, area sums, largest overlap, uncovered_rectangles_m [[x0, x1, y0, y1]] and uncovered_area_m2 |
| geometry_stats | smallest block dimension and edge counts (meshing aid) |
| closed_form_checks | analytic estimates from the inputs, not solver results: mean_excess_on_convective_face_K (P_total x r_convec) and die_through_thickness_scale_K (q'' t / k of the hottest die) |

**chiplets[j]**:
- index, flp_name (Chiplet_i), role, power_W;
- the rectangle: x0_m, y0_m, width_m, height_m, and its centre x_centre_m, y_centre_m;
- rotated_vs_tap_cfg (true when the TAP config's width and height are swapped) and catalog_width_height_mm;
- microbump_halo_m, power_density_W_m2, volumetric_source_W_m3;
- raw_centre_mm and raw_centre_uncertainty_mm (the decoded centre, SL:decode);
- hotspot_centre_cells_64, hotspot_cover_range_64 and hotspot_centre_cells_centroid_offset_m;
- src.

**layers["Ln"][k]** (a block):
- name, x0_m, y0_m, width_m, height_m, x_centre_m, y_centre_m;
- material, rho_mK_per_W, k_W_per_mK;
- power_W (non-zero only for L4 Chiplet_*);
- src;
- hotspot_centre_cells_64: the 1, 2 or 4 cells of HotSpot's 64 x 64 grid whose weighted mean HotSpot reports for this block in centre mode. Each cell has row_i, col_j, weight and bounds x0_m, x1_m, y0_m, y1_m. Row 0 is at the top (y = 0.045 m).

The same block names occur in L3 and L4 with identical rectangles but different materials.

## Building the FE model

1. **Layers 0-5.** Layer L occupies [stack[L].z_bottom_m, stack[L].z_top_m]. Tile [0, 0.045]^2 with the rectangles of cases.X.layers["L<L>"] and give each its k_W_per_mK.
2. **Case A gap.** Also fill cases.A.tiling_check.L3/L4.uncovered_rectangles_m with underfill, k = 1.6 (ambiguity A4).
3. **Spreader and sink.** They are the boxes stack[6] and stack[7] (extent_m and z).
4. **Source, primary.** Put volumetric_source_W_m3 in each chiplet's rectangle over the whole of layer 4.
5. **Source, variant S.** Put power_density_W_m2 as a flux on the chip's bottom face, z = metric.z_die_bottom_m.
6. **Boundaries.** Use a Robin condition on the sink top: boundary_conditions.convective_face (h_W_per_m2K, T_ref_K). Every other external face is adiabatic. Interfaces have perfect contact.
7. **Observables.**
   - T_i = T(x_centre_m, y_centre_m, z) at z = metric.z_die_bottom_m, z_die_mid_m and z_die_top_m.
   - M = max_i T_i(mid) - 273.15 is compared with paper_target_C.
   - T_cell(b) = sum over b's hotspot_centre_cells_64 of weight x (area mean of T over the cell footprint at the layer's mid-thickness).

## Regenerating the HotSpot inputs (optional)

This rebuilds the GEN files with the public generator. Run it from the root of a TAP-2.5D checkout at commit 3a3f4e5a4d019b2d9cda1f72d8d1755238845dbb; the generator reads util/hotspot.config relative to the working directory. TAP's config.py is not imported, because it imports the CPLEX router, so its two small parsers are repeated here.

```python
import configparser, json, os, sys
sys.path.insert(0, os.getcwd())
from passive_interposer import PassiveInterposer

def get_list(o): return list(map(float, "".join(o.split()).split(",")))                  # as TAP config.py
def get_matrix(o): return [list(map(int, r.split(","))) for r in "".join(o.split()).split(";")]

case_id, cfg_file, case_json = "A", "configs/sys_ascend910.cfg", sys.argv[1]   # B: configs/sys_micro150.cfg
chips = sorted(json.load(open(case_json))["cases"][case_id]["chiplets"], key=lambda d: d["index"])
cfg = configparser.ConfigParser(); cfg.read(cfg_file)
s = PassiveInterposer()
s.set_chiplet_count(cfg.getint("chiplets", "chiplet_count")); s.initialize()
s.set_interposer_size(45.0)                       # paper Sec. IV; TAP README: -g intp_size=45
s.set_chiplet_size(get_list(cfg.get("chiplets", "widths")), get_list(cfg.get("chiplets", "heights")))
s.set_chiplet_power(get_list(cfg.get("chiplets", "powers")))
s.set_connection_matrix(get_matrix(cfg.get("chiplets", "connections")))
s.set_granularity(cfg.getfloat("general", "placer_granularity"))
s.set_link_type(cfg.get("interposer", "link_type"))
s.compute_ubump_overhead()
for d in chips:
    if d["rotated_vs_tap_cfg"]:
        s.rotate(d["index"])
s.x = [d["raw_centre_mm"][0] for d in chips]
s.y = [d["raw_centre_mm"][1] for d in chips]
os.makedirs("out", exist_ok=True); s.set_path("out/")
s.gen_flp("sample"); s.gen_ptrace("sample")
```

To compare with the sha256 values in sources["GEN:..."], convert the files to LF line endings and reduce the floorplan paths in samplelayers.lcf to bare file names. gen_flp writes out/new_hotspot.config only when that file does not exist yet.

## Reporting results (suggested JSON)

```json
{"case": "A", "source_variant": "primary",
 "meshes": [{"label": "m1", "element": "P2 tet", "elements": 0, "nodes": 0, "dofs": 0,
             "h_min_m": 0.0, "h_max_m": 0.0, "elements_through_layer": {"L3": 0, "L4": 0, "L5": 0},
             "M_C": 0.0, "T_K": {"Chiplet_0": {"bottom": 0.0, "mid": 0.0, "top": 0.0}},
             "T_cell_K": {"Chiplet_0": 0.0}, "residual_rel": 0.0, "time_s": 0.0, "peak_memory_MB": 0.0}],
 "M_extrapolated_C": 0.0, "M_error_estimate_K": 0.0, "observed_order": 0.0,
 "energy_balance": {"P_in_W": 0.0, "Q_out_W": 0.0, "rel_error": 0.0, "mean_excess_convective_face_K": 0.0},
 "field_extremes": {"T_max_K": 0.0, "at_m": [0.0, 0.0, 0.0], "T_min_K": 0.0, "at_min_m": [0.0, 0.0, 0.0]},
 "versions": {"python": "", "numpy": "", "scipy": "", "scikit-fem": "", "mesher": ""},
 "case_json_sha256": "", "script_sha256": ""}
```
