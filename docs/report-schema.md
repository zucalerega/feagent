# `report.json` schema (version 1)

`feagent check SPEC` (and `feagent automesh PART`) write `report.json` and `REPORT.md` into the
spec's output directory: `<spec dir>/<output>`, `out/` by default. `report.json` is for programs
and the evaluator agent. `REPORT.md` holds the same content for people: the gate table first,
then the numbers, the boundary conditions and the file paths. The same JSON document is printed
to stdout by `feagent check SPEC --json`.

Producer: `feagent.check.CheckReport.to_dict()`. Round-trip: `CheckReport.from_dict(json)`.

## Top level

| field | type | meaning |
|---|---|---|
| `schema_version` | int | `1`. Bumped on any incompatible change. |
| `spec_path` | str | Absolute path of the checked `mesh.yaml`. |
| `status` | `"pass"` \| `"fail"` | `"fail"` if any gate failed or no mesh was written. Warnings and skips do not fail the check. |
| `gates` | list[Gate] | Every gate, in order: the mesher gates, then the quality gates (the quality gates are present only if a mesh exists). **This is the authoritative list.** |
| `mesh` | MeshResult | The mesher's result (the `manifest.json` content), with `geometry` set to `null`. The full geometry analysis is in `<mesh.out_dir>/manifest.json`. |
| `quality` | object \| null | Mesh quality measurements; `null` when no mesh was written. |
| `bcs` | object \| null | Boundary conditions declared in the spec's `bcs:` section, passed through for the downstream solver; `null` if none are declared. |
| `renders` | list[Render] | PNG renders (empty with `--no-render` or when no mesh was written). |
| `created_at` | str | ISO-8601 UTC time of the check, e.g. `2026-10-30T12:00:00+00:00`. |

## Gate

| field | type | meaning |
|---|---|---|
| `id` | str | Stable identifier (table below). Use it to match gates across reports. |
| `name` | str | Human-readable name. |
| `status` | `"pass"` \| `"warn"` \| `"fail"` \| `"skip"` | `skip`: the gate does not apply, or an earlier stage failed. |
| `value` | number \| null | Measured value. |
| `threshold` | number \| null | Limit it was compared with (a count for count gates). |
| `message` | str | What was measured, with numbers. A failing `groups` gate carries the near-miss explanation of the face query here (multi-line). |
| `hint` | str | The spec key to change. Empty when the gate passes. |

### Gate ids

Mesher gates (`feagent.mesher`). They run in order and a failure stops the pipeline: the gates
after it are `skip` and there is no mesh, so there are no quality gates either.

| id | statuses | value / threshold | fails when |
|---|---|---|---|
| `spec` | fail | — | The spec became invalid between validation and meshing. This is rare: `check` validates the spec first and exits 2 on a spec error. |
| `geometry` | pass, fail, skip | volume count / 1 | The CAD does not load, the domain cannot be built, or the part has no volume. |
| `groups` | pass, fail, skip | number of groups resolved | A group's query matches the wrong number of faces (`expect`), or a face is in two groups. |
| `mesh_generated` | pass, fail, skip | tet count / 1 | gmsh raises an error, produces 0 tets, or times out. |

Quality gates (`feagent.quality`), evaluated against the spec's `quality:` thresholds:

| id | statuses | value / threshold | meaning |
|---|---|---|---|
| `inverted` | pass, fail | inverted tets / 0 | Count of tets whose scaled Jacobian is ≤ 0. |
| `sicn` | pass, fail | min SICN / `quality.min_sicn` | Worst signed inverse condition number. |
| `gamma` | pass, warn | min gamma / `quality.min_gamma` | Inscribed/circumscribed radius ratio. A sliver warning; never fails. |
| `volume_fidelity` | pass, fail, skip | relative error / `quality.max_volume_error` | Mesh volume compared with the CAD volume. |
| `area_fidelity` | pass, fail, skip | worst relative error / `quality.max_area_error` | Area of each surface group compared with its CAD faces (skip if there are no groups). |
| `connectivity` | pass, fail | components / expected | Number of face-connected mesh pieces compared with the geometry. |
| `thin_walls` | pass, warn, skip | min thickness/edge / `quality.thin_wall_min_elements` | Elements across each thin-wall hazard (skip if there are none). |
| `non_ortho` | pass, warn, fail | max deg / `quality.max_non_ortho` | **cfd only.** Warns above 65°. |
| `skewness` | pass, fail | max / `quality.max_skewness` | **cfd only.** |

## MeshResult (`mesh`)

`ok` (bool), `spec_path`, `out_dir`, `mesh_path` (str | null; MSH 4.1), `exports` (format → path;
formats: `msh`, `msh22`, `inp`, `vtk`, `unv`, `su2`), `groups` (name → `{name, dim, tags,
elements, area}`; surface groups have `dim` 2, the volume group `domain` has `dim` 3),
`n_nodes`, `n_elements` (tets), `order` (1 | 2), `size_scale`, `elapsed_s`, `errors` (list[str]),
`warnings` (list[str]), `gates` (the mesher gates, also present in the top-level `gates`),
`geometry` (always `null` here).

Every group is exported under its own name: as a physical group in `.msh`, `.su2` and `.unv`, and
as an NSET and ELSET in `.inp`. The solver finds the `bcs` groups by these names.

## Quality (`quality`)

| field | meaning |
|---|---|
| `n_nodes`, `n_tets` | Mesh size. |
| `sicn`, `gamma`, `sj` | Stats `{min, p01, p05, median, max, bins (11 edges), hist (10 counts)}` per tet. |
| `inverted` | Tets with scaled Jacobian ≤ 0. |
| `worst` | The 10 lowest-SICN tets: `{tag, sicn, gamma, centroid, faces}`, where `faces` are the CAD faces each tet touches. |
| `volume_cad`, `volume_mesh`, `volume_error` | Volume fidelity (relative error). |
| `group_area` | name → `[cad_area, mesh_area, relative_error]`. |
| `components`, `expected_components` | Connectivity. |
| `non_ortho`, `skewness` | Stats (cfd), else `null`. |

## Boundary conditions (`bcs`)

The spec's `bcs:` section, passed through unchanged. The harness does not solve; it checks that
every group named here exists in the spec (a spec error otherwise), and the mesher checks each
group's face count.

```json
{"fea": {"fixed": ["mount"], "loads": [{"group": "load", "force": [0.0, 0.0, -100.0]}],
         "E": 210000.0, "nu": 0.3}}
{"cfd": {"inlet": ["inlet"], "outlet": ["outlet"], "velocity": 1.0}}
```

Units are mm, N and MPa. `force` is the total force on the group in N. For cfd, faces in no group
go to `default_group`, which is `walls` by default.

## Render (`renders[]`)

`{path, caption}`. `groups.png` shows the surface groups over a translucent shell.
`quality.png` shows per-tet SICN with mid-plane clips, the 10 worst tets marked. Each caption
gives the numbers shown in its image.

## Exit codes (`feagent check` / `automesh`)

`0` all gates pass (warnings allowed), `1` a gate failed or no mesh, `2` spec / CAD / usage error
(message and hint on stderr; with `--json`, `{"ok": false, "error", "hint"}` on stdout and no
report written).
