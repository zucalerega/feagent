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

**`mesh.inp` contains surface elements.** Besides the solid tets (C3D4 / C3D10, in the ELSET
`domain`), `mesh.inp` holds the triangles of every surface group as CPS3 (order 1) or CPS6
(order 2) elements, written per CAD face (`*ELEMENT, type=CPS6, ELSET=Surface<tag>`) and
collected in the group's ELSET, so the solver can locate the faces. These surface elements must
**not** get a section or a material: assign the solid section to `domain` only, and use the
surface groups' NSETs (or the faces of `domain` elements) for loads and supports.

## Quality (`quality`)

| field | meaning |
|---|---|
| `n_nodes`, `n_tets` | Mesh size. |
| `sicn`, `gamma`, `sj` | Stats `{min, p01, p05, median, max, bins (11 edges), hist (10 counts)}` per tet. |
| `inverted` | Tets with scaled Jacobian ≤ 0. |
| `worst` | The 10 lowest-SICN tets: `{tag, sicn, gamma, centroid, faces}`, where `faces` are the CAD faces each tet touches. |
| `volume_cad`, `volume_mesh`, `volume_error` | Volume fidelity (relative error). The mesh volume is integrated by Gauss quadrature of the element Jacobian, so curved order-2 tets are measured exactly (not as straight-sided tets). |
| `group_area` | name → `[cad_area, mesh_area, relative_error]`. Mesh areas are integrated the same way. |
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

## When to trust a `report.json`

A check that stops at spec validation (exit 2) writes no report. An older `report.json` from an
earlier run stays on disk, but the state entry is set to `status: "fail"` with the spec `error`.
A `report.json` describes the current spec only if all of the following hold:

1. the state entry's `report_created_at` equals the report's `created_at`;
2. the entry's `status` equals the report's `status`;
3. the spec file's current mtime equals the entry's `spec_mtime`. If the spec was edited during
   or after the check, the report is stale.

## State file (`state.json`, read by the Stop hook)

Location: `$FEAGENT_STATE` if set. Otherwise it is `<root>/.feagent/state.json`, where `<root>` is
the nearest ancestor of the working directory (the directory itself included) that contains
`.feagent/` or `pyproject.toml`. If there is none, the working directory is used. The file is
written atomically, and top-level keys other than `specs` (e.g. the hook's `hook_blocks`) are
kept.

```json
{"specs": {"/abs/path/mesh.yaml": {
  "status": "pass", "checked_at": 1793416446.96, "spec_mtime": 1793416443.98,
  "error": null, "report_created_at": "2026-10-31T03:14:06+00:00",
  "waiver": null, "waiver_spec_mtime": null}}}
```

| field | meaning |
|---|---|
| `status` | `unchecked`: `feagent mesh` succeeded but no check has run. `fail`: a check failed, meshing failed, or the spec is invalid. `pass`: a check passed. |
| `checked_at` | Epoch seconds when the entry was written. |
| `spec_mtime` | Epoch seconds: the spec's mtime, read before the run loaded the spec. The result is stale when the file's current mtime differs. |
| `error` | Spec validation error text (with its hint) when the spec was invalid, else `null`. |
| `report_created_at` | `created_at` of the `report.json` this entry describes (`check` only), else `null`. |
| `waiver`, `waiver_spec_mtime` | Set by `feagent waive SPEC --reason`. A waiver applies only while the spec's mtime equals `waiver_spec_mtime`, so editing the spec makes it stale again. A passing check clears the waiver; a failing check or a mesh run keeps it. |

### Stop hook (`feagent hook-stop`)

The hook reads the Claude Code Stop payload from stdin (`transcript_path`, `stop_hook_active`).
**Scope:** the session starts at the first timestamped entry of the transcript, and only entries
whose `checked_at` is at or after that time can block; entries from earlier sessions are ignored.
Without a payload or a readable transcript every entry counts. An entry blocks when the spec file
exists and the entry is `fail`, `unchecked` or stale (spec mtime changed), unless a valid waiver
covers it. Blocking means exit code 2 with the reasons on stderr.

**Escape:** after 3 consecutive blocks for the same blocking set (the exact `[spec, spec mtime,
status]` triples) the hook allows the stop and keeps allowing that set; it blocks again only when
the set changes (an edit, a new status, a new failing spec). `stop_hook_active: false` marks a
fresh stop attempt and restarts the count. The counter is the top-level `hook_blocks` key:
`{"count": n, "blocking": [[spec, mtime, status], ...] | null, "allowed": [...] | null}`. A
malformed state file (`specs` or an entry not a mapping) never blocks; the hook says so on stderr.

## Exit codes (every `feagent` command)

| code | meaning |
|---|---|
| `0` | OK. For `check` and `automesh`, every gate passed (warnings allowed). |
| `1` | A gate failed or no mesh was written (`check`, `automesh`, `mesh`). The report/result is still printed. |
| `2` | Usage, spec, query or CAD error. No report is written. |
| `3` | Internal error. The traceback goes to stderr. |

For exit codes 2 and 3, stderr gets `error: <message>` and `hint: <hint>`. With `--json`, stdout
gets exactly one document: `{"ok": false, "error": "...", "hint": "..."}`. For an internal error
the message starts with `internal: `.
