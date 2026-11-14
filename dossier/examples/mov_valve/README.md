# Example: the body of a main oxidizer valve

A complete `dossier` project, small enough to read in one sitting. Everything in it is example data made for
this demonstration: the engine, its interface numbers, the valve geometry, the material values and the
factor policy. Do not use any number from it for design.

| file | what it holds |
|---|---|
| `project.yaml` | the project: which component is verified, where the other files are |
| `system.yaml` | the oxidizer feed branch: 11 elements, 10 connections, 8 phases, 3 duty cycles, the named numbers with their sources, and the effects of each connection on the valve in each phase |
| `components/mov_body/component.yaml` | the valve body: CAD, material, 18 named features (face sets selected by geometry), 5 interfaces, 5 stress classification lines, 6 hand-calculation models |
| `components/mov_body/requirements.yaml` | 10 requirements of the valve specification: source text, parameters, criteria |
| `components/mov_body/geometry.step` | the CAD, made by `make_geometry.py` |
| `materials.yaml`, `policy.yaml`, `terms.yaml` | the material with its sources, the factor policy, the project's technical names for the STE check |

```bash
python -m dossier check dossier/examples/mov_valve          # read the inputs, evaluate every number
python -m dossier run   dossier/examples/mov_valve --quick  # every stage, coarse meshes
python -m dossier new   my_valve                            # a copy to edit
```

## The system in words

A turbopump delivers liquid oxygen at up to 10.4 MPa to the main oxidizer valve, a top-entry ball valve with
an extended neck, bolted to the pump discharge flange. The valve carries the injector feed line on its outlet
flange and a pneumatic actuator of 2.8 kg on top of its neck. At shutdown the actuator closes the valve
against the full flow of 5.3 kg/s, which adds a pressure surge of 0.32 MPa at the inlet. The valve body is
at 90 K inside and in ambient air outside. Each valve gets a proof test at 1.5 times MEOP.

## The six deliberate defects of the specification

The validation stage must find these. They are the kind of defect that a specification picks up when the
system around the component changes and the specification does not.

| # | requirement | defect |
|---|---|---|
| 1 | REQ-MOV-001 | MEOP is 10.5 MPa. The system context gives 10.72 MPa at the inlet during shutdown: the maximum pump pressure plus the closure surge. |
| 2 | REQ-MOV-004 | The factor of safety against yield is 1.1. The factor policy asks for 1.25. |
| 3 | REQ-MOV-005 | The fluid temperature range is 233 K to 323 K. The fluid is liquid oxygen at 90 K. |
| 4 | REQ-MOV-006 | The design life is 200 pressure cycles. The system context gives 267. |
| 5 | REQ-MOV-009 | "Sufficiently stiff": no measurable criterion. Analysis cannot verify it. |
| 6 | (none) | No requirement covers the transport phase, in which a 15 g shock loads the neck through the actuator. |
