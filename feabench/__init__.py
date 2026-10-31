"""feabench: benchmark cases and a deterministic evaluation harness for FEA-setup agents.

Layout:
  feabench.ccx        CalculiX I/O: run the fixed solver, read .frd/.dat, scan .inp decks
  feabench.fem        mesh utilities: geometric regions, point location/interpolation, volumes
  feabench.qoi        quantities of interest extracted from solver results
  feabench.scoring    per-case and population scoring (see docs/eval-design.md)
  feabench.evaluate   submission -> solver run -> QoIs -> score
  feabench.suites     importers/builders for the benchmark suites (calculix, code_aster, simjeb, verification)
"""

__version__ = "0.1.0"
