"""The system side of the validate stage: is the system model sound, and what does it look like?

    dossier.system.check.check(project, ctx, geo) -> Journal      findings and assumptions about the system model
    dossier.system.diagram.render(project, ctx) -> str            the system diagram, one SVG document
    dossier.system.diagram.effects_table(project, ctx) -> rows    every effect on the component, per phase

Import the modules, not names from this package (`from dossier.system import check` gives the module).
"""
