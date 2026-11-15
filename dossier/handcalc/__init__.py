"""Hand calculations: closed-form models that check the analysis from an independent side.

A hand calculation is the check that finds a wrong finite-element model, thus every formula here has a
stated source and a test against an independent number (tests/dossier/test_dossier_handcalc.py).

    from dossier import handcalc
    handcalc.applies(spec, load_case)                       # the `when` of the specification
    hc = handcalc.evaluate(spec, project, ctx, geo, load_case)   # a HandCalcResult record
    checks = handcalc.compare(hc, analysis, project)        # one Check per comparison
    handcalc.MODELS, handcalc.get("thick_cylinder_pressure")
    handcalc.get("sphere_pressure").calculate({"p": 2e6, "r_i": 0.1, "r_o": 0.11}).values   # SI numbers

A specification in component.yaml (dossier.model.HandCalcSpec):

    - id: HC-SHELL-HOOP
      model: thick_cylinder_pressure
      title: Hoop stress in the shell
      when: {analysis: [static_structural], needs: ["pressure@INT-INLET"]}
      inputs:
        p: {expr: 'load("INT-INLET", "pressure")'}
        r_i: 22 mm
        r_o: 28 mm
      compare:
        - {output: hoop_mean, with: "max_principal.membrane@SCL-SHELL", band: [0.6, 1.5], why: "..."}

The models (each module states its sources and how its tests verify it)

    pressure   thick_cylinder_pressure          Lamé cylinder: hoop_inner, hoop_outer, hoop_mean, radial_inner,
                                                axial_closed, von_mises_inner
               thin_cylinder_pressure           membrane stress: hoop, axial_closed, von_mises_closed,
                                                radial_growth_closed
               sphere_pressure                  Lamé sphere: hoop_inner, hoop_outer, hoop_mean, hoop_mean_area,
                                                radial_inner, von_mises_inner
    plates     circular_plate_pressure          sigma_center_clamped, sigma_edge_clamped,
                                                sigma_center_simply_supported, deflection_center_clamped,
                                                deflection_center_simply_supported
    beams      tube_bending_axial               sigma_bending, sigma_bending_outer, sigma_axial, sigma_tension,
                                                sigma_compression
               beam_bending                     sigma_bending = M / Z
               bolt_circle_loads                force_axial, force_moment, force_max, force_min
               euler_buckling                   P_cr, sigma_cr, slenderness
    thermal    fin_convective_tip               T_tip, heat_flow_base, m, mL
               thermal_stress_constrained       sigma_uniaxial, sigma_biaxial
               hollow_cylinder_radial_gradient  hoop_inner, hoop_outer, axial_inner, axial_outer
               conduction_cylinder_wall         heat_flow_per_length, T_wall_inner, T_wall_outer, dT_wall
               dittus_boelter                   reynolds, prandtl, nusselt, h
    fluid      pressure_drop_minor_loss         velocity, reynolds, friction_factor, dp_friction, dp_minor, dp_total
               rigid_column_surge               dp, dp_michaud
               joukowsky_surge                  dp
    fatigue    basquin_life                     sigma_ar, cycles

What a record holds (dossier.model.HandCalcResult): the inputs in SI with their display form and their
source, the steps (one sentence, the formula in LaTeX without delimiters, the result with its unit), the
outputs, the assumptions of the model, the warnings of its validity check, and the comparisons that the
analysis must serve. Text is ASD-STE100.
"""

from dossier.handcalc import beams, fatigue, fluid, plates, pressure, thermal  # noqa: F401  (they register the models)
from dossier.handcalc.base import MODELS, Calc, HandCalcError, Input, Model, Output, get, model, register
from dossier.handcalc.engine import applies, compare, compared, evaluate

__all__ = ["evaluate", "applies", "compare", "compared", "MODELS", "get", "Model", "Input", "Output", "Calc",
           "HandCalcError", "model", "register"]
