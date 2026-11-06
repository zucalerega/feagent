"""Box-beam (Euler-Bernoulli + web shear) tip deflection of the wingbox cantilever.

Section (x-z plane, mm): skins 600 x 20 at z = 0..20 and 130..150, spar webs 15 x 110 at
x = 0..15 and 585..600, z = 20..130. Ribs are ignored. Bending about the x axis through the
section centroid (z = 75, by symmetry), parallel-axis theorem per rectangle.

    delta = P L^3 / (3 E I) + P L / (G A_webs),  G = E / (2 (1 + nu))

Usage: python beam_theory.py  -> prints JSON.
"""
import json

E, NU = 71700.0, 0.33          # MPa, 7075-T6
P, L = 20000.0, 2000.0         # N (tip, +z), mm (root y = 0 to tip y = 2000)
G = E / (2 * (1 + NU))

# (name, width b along x, height h along z, centroid z)
RECTS = [("lower_skin", 600.0, 20.0, 10.0), ("upper_skin", 600.0, 20.0, 140.0),
         ("front_web", 15.0, 110.0, 75.0), ("rear_web", 15.0, 110.0, 75.0)]


def section():
    A = sum(b * h for _, b, h, _ in RECTS)
    zc = sum(b * h * z for _, b, h, z in RECTS) / A
    parts = {n: b * h**3 / 12 + b * h * (z - zc) ** 2 for n, b, h, z in RECTS}
    return A, zc, parts


def main():
    A, zc, parts = section()
    I = sum(parts.values())
    a_webs = sum(b * h for n, b, h, _ in RECTS if "web" in n)
    d_bend = P * L**3 / (3 * E * I)
    d_shear = P * L / (G * a_webs)
    # sensitivity of the shear term to the web-height convention (skin mid-plane distance 130 mm)
    d_shear_130 = P * L / (G * 2 * 15.0 * 130.0)
    m_root = P * L
    # web shear stress at the neutral axis (Jourawski, solid section): tau = V Q / (I * 2 t_web)
    q_na = 600.0 * 20.0 * 65.0 + 2 * 15.0 * 55.0 * 27.5
    tau_na = P * q_na / (I * 2 * 15.0)
    # Thin-walled shear-flow estimate (free warping, mid-line idealisation: skins at z = +-65,
    # webs at x = 7.5 / 592.5, skin half-width 292.5 from the symmetry plane to the corner).
    # The elementary term P L / (G A_webs) ignores the in-plane shear flow in the skins
    # (q_skin = V/I t_s 65 s), which is the same shear flow that drives shear lag.
    import numpy as np
    V, ts, tw, hz, bh = P, 20.0, 15.0, 65.0, 292.5
    s = np.linspace(0.0, bh, 4001)
    q_skin = V / I * ts * hz * s
    z = np.linspace(-hz, hz, 4001)
    q_web = q_skin[-1] + V / I * tw * (hz**2 - z**2) / 2
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    u_skin = 4 * trap(q_skin**2 / (2 * G * ts), s)   # 4 half-skins, energy per unit span
    u_web = 2 * trap(q_web**2 / (2 * G * tw), z)
    v_webs = 2 * trap(q_web, z)                       # vertical force carried (check ~ V)
    d_sh_skin, d_sh_web = 2 * u_skin * L / P, 2 * u_web * L / P   # Castigliano, V = P along the span
    out = {
        "E_MPa": E, "nu": NU, "G_MPa": G, "P_N": P, "L_mm": L,
        "area_mm2": A, "centroid_z_mm": zc,
        "I_parts_mm4": parts, "I_skins_mm4": parts["lower_skin"] + parts["upper_skin"],
        "I_webs_mm4": parts["front_web"] + parts["rear_web"], "I_mm4": I,
        "A_webs_mm2": a_webs,
        "delta_bending_mm": d_bend, "delta_shear_mm": d_shear, "delta_total_mm": d_bend + d_shear,
        "delta_shear_webs_130mm_mm": d_shear_130,
        "root_sigma_yy_outer_fibre_MPa": m_root * (150.0 - zc) / I,
        "root_sigma_yy_skin_midplane_MPa": m_root * (140.0 - zc) / I,
        "web_mean_shear_MPa": P / a_webs,
        "web_shear_neutral_axis_jourawski_MPa": tau_na,
        "thin_walled_free_warping": {
            "delta_shear_webs_mm": d_sh_web, "delta_shear_skins_inplane_mm": d_sh_skin,
            "delta_total_mm": d_bend + d_sh_web + d_sh_skin,
            "web_vertical_force_N": v_webs, "q_corner_N_per_mm": float(q_skin[-1]),
            "note": "upper bound on shear flexibility: assumes the section warps freely, "
                    "which the fully clamped root suppresses"},
    }
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
