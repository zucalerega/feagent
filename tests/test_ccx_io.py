"""CalculiX I/O: .frd/.dat readers, regression comparator, deck scanner, runner."""

import shutil

import numpy as np
import pytest

from feabench.ccx import compare_dat, ensure_output_requests, read_buckling, read_eigen, read_frd, run_ccx, scan_deck
from feabench.ccx.inp import read_deck_materials, read_deck_nodes
from feabench.paths import CALCULIX_DIR

DECKS = CALCULIX_DIR / "decks"
needs_ccx = pytest.mark.skipif(shutil.which("ccx") is None, reason="CalculiX not installed")


def _run(tmp_path, name):
    shutil.copy(DECKS / f"{name}.inp", tmp_path / f"{name}.inp")
    rr = run_ccx(tmp_path / f"{name}.inp")
    assert rr.ok, rr.errors
    return rr


@needs_ccx
def test_static_deck_matches_upstream_reference(tmp_path):
    rr = _run(tmp_path, "beamp")
    assert rr.cpu_s > 0 and rr.n_equations and rr.ccx_version
    cmp = compare_dat(tmp_path / "beamp.dat", DECKS / "beamp.dat.ref")
    assert cmp.passed, cmp


@needs_ccx
def test_comparator_flags_perturbed_output(tmp_path):
    _run(tmp_path, "beamp")
    text = (tmp_path / "beamp.dat").read_text().splitlines()
    done = False
    for i, line in enumerate(text):  # perturb the first significant number by 1 %
        if "time" in line:
            continue
        for tok in line.split():
            try:
                v = float(tok)
            except ValueError:
                continue
            if "E" in tok and abs(v) > 1.0:
                text[i] = line.replace(tok, f"{v * 1.01:.6E}", 1)
                done = True
                break
        if done:
            break
    assert done
    (tmp_path / "bad.dat").write_text("\n".join(text) + "\n")
    assert not compare_dat(tmp_path / "bad.dat", DECKS / "beamp.dat.ref").passed


@needs_ccx
def test_frd_reader_hex20_and_fields(tmp_path):
    src = (DECKS / "beamp.inp").read_text()
    (tmp_path / "m.inp").write_text(ensure_output_requests(src))
    assert run_ccx(tmp_path / "m.inp").ok
    r = read_frd(tmp_path / "m.frd")
    assert {"DISP", "STRESS", "FORC"} <= set(r.names())
    hexes = r.elements["hex20"]
    row = r.node_row()
    X = r.coords[[row[n] for n in hexes[0]]]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    for k, (a, b) in enumerate(edges):  # input node order restored (mid-edge nodes at edge centres)
        assert np.allclose(X[8 + k], (X[a] + X[b]) / 2)
    # internal forces sum to zero over the model (applied load is 9 N; .frd keeps 6 digits)
    assert np.allclose(np.nansum(r.field_on_nodes("FORC"), axis=0), 0, atol=1e-5 * 9.0)


@needs_ccx
def test_eigen_and_buckling_tables(tmp_path):
    _run(tmp_path, "beam8f")
    modes = read_eigen(tmp_path / "beam8f.dat")[0]
    assert len(modes) >= 4 and all(m.frequency > 0 for m in modes)
    assert compare_dat(tmp_path / "beam8f.dat", DECKS / "beam8f.dat.ref").passed
    _run(tmp_path, "beamb")
    factors = read_buckling(tmp_path / "beamb.dat")[0]
    assert factors[0][0] == 1 and factors[0][1] > 0


def test_deck_scanner_and_output_injection():
    info = scan_deck(DECKS / "beamp.inp")
    assert info.procedures == ["static"] and "linear_solid" in info.tags()
    assert info.n_nodes > 0 and info.element_types == {"C3D20R": 32}
    text = "*STEP\n*STATIC\n*CLOAD\n1,1,1.\n*END STEP\n*STEP\n*FREQUENCY\n4\n*END STEP\n"
    out = ensure_output_requests(text)
    assert out.count("*NODE FILE") == 2 and "U, RF" in out and out.count("*END STEP") == 2
    mats = read_deck_materials(DECKS / "beamp.inp")
    assert list(mats.values())[0]["E"] == pytest.approx(210000.0)
    nodes = read_deck_nodes(DECKS / "beamp.inp")
    assert len(nodes) == info.n_nodes
