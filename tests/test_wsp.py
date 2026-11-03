"""wsp: the student's task, scoring, the weight-space programming harness and the training pipeline.

The model tests use wsp.tiny (a random 4-layer Qwen3.5 with a tokenizer trained on our prompts), so they
run offline in about a minute; they check plumbing, not quality.
"""

from __future__ import annotations

import json

import pytest

from wsp import problems, task

PID = "train_stepped_shaft_axi_Dd1p2_rd0p05"     # small automesher mesh: cheap to score


def test_splits_disjoint():
    tr, te = set(problems.train_ids()), set(problems.test_ids())
    assert not tr & te
    assert all(p.startswith("train_") for p in tr)
    assert set(problems.TEST_OUT_OF_FAMILY) <= te
    # out-of-family test parts have no training variant of their family
    fams = {problems.family(p) for p in tr}
    assert not {problems.family(p) for p in problems.TEST_OUT_OF_FAMILY} & fams
    assert not problems.DEGENERATE & te


def test_parse_messy_answers():
    spec, err = task.parse('Sure! ```json\n{"order": 2, "h": 40, "refinements": []}\n``` done')
    assert err is None and spec["h"] == 40
    spec, _ = task.parse('<think>x {"a": 1</think>{"order": 1, "h": 9, "refinements": [{"curve": "c3", "size": 1, '
                         '"radius": 5}]} trailing {')
    assert spec["order"] == 1 and spec["refinements"][0]["curve"] == "c3"
    assert task.parse("no json here")[0] is None


def test_request_units_and_tags():
    geo = task.geometry("cooks_membrane")
    k = next(iter(geo.corners))
    c = next(iter(geo.curves))
    req = task.to_request("cooks_membrane", {"order": 2, "h": 50, "refinements": [
        {"corner": f"k{k}", "size": 5, "radius": 100}, {"curve": c, "size": 10, "radius": 50}]})
    assert req["h"] == pytest.approx(50 * geo.scale)
    assert req["refinements"][0]["point"] == list(geo.corners[k]["point"])
    assert req["refinements"][1]["edge"] == c
    with pytest.raises(task.SpecError):
        task.to_request("cooks_membrane", {"order": 2, "h": 50, "refinements": [{"corner": 999, "size": 1, "radius": 1}]})
    with pytest.raises(task.SpecError):
        task.to_request("cooks_membrane", {"order": 3, "h": 50})
    assert task.canonical({"order": 2, "h": 50.0004, "refinements": [{"curve": "c2", "size": 2.50, "radius": 30.0}]}) \
        == {"order": 2, "h": 50, "refinements": [{"curve": 2, "size": 2.5, "radius": 30}]}


def test_prompt_mentions_scoring_and_format():
    p = task.prompt(PID, playbook="- refine fillets")
    assert "PART GEOMETRY" in p and "x the automesher" in p and "ANSWER." in p and "refine fillets" in p
    assert "Report:" not in p


def test_score_valid_and_invalid():
    good = task.score(PID, {"order": 2, "h": 60, "refinements": []}, use_cache=False)
    assert good["valid"] and good["x_auto"] > 0 and good["dofs"] > 0
    bad = task.score(PID, {"order": 2, "h": 60, "refinements": [{"corner": 99, "size": 1, "radius": 9}]})
    assert not bad["valid"] and bad["x_auto"] == 0
    capped = task.score(PID, {"order": 2, "h": 0.5, "refinements": []})
    assert not capped["valid"] and "node cap" in capped["error"]


def test_kernel_guard():
    from wsp.kernel import ProtocolRefused, validate_code
    validate_code("import numpy as np\nx = load('a').abs().mean()\nsave_edit('a', load('a') * 0.5 + 1)")
    for bad in ("loss.backward()", "import torch.optim as o", "from torch import nn", "torch.nn.Linear(2, 2)",
                "import subprocess", "x = torch.zeros(2, requires_grad=True)", "exec('1')",
                "torch.set_grad_enabled(True)", "import transformers", "model.forward(x)"):
        with pytest.raises(ProtocolRefused):
            validate_code(bad)


def test_weight_store_rules(tmp_path):
    import torch
    from wsp.kernel import EditRefused, WeightStore
    tensors = {"model.embed_tokens.weight": torch.randn(5, 4), "model.layers.0.mlp.down_proj.weight": torch.randn(4, 6),
               "model.norm.weight": torch.ones(4), "model.layers.0.input_layernorm.weight": torch.ones(4)}
    st = WeightStore(tensors, tmp_path / "e")
    assert st.frozen == {"model.embed_tokens.weight", "model.norm.weight"}
    with pytest.raises(EditRefused):
        st.save_edit("model.embed_tokens.weight", torch.zeros(5, 4))
    W = st.load_original("model.layers.0.mlp.down_proj.weight")
    with pytest.raises(EditRefused):
        st.save_edit("model.layers.0.mlp.down_proj.weight", W * 1.5)            # a global knob
    W2 = W.clone()
    W2[:, 2] = 0.0                                                               # ablate one neuron
    r = st.save_edit("model.layers.0.mlp.down_proj.weight", W2)
    assert r["changed"] and st.edited() == ["model.layers.0.mlp.down_proj.weight"]
    assert torch.equal(st.load("model.layers.0.mlp.down_proj.weight"), W2)
    assert st.diff()["changed_entries"] == int((W[:, 2] != 0).sum())
    assert st.save_edit("model.layers.0.mlp.down_proj.weight", W)["changed"] is False   # back to the original
    assert st.edited() == []


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    from wsp import tiny
    return tiny.build(tmp_path_factory.mktemp("tiny") / "model")


@pytest.mark.slow
def test_tiny_pipeline(tiny, tmp_path):
    import torch
    from wsp import editor, sft
    from wsp.student import Student

    s = Student(tiny, threads=2)
    ans = s.generate([task.prompt(PID)], max_new_tokens=8)
    assert len(ans) == 1 and isinstance(ans[0][0], str)

    # a WSP session: kernel analysis + an edit + forward passes (teacher-forced and generate)
    sess = editor.Session(tmp_path / "sess", tiny, passes=3, teacher_file=str(tmp_path / "none.jsonl"))
    name = "model.layers.0.mlp.down_proj.weight"
    r = sess.step(f"look at one tensor\n```python\nW = load('{name}')\nW.shape\n```")
    assert r["ok"] and "torch.Size" in r["output"]
    r = sess.step(f"```python\nW = load('{name}').clone()\nW[:, 0] = 0\nsave_edit('{name}', W)\n```")
    assert r["ok"] and sess.store.edited() == [name]
    r = sess.step('```python\nimport torch\nx = torch.zeros(3, requires_grad=True)\n```')
    assert not r["ok"] and "ProtocolRefused" in r["error"] and r["runs"] == 2       # refused: not counted
    req = {"type": "teacher_forced", "model": "edited", "chats": [{"pid": PID, "answer": '{"order": 2}'}],
           "return": {"token_logprobs": True, "topk": 3, "hidden_states": [0, 2], "mlp_act": [1]}}
    r = sess.step("```forward\n" + json.dumps(req) + "\n```")
    assert r["ok"] and r["passes_left"] == 2
    files = r["files"]
    assert files["hidden_2.npy"][2] == 64 and files["mlp_act_1.npy"][2] == 128
    # the original weights are back in the model after the pass
    assert torch.equal(sess.s.model.state_dict()[name], sess.store.load_original(name).to(torch.float32))
    r = sess.step('```forward\n{"type": "generate", "model": "both", "problems": ["%s"], "max_tokens": 6}\n```' % PID)
    assert r["ok"] and "paired" in r and r["edited"]["valid"] == 0.0
    r = sess.step("I am done")
    assert r["final"] is False and "continues" in r["note"]
    r = sess.step('```forward\n{"type": "bogus"}\n```')
    assert not r["ok"] and r["passes_used"] == 2                                   # a failed request is free

    # LoRA distillation: a couple of steps on one example
    s2 = Student(tiny, threads=2)
    info = sft.train(s2, [{"pid": PID, "answer": '{"order": 2, "h": 60, "refinements": []}'}] * 2, tmp_path / "sft",
                     epochs=1, accum=1, lora_layers=2, log=lambda m: None)
    assert info["steps"] == 2 and (tmp_path / "sft" / "adapter").exists()
    s3 = Student(tiny, adapter=tmp_path / "sft" / "adapter", threads=2)
    assert s3.generate([task.prompt(PID)], max_new_tokens=4)


def test_evaluate_recipes(tmp_path):
    from wsp import evaluate
    summary = evaluate.run("recipes", [PID], tmp_path, answers={PID: ['{"order": 2, "h": 60, "refinements": []}',
                                                                      "nonsense"]}, workers=1)
    assert summary["all"]["n"] == 2 and summary["all"]["valid"] == 0.5
    rows = [json.loads(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    assert rows[0]["mesh_path"] and rows[1]["error"].startswith("no JSON")
