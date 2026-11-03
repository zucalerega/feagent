"""Run a weight-space programming session: the teacher (editor) edits the student's weights.

The editor answers each turn with exactly one fenced block (as in the paper's Experiment 8 driver):
  ```python   code for the persistent kernel (wsp.kernel.Kernel)
  ```forward  a JSON forward-pass request (wsp.kernel.ForwardTool)
and, once every forward pass is used, a final reply without a block that says what it changed and why.

Two ways to drive a session:
  run    automated: the editor is Claude Opus through the Anthropic API (needs ANTHROPIC_API_KEY)
           python -m wsp.editor run --session wsp_runs/wsp_s1 --passes 25
  serve  a persistent session that reads blocks from <session>/inbox/ (so any editor, e.g. a Claude Code
         session, can drive it with `send`), and writes each result to <session>/outbox/
           python -m wsp.editor serve --session wsp_runs/wsp_s1 --passes 25 &
           python -m wsp.editor send --session wsp_runs/wsp_s1 block.md
           python -m wsp.editor message --session wsp_runs/wsp_s1       # the task message
Every block and result is appended to <session>/log.jsonl; the edits are <session>/edits/<tensor>.npy.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ARCH = {
    "qwen3_5": ("{L} layers of width {d}; every fourth layer (3, 7, 11, ...) is a full-attention layer with a gated "
                "output, the others are Gated DeltaNet linear-attention layers; each layer has a SwiGLU MLP of "
                "width {f}; RMSNorm; vocabulary {V}; input embedding and output layer tied"),
    "default": "{L} layers of width {d}; MLP width {f}; vocabulary {V}",
}


def arch_text(model) -> str:
    cfg = getattr(model.config, "text_config", model.config)
    t = ARCH.get(getattr(cfg, "model_type", "").replace("_text", ""), ARCH["default"])
    return t.format(L=cfg.num_hidden_layers, d=cfg.hidden_size, f=cfg.intermediate_size, V=cfg.vocab_size)


def message(student_name: str, arch: str, n_practice: int, passes: int, max_gen: int, n_teacher: int) -> str:
    return f"""TASK. You are given the weights of a language model, {student_name} ({arch}). It is used as a cheap meshing assistant for finite-element analysis: given a statement and a geometry description of a 2D part, it must answer with one JSON mesh recipe ({{"order", "h", "refinements"}}). The recipe is meshed with gmsh, solved with a fixed solver and scored against a fine reference solution in "x the automesher" (the gmsh default mesh = 1x; higher is better; an invalid answer = 0). Your job is to make it better at this by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it writes poor recipes and repair that with targeted changes.

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model answer held-out benchmark parts (you never see them; they include part families that are not among the practice problems) with greedy decoding, the chat template without thinking, and the same prompt format as the practice problems (PROMPT[pid] below). The mean x the automesher over the held-out parts is the result. The config, tokenizer, chat template and decoding are fixed: only weight values can change (same tensor names and shapes).

WHAT COUNTS. Edits that come from understanding the network: find out what goes wrong when it writes poor recipes, identify the components responsible (neurons, heads, directions, circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope (e.g. it ignores the part's small features, copies the example's tags, picks linear elements, writes malformed JSON, chooses absurd sizes). Out of scope and refused by save_edit: the token embedding, the output layer and the final norm, and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU): variables survive between your code blocks, like a notebook. Available: numpy, scipy, scikit-learn, PyTorch (CPU), and these names: NAMES (all tensor names), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited(), tok (the tokenizer), PRACTICE ({n_practice} practice problem ids), PROMPT[pid] (the exact user message the model gets for a practice problem), TEACHER[pid] (the best recipe an expert teacher found for it, verified by the solver, with its score; {n_teacher} problems have one) and chat(pid, answer=None) (the full chat-template text the model reads; with an answer, the text of the model answering). Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. Output is cut at 10,000 characters. Your working folder is the kernel's current directory.
2. A forward-pass tool with a budget of {passes} forward passes in total. Each counts as one pass:
 (a) GENERATE: reply with exactly one block that starts with ```forward and contains JSON: {{"type": "generate", "model": "original", "edited" or "both", "problems": ["<practice id>", ...], "n": samples per problem (1 with greedy decoding), "temperature": 0 (greedy, as in the evaluation) or > 0}}. At most {max_gen} answers per model. Every answer is parsed, meshed, solved and scored. You get the mean and median x the automesher, the share of valid answers, the failure counts, and with "both" a paired comparison (mean difference edited - original with its standard error, and how many problems got better or worse); the answers and scores are in gen_<k>.jsonl (with "both": gen_<k>_original.jsonl and gen_<k>_edited.jsonl).
 (b) TEACHER_FORCED: reply with a ```forward block containing {{"type": "teacher_forced", "model": "original" or "edited", "texts": ["raw text", ...] or "chats": [{{"pid": "<practice id>", "answer": "<the assistant's answer text, e.g. TEACHER[pid]['answer']>"}}, ...], "return": {{"token_logprobs": true, "topk": k (at most 50), "hidden_states": [layer indices 0..L; 0 = embeddings, l = after layer l], "mlp_act": [layers: the inputs of each MLP's down_proj]}}}}. At most 64 texts. Results are .npy files (right-padded) with meta.json in fwd_<k>/.
A request that fails does not count against the budget.

HOW TO WORK. Diagnose first: look at what the model answers and why it scores badly, then localize the cause in the weights, then edit, then measure the edit with paired "both" generations on practice problems you did not use to design it. Keep notes.md in your folder up to date.

EXPLAIN EVERY STEP. In every reply, before your block, say in plain English (3 to 8 sentences) what the last result showed, what you conclude, what you will do now and why, and what outcome would confirm or refute your hypothesis.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward tool: do not implement or simulate the network's computation in the kernel; (c) internet access. Allowed: any analysis of the weights and of forward-pass results, any use of the practice problems and teacher recipes, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {passes} forward passes are used; then make your final choice (your current edits are submitted) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help."""


BLOCK = re.compile(r"```(python|forward)\s*\n(.*?)```", re.S)


def parse_block(text: str) -> tuple[str | None, str]:
    m = BLOCK.search(text)
    return (m.group(1), m.group(2)) if m else (None, text)


class Session:
    def __init__(self, session: str | Path, model: str, passes: int, practice_split: str = "train",
                 teacher_file: str | None = None):
        from wsp import problems, task
        from wsp.kernel import ForwardTool, Kernel, WeightStore
        from wsp.student import Student
        self.dir = Path(session)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.s = Student(model)
        practice = problems.train_ids() if practice_split == "train" else practice_split.split(",")
        self.store = WeightStore(self.s.model.state_dict(), self.dir / "edits")
        teacher = load_teacher(teacher_file)
        teacher = {p: v for p, v in teacher.items() if p in practice}
        prompts = {p: task.prompt(p) for p in practice}
        extra = {"tok": self.s.tok, "PRACTICE": practice, "PROMPT": prompts, "TEACHER": teacher,
                 "chat": lambda pid, answer=None: self.s.chat_text(prompts[pid], answer=answer)}
        self.kernel = Kernel(self.store, self.dir / "work", extra)
        self.fwd = ForwardTool(self.s, self.store, self.dir / "work", practice, passes)
        self.msg = message(self.s.name, arch_text(self.s.model), len(practice), passes, ForwardTool.MAX_GEN,
                           len(teacher))
        (self.dir / "message.txt").write_text(self.msg)
        self.log = self.dir / "log.jsonl"

    def step(self, text: str) -> dict:
        kind, body = parse_block(text)
        t0 = time.perf_counter()
        if kind == "python":
            res = self.kernel.run(body)
        elif kind == "forward":
            try:
                req = json.loads(body)
            except json.JSONDecodeError as e:
                res = {"ok": False, "error": f"forward block is not JSON: {e}"}
            else:
                res = self.fwd(req)
        else:
            res = {"final": self.fwd.used >= self.fwd.budget,
                   "note": None if self.fwd.used >= self.fwd.budget else
                   f"no code block: the session continues until all {self.fwd.budget} passes are used "
                   f"({self.fwd.budget - self.fwd.used} left)"}
        res["seconds"] = round(time.perf_counter() - t0, 1)
        res["passes_used"] = self.fwd.used
        with open(self.log, "a") as fh:
            fh.write(json.dumps({"t": time.time(), "kind": kind, "text": text, "result": res}, default=str) + "\n")
        if kind is None and res.get("final"):
            (self.dir / "final.md").write_text(text)
            (self.dir / "diff.json").write_text(json.dumps(self.store.diff(), indent=1))
        return res


def load_teacher(path: str | None) -> dict:
    """{pid: {"answer": JSON text, "x_auto": float}} from the demos file (wsp.demos)."""
    from wsp import task
    p = Path(path) if path else Path(__file__).with_name("data") / "demos.jsonl"
    out = {}
    if p.exists():
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if r.get("rank", 0) == 0:
                out[r["pid"]] = {"answer": task.dumps(r["spec"]), "x_auto": r["x_auto"]}
    return out


def result_text(res: dict) -> str:
    """What the editor reads back after a block."""
    if "output" in res or "error" in res and "passes_left" not in res:
        s = res.get("output") or ""
        if res.get("error"):
            s += ("\n" if s else "") + f"ERROR: {res['error']}"
        return (s or "(no output)") + f"\n[code runs used: {res.get('runs')}; forward passes used: {res['passes_used']}]"
    return json.dumps(res, indent=1)


# --------------------------------------------------------------------------- drivers


def serve(args) -> None:
    sess = Session(args.session, args.model, args.passes, teacher_file=args.teacher)
    inbox, outbox = sess.dir / "inbox", sess.dir / "outbox"
    inbox.mkdir(exist_ok=True)
    outbox.mkdir(exist_ok=True)
    (sess.dir / "READY").write_text(str(time.time()))
    print(f"serving {sess.dir} ({args.passes} passes)", flush=True)
    while True:
        for f in sorted(inbox.glob("*.md")):
            out = outbox / (f.stem + ".json")
            if out.exists():
                continue
            text = f.read_text()
            if text.strip() == "STOP":
                (sess.dir / "diff.json").write_text(json.dumps(sess.store.diff(), indent=1))
                return
            res = sess.step(text)
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps({"result": res, "text": result_text(res)}, default=str))
            tmp.replace(out)
            print(f"{f.stem}: {res.get('ok', res.get('final'))} passes_used={res['passes_used']}", flush=True)
        time.sleep(0.3)


def send(args) -> None:
    d = Path(args.session)
    inbox, outbox = d / "inbox", d / "outbox"
    n = len(list(inbox.glob("*.md")))
    f = inbox / f"{n:04d}.md"
    f.write_text(Path(args.file).read_text() if args.file != "-" else sys.stdin.read())
    out = outbox / f"{n:04d}.json"
    t0 = time.time()
    while not out.exists():
        if time.time() - t0 > args.timeout:
            raise SystemExit(f"no answer after {args.timeout} s (the server keeps working; check {out} later)")
        time.sleep(0.5)
    print(json.loads(out.read_text())["text"])


def run(args) -> None:
    """The editor is Claude (Anthropic API): one block per turn until the passes are used."""
    import anthropic
    sess = Session(args.session, args.model, args.passes, teacher_file=args.teacher)
    client = anthropic.Anthropic()
    msgs = [{"role": "user", "content": sess.msg}]
    for turn in range(args.max_turns):
        resp = client.messages.create(model=args.editor, max_tokens=32000, messages=msgs,
                                      thinking={"type": "adaptive"}, output_config={"effort": args.effort})
        text = "".join(b.text for b in resp.content if b.type == "text")
        msgs.append({"role": "assistant", "content": resp.content})
        res = sess.step(text)
        print(f"turn {turn}: passes {res['passes_used']}/{args.passes}", flush=True)
        if res.get("final"):
            break
        msgs.append({"role": "user", "content": result_text(res)})


def main(argv=None) -> int:
    from wsp.student import DEFAULT_MODEL
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("serve", "run", "message"):
        p = sub.add_parser(name)
        p.add_argument("--session", required=True)
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--passes", type=int, default=25)
        p.add_argument("--teacher", default=None)
        p.add_argument("--editor", default="claude-opus-5-5")
        p.add_argument("--effort", default="high")
        p.add_argument("--max-turns", type=int, default=400)
    p = sub.add_parser("send")
    p.add_argument("--session", required=True)
    p.add_argument("file")
    p.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args(argv)
    if a.cmd == "serve":
        serve(a)
    elif a.cmd == "send":
        send(a)
    elif a.cmd == "run":
        run(a)
    else:
        print(Session(a.session, a.model, a.passes, teacher_file=a.teacher).msg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
