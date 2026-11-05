"""Weight-space programming (WSP) harness: an editor model reads and edits the student's weights directly.

After the harness of Experiment 8 in "Sparks of Weight-Space Programming" (Takehi, Teoh, Zhang, Khattab,
MIT CSAIL, 2026-11-02) as replicated by openwsp/openwsp (MIT license); rewritten here for the meshing task.
The editor (the teacher, Claude Opus) gets:

  * a persistent Python kernel over the weights: NAMES, load(name), load_original(name),
    save_edit(name, tensor), drop_edit(name), edited(), tok, np, torch, plus the task's PRACTICE ids,
    PROMPT[pid] (the student's message), TEACHER[pid] (the best solver-verified recipe and its score)
    and chat(pid, answer=None) (the chat-template text). Gradients are off and an AST guard refuses
    training, autograd, model layers, process / network modules and dynamic exec (a protocol guard, not
    a security boundary: run untrusted editors in a sandbox);
  * a forward-pass tool with a budget of passes: GENERATE (the original and/or edited student answers
    practice problems; every answer is meshed, solved and scored, x the automesher) or TEACHER_FORCED
    (the model reads given texts / chats and returns token log-probs, top-k, residual-stream hidden
    states and MLP down_proj inputs as .npy files).

Rules (as in the paper): no gradient descent or other training, no backward passes or optimizers, no
iterative fitting of weights to a loss; no running the network except through the forward tool. Any
edit computed by analysis is allowed. Frozen: the token embedding, the output layer and the final norm.
A whole-tensor rescale is refused (a global knob).
"""

from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import signal
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np


class EditRefused(ValueError):
    pass


class ProtocolRefused(ValueError):
    pass


class KernelTimeout(TimeoutError):
    pass


def frozen_names(names) -> set[str]:
    """Embeddings, output layer and final norm (any prefix: model., model.language_model., ...)."""
    out = set()
    for n in names:
        final_norm = n.endswith(".norm.weight") and ".layers." not in n
        if n.endswith("embed_tokens.weight") or n == "lm_head.weight" or final_norm:
            out.add(n)
    return out


# --------------------------------------------------------------------------- weight store


class WeightStore:
    """The student's tensors (read-only originals) and the editor's float32 edits (one .npy each)."""

    def __init__(self, tensors: dict, edit_dir: str | Path, *, refuse_rescale: bool = True):
        import torch
        self._orig = {k: v.detach() for k, v in tensors.items()}
        self.names = sorted(self._orig)
        self.frozen = frozen_names(self.names)
        self.refuse_rescale = refuse_rescale
        self.edit_dir = Path(edit_dir)
        self.edit_dir.mkdir(parents=True, exist_ok=True)
        self._torch = torch

    def _check(self, name: str) -> None:
        if name not in self._orig:
            raise EditRefused(f"unknown tensor {name!r}")

    def load_original(self, name: str):
        self._check(name)
        return self._orig[name].float().clone()

    def load(self, name: str):
        self._check(name)
        f = self.edit_dir / f"{name}.npy"
        if f.exists():
            return self._torch.from_numpy(np.load(f)).float()
        return self.load_original(name)

    def save_edit(self, name: str, tensor) -> dict:
        torch = self._torch
        self._check(name)
        if name in self.frozen:
            raise EditRefused(f"{name} is frozen (embeddings, output layer and final norm are out of scope)")
        t = torch.as_tensor(tensor).detach().float().cpu().contiguous()
        if t.requires_grad:
            raise EditRefused("edits must not carry gradients")
        orig = self.load_original(name)
        if tuple(t.shape) != tuple(orig.shape):
            raise EditRefused(f"shape {tuple(t.shape)} != {tuple(orig.shape)}")
        if not torch.isfinite(t).all() or not torch.isfinite(t.bfloat16().float()).all():
            raise EditRefused("non-finite values")
        if torch.equal(t, orig):
            self.drop_edit(name)
            return {"name": name, "changed": False}
        if self.refuse_rescale:
            nz = orig.abs() > 0
            if nz.any():
                s = float((t[nz] / orig[nz]).median())
                if torch.allclose(t, orig * s, rtol=1e-5, atol=1e-7):
                    raise EditRefused("a whole-tensor rescale is a global knob, not a targeted edit")
        f = self.edit_dir / f"{name}.npy"
        tmp = f.with_suffix(".tmp.npy")
        np.save(tmp, t.numpy())
        os.replace(tmp, f)
        changed = int((t != orig).sum())
        return {"name": name, "changed": True, "changed_entries": changed,
                "relative_change": float((t - orig).norm() / max(orig.norm(), 1e-12))}

    def drop_edit(self, name: str) -> bool:
        self._check(name)
        f = self.edit_dir / f"{name}.npy"
        if f.exists():
            f.unlink()
            return True
        return False

    def edited(self) -> list[str]:
        return sorted(p.name[:-4] for p in self.edit_dir.glob("*.npy"))

    def snapshot(self, tag: str) -> list[str]:
        """Copy the current edits to <edit_dir>/../snapshots/<tag>/ (the kernel cannot import shutil)."""
        import re
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", tag):
            raise EditRefused("snapshot names are letters, digits, _ . -")
        d = self.edit_dir.parent / "snapshots" / tag
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.npy"):
            old.unlink()
        for name in self.edited():
            np.save(d / f"{name}.npy", np.load(self.edit_dir / f"{name}.npy"))
        return self.edited()

    def restore(self, tag: str) -> list[str]:
        """Make a snapshot the current edits again."""
        d = self.edit_dir.parent / "snapshots" / tag
        if not d.is_dir():
            raise EditRefused(f"no snapshot {tag!r}")
        for f in self.edit_dir.glob("*.npy"):
            f.unlink()
        for f in d.glob("*.npy"):
            self.save_edit(f.name[:-4], self._torch.from_numpy(np.load(f)))
        return self.edited()

    def diff(self) -> dict:
        """Per edited tensor: changed entries and relative change; the total share of weights changed."""
        rows, total = {}, 0
        n_all = sum(int(v.numel()) for v in self._orig.values())
        for name in self.edited():
            t, o = self.load(name), self.load_original(name)
            ch = int((t != o).sum())
            total += ch
            rows[name] = {"changed_entries": ch, "relative_change": float((t - o).norm() / max(o.norm(), 1e-12)),
                          "sha256": hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]}
        return {"tensors": rows, "changed_entries": total, "share_of_weights": total / max(n_all, 1)}


# --------------------------------------------------------------------------- protocol guard

_FORBIDDEN_IMPORTS = {"requests", "httpx", "urllib", "http", "socket", "ftplib", "aiohttp", "subprocess",
                      "multiprocessing", "transformers", "vllm", "tensorflow", "jax", "peft", "anthropic",
                      "shutil", "ctypes"}
_FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "breakpoint", "input", "open_model"}
_FORBIDDEN_ATTRS = {"backward", "autograd", "optim", "optimizer", "enable_grad", "forward", "system", "popen",
                    "spawn", "fork", "forkpty", "execl", "execv", "execve", "execvp", "execvpe", "exit", "_exit",
                    "quit", "rmtree", "remove", "unlink"}
_LAYER_CALLS = {"Linear", "Embedding", "Conv1d", "Conv2d", "RNN", "GRU", "LSTM", "Transformer", "Sequential",
                "Module", "linear", "embedding", "conv1d", "scaled_dot_product_attention", "silu", "gelu"}
_GRAD_TOGGLES = {"set_grad_enabled", "requires_grad_"}


def validate_code(code: str) -> ast.Module:
    """Refuse training, model execution, process / network tools (after openwsp's guard)."""
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                parts = m.split(".")
                if parts[0] in _FORBIDDEN_IMPORTS or any(p in {"optim", "autograd"} for p in parts):
                    raise ProtocolRefused(f"import not allowed: {m}")
                if m.startswith("torch.nn") and m != "torch.nn.functional":
                    raise ProtocolRefused("model layers are not available in the kernel; use the forward tool")
            if isinstance(node, ast.ImportFrom) and node.module in ("torch", "torch.nn.functional"):
                for a in node.names:
                    if a.name in {"optim", "autograd", "nn", "enable_grad"} | _LAYER_CALLS | _GRAD_TOGGLES:
                        raise ProtocolRefused(f"not available: {a.name}")
        if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_ATTRS:
            raise ProtocolRefused(f"operation not allowed: {node.attr}")
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else f.id if isinstance(f, ast.Name) else None
            if isinstance(f, ast.Name) and f.id in _FORBIDDEN_CALLS:
                raise ProtocolRefused(f"not allowed: {f.id}")
            if name == "getattr" and len(node.args) > 1 and isinstance(node.args[1], ast.Constant) \
                    and node.args[1].value in _FORBIDDEN_ATTRS:
                raise ProtocolRefused("dynamic access to a forbidden operation")
            if name in _LAYER_CALLS:
                raise ProtocolRefused("running model layers in the kernel is forbidden; use the forward tool")
            if name in _GRAD_TOGGLES:
                args = list(node.args) + [k.value for k in node.keywords]
                if not args or not all(isinstance(a, ast.Constant) and a.value is False for a in args):
                    raise ProtocolRefused("enabling gradients is forbidden")
            for k in node.keywords:
                if k.arg == "requires_grad" and not (isinstance(k.value, ast.Constant) and k.value.value is False):
                    raise ProtocolRefused("tensors with gradients are forbidden")
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Attribute) and t.attr == "requires_grad" for t in targets):
                raise ProtocolRefused("changing gradient flags is forbidden")
    return tree


class _Bounded(io.TextIOBase):
    def __init__(self, limit: int):
        self.limit, self.parts, self.n, self.cut = limit, [], 0, False

    def write(self, s: str) -> int:
        room = max(0, self.limit - self.n)
        if room:
            self.parts.append(s[:room])
            self.n += min(room, len(s))
        self.cut |= len(s) > room
        return len(s)

    def getvalue(self) -> str:
        return "".join(self.parts) + ("\n[output truncated]" if self.cut else "")


@contextmanager
def _deadline(seconds: float):
    if threading.current_thread() is not threading.main_thread() or not hasattr(signal, "setitimer"):
        yield
        return

    def expired(*_):
        raise KernelTimeout(f"execution exceeded {seconds:g} s")

    old = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


class Kernel:
    """A persistent namespace: variables survive between code blocks, like a notebook."""

    def __init__(self, store: WeightStore, work_dir: str | Path, extra: dict | None = None, *,
                 max_runs: int = 300, timeout: float = 900, output_limit: int = 10_000):
        import torch
        self.store, self.work = store, Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.max_runs, self.timeout, self.limit, self.runs = max_runs, timeout, output_limit, 0
        self.ns: dict[str, Any] = {"__name__": "__wsp_kernel__", "NAMES": store.names, "load": store.load,
                                   "load_original": store.load_original, "save_edit": store.save_edit,
                                   "drop_edit": store.drop_edit, "edited": store.edited, "snapshot": store.snapshot,
                                   "restore": store.restore, "np": np, "torch": torch, **(extra or {})}

    def run(self, code: str) -> dict:
        import torch
        t0 = time.perf_counter()
        if self.runs >= self.max_runs:
            return {"ok": False, "output": "", "error": "code-run budget exhausted", "runs": self.runs}
        try:
            tree = validate_code(code)
        except (ProtocolRefused, SyntaxError) as e:     # a refused block costs nothing
            return {"ok": False, "output": "", "error": f"{type(e).__name__}: {e}", "runs": self.runs}
        self.runs += 1
        out, err, cwd = _Bounded(self.limit), None, os.getcwd()
        try:
            os.chdir(self.work)
            with redirect_stdout(out), redirect_stderr(out), torch.no_grad(), _deadline(self.timeout):
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                    ast.fix_missing_locations(last)
                exec(compile(tree, "<wsp>", "exec"), self.ns, self.ns)  # noqa: S102  the kernel's purpose
                if last is not None:
                    v = eval(compile(last, "<wsp>", "eval"), self.ns, self.ns)  # noqa: S307
                    self.ns["_"] = v
                    if v is not None:
                        print(repr(v))
        except (Exception, SystemExit, KeyboardInterrupt) as e:  # noqa: BLE001  the cell's error is its output
            err = f"{type(e).__name__}: {e}"[: self.limit]
        finally:
            os.chdir(cwd)
        return {"ok": err is None, "output": out.getvalue(), "error": err, "runs": self.runs,
                "seconds": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------------------- forward tool


class ForwardTool:
    """The only way to run the student: budgeted GENERATE and TEACHER_FORCED passes."""

    MAX_GEN = 64            # answers per generate pass (problems x samples); per model with "both"
    MAX_TF_TEXTS = 64
    MAX_TF_TOKENS = 131072

    def __init__(self, student, store: WeightStore, work_dir: str | Path, practice: list[str], budget: int,
                 score_workers: int = 3):
        self.s, self.store, self.work = student, store, Path(work_dir)
        self.practice, self.budget, self.used, self.workers = list(practice), budget, 0, score_workers

    # swap the editor's tensors in for one pass, then restore the originals
    @contextmanager
    def _weights(self, which: str):
        import torch
        sd = self.s.model.state_dict()
        names = self.store.edited() if which == "edited" else []
        try:
            with torch.no_grad():
                for n in names:
                    sd[n].copy_(self.store.load(n).to(sd[n].dtype))
            yield
        finally:
            with torch.no_grad():
                for n in names:
                    sd[n].copy_(self.store.load_original(n).to(sd[n].dtype))

    def __call__(self, req: dict) -> dict:
        if self.used >= self.budget:
            return {"ok": False, "error": f"forward budget exhausted ({self.budget} passes)"}
        kind = req.get("type")
        try:
            if kind == "generate":
                out = self._generate(req)
            elif kind == "teacher_forced":
                out = self._teacher_forced(req)
            else:
                return {"ok": False, "error": "type must be 'generate' or 'teacher_forced'"}
        except (ValueError, KeyError, TypeError) as e:      # a refused request does not count
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        self.used += 1
        return {"ok": True, "pass": self.used, "passes_left": self.budget - self.used, **out}

    def _generate(self, req: dict) -> dict:
        from wsp import task
        which = req.get("model", "edited")
        if which not in ("original", "edited", "both"):
            raise ValueError("model must be original, edited or both")
        pids = list(req.get("problems") or [])
        bad = [p for p in pids if p not in self.practice]
        if bad:
            raise ValueError(f"not practice problems: {bad[:5]}")
        n = int(req.get("n", 1))
        temp = float(req.get("temperature", 0.0))
        n = 1 if temp == 0 else n
        if not pids or len(pids) * n > self.MAX_GEN:
            raise ValueError(f"1 to {self.MAX_GEN} answers per model (problems x n)")
        k = self.used + 1
        res = {}
        for model in (["original", "edited"] if which == "both" else [which]):
            with self._weights(model):
                answers = self.s.generate([task.prompt(p) for p in pids], temperature=temp, n=n,
                                          max_new_tokens=int(req.get("max_tokens", 320)), seed=k)
            jobs = [(p, task.parse(a)[0], None) for p, al in zip(pids, answers) for a in al]
            scored = task.score_many(jobs, workers=self.workers)
            rows, i = [], 0
            for p, al in zip(pids, answers):
                for a in al:
                    r = scored[i]
                    i += 1
                    if task.parse(a)[0] is None:
                        r = task.invalid(p, "no JSON object in the answer")
                    rows.append({"pid": p, "answer": a, "x_auto": r["x_auto"], "valid": r["valid"],
                                 "error": r.get("error"), "gain": r.get("gain"), "cost": r.get("cost"),
                                 "nodes": r.get("nodes")})
            name = f"gen_{k}.jsonl" if which != "both" else f"gen_{k}_{model}.jsonl"
            with open(self.work / name, "w") as fh:
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
            xs = np.array([r["x_auto"] for r in rows])
            fails: dict[str, int] = {}
            for r in rows:
                if not r["valid"]:
                    key = (r["error"] or "invalid").split(":")[0][:60]
                    fails[key] = fails.get(key, 0) + 1
            res[model] = {"file": name, "mean_x_auto": float(xs.mean()), "median_x_auto": float(np.median(xs)),
                          "valid": float(np.mean([r["valid"] for r in rows])), "n": len(rows),
                          "mean_answer_chars": float(np.mean([len(r["answer"]) for r in rows])), "failures": fails}
            res[model]["_x"] = xs
        if which == "both":
            d = res["edited"]["_x"] - res["original"]["_x"]
            res["paired"] = {"mean_diff": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1
                             else None, "better": int((d > 0.05).sum()), "worse": int((d < -0.05).sum())}
        for v in res.values():
            v.pop("_x", None)
        return res

    def _teacher_forced(self, req: dict) -> dict:
        import torch
        which = req.get("model", "edited")
        if which not in ("original", "edited"):
            raise ValueError("model must be original or edited")
        tok = self.s.tok
        enc, prompt_lens = [], []
        if req.get("texts") is not None:
            for t in req["texts"]:
                enc.append(torch.tensor(tok(t, add_special_tokens=False)["input_ids"]))
                prompt_lens.append(0)
        elif req.get("chats") is not None:
            from wsp import task
            for c in req["chats"]:
                if not (isinstance(c, dict) and c.get("pid") in self.practice):
                    raise ValueError('chats are {"pid": <practice id>, "answer": text or null}')
                # prompt tokens + answer tokens: what generation continues from (as in wsp.sft)
                p_ids = tok(self.s.chat_text(task.prompt(c["pid"])), add_special_tokens=False)["input_ids"]
                a_ids = tok(c["answer"] + self.s.end_of_turn, add_special_tokens=False)["input_ids"] \
                    if c.get("answer") else []
                enc.append(torch.tensor(p_ids + a_ids))
                prompt_lens.append(len(p_ids))
        if not enc or len(enc) > self.MAX_TF_TEXTS:
            raise ValueError(f"1 to {self.MAX_TF_TEXTS} texts")
        ret = req.get("return") or {}
        layers = [int(x) for x in ret.get("hidden_states", [])]
        mlp_layers = [int(x) for x in ret.get("mlp_act", [])]
        topk = min(int(ret.get("topk", 0)), 50)
        if sum(len(e) for e in enc) > self.MAX_TF_TOKENS:
            raise ValueError("too many tokens")
        k = self.used + 1
        d = self.work / f"fwd_{k}"
        d.mkdir(exist_ok=True)
        model = self.s.model
        blocks = _decoder_layers(model)
        caps: dict[int, list] = {L: [] for L in mlp_layers}
        hooks = []
        for L in mlp_layers:
            mod = blocks[L].mlp.down_proj
            hooks.append(mod.register_forward_hook(lambda m, inp, out, L=L: caps[L].append(inp[0][0].float().cpu())))
        meta = {"texts": len(enc), "lengths": [int(len(e)) for e in enc], "prompt_lengths": prompt_lens,
                "files": {}}
        hs: dict[int, list] = {L: [] for L in layers}
        lps, tops = [], []
        try:
            with self._weights(which), torch.no_grad():
                for ids in enc:
                    o = model(ids[None].to(self.s.device), output_hidden_states=bool(layers))
                    if layers:
                        for L in layers:
                            hs[L].append(o.hidden_states[L][0].float().cpu())
                    logp = torch.log_softmax(o.logits[0].float(), -1).cpu()
                    if ret.get("token_logprobs"):
                        lps.append(logp[:-1].gather(1, ids[1:, None])[:, 0])
                    if topk:
                        v, i = logp.topk(topk, -1)
                        tops.append((v, i))
        finally:
            for h in hooks:
                h.remove()

        def pad_save(name, seqs):
            T = max(s.shape[0] for s in seqs)
            arr = np.zeros((len(seqs), T) + tuple(seqs[0].shape[1:]), dtype=np.float32)
            for j, s in enumerate(seqs):
                arr[j, : s.shape[0]] = s.numpy()
            np.save(d / name, arr)
            meta["files"][name] = list(arr.shape)

        for L in layers:
            pad_save(f"hidden_{L}.npy", hs[L])
        for L in mlp_layers:
            pad_save(f"mlp_act_{L}.npy", caps[L])
        if lps:
            pad_save("token_logprobs.npy", [x[:, None] for x in lps])
        if tops:
            pad_save("topk_logprobs.npy", [v for v, _ in tops])
            pad_save("topk_ids.npy", [i.float() for _, i in tops])
        (d / "meta.json").write_text(json.dumps(meta))
        return {"dir": d.name, **meta}


def _decoder_layers(model):
    """The list of decoder blocks of a HF causal LM (Qwen3 / Qwen3.5 / Llama / Gemma layouts)."""
    m = model
    for attr in ("model", "language_model"):
        if hasattr(m, attr) and not hasattr(m, "layers"):
            m = getattr(m, attr)
    if hasattr(m, "language_model") and not hasattr(m, "layers"):
        m = m.language_model
    if not hasattr(m, "layers"):
        raise AttributeError("cannot find the decoder layers")
    return m.layers
