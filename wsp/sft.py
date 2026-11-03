"""LoRA distillation: fine-tune the student on solver-verified teacher recipes (and its own good samples).

    python -m wsp.sft --data wsp/data/demos.jsonl --out wsp_runs/sft1            # teacher demos
    python -m wsp.sft --data wsp/data/demos.jsonl,wsp_runs/rft1/keep.jsonl ...  # + RFT samples

Each example is (the student's exact prompt for a training part, the recipe as compact JSON). The loss is
on the answer tokens only, and the logits are computed for those positions only (the 248k-token output
layer over a 1k-token prompt would cost more than the rest of the step). LoRA goes in the top
--lora-layers decoder blocks, so backpropagation stops there: on a CPU this halves the step time.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def load_examples(paths: list[str], max_rank: int = 1, min_x: float = 0.0) -> list[dict]:
    """Rows {pid, spec, x_auto, rank} -> examples {pid, answer, weight}. rank = 0 is a problem's best recipe."""
    from wsp import task
    out = []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            if r.get("rank", 0) < max_rank and r.get("x_auto", 0) >= min_x:
                out.append({"pid": r["pid"], "answer": task.dumps(r["spec"]), "x_auto": r.get("x_auto"),
                            "source": r.get("source", Path(p).stem)})
    return out


def train(student, examples: list[dict], out: str | Path, *, epochs: int = 3, lr: float = 2e-4, rank: int = 16,
          alpha: int = 32, lora_layers: int = 12, accum: int = 4, seed: int = 0, log=print) -> dict:
    import torch
    from peft import LoraConfig, get_peft_model

    from wsp import task
    random.seed(seed)
    torch.manual_seed(seed)
    model = student.model
    cfg = getattr(model.config, "text_config", model.config)
    L = cfg.num_hidden_layers
    present = {n.split(".")[-1] for n, _ in model.named_modules()}
    lcfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.05, target_modules=[t for t in TARGETS if t in present],
                      layers_to_transform=list(range(max(0, L - lora_layers), L)), task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # tokenize: prefix (generation prompt) + answer + end of turn; loss on the answer part
    data = []
    for ex in examples:
        prefix = student.chat_text(task.prompt(ex["pid"]))
        full = prefix + ex["answer"] + student.end_of_turn
        p_ids = student.tok(prefix, add_special_tokens=False)["input_ids"]
        f_ids = student.tok(full, add_special_tokens=False)["input_ids"]
        n_ans = len(f_ids) - len(p_ids)
        if f_ids[: len(p_ids)] != p_ids or n_ans <= 0:
            raise ValueError(f"{ex['pid']}: the answer does not tokenize as a continuation of the prompt")
        data.append((torch.tensor(f_ids), n_ans))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = math.ceil(len(data) * epochs / accum)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * max(0.0, (steps - s) / steps) ** 0.5)
    hist, t0, step = [], time.perf_counter(), 0
    order = []
    for e in range(epochs):
        idx = list(range(len(data)))
        random.shuffle(idx)
        order += idx
    opt.zero_grad()
    run_loss, run_n = 0.0, 0
    for i, j in enumerate(order):
        ids, n_ans = data[j]
        o = model(input_ids=ids[None].to(student.device), logits_to_keep=n_ans + 1)
        logits = o.logits[0, :-1].float()
        target = ids[-n_ans:].to(student.device)
        loss = torch.nn.functional.cross_entropy(logits, target)
        (loss / accum).backward()
        run_loss += float(loss.detach())
        run_n += 1
        if (i + 1) % accum == 0 or i == len(order) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            hist.append({"step": step, "loss": run_loss / run_n, "lr": sched.get_last_lr()[0],
                         "seconds": round(time.perf_counter() - t0, 1)})
            log(json.dumps(hist[-1]))
            run_loss, run_n = 0.0, 0
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    info = {"examples": len(data), "epochs": epochs, "steps": step, "lr": lr, "rank": rank, "alpha": alpha,
            "lora_layers": lora_layers, "trainable_params": n_train, "seconds": round(time.perf_counter() - t0, 1),
            "history": hist, "model": student.path}
    (out / "train.json").write_text(json.dumps(info, indent=1))
    model.eval()
    return info


def main(argv=None) -> int:
    from wsp.student import DEFAULT_MODEL, Student
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="comma-separated demo / RFT files")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--edits", default=None, help="train on top of WSP edits")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-layers", type=int, default=12)
    ap.add_argument("--max-rank", type=int, default=2, help="use each problem's best N recipes")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    ex = load_examples(a.data.split(","), max_rank=a.max_rank, min_x=1.0)
    print(f"{len(ex)} examples from {len({e['pid'] for e in ex})} problems", flush=True)
    s = Student(a.model, edits=a.edits, threads=a.threads)
    info = train(s, ex, a.out, epochs=a.epochs, lr=a.lr, rank=a.rank, lora_layers=a.lora_layers,
                 log=lambda m: print(m, flush=True))
    print(json.dumps({k: v for k, v in info.items() if k != "history"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
