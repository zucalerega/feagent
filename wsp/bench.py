"""What the student costs to run and to train on this machine, measured on the model's architecture.

    python -m wsp.bench                       # Qwen3.5-0.8B and Qwen3-0.6B shapes, 4 threads
    python -m wsp.bench --arch qwen3.5-0.8b --prompt 1000 --answer 120

Latency depends on the architecture (layer types, widths, vocabulary), not on the weight values, so the
models are built from their published configurations with random weights: no download needed. Measured:
prefill of a task prompt, greedy decoding of a recipe, a batch of 6 recipes, and one LoRA training step
(LoRA in the top 12 blocks, loss on the answer tokens only, as wsp.sft trains).
"""

from __future__ import annotations

import argparse
import json
import time

ARCHS = {
    # the configuration given in the WSP paper's Experiment 8 / openwsp (Qwen3.5-0.8B, text part)
    "qwen3.5-0.8b": dict(kind="qwen3_5", vocab_size=248320, hidden_size=1024, intermediate_size=3584,
                         num_hidden_layers=24, num_attention_heads=8, num_key_value_heads=2, head_dim=256,
                         linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=128,
                         linear_value_head_dim=128, tie_word_embeddings=True),
    # Qwen3-0.6B (plain transformer)
    "qwen3-0.6b": dict(kind="qwen3", vocab_size=151936, hidden_size=1024, intermediate_size=3072,
                       num_hidden_layers=28, num_attention_heads=16, num_key_value_heads=8, head_dim=128,
                       tie_word_embeddings=True),
}


def build(arch: str, layers: int | None = None):
    import torch
    a = dict(ARCHS[arch])
    kind = a.pop("kind")
    if layers:
        a["num_hidden_layers"] = layers
    if kind == "qwen3_5":
        from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig
        L = a["num_hidden_layers"]
        a["layer_types"] = ["full_attention" if i % 4 == 3 else "linear_attention" for i in range(L)]
        model = Qwen3_5ForCausalLM(Qwen3_5TextConfig(**a, max_position_embeddings=32768))
    else:
        from transformers import Qwen3Config, Qwen3ForCausalLM
        model = Qwen3ForCausalLM(Qwen3Config(**a, max_position_embeddings=32768))
    return model.to(torch.float32).eval()


def bench(arch: str, prompt: int, answer: int, threads: int, layers: int | None = None, train: bool = True) -> dict:
    import torch
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    model = build(arch, layers)
    V = model.config.vocab_size
    n_params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, V, (1, prompt))
    out = {"arch": arch, "params": n_params, "threads": threads, "prompt_tokens": prompt, "answer_tokens": answer}
    with torch.no_grad():
        model(ids[:, :32])                                   # warm-up
        t = time.perf_counter()
        model(ids, logits_to_keep=1)
        out["prefill_s"] = round(time.perf_counter() - t, 2)
        t = time.perf_counter()
        g = model.generate(ids, max_new_tokens=answer, min_new_tokens=answer, do_sample=False, pad_token_id=0)
        dt = time.perf_counter() - t
        out["recipe_s"] = round(dt, 2)                       # prefill + decode of one recipe
        out["decode_tok_s"] = round(answer / max(dt - out["prefill_s"], 1e-6), 1)
        b = torch.randint(0, V, (6, prompt))
        t = time.perf_counter()
        model.generate(b, attention_mask=torch.ones_like(b), max_new_tokens=answer, min_new_tokens=answer,
                       do_sample=False, pad_token_id=0)
        out["batch6_s_per_recipe"] = round((time.perf_counter() - t) / 6, 2)
    assert g.shape[1] == prompt + answer
    if train:
        from peft import LoraConfig, get_peft_model

        from wsp.sft import TARGETS
        L = model.config.num_hidden_layers
        present = {n.split(".")[-1] for n, _ in model.named_modules()}
        pm = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=[x for x in TARGETS if x in present],
                                              layers_to_transform=list(range(max(0, L - 12), L)),
                                              layers_pattern="layers", task_type="CAUSAL_LM"))
        pm.train()
        seq = torch.randint(0, V, (1, prompt + answer))
        opt = torch.optim.AdamW([p for p in pm.parameters() if p.requires_grad], lr=1e-4)
        times = []
        for _ in range(2):
            t = time.perf_counter()
            o = pm(input_ids=seq, logits_to_keep=answer + 1)
            loss = torch.nn.functional.cross_entropy(o.logits[0, :-1].float(), seq[0, -answer:])
            loss.backward()
            opt.step()
            opt.zero_grad()
            times.append(time.perf_counter() - t)
        out["lora_step_s"] = round(min(times), 2)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="qwen3.5-0.8b,qwen3-0.6b")
    ap.add_argument("--prompt", type=int, default=1000)
    ap.add_argument("--answer", type=int, default=120)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=None, help="fewer layers (a quick check of the script)")
    ap.add_argument("--no-train", action="store_true")
    a = ap.parse_args(argv)
    for arch in a.arch.split(","):
        print(json.dumps(bench(arch, a.prompt, a.answer, a.threads, a.layers, train=not a.no_train)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
