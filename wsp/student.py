"""The student: a small open causal LM from Hugging Face, run on CPU (or a GPU when there is one).

Default model: Qwen/Qwen3.5-0.8B (Apache-2.0, 24 layers: 18 Gated DeltaNet + 6 full attention, the model
of the WSP paper's Experiment 8). Any causal LM with a chat template works (Qwen/Qwen3-0.6B is a plain
transformer and faster on CPU). The student answers in non-thinking mode: the answer is a ~100-token
JSON object, and thinking would multiply the cost (and, on Qwen3.5-0.8B, loop; see the paper).

Variants of one base model, all through `Student(...)`:
  edits=DIR     weight-space edits (wsp.kernel layout: one float32 <tensor name>.npy per edited tensor)
  adapter=DIR   a LoRA adapter (wsp.sft), merged into the weights for speed
"""

from __future__ import annotations

import os
import time
from pathlib import Path

DEFAULT_MODEL = os.environ.get("WSP_MODEL", "Qwen/Qwen3.5-0.8B")


def resolve_model(path_or_id: str) -> str:
    """A local directory, or a hub id (downloaded once into models/, gitignored)."""
    p = Path(path_or_id)
    if p.is_dir():
        return str(p)
    local = Path(__file__).resolve().parents[1] / "models" / path_or_id.replace("/", "__")
    if local.is_dir() and any(local.glob("*.safetensors")):
        return str(local)
    from huggingface_hub import snapshot_download
    return snapshot_download(path_or_id, local_dir=str(local),
                             allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"])


def apply_edits(model, edits_dir: str | Path) -> list[str]:
    """Overwrite the named tensors with the saved edits (float32 .npy, cast to the model's dtype)."""
    import numpy as np
    import torch
    sd = model.state_dict()
    done = []
    for f in sorted(Path(edits_dir).glob("*.npy")):
        name = f.name[:-4]
        if name not in sd:
            raise KeyError(f"edit {name} is not a tensor of this model")
        t = torch.from_numpy(np.load(f))
        if tuple(t.shape) != tuple(sd[name].shape):
            raise ValueError(f"edit {name}: shape {tuple(t.shape)} != {tuple(sd[name].shape)}")
        with torch.no_grad():
            sd[name].copy_(t.to(sd[name].dtype))
        done.append(name)
    return done


class Student:
    def __init__(self, model: str = DEFAULT_MODEL, *, edits: str | Path | None = None,
                 adapter: str | Path | None = None, dtype: str | None = None, device: str | None = None,
                 threads: int | None = None, thinking: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads:
            torch.set_num_threads(threads)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # CPU without bf16 instructions (this box): fp32 is much faster than emulated bf16
        dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[
            dtype or ("bfloat16" if self.device == "cuda" else "float32")]
        self.path = resolve_model(model)
        self.tok = AutoTokenizer.from_pretrained(self.path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.path, dtype=dt).to(self.device).eval()
        self.edited: list[str] = apply_edits(self.model, edits) if edits else []
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter)).merge_and_unload().eval()
        self.thinking = thinking
        self.name = Path(self.path).name

    # ------------------------------------------------------------------ prompts
    def chat_text(self, user: str, system: str | None = None, answer: str | None = None) -> str:
        """The chat-template text the model reads: the generation prompt, plus `answer` and the end-of-turn
        token when given. Training (wsp.sft) and teacher-forced passes use exactly this, so the answer
        tokens follow the same prefix as at generation time (chat templates may render a past assistant
        turn differently, e.g. Qwen's empty think block)."""
        from wsp.task import SYSTEM
        msgs = [{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": user}]
        prefix = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=self.thinking)
        return prefix if answer is None else prefix + answer + self.end_of_turn

    @property
    def end_of_turn(self) -> str:
        vocab = self.tok.get_vocab()
        return "<|im_end|>" if "<|im_end|>" in vocab else (self.tok.eos_token or "")

    # ------------------------------------------------------------------ generation
    def generate(self, users: list[str], *, max_new_tokens: int = 320, temperature: float = 0.0,
                 top_p: float = 0.95, top_k: int = 20, n: int = 1, seed: int = 0, batch_size: int = 6,
                 system: str | None = None) -> list[list[str]]:
        """Answers for each user message: n samples each (greedy when temperature == 0, then n = 1)."""
        import torch
        texts = [self.chat_text(u, system) for u in users]
        n = 1 if temperature == 0 else n
        jobs = [(i, t) for i, t in enumerate(texts) for _ in range(n)]
        out: list[list[str]] = [[] for _ in users]
        torch.manual_seed(seed)
        for b in range(0, len(jobs), batch_size):
            chunk = jobs[b:b + batch_size]
            enc = self.tok([t for _, t in chunk], return_tensors="pt", padding=True).to(self.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            if temperature > 0:
                kw.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k)
            else:
                kw.update(do_sample=False)
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            for (i, _), row in zip(chunk, gen[:, enc["input_ids"].shape[1]:]):
                out[i].append(self.tok.decode(row, skip_special_tokens=True))
        return out

    def timed_generate(self, users: list[str], **kw) -> tuple[list[list[str]], float]:
        t = time.perf_counter()
        return self.generate(users, **kw), time.perf_counter() - t
