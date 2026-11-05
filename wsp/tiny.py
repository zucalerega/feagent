"""A tiny random Qwen3.5-architecture model with a tokenizer trained on the task's own prompts.

For pipeline tests only (tests/test_wsp.py and `--model tiny`): it exercises every code path (chat
template, generation, LoRA training, the WSP kernel and forward tool, weight edits) in seconds, on a
machine without access to the model hub. Its answers are random text; never report them as results.
"""

from __future__ import annotations

from pathlib import Path

CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
                 "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


def build(path: str | Path, pids: list[str] | None = None, layers: int = 4) -> str:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from wsp import problems, task
    path = Path(path)
    if (path / "config.json").exists():
        return str(path)
    path.mkdir(parents=True, exist_ok=True)
    pids = pids or problems.train_ids()[:3]
    corpus = [task.prompt(p) for p in pids] + [task.ANSWER_FORMAT, task.SYSTEM] * 3
    corpus += ['{"order": 2, "h": %d, "refinements": [{"curve": %d, "size": %g, "radius": %d}]}' % (h, c, s, r)
               for h in (20, 40, 60, 80) for c in (1, 3, 5) for s in (2.5, 4) for r in (30, 60)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=800, special_tokens=["<|endoftext|>", "<|im_start|>", "<|im_end|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(corpus, trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                   bos_token=None)
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(path)
    cfg = Qwen3_5TextConfig(vocab_size=len(fast), hidden_size=64, intermediate_size=128, num_hidden_layers=layers,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_key_head_dim=16,
                            linear_value_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4,
                            layer_types=(["linear_attention"] * 3 + ["full_attention"]) * (layers // 4),
                            tie_word_embeddings=True, max_position_embeddings=8192,
                            eos_token_id=fast.eos_token_id, pad_token_id=fast.pad_token_id)
    Qwen3_5ForCausalLM(cfg).save_pretrained(path)
    return str(path)
