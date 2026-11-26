"""Random-weight HY-V3 checkpoints in the released layout, for tests and pipeline dry runs."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import torch

from .engine import expected_keys
from .prune import AUX_FILES, save_shard


def tiny_config(vocab_size: int, num_experts=8, top_k=2, layers=3, hidden_size=64) -> dict:
    """A HY-V3 config with the same option values as the real checkpoint, at toy dimensions."""
    return {
        "architectures": ["HYV3ForCausalLM"], "model_type": "hy_v3",
        "attn_impl": "eager", "bos_token_id": 120000, "enable_attention_fp32_softmax": False,
        "enable_lm_head_fp32": True, "enable_moe_fp32_combine": False, "eod_token_id": 120026,
        "eos_token_id": 120025, "expert_hidden_dim": 32, "first_k_dense_replace": 1, "head_dim": 16,
        "hidden_act": "silu", "hidden_size": hidden_size, "initializer_range": 0.006, "intermediate_size": 128,
        "max_position_embeddings": 4096, "moe_intermediate_size": 32,
        "moe_router_enable_expert_bias": True, "moe_router_use_sigmoid": True,
        "num_attention_heads": 4, "num_experts": num_experts, "num_experts_per_tok": top_k,
        "num_hidden_layers": layers, "num_key_value_heads": 2, "num_shared_experts": 1, "output_router_logits": True,
        "pad_token_id": 120002, "qk_norm": True, "rms_norm_eps": 1e-05, "rope_scaling": None,
        "rope_theta": 11158840.0, "route_norm": True, "router_scaling_factor": 2.826, "sep_token_id": 120007,
        "tie_word_embeddings": False, "use_cache": True, "use_grouped_mm": False, "vocab_size": vocab_size,
    }


def tiny_checkpoint(out_dir, tokenizer_dir, vocab_size=120832, seed=0, num_experts=8, top_k=2, layers=3,
                      router_gain=3.0, hidden_size=64) -> Path:
    """Write a two-shard random checkpoint plus the tokenizer files of ``tokenizer_dir``; returns its path."""
    out_dir, tokenizer_dir = Path(out_dir), Path(tokenizer_dir)
    config = tiny_config(vocab_size, num_experts, top_k, layers, hidden_size)
    generator = torch.Generator().manual_seed(seed)
    hidden, inter, experts = config["hidden_size"], config["moe_intermediate_size"], config["num_experts"]

    def normal(*shape, gain=0.5):
        weights = torch.randn(*shape, generator=generator) * gain / math.sqrt(shape[-1])
        return weights.bfloat16()

    def mlp(prefix, size):
        return {f"{prefix}gate_proj.weight": normal(size, hidden), f"{prefix}up_proj.weight": normal(size, hidden),
                f"{prefix}down_proj.weight": normal(hidden, size)}

    tensors = {"model.embed_tokens.weight": normal(vocab_size, hidden, gain=1.0)}
    for layer in range(layers):
        prefix = f"model.layers.{layer}."
        heads, kv = config["num_attention_heads"] * config["head_dim"], config["num_key_value_heads"] * config["head_dim"]
        tensors.update({prefix + "input_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
                        prefix + "post_attention_layernorm.weight": torch.ones(hidden, dtype=torch.bfloat16),
                        prefix + "self_attn.q_norm.weight": torch.ones(config["head_dim"], dtype=torch.bfloat16),
                        prefix + "self_attn.k_norm.weight": torch.ones(config["head_dim"], dtype=torch.bfloat16),
                        prefix + "self_attn.q_proj.weight": normal(heads, hidden),
                        prefix + "self_attn.o_proj.weight": normal(hidden, heads),
                        prefix + "self_attn.k_proj.weight": normal(kv, hidden),
                        prefix + "self_attn.v_proj.weight": normal(kv, hidden)})
        if layer == 0:
            tensors.update(mlp(prefix + "mlp.", config["intermediate_size"]))
        else:
            tensors[prefix + "mlp.router.gate.weight"] = normal(experts, hidden, gain=router_gain)
            # a small expert bias: with a large one the same experts win for every token and pruning
            # would never change a routing decision, which makes a dry run of the mechanism vacuous
            tensors[prefix + "mlp.expert_bias"] = ((torch.rand(experts, generator=generator) - 0.5) * 0.1).float()
            tensors.update(mlp(prefix + "mlp.shared_mlp.", inter))
            for expert in range(experts):
                tensors.update(mlp(f"{prefix}mlp.experts.{expert}.", inter))
    tensors["model.norm.weight"] = torch.ones(hidden, dtype=torch.bfloat16)
    tensors["lm_head.weight"] = normal(vocab_size, hidden)
    sparse = [layer > 0 for layer in range(layers)]
    if set(tensors) != expected_keys(config, sparse):
        raise AssertionError("tiny checkpoint keys do not match the released layout")
    names = list(tensors)
    cut = names.index(f"model.layers.{layers - 1}.input_layernorm.weight")
    out_dir.mkdir(parents=True)
    weight_map = {}
    for number, group in enumerate((names[:cut], names[cut:]), 1):
        shard = f"model-{number:05d}-of-00002.safetensors"
        save_shard(out_dir / shard, [(name, tensors[name]) for name in group], {"format": "pt"})
        weight_map.update((name, shard) for name in group)
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
         "weight_map": weight_map}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    for name in AUX_FILES:
        if (tokenizer_dir / name).exists():
            shutil.copyfile(tokenizer_dir / name, out_dir / name)
    # Declares these weights as synthetic, so identity.checkpoint_identity can bind them
    # without mistaking them for anything real.
    declared = ["config.json", "model.safetensors.index.json", *sorted(set(weight_map.values()))]
    provenance = {"kind": "synthetic", "seed": seed,
                  "files": {name: hashlib.sha256((out_dir / name).read_bytes()).hexdigest() for name in declared}}
    (out_dir / "checkpoint_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n",
                                                        encoding="utf-8")
    return out_dir
