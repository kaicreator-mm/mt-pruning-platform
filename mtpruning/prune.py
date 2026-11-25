"""Physical expert pruning of a HF-layout HY-V3 checkpoint.

Given the kept original expert ids of every sparse layer, write a new checkpoint
directory: router rows and expert biases are subset, each expert's three matrices
are copied under its new index, and every other tensor is copied unchanged. The
result is an ordinary HY-V3 checkpoint (``num_experts`` = kept count) that the
reference loader opens like any other, plus a ``prune_manifest.json`` recording
the mapping. Nothing is trained or rescaled.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import torch

from .engine import Checkpoint

AUX_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
_EXPERT = re.compile(r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")
_ROUTER = re.compile(r"^model\.layers\.(\d+)\.mlp\.router\.gate\.weight$")
_BIAS = re.compile(r"^model\.layers\.(\d+)\.mlp\.expert_bias$")
_LAYER = re.compile(r"^model\.layers\.(\d+)\.")
_DTYPES = {torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.int64: "I64", torch.int32: "I32"}


def validate_kept(kept: dict[int, list[int]], sparse_layers: list[int], num_experts: int, top_k: int) -> None:
    """Raise ValueError unless ``kept`` is an order-preserving expert mapping for every sparse layer."""
    if set(kept) != set(sparse_layers):
        raise ValueError("expert mapping must cover every sparse layer")
    sizes = set()
    for ids in kept.values():
        if not isinstance(ids, (list, tuple)) or not ids:
            raise ValueError("invalid expert mapping")
        if any(type(i) is not int or not 0 <= i < num_experts for i in ids):
            raise ValueError("invalid expert mapping")
        if len(set(ids)) != len(ids) or list(ids) != sorted(ids):
            raise ValueError("invalid expert mapping")
        if len(ids) < top_k:
            raise ValueError("invalid expert count")
        sizes.add(len(ids))
    if len(sizes) != 1:
        # config.num_experts is a single number, so a loadable checkpoint needs one width for all layers.
        raise ValueError("every sparse layer must keep the same number of experts")


def pruned_tensors(source: Checkpoint, kept: dict[int, list[int]]) -> dict[str, torch.Tensor]:
    """Every tensor of the pruned checkpoint, keyed by its name there."""
    local = {layer: {origin: i for i, origin in enumerate(ids)} for layer, ids in kept.items()}
    tensors = {}
    for source_name in source.weight_map:
        name, tensor = source_name, source.tensor(source_name)
        expert = _EXPERT.match(name)
        if expert:
            layer, origin = int(expert[1]), int(expert[2])
            if origin not in local[layer]:
                continue
            name = f"model.layers.{layer}.mlp.experts.{local[layer][origin]}.{expert[3]}_proj.weight"
        elif _ROUTER.match(name) or _BIAS.match(name):
            tensor = tensor[torch.tensor(kept[int(_LAYER.match(name)[1])])]
        tensors[name] = tensor
    return tensors


def source_tensor(name: str, source: Checkpoint, kept: dict[int, list[int]]) -> torch.Tensor:
    """The source tensor that the tensor stored under ``name`` must equal."""
    expert, router, bias = _EXPERT.match(name), _ROUTER.match(name), _BIAS.match(name)
    if expert:
        layer, local, kind = int(expert[1]), int(expert[2]), expert[3]
        return source.tensor(f"model.layers.{layer}.mlp.experts.{kept[layer][local]}.{kind}_proj.weight")
    if router or bias:
        layer = int(_LAYER.match(name)[1])
        suffix = "mlp.router.gate.weight" if router else "mlp.expert_bias"
        return source.tensor(f"model.layers.{layer}.{suffix}")[torch.tensor(kept[layer])]
    return source.tensor(name)


def save_shard(path: Path, named: list[tuple[str, torch.Tensor]], metadata: dict | None = None) -> None:
    """Stream tensors into a safetensors file.

    Written by hand rather than with ``safetensors.torch.save_file`` because the
    source tensors are views into shared mmaps, which that function rejects.
    """
    header, offset = ({"__metadata__": metadata} if metadata else {}), 0
    plan = []
    for name, tensor in named:
        if not tensor.is_contiguous() or tensor.dtype not in _DTYPES:
            raise ValueError(f"cannot store {name}")
        end = offset + tensor.numel() * tensor.element_size()
        header[name] = {"dtype": _DTYPES[tensor.dtype], "shape": list(tensor.shape), "data_offsets": [offset, end]}
        plan.append((tensor, end - offset))
        offset = end
    raw = json.dumps(header, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as handle:
        handle.write(len(raw).to_bytes(8, "little"))
        handle.write(raw)
        for tensor, nbytes in plan:
            raw = tensor.view(torch.uint8).reshape(-1)
            for start in range(0, nbytes, 64 << 20):
                handle.write(memoryview(raw[start:start + (64 << 20)].numpy()))
    return offset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_pruned(source: Checkpoint, kept: dict[int, list[int]], out_dir: Path,
                 max_shard_bytes: int = 5_000_000_000) -> dict:
    """Write the pruned checkpoint and return its manifest."""
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError(f"{out_dir} is not empty")
    validate_kept(kept, source.sparse_layers, source.num_experts, int(source.config["num_experts_per_tok"]))
    tensors = pruned_tensors(source, kept)
    groups, current, size = [], [], 0
    for name, tensor in tensors.items():
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > max_shard_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(name)
        size += nbytes
    groups.append(current)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight_map, files, total = {}, [], 0
    for number, group in enumerate(groups, 1):
        shard = f"model-{number:05d}-of-{len(groups):05d}.safetensors"
        total += save_shard(out_dir / shard, [(name, tensors[name]) for name in group], {"format": "pt"})
        weight_map.update((name, shard) for name in group)
    for shard in sorted(set(weight_map.values())):
        files.append({"name": shard, "bytes": (out_dir / shard).stat().st_size, "sha256": sha256_file(out_dir / shard)})
    (out_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    config = dict(source.config)
    config["num_experts"] = len(next(iter(kept.values())))
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name in AUX_FILES:
        if (source.path / name).exists():
            shutil.copyfile(source.path / name, out_dir / name)
    manifest = {
        "source_path": str(source.path), "source_num_experts": source.num_experts, "num_experts": config["num_experts"],
        "kept": {str(layer): list(ids) for layer, ids in sorted(kept.items())},
        "tensors": len(weight_map), "tensor_bytes": total, "files": files,
        "index_sha256": sha256_file(out_dir / "model.safetensors.index.json"),
        "config_sha256": sha256_file(out_dir / "config.json"),
    }
    (out_dir / "prune_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def verify_pruned(source: Checkpoint, pruned: Checkpoint) -> dict:
    """Compare every tensor of a reloaded pruned checkpoint with its source tensor."""
    if pruned.kept is None:
        raise ValueError("not a pruned checkpoint (no prune_manifest.json)")
    kept = pruned.kept
    validate_kept(kept, source.sparse_layers, source.num_experts, int(source.config["num_experts_per_tok"]))
    mismatches = [name for name in pruned.weight_map
                  if not torch.equal(pruned.tensor(name), source_tensor(name, source, kept))]
    result = {
        "tensors_checked": len(pruned.weight_map),
        "tensor_bytes": sum(pruned.tensor(name).numel() * pruned.tensor(name).element_size()
                            for name in pruned.weight_map),
        "mismatched_tensors": len(mismatches), "first_mismatches": mismatches[:8],
        "identity_mapping": all(kept[layer] == list(range(source.num_experts)) for layer in source.sparse_layers),
    }
    result["pass"] = result["mismatched_tensors"] == 0
    return result
