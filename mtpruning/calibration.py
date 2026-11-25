"""What the original model actually routes, and the direction-balanced selection built from it.

Statistics are never carried across process restarts in memory. Every sample's
routing is committed to the run journal together with its output; the
per-direction aggregate is rebuilt from those committed samples, in sample ID
order, whenever it is needed (see pipeline.run and pipeline.select).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .corpus import DIRECTIONS
from .reviewed import handoff

PHASES = ("prefill", "decode")
AUDIT_FIELDS = ("decisions", "changed", "lost_mass")
CELL_FIELDS = ("tokens", "layer", "expert", "count", "mass")


class SampleRouting:
    """Routing statistics of one sample: per phase, the (layer, expert) cells that were selected.

    Experts are numbered the way the checkpoint numbers them. For the original model
    that is the released numbering; for a pruned checkpoint it is the re-indexed one.
    """

    def __init__(self, engine):
        layers, experts = len(engine.ckpt.sparse_layers), engine.ckpt.num_experts
        self.top_k = engine.top_k
        self.position = {layer: i for i, layer in enumerate(engine.ckpt.sparse_layers)}
        self.layer_ids = torch.tensor(engine.ckpt.sparse_layers)     # cells are named by layer index, not position
        self.count = {phase: torch.zeros(layers, experts, dtype=torch.int64) for phase in PHASES}
        self.mass = {phase: torch.zeros(layers, experts, dtype=torch.float64) for phase in PHASES}
        self.tokens = {phase: 0 for phase in PHASES}

    def hook(self, layer_idx, phase, top, weights):
        # Summed per expert over a sorted, contiguous slice: a recomputation of the same
        # sample yields the same bits, which is what the journal's conflict check relies on.
        flat = top.reshape(-1)
        values = weights.reshape(-1).double()
        row = self.position[layer_idx]
        order = torch.argsort(flat, stable=True)
        experts, counts = torch.unique_consecutive(flat[order], return_counts=True)
        sorted_values = values[order]
        starts = (torch.cumsum(counts, 0) - counts).tolist()
        sums = torch.stack([sorted_values[start:start + n].sum() for start, n in zip(starts, counts.tolist())])
        self.count[phase][row][experts] += counts
        self.mass[phase][row][experts] += sums
        if row == 0:
            self.tokens[phase] += top.shape[0]

    def payload(self) -> dict:
        result = {}
        for phase in PHASES:
            count, mass = self.count[phase], self.mass[phase]
            consistent = bool((count.sum(-1) == self.tokens[phase] * self.top_k).all())
            if not consistent or bool((mass[count == 0] != 0).any()) or not bool(torch.isfinite(mass).all()):
                raise RuntimeError("inconsistent routing statistics")
            layers, experts = count.nonzero(as_tuple=True)
            result[phase] = {"tokens": self.tokens[phase], "layer": self.layer_ids[layers].tolist(),
                             "expert": experts.tolist(),
                             "count": count[layers, experts].tolist(), "mass": mass[layers, experts].tolist()}
        return result


class SampleAudit:
    """What an active mask changes in one sample."""

    def __init__(self):
        self.changed = {phase: 0 for phase in PHASES}
        self.lost = {phase: 0.0 for phase in PHASES}

    def hook(self, phase, changed, lost):
        self.changed[phase] += changed
        self.lost[phase] += lost

    def payload(self, routing: dict, depth: int) -> dict:
        return {phase: {"decisions": routing[phase]["tokens"] * depth, "changed": self.changed[phase],
                        "lost_mass": self.lost[phase]} for phase in PHASES}


def validate_payload(payload, context: dict, sparse_layers: list[int], num_experts: int, top_k: int,
                     max_new_tokens: int) -> None:
    """Raise ValueError unless a committed sample is well formed and is the corpus record it names."""
    need = handoff.need
    need(isinstance(payload, dict) and isinstance(payload.get("identity_key"), str), "malformed sample payload")
    need(all(payload.get(k) == context[k] for k in ("split", "direction", "line")), "sample identity mismatch")
    need(type(payload.get("prompt_tokens")) is int and payload["prompt_tokens"] == len(context["prompt_ids"]) > 0,
         "prompt token mismatch")
    tokens = payload.get("tokens")
    need(isinstance(tokens, list) and all(type(t) is int and t >= 0 for t in tokens)
         and type(payload.get("new_tokens")) is int and payload["new_tokens"] == len(tokens) <= max_new_tokens,
         "generated token mismatch")
    need(payload.get("finish") in ("eos", "length") and isinstance(payload.get("text"), str)
         and (payload["finish"] != "length" or payload["new_tokens"] == max_new_tokens), "bad generation record")
    routing = payload.get("routing")
    need(isinstance(routing, dict) and set(routing) == set(PHASES), "missing routing statistics")
    need(routing["prefill"].get("tokens") == payload["prompt_tokens"]
         and routing["decode"].get("tokens") == payload["new_tokens"], "routing token counts disagree with the record")
    known = set(sparse_layers)
    for phase in PHASES:
        cells = routing[phase]
        need(set(cells) == set(CELL_FIELDS)
             and len({len(cells[name]) for name in CELL_FIELDS if name != "tokens"}) == 1, "malformed routing cells")
        need(type(cells["tokens"]) is int and cells["tokens"] >= 0, "bad routing token count")
        need(all(type(v) is int and v >= 0 for v in cells["layer"] + cells["expert"] + cells["count"]),
             "invalid routing cells")
        need(all(type(v) is float and math.isfinite(v) and v > 0 for v in cells["mass"]), "invalid routing mass")
        pairs = list(zip(cells["layer"], cells["expert"]))
        need(len(set(pairs)) == len(pairs), "duplicate routing cell")
        need(all(layer in known and expert < num_experts for layer, expert in pairs), "routing cell out of range")
        per_layer = Counter()
        for layer, count in zip(cells["layer"], cells["count"]):
            per_layer[layer] += count
        # every token selects exactly top_k experts in every sparse layer
        need(len(per_layer) == (len(sparse_layers) if cells["tokens"] else 0)
             and all(count == cells["tokens"] * top_k for count in per_layer.values()), "incomplete routing coverage")
    audit = payload.get("audit")
    if audit is not None:
        need(isinstance(audit, dict) and set(audit) == set(PHASES), "malformed mask audit")
        for phase in PHASES:
            entry = audit[phase]
            need(set(entry) == set(AUDIT_FIELDS) and all(
                type(entry[name]) in (int, float) and math.isfinite(entry[name]) and entry[name] >= 0
                for name in AUDIT_FIELDS), "invalid mask audit")
            need(entry["changed"] <= entry["decisions"], "mask audit changed more decisions than it saw")


def rebuild(samples: list[dict], sparse_layers: list[int], num_experts: int, top_k: int,
            max_new_tokens: int, contexts: dict) -> dict:
    """Per-direction statistics of the committed ``samples`` (in the order given).

    With the audit, every sample must carry one; without it, none may.
    """
    shape = (len(DIRECTIONS), len(sparse_layers), num_experts)
    stats = {"count": {phase: torch.zeros(shape, dtype=torch.int64) for phase in PHASES},
             "mass": {phase: torch.zeros(shape, dtype=torch.float64) for phase in PHASES},
             "tokens": {phase: torch.zeros(len(DIRECTIONS), dtype=torch.int64) for phase in PHASES},
             "sentences": torch.zeros(len(DIRECTIONS), dtype=torch.int64),
             "audit": torch.zeros(len(DIRECTIONS), len(PHASES), len(AUDIT_FIELDS), dtype=torch.float64)}
    position = {layer: i for i, layer in enumerate(sparse_layers)}
    audited = [sample["audit"] is not None for sample in samples]
    if any(audited) and not all(audited):
        raise ValueError("mixed masked and unmasked samples")
    for sample in samples:
        validate_payload(sample, contexts[sample["id"]], sparse_layers, num_experts, top_k, max_new_tokens)
        row = DIRECTIONS.index(sample["direction"])
        stats["sentences"][row] += 1
        for phase_index, phase in enumerate(PHASES):
            cells = sample["routing"][phase]
            if cells["layer"]:
                index = (torch.tensor([row] * len(cells["layer"])),
                         torch.tensor([position[layer] for layer in cells["layer"]]),
                         torch.tensor(cells["expert"]))
                # cells of one sample are unique, so the order of these additions is fixed
                stats["count"][phase][index] += torch.tensor(cells["count"])
                # float64 on purpose: the statistics are accumulated in float64
                stats["mass"][phase].index_put_(index, torch.tensor(cells["mass"], dtype=torch.float64),
                                                accumulate=True)
            stats["tokens"][phase][row] += cells["tokens"]
            if sample["audit"] is not None:
                stats["audit"][row, phase_index] = torch.tensor(
                    [sample["audit"][phase][name] for name in AUDIT_FIELDS], dtype=torch.float64)
    return stats


def save_stats(base, stats: dict, sparse_layers: list[int]) -> dict:
    """Write the aggregate in the layout the reviewed r6 adapter reads; returns the summary."""
    base = Path(base)
    total = stats["count"]["prefill"] + stats["count"]["decode"]
    summary = {"directions": list(DIRECTIONS),
               "sentences": {d: int(stats["sentences"][i]) for i, d in enumerate(DIRECTIONS)},
               "tokens": {d: {phase: int(stats["tokens"][phase][i]) for phase in PHASES}
                          for i, d in enumerate(DIRECTIONS)},
               "expert_slots_never_selected": {d: int((total[i] == 0).sum()) for i, d in enumerate(DIRECTIONS)},
               "expert_slots_total": int(total[0].numel()), "layers": sparse_layers}
    arrays = {f"{name}_{phase}": stats[name][phase].numpy() for name in ("count", "mass", "tokens") for phase in PHASES}
    arrays["sentences"] = stats["sentences"].numpy()
    temporary = base.with_name(base.name + f".{np.random.randint(0, 1 << 30)}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(str(base) + ".npz")
    base.with_name(base.name + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def mask_audit_report(stats: dict) -> dict:
    """Per direction and phase: decisions seen, changed by the mask, and the routing weight it removed."""
    return {d: {phase: {"decisions": int(stats["audit"][i, p, 0]), "changed": int(stats["audit"][i, p, 1]),
                        "changed_fraction": float(stats["audit"][i, p, 1] / stats["audit"][i, p, 0])
                        if stats["audit"][i, p, 0] else None,
                        "lost_weight_mass": round(float(stats["audit"][i, p, 2]), 4)}
                for p, phase in enumerate(PHASES)} for i, d in enumerate(DIRECTIONS)}


def select(mass: torch.Tensor, keep: int, top_k: int) -> list[list[int]]:
    """Direction-balanced selection per sparse layer.

    ``mass`` is [directions, sparse layers, experts] routing weight. Each direction is
    normalised on its own first, so a direction with more tokens (Burmese) cannot
    dominate; the kept experts are the top ``keep`` by the mean of those shares.
    Same rule as the D01a synthetic demo.
    """
    values = torch.as_tensor(mass, dtype=torch.float64)
    if values.ndim != 3 or values.shape[0] != len(DIRECTIONS):
        raise ValueError("all six directions required")
    if not torch.isfinite(values).all() or (values < 0).any() or (values.sum(-1) <= 0).any():
        raise ValueError("invalid profile")
    experts = values.shape[-1]
    if not top_k <= keep <= experts:
        raise ValueError("invalid expert count")
    normalized = values / values.sum(-1, keepdim=True)
    scores = normalized.mean(0)
    return [sorted(sorted(range(experts), key=lambda i: (-float(scores[layer, i]), i))[:keep])
            for layer in range(values.shape[1])]


def retained_mass(mass: torch.Tensor, kept: list[list[int]]) -> torch.Tensor:
    """Share of each direction's routing weight that still has an expert, [directions, sparse layers]."""
    values = torch.as_tensor(mass, dtype=torch.float64)
    rows = torch.stack([torch.tensor(ids) for ids in kept])
    kept_mass = torch.stack([values[:, layer].index_select(1, rows[layer]).sum(-1) for layer in range(len(kept))], 1)
    return kept_mass / values.sum(-1)
