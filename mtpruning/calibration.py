"""What the original model actually routes, and the direction-balanced selection built from it."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .corpus import DIRECTIONS

PHASES = ("prefill", "decode")


class RoutingStats:
    """Per direction, phase and sparse layer: how often each expert is chosen and with what weight."""

    def __init__(self, engine):
        checkpoint = engine.ckpt
        self.sparse_layers = checkpoint.sparse_layers
        self.num_experts = checkpoint.num_experts
        shape = (len(DIRECTIONS), len(self.sparse_layers), self.num_experts)
        self.count = {phase: torch.zeros(shape, dtype=torch.int64) for phase in PHASES}
        self.mass = {phase: torch.zeros(shape, dtype=torch.float64) for phase in PHASES}
        self.tokens = {phase: torch.zeros(len(DIRECTIONS), dtype=torch.int64) for phase in PHASES}
        self.sentences = torch.zeros(len(DIRECTIONS), dtype=torch.int64)
        self._direction, self._origin = None, engine.origin

    def begin(self, direction: str) -> None:
        self._direction = DIRECTIONS.index(direction)

    def hook(self, layer_idx: int, phase: str, top: torch.Tensor, weights: torch.Tensor) -> None:
        row, layer = self._direction, self.sparse_layers.index(layer_idx)
        flat = self._origin[layer_idx][top].reshape(-1)
        self.count[phase][row, layer].index_add_(0, flat, torch.ones_like(flat))
        self.mass[phase][row, layer].index_add_(0, flat, weights.reshape(-1).double())
        if layer == 0:
            self.tokens[phase][row] += top.shape[0]

    def save(self, base: Path) -> dict:
        arrays = {f"count_{phase}": self.count[phase].numpy() for phase in PHASES}
        arrays.update({f"mass_{phase}": self.mass[phase].numpy() for phase in PHASES})
        arrays.update({f"tokens_{phase}": self.tokens[phase].numpy() for phase in PHASES})
        arrays["sentences"] = self.sentences.numpy()
        np.savez_compressed(Path(str(base) + ".npz"), **arrays)
        summary = self.summary()
        Path(str(base) + ".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        return summary

    def summary(self) -> dict:
        total = self.mass["prefill"] + self.mass["decode"]
        never = (self.count["prefill"] + self.count["decode"]) == 0
        return {
            "directions": list(DIRECTIONS),
            "sentences": {d: int(self.sentences[i]) for i, d in enumerate(DIRECTIONS)},
            "tokens": {d: {phase: int(self.tokens[phase][i]) for phase in PHASES} for i, d in enumerate(DIRECTIONS)},
            "expert_slots_never_selected": {d: int(never[i].sum()) for i, d in enumerate(DIRECTIONS)},
            "expert_slots_total": int(never[0].numel()),
            "layers": self.sparse_layers,
        }

    @classmethod
    def load(cls, path_base: Path) -> "RoutingStats":
        with np.load(Path(str(path_base) + ".npz")) as data:
            stats = cls.__new__(cls)
            stats.count = {phase: torch.from_numpy(data[f"count_{phase}"]) for phase in PHASES}
            stats.mass = {phase: torch.from_numpy(data[f"mass_{phase}"]) for phase in PHASES}
            stats.tokens = {phase: torch.from_numpy(data[f"tokens_{phase}"]) for phase in PHASES}
            stats.sentences = torch.from_numpy(data["sentences"])
        stats.sparse_layers = json.loads(Path(str(path_base) + ".json").read_text(encoding="utf-8"))["layers"]
        stats.num_experts = stats.count["prefill"].shape[-1]
        return stats


class MaskAudit:
    """While a mask is active: how many routing decisions the mask changes, and how much weight it removes."""

    def __init__(self):
        self.changed = {phase: torch.zeros(len(DIRECTIONS), dtype=torch.float64) for phase in PHASES}
        self.lost = {phase: torch.zeros(len(DIRECTIONS), dtype=torch.float64) for phase in PHASES}
        self.decisions = {phase: torch.zeros(len(DIRECTIONS), dtype=torch.float64) for phase in PHASES}
        self._direction = None

    def begin(self, direction: str) -> None:
        self._direction = DIRECTIONS.index(direction)

    def hook(self, phase: str, changed_tokens: int, lost_mass: float) -> None:
        self.changed[phase][self._direction] += changed_tokens
        self.lost[phase][self._direction] += lost_mass
        self.decisions[phase][self._direction] += 1   # one layer call = one decision per token

    def report(self, token_layers: dict) -> dict:
        """``token_layers[direction][phase]``: tokens × sparse layers processed for that direction."""
        return {d: {phase: {
            "decisions": int(token_layers[d][phase]),
            "changed": int(self.changed[phase][i]),
            "changed_fraction": float(self.changed[phase][i] / token_layers[d][phase])
            if token_layers[d][phase] else None,
            "lost_weight_mass": round(float(self.lost[phase][i]), 4),
        } for phase in PHASES} for i, d in enumerate(DIRECTIONS)}


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
