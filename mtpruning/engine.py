"""Lean CPU engine for HF-layout HY-V3 checkpoints.

Op-for-op mirror of what ``transformers`` 5.24.1 actually runs for this model
with ``attn_implementation="eager"``: eager attention, the sigmoid + expert-bias
router, the default ``grouped_mm`` experts path. The one structural difference:
expert weights stay zero-copy mmap views, one 2D tensor per expert, instead of
the fused 3D ``gate_up_proj``/``down_proj`` the reference builds on the heap.
That is what lets a 60 GB checkpoint run from the page cache of a 64 GB
machine. ``tests/test_engine.py`` holds this engine and the reference
implementation to identical logits on random checkpoints.

Weight views come from a writable private mmap, so nothing here may write to
them. The masked baseline only adds a ``denied`` vector per layer (0 for kept
experts, -inf for pruned ones) to the selection scores before top-k; routing
itself is the reference sigmoid + expert-bias rule everywhere.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

# safe_open objects own the mmaps that tensor views point into. Keep every handle
# alive for the rest of the process so a dropped Checkpoint cannot unmap memory
# that a view still references.
_HANDLES: list = []

# Reference-implementation options this engine relies on. A checkpoint that
# disagrees is not supported and must fail closed rather than run wrongly.
_REQUIRED_CONFIG = {
    "model_type": "hy_v3",
    "architectures": ["HYV3ForCausalLM"],
    "hidden_act": "silu",
    "moe_router_use_sigmoid": True,
    "moe_router_enable_expert_bias": True,
    "num_shared_experts": 1,
    "enable_moe_fp32_combine": False,
    "use_grouped_mm": False,
    "qk_norm": True,
    "tie_word_embeddings": False,
}
_DEFAULT_CONFIG = {"attention_bias": False, "mlp_bias": False}


def check_config(config: dict) -> None:
    """Raise ValueError unless the checkpoint config is one this engine implements."""
    for key, want in _REQUIRED_CONFIG.items():
        if config.get(key) != want:
            raise ValueError(f"unsupported config: {key}={config.get(key)!r}, expected {want!r}")
    for key, want in _DEFAULT_CONFIG.items():
        if config.get(key, want) != want:
            raise ValueError(f"unsupported config: {key}={config.get(key)!r}")
    for key in ("rope_scaling", "rope_parameters"):
        if config.get(key) is not None:
            raise ValueError(f"unsupported config: {key}={config.get(key)!r}")
    if int(config["num_attention_heads"]) % int(config["num_key_value_heads"]):
        raise ValueError("num_attention_heads must be a multiple of num_key_value_heads")


def expected_keys(config: dict, sparse: list[bool]) -> set[str]:
    """Checkpoint keys of a HY-V3 model with the given per-layer MLP types."""
    experts = int(config["num_experts"])
    keys = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    for layer, is_sparse in enumerate(sparse):
        prefix = f"model.layers.{layer}."
        keys.update(prefix + name for name in (
            "input_layernorm.weight", "post_attention_layernorm.weight",
            "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
        ))
        if is_sparse:
            keys.update(prefix + name for name in (
                "mlp.router.gate.weight", "mlp.expert_bias",
                "mlp.shared_mlp.gate_proj.weight", "mlp.shared_mlp.up_proj.weight",
                "mlp.shared_mlp.down_proj.weight",
            ))
            for expert in range(experts):
                for name in ("gate_proj", "up_proj", "down_proj"):
                    keys.add(f"{prefix}mlp.experts.{expert}.{name}.weight")
        else:
            keys.update(f"{prefix}mlp.{name}_proj.weight" for name in ("gate", "up", "down"))
    return keys


def sparse_layers_from_config(config: dict) -> list[bool]:
    """Reference default: the first layer is dense when there is more than one, the rest sparse."""
    layers = int(config["num_hidden_layers"])
    if config.get("mlp_layer_types") is not None:
        types = list(config["mlp_layer_types"])
        if len(types) != layers or set(types) - {"dense", "sparse"}:
            raise ValueError("invalid mlp_layer_types")
        return [t == "sparse" for t in types]
    return [layer > 0 for layer in range(layers)]


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return weight * x.to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """cos/sin are [seq, head_dim] and broadcast over the head axis of q/k ([heads, seq, dim])."""
    cos, sin = cos.unsqueeze(0), sin.unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    heads, seq, dim = x.shape
    if n_rep == 1:
        return x
    return x[:, None, :, :].expand(heads, n_rep, seq, dim).reshape(heads * n_rep, seq, dim)


def eager_attention(query, key, value, n_rep, scaling, mask=None):
    key, value = repeat_kv(key, n_rep), repeat_kv(value, n_rep)
    weights = torch.matmul(query, key.transpose(1, 2)) * scaling
    if mask is not None:
        weights = weights + mask
    weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
    return torch.matmul(weights, value)


def expert_combine(x: torch.Tensor, top: torch.Tensor, weights: torch.Tensor, gate, up, down) -> torch.Tensor:
    """Routed-expert output of transformers' default experts path (``grouped_mm``), bit for bit.

    ``top``/``weights`` are [tokens, k] local expert ids and routing weights. The
    reference computes every expert's contribution, scales it in fp32 and sums the
    k slots in fp32 before rounding to bf16 once. Per-expert GEMMs by ``F.linear``
    on the separate gate/up matrices equal its fused ``torch._grouped_mm`` bit for
    bit at this model's shapes (checked in ``tests/test_engine.py``), so weights
    can stay zero-copy views. The eager loop in ``modeling_hy_v3`` that accumulates
    in bf16 expert by expert is a different, equally official path and differs in
    the last bit; it is covered by a tolerance test.
    """
    tokens, k = top.shape
    flat = top.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=len(gate))
    starts = torch.cumsum(counts, 0) - counts
    token = order // k
    contribution = torch.empty((flat.numel(), x.shape[1]), dtype=x.dtype)
    for e in counts.nonzero().flatten().tolist():
        lo, hi = int(starts[e]), int(starts[e] + counts[e])
        rows = token[lo:hi]
        contribution[lo:hi] = F.linear(F.silu(F.linear(x[rows], gate[e])) * F.linear(x[rows], up[e]), down[e])
    weighted = contribution * weights.reshape(-1)[order][:, None]          # fp32 [tokens * k, hidden]
    original = torch.empty_like(order)
    original[order] = torch.arange(order.numel())
    return weighted[original].view(tokens, k, x.shape[1]).sum(dim=1).to(x.dtype)


class Checkpoint:
    """Read-only, mmap-backed view of a checkpoint directory in the released HF layout."""

    def __init__(self, path):
        self.path = Path(path)
        self.config = json.loads((self.path / "config.json").read_text(encoding="utf-8"))
        check_config(self.config)
        index = json.loads((self.path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = index["weight_map"]
        prefix = "model.layers."
        routed = {int(name[len(prefix):].split(".", 1)[0]) for name in self.weight_map
                  if name.startswith(prefix) and name.endswith(".mlp.router.gate.weight")}
        self.sparse = [layer in routed for layer in range(int(self.config["num_hidden_layers"]))]
        if self.sparse != sparse_layers_from_config(self.config):
            raise ValueError("config and tensor layout disagree about which layers are sparse")
        expected = expected_keys(self.config, self.sparse)
        if set(self.weight_map) != expected:
            missing, extra = sorted(expected - set(self.weight_map)), sorted(set(self.weight_map) - expected)
            raise ValueError(f"unknown tensor layout: missing={missing[:4]} unexpected={extra[:4]}")
        self.sparse_layers = [layer for layer, is_sparse in enumerate(self.sparse) if is_sparse]
        self.num_experts = int(self.config["num_experts"])
        manifest = self.path / "prune_manifest.json"
        # Original expert id of every local expert, per sparse layer. Identity for a released checkpoint.
        self.kept = ({int(k): v for k, v in json.loads(manifest.read_text(encoding="utf-8"))["kept"].items()}
                     if manifest.exists() else None)
        self._handles: dict[str, object] = {}

    def tensor(self, name: str) -> torch.Tensor:
        shard = self.weight_map[name]
        handle = self._handles.get(shard)
        if handle is None:
            handle = safe_open(self.path / shard, framework="pt", device="cpu")
            self._handles[shard] = handle
            _HANDLES.append(handle)
        return handle.get_tensor(name)

    def origin_ids(self, layer: int) -> list[int]:
        return self.kept[layer] if self.kept else list(range(self.num_experts))


@dataclass
class Layer:
    idx: int
    sparse: bool
    ln1: torch.Tensor
    ln2: torch.Tensor
    wq: torch.Tensor
    wk: torch.Tensor
    wv: torch.Tensor
    wo: torch.Tensor
    q_norm: torch.Tensor
    k_norm: torch.Tensor
    router: torch.Tensor | None = None       # fp32 [experts, hidden]
    bias: torch.Tensor | None = None         # fp32 [experts]
    gate: list[torch.Tensor] | None = None   # experts x [intermediate, hidden]
    up: list[torch.Tensor] | None = None
    down: list[torch.Tensor] | None = None   # experts x [hidden, intermediate]
    shared: tuple | None = None
    dense: tuple | None = None
    denied: torch.Tensor | None = None       # fp32 [experts]: 0 kept, -inf pruned


class KVCache:
    """Pre-allocated key/value store. ``len`` counts the tokens every layer has already stored."""

    def __init__(self, layers, kv_heads, head_dim, capacity, dtype):
        self.k = torch.zeros(layers, kv_heads, capacity, head_dim, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.len = 0

    def append(self, layer, k, v):
        """Store one chunk for ``layer`` and return everything stored so far, the chunk included."""
        start, n = self.len, k.shape[1]
        self.k[layer, :, start:start + n] = k
        self.v[layer, :, start:start + n] = v
        return self.k[layer, :, :start + n], self.v[layer, :, :start + n]


@dataclass
class Generation:
    tokens: list[int]
    finish: str                 # "eos" or "length" (hit max_new_tokens without EOS)
    prompt_tokens: int
    prefill_seconds: float
    decode_seconds: float
    trace: dict | None = None   # logits [steps, vocab] bf16, route_ids [steps, layers, k], route_w same


class Engine:
    def __init__(self, checkpoint: Checkpoint, threads: int | None = None):
        self.ckpt = checkpoint
        if threads is not None:
            torch.set_num_threads(threads)
        cfg = checkpoint.config
        self.top_k = int(cfg["num_experts_per_tok"])
        self.router_scaling = float(cfg["router_scaling_factor"])
        self.eps = float(cfg["rms_norm_eps"])
        self.hidden = int(cfg["hidden_size"])
        self.n_heads = int(cfg["num_attention_heads"])
        self.n_kv = int(cfg["num_key_value_heads"])
        self.head_dim = int(cfg["head_dim"])
        self.n_rep = self.n_heads // self.n_kv
        self.scaling = self.head_dim ** -0.5
        self.max_context = min(int(cfg["max_position_embeddings"]), 8192)   # model card: max_context 8192
        eos = cfg["eos_token_id"]
        self.stop_ids = set(eos if isinstance(eos, list) else [eos])
        theta = float(cfg.get("rope_theta", 11158840.0))
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim))
        self._rope_cap, self._cos, self._sin = 0, None, None
        self.on_route = None       # fn(layer_idx, phase, top_ids [tokens, k], weights [tokens, k]); local ids
        self.audit = None          # fn(phase, changed_tokens, lost_mass) while a mask is active
        # Original expert id of every local expert; lets hooks and traces speak in released-checkpoint ids.
        self.origin = {layer: torch.tensor(checkpoint.origin_ids(layer)) for layer in checkpoint.sparse_layers}
        tensor = checkpoint.tensor
        self.embed = tensor("model.embed_tokens.weight")
        self.final_norm = tensor("model.norm.weight")
        self.lm_head = tensor("lm_head.weight")
        self.layers = [self._load_layer(i) for i in range(int(cfg["num_hidden_layers"]))]

    def _load_layer(self, idx: int) -> Layer:
        tensor, prefix = self.ckpt.tensor, f"model.layers.{idx}."
        layer = Layer(
            idx, self.ckpt.sparse[idx],
            tensor(prefix + "input_layernorm.weight"), tensor(prefix + "post_attention_layernorm.weight"),
            tensor(prefix + "self_attn.q_proj.weight"), tensor(prefix + "self_attn.k_proj.weight"),
            tensor(prefix + "self_attn.v_proj.weight"), tensor(prefix + "self_attn.o_proj.weight"),
            tensor(prefix + "self_attn.q_norm.weight"), tensor(prefix + "self_attn.k_norm.weight"),
        )
        if layer.sparse:
            # The reference keeps the router weight and bias in fp32 (bf16 values upcast exactly).
            layer.router = tensor(prefix + "mlp.router.gate.weight").float()
            layer.bias = tensor(prefix + "mlp.expert_bias").float()
            names = ("gate_proj", "up_proj", "down_proj")
            experts = [[tensor(f"{prefix}mlp.experts.{e}.{name}.weight") for name in names]
                       for e in range(self.ckpt.num_experts)]
            layer.gate, layer.up, layer.down = ([expert[i] for expert in experts] for i in range(3))
            layer.shared = tuple(tensor(f"{prefix}mlp.shared_mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        else:
            layer.dense = tuple(tensor(f"{prefix}mlp.{n}_proj.weight") for n in ("gate", "up", "down"))
        return layer

    def set_mask(self, kept: dict[int, list[int]] | None):
        """Restrict routing to ``kept[layer]`` (original expert ids). None lifts the mask."""
        if kept is None:
            for layer in self.layers:
                layer.denied = None
            return
        if set(kept) != set(self.ckpt.sparse_layers):
            raise ValueError("mask must name every sparse layer")
        for layer in self.layers:
            if not layer.sparse:
                continue
            ids = kept[layer.idx]
            valid = ids and all(type(i) is int and 0 <= i < self.ckpt.num_experts for i in ids)
            if not valid or len(set(ids)) != len(ids) or len(ids) < self.top_k:
                raise ValueError("invalid mask")
            allowed = torch.zeros(self.ckpt.num_experts, dtype=torch.bool)
            allowed[list(ids)] = True
            layer.denied = torch.where(allowed, 0.0, float("-inf")).float()

    def new_cache(self, capacity: int) -> KVCache:
        return KVCache(len(self.layers), self.n_kv, self.head_dim, capacity, torch.bfloat16)

    def _rope(self, positions: torch.Tensor):
        need = int(positions.max()) + 1
        if need > self._rope_cap:
            self._rope_cap = need + 1024
            freqs = torch.arange(self._rope_cap, dtype=torch.float)[:, None] * self.inv_freq
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos, self._sin = emb.cos().bfloat16(), emb.sin().bfloat16()
        return self._cos[positions], self._sin[positions]

    def _route(self, layer: Layer, x: torch.Tensor, phase: str):
        scores = torch.sigmoid(F.linear(x.float(), layer.router))
        choice = scores + layer.bias
        if layer.denied is not None:
            choice = choice + layer.denied
        top = torch.topk(choice, self.top_k, dim=-1, sorted=False).indices
        weights = scores.gather(1, top)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20) * self.router_scaling
        if layer.denied is not None and self.audit is not None:
            # What the unmasked model would have picked on the same hidden states.
            free = torch.topk(scores + layer.bias, self.top_k, dim=-1, sorted=False).indices
            changed = ~(torch.sort(top, dim=-1).values == torch.sort(free, dim=-1).values).all(dim=-1)
            free_w = scores.gather(1, free)
            free_w = free_w / (free_w.sum(dim=-1, keepdim=True) + 1e-20)
            self.audit(phase, int(changed.sum()), float(free_w[torch.isinf(layer.denied[free])].sum()))
        return top, weights

    def _moe(self, layer: Layer, x: torch.Tensor, phase: str) -> torch.Tensor:
        top, weights = self._route(layer, x, phase)
        if self.on_route is not None:
            self.on_route(layer.idx, phase, top, weights)
        out = expert_combine(x, top, weights, layer.gate, layer.up, layer.down)
        gate, up, down = layer.shared
        return out + F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)

    @torch.inference_mode()
    def forward(self, ids, cache: KVCache, phase: str = "decode") -> torch.Tensor:
        """Run one chunk of tokens through the model; returns bf16 logits [tokens, vocab]."""
        ids = ids if torch.is_tensor(ids) else torch.tensor(ids, dtype=torch.long)
        n = ids.numel()
        if cache.len + n > self.max_context:
            raise ValueError(f"sequence longer than max_context={self.max_context}")
        start = cache.len
        cos, sin = self._rope(torch.arange(start, start + n))
        # [query, key] additive causal mask; 2-D so it broadcasts over heads without adding a dimension.
        mask = None
        if n > 1:
            mask = torch.full((n, start + n), float("-inf"), dtype=torch.float32).triu(diagonal=start + 1)
        h = self.embed[ids]
        for layer in self.layers:
            residual = h
            x = rms_norm(h, layer.ln1, self.eps)
            shape = (n, -1, self.head_dim)
            q = F.linear(x, layer.wq).view(shape).transpose(0, 1)
            k = F.linear(x, layer.wk).view(shape).transpose(0, 1)
            v = F.linear(x, layer.wv).view(shape).transpose(0, 1)
            q, k = rms_norm(q, layer.q_norm, self.eps), rms_norm(k, layer.k_norm, self.eps)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            keys, values = cache.append(layer.idx, k, v)
            attn = eager_attention(q, keys, values, self.n_rep, self.scaling, mask)
            if attn.shape != (self.n_heads, n, self.head_dim):
                raise RuntimeError(f"attention output has shape {tuple(attn.shape)}, expected "
                                   f"{(self.n_heads, n, self.head_dim)}")
            h = residual + F.linear(attn.transpose(0, 1).reshape(n, self.hidden), layer.wo)
            residual = h
            x = rms_norm(h, layer.ln2, self.eps)
            if layer.sparse:
                h = residual + self._moe(layer, x, phase)
            else:
                gate, up, down = layer.dense
                h = residual + F.linear(F.silu(F.linear(x, gate)) * F.linear(x, up), down)
        cache.len = start + n
        return F.linear(rms_norm(h, self.final_norm, self.eps), self.lm_head)

    def generate(self, prompt_ids, max_new_tokens: int, trace: bool = False) -> Generation:
        """Greedy decoding: argmax of the bf16 logits, as the reference generate() does."""
        prompt_ids = [int(i) for i in prompt_ids]
        depth = len(self.ckpt.sparse_layers)
        if trace and depth == 0:
            raise ValueError("nothing to trace: the checkpoint has no sparse layers")
        pending = {"phase": None, "rows": None}
        id_steps, weight_steps, logits_steps = [], [], []
        previous = self.on_route

        def collector(layer_idx, phase, top, weights):
            if previous is not None:
                previous(layer_idx, phase, top, weights)
            if pending["phase"] != phase:
                pending["phase"], pending["rows"] = phase, [None] * depth
            pending["rows"][self.ckpt.sparse_layers.index(layer_idx)] = (self.origin[layer_idx][top], weights)

        def flush():
            ids, ws = (torch.stack(part) for part in zip(*pending["rows"]))   # [layers, tokens, k]
            id_steps.append(ids.transpose(0, 1))
            weight_steps.append(ws.transpose(0, 1))
            pending["phase"], pending["rows"] = None, None

        self.on_route = collector if trace else previous
        try:
            with torch.inference_mode():
                cache = self.new_cache(len(prompt_ids) + max_new_tokens)
                t0 = time.perf_counter()
                logits = self.forward(prompt_ids, cache, "prefill")
                prefill_seconds = time.perf_counter() - t0
                if trace:
                    logits_steps.append(logits)
                    flush()
                tokens, finish, t0 = [], "length", time.perf_counter()
                for _ in range(max_new_tokens):
                    nxt = int(logits[-1].float().argmax())
                    if nxt in self.stop_ids:
                        finish = "eos"
                        break
                    tokens.append(nxt)
                    logits = self.forward([nxt], cache, "decode")
                    if trace:
                        logits_steps.append(logits[-1:])
                        flush()
                decode_seconds = time.perf_counter() - t0
        finally:
            self.on_route = previous
        record = None
        if trace:
            record = {"logits": torch.cat(logits_steps),
                      "route_ids": torch.cat(id_steps),        # [steps, layers, k], original expert ids
                      "route_w": torch.cat(weight_steps)}      # [steps, layers, k]
        return Generation(tokens, finish, len(prompt_ids), prefill_seconds, decode_seconds, record)
