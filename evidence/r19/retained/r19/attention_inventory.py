import datetime
import json
from contextlib import ExitStack
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
source_identity = sys.argv[2]
config = json.loads((model_dir / "config.json").read_text())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
weight_map = index["weight_map"]
hidden = int(config["hidden_size"])
heads = int(config["num_attention_heads"])
kv_heads = int(config["num_key_value_heads"])
head_dim = int(config["head_dim"])
if min(hidden, heads, kv_heads, head_dim) <= 0 or heads % kv_heads:
    raise ValueError("invalid attention dimensions in config")
attention_width = heads * head_dim
kv_width = kv_heads * head_dim
expected = {
    "q_proj": (attention_width, hidden),
    "k_proj": (kv_width, hidden),
    "v_proj": (kv_width, hidden),
    "o_proj": (hidden, attention_width),
}
handles = {}
layers = []
mismatches = []
with ExitStack() as stack:
    for layer in range(int(config["num_hidden_layers"])):
        shapes = {}
        for projection, expected_shape in expected.items():
            name = f"model.layers.{layer}.self_attn.{projection}.weight"
            shard = weight_map[name]
            if shard not in handles:
                handles[shard] = stack.enter_context(safe_open(model_dir / shard, framework="pt", device="cpu"))
            actual = tuple(handles[shard].get_slice(name).get_shape())
            shapes[projection] = {"name": name, "expected": expected_shape, "actual": actual,
                                  "pass": actual == expected_shape}
            if actual != expected_shape:
                mismatches.append(shapes[projection])
        layers.append({"layer": layer, "pass": all(item["pass"] for item in shapes.values()),
                       "projections": shapes})
result = {
    "status": "PASS" if not mismatches else "FAIL",
    "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    "source_identity": source_identity,
    "model_dir": str(model_dir),
    "weights_read": "safetensors metadata only; no tensor payload loaded and no files modified",
    "hidden_size": hidden,
    "num_attention_heads": heads,
    "num_key_value_heads": kv_heads,
    "head_dim": head_dim,
    "attention_width": attention_width,
    "kv_width": kv_width,
    "layers_checked": len(layers),
    "tensors_checked": sum(len(layer["projections"]) for layer in layers),
    "mismatches": mismatches,
    "layers": layers,
}
print(json.dumps(result, indent=2, sort_keys=True))
sys.exit(0 if not mismatches else 1)
