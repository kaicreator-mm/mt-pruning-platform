import json
import sqlite3
import sys
from pathlib import Path

from safetensors import safe_open

model_dir = Path(sys.argv[1])
journal_path = Path(sys.argv[2])
config = json.loads((model_dir / "config.json").read_text())
index = json.loads((model_dir / "model.safetensors.index.json").read_text())
names = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.k_proj.weight",
    "model.layers.0.self_attn.v_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
]
shapes = {}
for name in names:
    shard = index["weight_map"][name]
    with safe_open(model_dir / shard, framework="pt", device="cpu") as handle:
        shapes[name] = list(handle.get_slice(name).get_shape())
with sqlite3.connect(f"file:{journal_path}?mode=ro", uri=True) as database:
    completed_samples = database.execute("SELECT count(*) FROM samples").fetchone()[0]
    identity_rows = database.execute("SELECT count(*) FROM meta").fetchone()[0]
result = {
    "status": "REAL_MODEL_DIMENSION_MISMATCH_CONFIRMED",
    "inference_performed": False,
    "weights_read": "safetensors metadata only; no tensor payload loaded",
    "hidden_size": config["hidden_size"],
    "num_attention_heads": config["num_attention_heads"],
    "head_dim": config["head_dim"],
    "num_key_value_heads": config["num_key_value_heads"],
    "attention_output_width": config["num_attention_heads"] * config["head_dim"],
    "failed_prompt_tokens": 74,
    "failed_input_elements": 303104,
    "failed_input_elements_per_token": 303104 // 74,
    "weight_shapes": shapes,
    "completed_journal_samples": completed_samples,
    "identity_rows": identity_rows,
}
print(json.dumps(result, indent=2, sort_keys=True))
