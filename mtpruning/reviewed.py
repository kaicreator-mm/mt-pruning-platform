"""The reviewed offline helpers merged in demos/followup (PR #3, main b2b486fc).

They are loaded from their files, so this branch runs exactly the reviewed code,
and their hashes are part of every run's code identity. What they check:
  r6/bridge.py      aggregate calibration coverage before selection; the
                    independent chrF++/diagnostics gate of report rows
  r7/handoff.py     the frozen-corpus contract with the R8 duplicate rules,
                    run records, and the validation-only candidate selection
  r4/trace.py       Journal: one committed sample per identity and sample ID
  r5/cache_plan.py  the experiment identity fields and their key
None of them proves where a checkpoint came from; identity.py does that.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "demos" / "followup"
FILES = {"bridge": "r6/bridge.py", "handoff": "r7/handoff.py", "trace": "r4/trace.py",
         "cache_plan": "r5/cache_plan.py"}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(f"d01b_reviewed_{name}", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bridge = _load("bridge", FILES["bridge"])
handoff = _load("handoff", FILES["handoff"])
trace = _load("trace", FILES["trace"])
cache_plan = _load("cache_plan", FILES["cache_plan"])


def reviewed_file_sha256() -> dict[str, str]:
    return {f"demos/followup/{relative}": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            for relative in FILES.values()}
