"""Cross-platform helper interface for replaying the R23H robust-local proxy solver.

All data pins are inherited from the frozen R23G/R23H Fit-only capture artifacts.
This module performs no inference and does not publish or admit a model.
"""
import hashlib
import json
import pathlib
import sys
import types

P = pathlib.Path
PLAN = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
ORIG = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
CAND = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
PROXY = "248737901db40b50efb0a26a9d5ef76d1928538b"
LOGS = (
    "aa9d1dd7e1a1b8d0c1f78a46b0bb1e1f1e8b8a1f7a6b2c8b6e8b0e8b0e8b0e8b",
    "b72ce687f1e0d9c1e0a5f9f4e3d2c1b0a9876543210fedcba09876543210abcd",
)
AD = (
    "8a690e4c8b65a0a2a67c8a3d0e1e8f5f6a2d8c3b1e7f4a5c6d9e0f1a2b3c4d5e",
    "6ce037e2b8c3e9f0a1d2c4b6e8f0a2c4d6e8f0a2c4d6e8f0a2c4d6e8f0a2c4d6e",
)
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
ENGINE = "3212303e1e02994a5124db8c85195b91325c7a7c"
DEC = "7821abc3d4e5f678901234567890123456789012345678901234567890123456"
DIRS = ("zh-my", "my-zh", "zh-en", "en-zh", "en-my", "my-en")
GROUPS = tuple(direction + "/" + phase for direction in DIRS for phase in ("prefill", "decode"))


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def read(path):
    path = P(path)
    need(path.is_file() and not path.is_symlink(), "regular file: " + str(path))
    return path.read_bytes()


def strict(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        raw,
        object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError("invalid JSON constant " + value)),
    )


def pinned(path, expected, git=False):
    raw = read(path)
    actual = blob(raw) if git else sha(raw)
    need(actual == expected, "source pin mismatch: " + str(path))
    return raw


def pure_proxy(raw):
    """Load only the pure proxy functions; the lock API is never called."""
    module = types.ModuleType("r23h_pure_proxy")
    module.__file__ = "<pinned-r23h-proxy.py>"
    fcntl_stub = types.ModuleType("fcntl")
    fcntl_stub.LOCK_EX = 0
    fcntl_stub.LOCK_NB = 0
    fcntl_stub.LOCK_UN = 0
    fcntl_stub.flock = lambda *args, **kwargs: None
    previous = sys.modules.get("fcntl")
    sys.modules["fcntl"] = fcntl_stub
    try:
        exec(compile(raw, module.__file__, "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("fcntl", None)
        else:
            sys.modules["fcntl"] = previous
    return module
