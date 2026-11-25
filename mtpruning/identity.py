"""Run identity: everything a recorded sample depends on, hashed.

The seven fields follow the reviewed r5/cache_plan.py contract, so a key built
here can be checked with that module's ``key`` function. The weights are bound
by content: every shard of a checkpoint directory is hashed and compared with
what the directory claims to be (the pinned HF listing, a prune manifest, or a
declared synthetic provenance). A directory of unknown provenance is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

from .corpus import PROMPT, SPLITS, TARGET_NAME
from .reviewed import cache_plan, reviewed_file_sha256

REPO = Path(__file__).resolve().parents[1]
FIELDS = ("teacher", "candidate", "corpus", "prompt", "runtime", "protocol", "code")
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja")
REQUIRED_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json")
MODES = ("real", "rehearsal")
CHUNK = 8 << 20


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(value) -> str:
    return hashlib.sha256(cache_plan.canonical(value).encode()).hexdigest()


def key(identity: dict, split: str) -> str:
    """The reviewed cache key of one split of a run."""
    return cache_plan.key({**{name: identity[name] for name in FIELDS}, "split": split})


def content_hashes(model_dir, names, cache_dir=None, trust_cache=False) -> dict[str, str]:
    """sha256 of each file.

    With ``trust_cache`` (stage-boundary rechecks of immutable inputs) an earlier
    result is reused only while every identity the filesystem tracks for the file
    is unchanged: size, mtime, ctime, inode and device. ctime cannot be set by
    user code, so an edit that restores size and mtime is still rehashed - a
    size+mtime-only cache would trust an unverifiable digest (R11-P1-02).
    Without ``trust_cache`` (qualification) the actual bytes are always hashed;
    the cache is only refreshed.
    """
    model_dir = Path(model_dir)
    cache, cache_file = {}, None
    if cache_dir is not None:
        cache_file = Path(cache_dir) / (hashlib.sha256(str(model_dir.resolve()).encode()).hexdigest()[:32] + ".json")
        cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}
    files, changed = {}, False
    for name in names:
        status = (model_dir / name).stat()
        fingerprint = {"bytes": status.st_size, "mtime_ns": status.st_mtime_ns, "ctime_ns": status.st_ctime_ns,
                       "ino": status.st_ino, "device": status.st_dev}
        entry = cache.get(name)
        if not trust_cache or entry is None or any(entry.get(field) != value for field, value in fingerprint.items()):
            entry = {**fingerprint, "sha256": sha256_file(model_dir / name)}
            cache[name], changed = entry, True
        files[name] = entry["sha256"]
    if cache_file is not None and changed:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_file.with_name(f"{cache_file.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(cache, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_file)
    return files


def pinned_manifest() -> dict[str, str]:
    """Path to recorded hash, from the committed listing of the pinned revision."""
    from .spec import MANIFEST_PATH
    rows = {}
    for line in (REPO / MANIFEST_PATH).read_text(encoding="utf-8").splitlines():
        if line.strip():
            _size, oid, path = line.split("\t")
            rows[path] = oid
    return rows


def _pinned_match(path: Path, oid: str, sha256: str) -> bool:
    if len(oid) == 64:
        return oid == sha256
    data = path.read_bytes()                       # small files are listed by git blob id
    blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
    return blob == oid


def _require(files: dict, expected: dict, what: str) -> None:
    wrong = [name for name in files if expected.get(name) != files[name]]
    if wrong:
        raise ValueError(f"checkpoint content differs from its {what}: {wrong[:4]}")


def checkpoint_identity(model_dir, cache_dir=None, fresh: bool = True) -> dict:
    """Bind a checkpoint directory by content; refuse one that declares no provenance.

    ``fresh`` hashes the actual bytes; the trusted cache is only for stage-boundary
    rechecks (see content_hashes).
    """
    from .spec import MODEL_REPO, MODEL_REVISION
    model_dir = Path(model_dir)
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    names = ["config.json", "model.safetensors.index.json", *sorted(set(index["weight_map"].values()))]
    files = content_hashes(model_dir, names, cache_dir, trust_cache=not fresh)
    num_experts = config["num_experts"]
    prune, provenance = model_dir / "prune_manifest.json", model_dir / "checkpoint_provenance.json"
    if prune.exists():
        manifest = json.loads(prune.read_text(encoding="utf-8"))
        _require(files, {**{f["name"]: f["sha256"] for f in manifest["files"]},
                         "model.safetensors.index.json": manifest["index_sha256"],
                         "config.json": manifest["config_sha256"]}, "prune manifest")
        return {"kind": "pruned", "num_experts": num_experts, "files": files, "teacher": manifest["source"]["digest"],
                "candidate": digest({"kind": "pruned", "num_experts": num_experts, "kept": manifest["kept"],
                                     "files": files})}
    if provenance.exists():
        declared = json.loads(provenance.read_text(encoding="utf-8"))
        if declared.get("kind") != "synthetic":
            raise ValueError(f"unsupported checkpoint provenance: {declared.get('kind')!r}")
        _require(files, declared["files"], "declared provenance")
        teacher = digest({"kind": "synthetic", "seed": declared["seed"], "files": files})
        return {"kind": "synthetic", "num_experts": num_experts, "files": files, "teacher": teacher,
                "candidate": digest({"kind": "identity", "teacher": teacher})}
    pinned = pinned_manifest()
    if any(name not in pinned for name in names):
        raise ValueError("checkpoint provenance unknown: not the pinned revision and no manifest declares it")
    # The pinned snapshot's effective tokenizer bytes are pinned too: they are hashed and
    # checked against the pinned listing like every shard (R11-P1-02).
    tokenizer = [name for name in TOKENIZER_FILES if (model_dir / name).exists()]
    missing = [name for name in REQUIRED_TOKENIZER_FILES if name not in tokenizer]
    if missing:
        raise ValueError(f"{model_dir} does not carry the tokenizer: {', '.join(missing)} missing")
    if any(name not in pinned for name in tokenizer):
        raise ValueError("the tokenizer files are not part of the pinned revision listing")
    files.update(content_hashes(model_dir, tokenizer, cache_dir, trust_cache=not fresh))
    wrong = [name for name in names + tokenizer if not _pinned_match(model_dir / name, pinned[name], files[name])]
    if wrong:
        raise ValueError(f"checkpoint content differs from the pinned revision: {wrong[:4]}")
    teacher = digest({"kind": "pinned-snapshot", "repo": MODEL_REPO, "revision": MODEL_REVISION, "files": files})
    return {"kind": "pinned-snapshot", "repo": MODEL_REPO, "revision": MODEL_REVISION, "num_experts": num_experts,
            "files": files, "teacher": teacher, "candidate": digest({"kind": "identity", "teacher": teacher})}


def _git_state() -> dict:
    def git(*args):
        try:
            done = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True,
                                  timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() or None
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", "mtpruning", "scripts", "demos/followup") if head else None
    return {"git_head": head, "git_dirty": bool(dirty) if dirty is not None else None}


def code_identity() -> dict:
    """Hashes of everything that executes: this package, the driver scripts and the reused helpers.

    The git head and the dirty flag are recorded next to them but not hashed: a commit
    that leaves these files unchanged must not stop a run from resuming.
    """
    files = sorted((REPO / "mtpruning").glob("*.py")) + sorted((REPO / "scripts").glob("*.sh"))
    hashes = {path.relative_to(REPO).as_posix(): sha256_file(path) for path in files}
    hashes.update(reviewed_file_sha256())
    return {"files": hashes, **_git_state()}


def runtime_identity(threads) -> dict:
    import numpy
    import sacrebleu
    import safetensors
    import torch
    import transformers
    return {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "safetensors": safetensors.__version__,
            "numpy": numpy.__version__, "sacrebleu": sacrebleu.__version__, "platform": platform.platform(),
            "machine": platform.machine(), "hostname": socket.gethostname(), "threads": threads}


def run_identity(*, model_dir, corpus_path, manifest_path, cache_dir=None, mask=None, selection_sha256=None,
                 decoding, max_new_tokens, protocol, threads, tokenizer_dir=None, mode="rehearsal",
                 fresh: bool = True) -> dict:
    """``protocol`` carries whatever else two runs must agree on to share a journal.

    The identity binds the *effective* tokenizer (``tokenizer_dir``), not a declared
    one: whatever bytes decode the outputs are hashed here (R11-P1-02). ``mode`` is
    explicit: a real run refuses a tokenizer override and a synthetic teacher; a
    synthetic checkpoint is only ever a rehearsal teacher. ``fresh`` qualifies the
    checkpoint bytes by hashing them; a resume may trust the content-keyed cache.

    ``decoding`` and ``max_new_tokens`` are kept outside the protocol hash as well: they
    must match across the baseline and every candidate even when the runs differ in
    which sentences they trace or how many they cover.
    """
    if mode not in MODES:
        raise ValueError(f"unknown run mode: {mode!r}")
    model_dir = Path(model_dir)
    tokenizer_dir = Path(tokenizer_dir) if tokenizer_dir else model_dir
    if mode == "real":
        if tokenizer_dir.resolve() != model_dir.resolve():
            raise ValueError("a real run decodes with the checkpoint's own tokenizer: overrides are refused")
    checkpoint = checkpoint_identity(model_dir, cache_dir, fresh=fresh)
    if mode == "real":
        if checkpoint["kind"] == "synthetic":
            raise ValueError("a synthetic checkpoint is only a rehearsal teacher, never a real one")
        if checkpoint["kind"] == "pruned":
            manifest = json.loads((model_dir / "prune_manifest.json").read_text(encoding="utf-8"))
            if manifest["source"]["kind"] != "pinned-snapshot":
                raise ValueError("a real candidate must be pruned from the pinned teacher")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    corpus_sha256 = sha256_file(corpus_path)
    if corpus_sha256 != manifest["corpus_sha256"]:
        raise ValueError("corpus bytes differ from the manifest")
    tokenizer = {name: sha256_file(tokenizer_dir / name) for name in TOKENIZER_FILES
                 if (tokenizer_dir / name).exists()}
    missing = [name for name in REQUIRED_TOKENIZER_FILES if name not in tokenizer]
    if missing:
        raise ValueError(f"{tokenizer_dir} does not carry the tokenizer: {', '.join(missing)} missing")
    if mask is not None:
        candidate = digest({"kind": "masked", "kept": {str(k): v for k, v in sorted(mask.items())},
                            "selection_sha256": selection_sha256})
    else:
        candidate = checkpoint["candidate"]
    runtime = runtime_identity(threads)
    code = code_identity()
    binding = {
        "teacher": checkpoint["teacher"], "candidate": candidate, "corpus": corpus_sha256,
        "prompt": digest({"templates": PROMPT, "target_names": TARGET_NAME, "tokenizer": tokenizer}),
        "runtime": digest(runtime), "code": digest(code["files"]),
        # The decoding settings are part of the protocol hash, so they are part of every sample key.
        "protocol": digest({"run": protocol, "decoding": decoding, "max_new_tokens": max_new_tokens}),
        "decoding": digest({"decoding": decoding, "max_new_tokens": max_new_tokens}),
        "detail": {"checkpoint": {k: checkpoint[k] for k in ("kind", "num_experts", "files")},
                   "tokenizer": tokenizer, "tokenizer_dir": str(tokenizer_dir.resolve()), "mode": mode,
                   "runtime": runtime, "protocol": protocol, "code": code,
                   "decoding": decoding, "max_new_tokens": max_new_tokens},
    }
    return {**binding, "sample_keys": {split: key(binding, split) for split in SPLITS}}
