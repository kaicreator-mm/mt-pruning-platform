"""The frozen corpus: checked byte for byte before anything reads it."""
from __future__ import annotations

from pathlib import Path

from .identity import sha256_file
from .reviewed import handoff


class BlockedCorpus(RuntimeError):
    """The corpus on disk is not the frozen one. Nothing may be resampled: stop and report."""
    code = "BLOCKED_CORPUS_CONTRACT"


def load_frozen_corpus(corpus_dir, manifest_sha256=None) -> tuple[dict, dict]:
    """The manifest and the records by ID of the frozen corpus, or BlockedCorpus.

    The manifest bytes must hash to the pinned value and the corpus bytes to the
    hash inside the manifest. The reviewed r7 check then validates every record
    against the manifest (exact IDs, sizes, line numbers) and applies the R8
    duplicate rules to the normalized text.
    """
    from .spec import CORPUS_MANIFEST_SHA256
    manifest_sha256 = manifest_sha256 or CORPUS_MANIFEST_SHA256
    corpus_dir = Path(corpus_dir)
    manifest_path, corpus_path = corpus_dir / "corpus_manifest.json", corpus_dir / "corpus.jsonl"
    try:
        actual = sha256_file(manifest_path)
        if actual != manifest_sha256:
            raise ValueError(f"corpus manifest sha256 {actual} is not the frozen {manifest_sha256}")
        return handoff.corpus(manifest_path, manifest_sha256, corpus_path)
    except Exception as error:      # the reviewed check signals every violation with ValueError
        raise BlockedCorpus(f"{type(error).__name__}: {error}") from error
