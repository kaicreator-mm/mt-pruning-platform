"""FLORES-200 sample for the six zh/en/my directions.

The public FLORES-200 release has two splits: dev (997 sentences) and devtest
(1012); test is withheld. Calibration comes from dev, validation and test are
two disjoint subsets of devtest. Sentences are drawn once per *source* language
with a seeded RNG, so directions sharing a source share their source sentences,
and references are the line-aligned target-language text.
"""
from __future__ import annotations

import hashlib
import json
import random
import tarfile
from pathlib import Path

from .spec import FLORES_FILES, FLORES_LICENSE, FLORES_TARBALL_SHA256, FLORES_URL

FLORES_LANG = {"zh": "zho_Hans", "en": "eng_Latn", "my": "mya_Mymr"}
LANGUAGES = ("zh", "en", "my")
DIRECTIONS = ("zh-my", "my-zh", "zh-en", "en-zh", "en-my", "my-en")   # zh→my first: the priority direction
SPLITS = ("calibration", "validation", "test")
# Official "Default Translation" prompts from the model card, verbatim apart from the named placeholders.
PROMPT = {
    "zh": "将以下文本翻译为 `{target}`，注意**只需要输出翻译后的结果，不要额外解释**：\n\n`{source}`",
    "en": "Translate the following text into `{target}`. Note that you should **only output the translated "
          "result without any additional explanation**:\n\n`{source}`",
}
TARGET_NAME = {"zh": {"zh": "中文", "en": "英语", "my": "缅甸语"},
               "en": {"zh": "Chinese", "en": "English", "my": "Burmese"}}


def direction(src_lang: str, tgt_lang: str) -> str:
    return f"{src_lang}-{tgt_lang}"


def prompt_lang(src_lang: str) -> str:
    """Chinese prompt when the source is Chinese, English otherwise (the two usages the model card shows)."""
    return "zh" if src_lang == "zh" else "en"


def build_prompt(src_lang: str, tgt_lang: str, source: str) -> tuple[str, str]:
    lang = prompt_lang(src_lang)
    return PROMPT[lang].format(target=TARGET_NAME[lang][tgt_lang], source=source), lang


def read_flores(tarball: Path) -> dict[tuple[str, str], list[str]]:
    """Verify the tarball and return the six line-aligned language files, keyed by (split, code)."""
    if hashlib.sha256(Path(tarball).read_bytes()).hexdigest() != FLORES_TARBALL_SHA256:
        raise ValueError("FLORES-200 tarball sha256 mismatch")
    texts = {}
    with tarfile.open(tarball, "r:gz") as archive:
        for (split, code), want in FLORES_FILES.items():
            data = archive.extractfile(f"./flores200_dataset/{split}/{code}.{split}").read()
            if hashlib.sha256(data).hexdigest() != want:
                raise ValueError(f"FLORES-200 file hash mismatch: {split}/{code}")
            lines = data.decode("utf-8").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            texts[(split, code)] = lines
    for split in ("dev", "devtest"):
        if len({len(text) for (name, _), text in texts.items() if name == split}) != 1:
            raise ValueError("FLORES-200 files are line-aligned; unequal line counts")
    return texts


def sample_indices(texts, n_calibration: int, n_validation: int, n_test: int, seed: int) -> dict:
    """Per source language: line numbers of each split, all disjoint by construction."""
    rng = random.Random(seed)
    picks = {}
    for lang in LANGUAGES:
        dev, devtest = texts[("dev", FLORES_LANG[lang])], texts[("devtest", FLORES_LANG[lang])]
        if not (n_calibration <= len(dev) and n_validation + n_test <= len(devtest)):
            raise ValueError("sample larger than split")
        drawn = sorted(rng.sample(range(len(devtest)), n_validation + n_test))
        picks[lang] = {
            "calibration": sorted(rng.sample(range(len(dev)), n_calibration)),
            "validation": drawn[:n_validation],
            "test": drawn[n_validation:],
        }
        sets = [set(v) for v in picks[lang].values()]
        if sets[1] & sets[2]:
            raise AssertionError("validation and test overlap")
    return picks


def build_corpus(tarball: Path, tokenizer, out_dir: Path, n_calibration=24, n_validation=20, n_test=20,
                 seed=42) -> dict:
    texts = read_flores(tarball)
    picks = sample_indices(texts, n_calibration, n_validation, n_test, seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for split in SPLITS:
        for name in DIRECTIONS:
            src_lang, tgt_lang = name.split("-")
            for line in picks[src_lang][split]:
                flores_split = "dev" if split == "calibration" else "devtest"
                source = texts[(flores_split, FLORES_LANG[src_lang])][line]
                reference = texts[(flores_split, FLORES_LANG[tgt_lang])][line]
                text, lang = build_prompt(src_lang, tgt_lang, source)
                encoded = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                        add_generation_prompt=True, return_dict=True,
                                                        return_tensors="pt")
                records.append({
                    "id": f"{split}:{name}:{line}", "split": split, "direction": name,
                    "src_lang": src_lang, "tgt_lang": tgt_lang,
                    "flores_split": "dev" if split == "calibration" else "devtest", "line": line,
                    "source": source, "reference": reference, "prompt_lang": lang, "prompt": text,
                    "prompt_ids": encoded["input_ids"][0].tolist(),
                })
    body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
    (out_dir / "corpus.jsonl").write_text(body, encoding="utf-8", newline="\n")
    manifest = {
        "corpus": "FLORES-200", "license": FLORES_LICENSE, "url": FLORES_URL,
        "tarball_sha256": FLORES_TARBALL_SHA256,
        "files": {f"{split}/{code}": digest for (split, code), digest in FLORES_FILES.items()},
        "split_sizes": {split: len(next(text for (name, _), text in texts.items() if name == split))
                        for split in ("dev", "devtest")},
        "note": "public FLORES-200 has dev and devtest only; calibration=dev, validation/test=disjoint devtest subsets",
        "seed": seed, "sizes_per_direction": {"calibration": n_calibration, "validation": n_validation,
                                              "test": n_test},
        "directions": list(DIRECTIONS),
        "prompt_policy": "model-card Default Translation prompt; Chinese prompt iff source is Chinese",
        "indices": picks, "records": len(records),
        "corpus_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    }
    (out_dir / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                                                  encoding="utf-8", newline="\n")
    return manifest


def load_corpus(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
