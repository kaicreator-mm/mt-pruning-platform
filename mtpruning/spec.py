"""Pinned inputs. Changing anything here means recorded evidence no longer describes the run."""

MODEL_REPO = "tencent/Hy-MT3-30B-A3B"
MODEL_REVISION = "08ae9acea91abe265c6236e757e49d8851e1ce06"
# Per-file sizes and hashes of the pinned revision, as listed by huggingface.co.
MANIFEST_PATH = "manifests/hf_manifest.tsv"

# The frozen pilot corpus. Only its manifest is committed (line numbers, hashes, rules);
# the text is rebuilt from the FLORES-200 tarball and is bound by these two hashes.
CORPUS_MANIFEST_SHA256 = "e516cc92a805eb3273c9376741ba5b4f24229d03d7189f0275d2829de848b901"
CORPUS_SHA256 = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"

DECODING = "greedy (argmax of bf16 logits)"
# Provisional gate of issue #1: chrF++ may not drop by more than this many points in any direction.
MAX_CHRF_DROP = 1.0

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
FLORES_LICENSE = "CC-BY-SA 4.0"
FLORES_TARBALL_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
# sha256 of each language file inside the tarball (member ./flores200_dataset/<split>/<code>.<split>)
FLORES_FILES = {
    ("dev", "zho_Hans"): "ae677073476a27d6d11db642df5c18b86ca5dfcbbeefeba7d99ec93b7f5bd177",
    ("dev", "eng_Latn"): "51a25f527fd81f428b6933d85a2f5e5544af92aa82042c7fbf0c4120327fba0a",
    ("dev", "mya_Mymr"): "fb8aeaf0144f1236287645eb8e91733276907448ad50bcd978ff9309d26487b9",
    ("devtest", "zho_Hans"): "b30158b1b76f7cee08530d8127a9769f4512597c7eefe97ae1ccf630d5b6e7fd",
    ("devtest", "eng_Latn"): "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88",
    ("devtest", "mya_Mymr"): "bbccb36a909003dfbaf68cf59b63729769cab723a0f90a85f68efd9ded7b1512",
}
