#!/usr/bin/env bash
# Pinned runtime for the D01b validation (CPU only). Verified on CC-LAN200, x86_64 Linux.
set -euxo pipefail
CONDA=${CONDA:-$HOME/miniconda3/bin/conda}
ENV=${ENV:-$HOME/miniconda3/envs/mt-pruning}
OUT=${OUT:-$(dirname "$0")/../evidence/env}
mkdir -p "$OUT"
if [ ! -x "$ENV/bin/python" ]; then "$CONDA" create -y -q -n "$(basename "$ENV")" python=3.13; fi
PIP="$ENV/bin/python -m pip"
$PIP install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://mirrors.nju.edu.cn/pytorch/whl/cpu "torch==2.10.0+cpu"
$PIP install --no-cache-dir --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    "transformers==5.24.1" "safetensors==0.8.0" "sacrebleu==2.6.0"
$PIP freeze > "$OUT/pip-freeze.txt"
"$ENV/bin/python" -I -m pip check
echo ENV_OK
