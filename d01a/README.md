# D01a synthetic expert-pruning demo

Run `python -m pip install torch safetensors`, then `python -m py_compile demo.py` and `python demo.py > result.json`.
Tested with Python 3.13.5, PyTorch 2.10.0+cpu. No actual translation model or text is used. Six direction labels designate synthetic feature distributions, not actual linguistic features. One synthetic MoE block, 16 experts/top-4. The baseline mask is applied before top-k; routing weights are normalized sigmoid scores, while bias is used only for selection. The compact version retains the same expert weights and shared experts. A zero masked error does not establish zero error against the full model or translation quality. See GitHub issue #1 for D01b requirements. No model weights are included.
