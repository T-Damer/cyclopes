#!/usr/bin/env bash
set -euo pipefail

: "${HF_READ_ONLY_TOKEN:?HF_READ_ONLY_TOKEN is required}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

python - <<'PY'
import os
import shutil
import torch

free_gb = shutil.disk_usage(".").free / 1024**3
minimum_free_gb = float(os.environ.get("MIN_FREE_GB", "40"))
assert free_gb >= minimum_free_gb, f"need at least {minimum_free_gb:.0f} GiB free, found {free_gb:.1f}"
assert torch.cuda.is_available(), "CUDA is unavailable"
major, _minor = torch.cuda.get_device_capability()
assert major >= 9, f"expected Hopper or newer, got compute capability {major}"
print({"torch": torch.__version__, "cuda": torch.version.cuda, "free_gb": round(free_gb, 1)})
PY

python -m pip install --no-cache-dir -r requirements-vast.txt
python -m pip install --no-deps -e .
python -m pytest -q tests/test_vit_modeling.py tests/test_cli.py tests/test_core.py

HF_TOKEN="$HF_READ_ONLY_TOKEN" python - <<'PY'
from cyclopes.vit_modeling import MultiLayerScalePairViT

model = MultiLayerScalePairViT.from_pretrained()
print({"parameters": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)})
PY
