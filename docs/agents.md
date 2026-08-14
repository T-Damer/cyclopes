# Development and reproducibility

Use Python 3.11 and the pinned training requirements:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements-train.txt
npm ci
```

## Data

Source revisions and terms are frozen in [`DATASETS.md`](../DATASETS.md). Materialize every class as the same RGB JPEG codec, then merge and audit:

```bash
.venv/bin/python tools/prepare_training.py --docci-root data/raw/docci --cocoxgen-root data/raw/cocoxgen --output data/core --nano-limit 1000
.venv/bin/python tools/add_met.py data/raw/met/openaccess.parquet data/core/manifest.csv data/met/manifest.csv --limit 5000
.venv/bin/python tools/add_pamela.py data/raw/pamela data/met/manifest.csv data/current/manifest.csv
.venv/bin/python tools/prepare_blender_frames.py data/blender
.venv/bin/python tools/generate_graphics_negatives.py data/graphics --count 2000
.venv/bin/python tools/generate_legacy.py biggan data/legacy-biggan --count 5000 --batch-size 64
.venv/bin/python tools/generate_legacy.py ldm data/legacy-ldm --count 2000
.venv/bin/python tools/generate_glide.py data/legacy-glide --count 2000
.venv/bin/python tools/import_legacy_npz.py data/raw/adm-samples.npz data/legacy-adm --generator adm
.venv/bin/python tools/merge_manifests.py data/final/manifest.csv data/current/manifest.csv data/blender/manifest.csv data/graphics/manifest.csv data/legacy-biggan/manifest.csv data/legacy-ldm/manifest.csv data/legacy-adm/manifest.csv data/legacy-glide/manifest.csv
.venv/bin/python tools/audit_manifest.py data/final/manifest.csv data/final/clean.csv --report reports/data-preflight.json --workers 24

# V5 replay set: licensed digital-art/CGI negatives plus modern AI replay
.venv/bin/python tools/prepare_hard_negatives.py data/hard-negatives --digital-art 2600 --cgi 1200 --photos 0
.venv/bin/python tools/prepare_ai_replay.py data/ai-replay --cocoxgen 2750 --pamela 2750
.venv/bin/python tools/merge_manifests.py data/replay/manifest.csv data/hard-negatives/manifest.csv data/ai-replay/manifest.csv
.venv/bin/python tools/audit_manifest.py data/replay/manifest.csv data/replay/clean.csv --report reports/data-preflight-replay.json --workers 12 --min-class-count 3000
```

`audit_manifest.py` verifies bytes and declared hashes, rejects split leakage and cross-label matches confirmed by both pHash and dHash, removes confirmed same-label duplicates and near-empty frames, and requires 20,000 images per class. Related video frames and DOCCI bursts remain in one split.

## Train and export

```bash
.venv/bin/python -m cyclopes.cli train --manifest data/final/clean.csv --output artifacts/cyclopes-v2.pt --report reports/train-v2.json --device cuda --batch-size 64 --workers 40 --epochs 6 --max-steps 4000
.venv/bin/python -m cyclopes.cli calibrate --manifest data/final/clean.csv --checkpoint artifacts/cyclopes-v2.pt --split calibration --device cuda --batch-size 128 --output artifacts/calibration.json --report reports/calibration-v2.json
.venv/bin/python -m cyclopes.cli evaluate --manifest data/final/clean.csv --checkpoint artifacts/cyclopes-v2.pt --split test --device cuda --batch-size 128 --calibration artifacts/calibration.json --report reports/test-v2.json
.venv/bin/python -m cyclopes.cli export --checkpoint artifacts/cyclopes-v2.pt --calibration artifacts/calibration.json --output extension/models/cyclopes.onnx --report reports/export-v2.json
.venv/bin/python -m cyclopes.cli onnx-parity --manifest data/final/clean.csv --checkpoint artifacts/cyclopes-v2.pt --calibration artifacts/calibration.json --onnx extension/models/cyclopes.onnx --split test --report reports/parity-v2.json --tolerance 0.01

# Accepted V5 replay fine-tune: one bounded, frozen-backbone pass from V2
.venv/bin/python -m cyclopes.cli train --manifest data/replay/clean.csv --initial-checkpoint data/artifacts/cyclopes-v2.pt --output data/artifacts/cyclopes-v5.pt --report reports/train-v5.json --device mps --epochs 1 --max-steps 100 --batch-size 16 --backbone-lr 0 --head-lr 0.00001 --freeze-steps 100 --workers 0
.venv/bin/python -m cyclopes.cli calibrate --manifest data/replay/clean.csv --checkpoint data/artifacts/cyclopes-v5.pt --split calibration --device mps --batch-size 32 --workers 0 --output data/artifacts/calibration-v5.json --report reports/calibration-v5.json
.venv/bin/python -m cyclopes.cli export --checkpoint data/artifacts/cyclopes-v5.pt --calibration data/artifacts/calibration-v5.json --output extension/models/cyclopes.onnx --report reports/export-v5.json
```

The checkpoint is selected by the worse of clean and web-transcoded validation balanced accuracy. Calibration, validation, test, and the fixed 65% browser boundary remain separate.

## vNext ViT adaptation on Vast

The released MobileNet ScalePair model remains the v0.1 baseline. The bounded vNext experiment freezes the directly licensed Community Forensics ViT-S prior and trains only Cyclopes multi-layer/scale residual heads. Build the independent evaluation corpus first; its procedure is in [`EVALUATION.md`](EVALUATION.md).

```bash
bash vast/bootstrap.sh
TRAIN_MANIFEST=/workspace/data/training/clean.csv \
EVAL_MANIFEST=/workspace/data/evaluation/clean.csv \
MAX_SECONDS=5400 bash vast/run.sh
```

The bootstrap requires an H200-class CUDA device, at least 180 GB free disk, and `HF_READ_ONLY_TOKEN`. It never reads or modifies the local Vast API key or SSH keys. Copy the result archive before destroying the rented instance.

## Field and browser checks

```bash
.venv/bin/python tools/prepare_field_regression.py tests/field-regression.json data/field-regression
.venv/bin/python -m cyclopes.cli evaluate --manifest data/field-regression/manifest.csv --checkpoint data/artifacts/cyclopes-v5.pt --calibration data/artifacts/calibration-v5.json --browser-view --predictions reports/field-browser-predictions-v5.json --report reports/field-browser-v5.json
.venv/bin/python -m cyclopes.cli onnx-parity --manifest data/field-regression/manifest.csv --checkpoint data/artifacts/cyclopes-v5.pt --calibration data/artifacts/calibration-v5.json --onnx extension/models/cyclopes.onnx --browser-view --report reports/parity-field-browser-v5.json --tolerance 0.001
.venv/bin/python -m pytest -q
npm test
```

Load `dist/` at `chrome://extensions` → Developer mode → Load unpacked. Reload a local fixture page once, click the Cyclopes icon, and verify loading badges, confidence badges, blur at ≥65%, dynamic images, scrolling, thumbnails, full images, and the GIF first frame. Field cases are frozen evaluation-only data and must never influence training, calibration, threshold, or checkpoint selection. No live Rule34 check is part of the vNext training gate.
