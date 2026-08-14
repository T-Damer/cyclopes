# Development and reproducibility

## Data

Training uses the pinned sources in [`DATASETS.md`](../DATASETS.md). After acquiring the archives, build the manifests in order:

```bash
python tools/prepare_training.py \
  --docci-root data/raw/docci \
  --cocoxgen-root data/raw/cocoxgen \
  --output data/training
python tools/prepare_fakeclue.py data/raw/fakeclue data/training/manifest.csv data/training/fakeclue-manifest.csv
python tools/add_pamela.py data/raw/pamela data/training/fakeclue-manifest.csv data/training/fastvit-manifest.csv
python tools/add_met.py data/raw/met/00000.parquet data/training/fastvit-manifest.csv data/training/fastvit-met-manifest.csv
python tools/prepare_arena.py data/arena/benchmark-v0.1
```

The materializer decodes both labels and rewrites them with identical JPEG settings. It records revisions, licenses, row counts, and the manifest hash. The loader rejects duplicate paths, duplicate bytes, and groups crossing splits.

## Train, calibrate, and evaluate

```bash
python -m cyclopes.cli train --manifest data/training/fastvit-manifest.csv --output runs/base.pt --report runs/base-train.json --epochs 4 --batch-size 128 --device cuda --pretrained --workers 8
python -m cyclopes.cli train --manifest data/training/fastvit-met-manifest.csv --initial-checkpoint runs/base.pt --output runs/cyclopes.pt --report runs/train.json --epochs 8 --batch-size 128 --learning-rate 5e-5 --device cuda --workers 8
python -m cyclopes.cli calibrate --manifest data/training/fastvit-met-manifest.csv --checkpoint runs/cyclopes.pt --split calibration --output runs/calibration.json --report runs/calibrate-report.json --device cuda
python -m cyclopes.cli evaluate --manifest data/arena/benchmark-v0.1/manifest.csv --checkpoint runs/cyclopes.pt --calibration runs/calibration.json --split test --report runs/arena-v0.1.json --device cuda
python -m cyclopes.cli export --checkpoint runs/cyclopes.pt --calibration runs/calibration.json --output extension/models/cyclopes-fastvit.onnx --report runs/export.json
pip install 'git+https://github.com/lynote-ai/ai-image-detector.git@d3f4976d36c59974a25f55a0a7850b9866d3223b' onnxconverter-common
python tools/export_sentry.py extension/models/cyclopes-sentry.onnx
python -m cyclopes.cli onnx-parity --manifest data/training/fastvit-met-manifest.csv --checkpoint runs/cyclopes.pt --calibration runs/calibration.json --onnx extension/models/cyclopes-fastvit.onnx --split test --report runs/onnx-parity.json
```

Arena is a development OOD benchmark and must not be presented as a frozen test. The final Synthbuster + RAISE-1k gate is opened only after model and calibration rules are frozen. The release threshold is exactly `0.65`.

The second graph is the Apache-2.0 `InfImagine/Sentry_image_models` ConvNeXt-Small, converted through the MIT-licensed `lynote-ai/ai-image-detector` adapter. The fixed browser blend is 68% FastViT and 32% ConvNeXt; its raw operating point `0.6794350376527292` is aligned to the displayed threshold `0.65`.

## Build and test

```bash
python -m pytest -q
npm ci
npm test
```

Load `dist/` at `chrome://extensions` → Developer mode → Load unpacked. Turn **Filter: ON**, disconnect networking, reload a page, and confirm that every analyzed image gets a score and AI scores at or above 65% are blurred. The final manual check compares Cyclopes with Rule34's **Filter AI posts** switch on the same result page.
