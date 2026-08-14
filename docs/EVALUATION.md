# Independent evaluation corpus

The release gate is 50,000–100,000 source-disjoint originals. No evaluation image, byte hash, or source name may appear in the adaptation manifest. The frozen target is 80,000 images: 42,000 real and 38,000 AI.

Required real scenarios are photographs, UI/screenshots, human illustration/comics, film/video frames, conventional 3D/game renders, memes/composites, and logos/text/diagrams. AI scenarios are legacy GAN/diffusion, 2025–2026 generators, anime/CGI, and web-degraded outputs.

`evaluation/sources.toml` defines the slots and quotas. Put licensed source files under `data/evaluation/raw/<slot>/`, then build and audit:

```bash
python tools/build_evaluation.py \
  --config evaluation/sources.toml \
  --training-manifest data/training/manifest.csv \
  --output data/evaluation/manifest.csv \
  --report reports/evaluation-corpus.json
python tools/audit_manifest.py data/evaluation/manifest.csv data/evaluation/clean.csv \
  --report reports/evaluation-audit.json --workers 24 --min-class-count 16000
```

For Hugging Face image datasets, materialize a pinned revision with identical JPEG settings for both labels:

```bash
HF_TOKEN="$HF_READ_ONLY_TOKEN" python tools/materialize_hf_images.py \
  OWNER/DATASET data/evaluation/raw/ui \
  --revision FULL_COMMIT_SHA --split test --image-column image --limit 7000 \
  --license "SPDX-ID"
```

Every slot must retain its source URL, revision, and license in its local `provenance.json`. `source-specific` in the template must be replaced before the corpus is accepted. Non-commercial, unknown-license, and training-overlapping datasets are rejected for the paid bounty.

Evaluation runs both the clean original and a deterministic JPEG/WebP/downscale view, reporting metrics per source, generator family, and scenario:

```bash
python -m cyclopes.cli evaluate \
  --manifest data/evaluation/clean.csv --split test --paired-views \
  --checkpoint runs/vit-v1/cyclopes-vit.pt \
  --calibration runs/vit-v1/calibration.json \
  --report reports/vit-v1-evaluation.json --device cuda
```

The private bounty test remains the only blind final score. This corpus is a regression gate, not evidence that any public benchmark result transfers unchanged.
