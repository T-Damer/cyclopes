# Dataset plan

The private bounty benchmark is unavailable, so Cyclopes uses multiple independent distributions and never treats one random train/test split as proof of generalization.

## Training and model selection

| Source | Role | Why | License / terms |
| --- | --- | --- | --- |
| [Google DOCCI](https://huggingface.co/datasets/google/docci) at `a0a43eaf34676ffd008fb6565dd8c2ba00d09100` | Real train/calibration/validation | 14,847 original photographs | CC BY 4.0 |
| [BitMind Nano Banana](https://huggingface.co/datasets/bitmind/nano-banana) at `9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4` | AI train/calibration/validation | Deterministic first 300 Gemini 2.5 Flash Image outputs | MIT |
| [Nano Banana Pro 1K](https://huggingface.co/datasets/ash12321/nano-banana-pro-generated-1k) at `01f43edd35eb2f47f5ea77cfb223d173083d61ad` | AI train/calibration/validation | 200 newer Nano Banana Pro outputs | MIT |
| [COCOXGEN](https://huggingface.co/datasets/heikeadel/cocoxgen) at `c336ad187c2ab298ce825df65088bdacbae104f6` | AI train/calibration/validation | 2,406 available Fooocus and SDXL outputs | CC BY 4.0 |
| [FakeClue GenImage](https://huggingface.co/datasets/bitmind/FakeClue) at `cef6f2303971a38e76616d507787cdc22544a5c5` | Balanced train/calibration/validation | 9,000 real and 9,000 AI images from six generator families | Apache 2.0 |
| [PAMELA](https://huggingface.co/datasets/pamela-dataset/pamela) at `14ebd68d2a2c34367d41020b62ee60b7504725fb` | AI train/calibration/validation | 1,800 available Flux 2 and Nano Banana Pro images | Annotations CC BY 4.0; image use follows generator terms |
| [The Met Open Access](https://huggingface.co/datasets/metmuseum/openaccess) at `c65f8d6041aea7b3bc767a54d93772c3c6a365f6` | Real train/calibration/validation | 6,626 unique public-domain artworks from the first pinned shard | CC0 1.0 |

All classes are decoded and rewritten as RGB JPEG at the same settings, removing EXIF, C2PA, text chunks, filenames, and source-codec shortcuts. The CSV retains only fields needed to audit splits. Files and related groups cannot cross splits.

The resulting weights are original Cyclopes training output. Training images are not redistributed; attribution and pinned revisions are retained here and in the generated provenance report. OpenFake is excluded from training because its CC BY-NC 4.0 license is unsuitable for a paid bounty release.

## Frozen tests

The final frozen source is never used for training, checkpoint selection, calibration, fusion, or threshold tuning. Arena is a development OOD benchmark because it has already informed model selection.

| Source | Gate | Coverage |
| --- | --- | --- |
| [AI Detector Arena v0.1](https://huggingface.co/datasets/aidetectarena/ai-image-detector-benchmark) | Development OOD | 1,018 AI and 1,013 available unique real images; current commercial/open generators; CC BY 4.0 |
| OpenFake at `3fd1109dc3258874243fa31c5bda9ee24260163b` | Optional research-only OOD | Frontier generators and Reddit images; **CC BY-NC 4.0**, never used for release training |
| [Synthbuster](https://doi.org/10.5281/zenodo.10066460) + RAISE-1k | Frozen external | 4,000 DALL-E 2/3, Firefly, and Midjourney v5 images plus 1,000 matched real photographs; non-commercial evaluation only |

Each clean frozen test also gets deterministic JPEG, resize, crop, screenshot-like resampling, and WebP variants. Transformations are identical for both labels and grouped with their original so they cannot cross splits. The manifest loader rejects duplicate paths, duplicate file content, and any group appearing in more than one split before training or evaluation starts.

## Required reports

Every evaluation report records:

- SHA-256 of the manifest, checkpoint, calibration, and ONNX model;
- exact `0.65` threshold confusion matrix and balanced accuracy;
- AI recall and real specificity;
- per-source and per-generator counts/metrics;
- clean versus transformed metrics;
- duplicate-content and group-leak audit;
- Python/ONNX/Chrome score parity and browser latency.

A release is rejected if balanced accuracy is below 75%, either class recall is below 70%, the manifest leaks a group across splits, or browser scores differ from Python by more than 0.01.
