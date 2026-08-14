# Cyclopes AI contract

This document freezes the ScalePair detector and its evaluation rules. A result may be reported only when produced by the commands and split rules below.

## Objective

Given pixels from an ordinary webpage image, return one number `p_ai` in `[0, 1]`. The extension labels an image AI-generated exactly when `p_ai >= 0.65`.

The target is at least `0.800` balanced accuracy; the minimum release gate is `0.750`:

```text
balanced_accuracy = (AI recall + real-image specificity) / 2
```

We also report AI recall, real specificity, ROC-AUC, per-source accuracy, model size, and browser latency. The fixed-threshold balanced accuracy is the primary metric.

## Model

Cyclopes v0.2 uses one browser-local multi-layer ScalePair ViT, not an ensemble:

- The MIT-licensed Community Forensics ViT-S prior is pinned and frozen.
- Trainable Cyclopes projections read hidden layers 4, 8, and 12 at 384×384.
- A residual head compares the displayed image embedding with an internal 160→384 thumbnail probe and corrects the frozen prior logit.
- Training pairs every clean image with a JPEG/WebP/downscale variant and penalizes inconsistent embeddings.
- Calibration selects a fixed fused/current-head blend, temperature, and bias; those constants are embedded in one ONNX graph.

The model sees pixels only. It cannot read metadata, paths, filenames, dimensions, hashes, watermarks, prompts, tags, or generator names.

## Data contract

Every sample is recorded in a CSV manifest with:

```text
path,label,source_dataset,generator_model,content_group,split,family,domain,license,sha256
```

- `label` is `0` for real and `1` for AI-generated.
- `source_dataset` and `generator_model` identify provenance.
- `content_group` binds related images, including thumbnails and adjacent video frames.
- `split` is `train`, `calibration`, `validation`, or `test`.
- `family`, `domain`, `license`, and `sha256` support family supervision and audit evidence.

No group may cross splits. Test sources and synthetic generators must be absent from training where the dataset permits it. Training batches are balanced by label and source.

Before the model sees an image, it is decoded to RGB pixels. EXIF, C2PA, PNG text chunks, filenames, paths, and file sizes are unavailable to the model. Both classes receive the same randomized JPEG/WebP recompression, resizing, blur, and color transforms. This prevents codec or dataset provenance from becoming a class shortcut.

## Splits and gates

1. **Smoke split**: small data only; proves training, export, and browser parity.
2. **Calibration split**: selects the head blend and fits temperature/bias. It maps the operating point to the displayed threshold of `0.65`.
3. **Validation split**: selects checkpoint and quantization. It is never used to fit calibration.
4. **Frozen test split**: opened once after all choices are frozen. It must include untouched images plus JPEG, resize, crop, screenshot, and social-media-style transcode variants.

Required pre-submission gates:

- Frozen-test balanced accuracy at `0.65` is at least `0.75`.
- Both AI recall and real specificity are at least `0.70`.
- No evaluated source is below `0.60` balanced accuracy when it contains both classes.
- Clean and browser-transcoded validation are both reported; checkpoint selection uses the worse balanced accuracy.
- Python ONNX and Chrome scores differ by at most `0.01` per test image.
- The unpacked extension works in a fresh Chrome profile after network access is disabled.
- No runtime request is made except fetching an image already displayed by the active webpage.

## Browser behavior

- Native Manifest V3 extension.
- ONNX Runtime Web with WebGPU first and WASM fallback; all executable and inference assets are packaged locally.
- An offscreen extension document fetches displayed image bytes, decodes pixels, runs inference, and returns only the score to the content script.
- Static and dynamically inserted webpage images of at least 64 by 64 pixels are analyzed when they enter a 200px viewport margin.
- The extension icon is the only control: click toggles Filter ON/OFF and exposes warming/error state.
- Every visible image receives a loading badge and then an AI confidence badge; scores at least 65% are blurred.
- Results are cached by image-byte SHA-256 with a 256-entry LRU; inference concurrency is one to keep page interaction responsive.

## Reproducibility evidence

The release must contain:

- exact dependency lock files;
- dataset acquisition/provenance instructions and generated manifest hashes;
- training configuration, seed, checkpoint hash, calibration parameters, and ONNX hash;
- machine-readable validation and frozen-test summaries;
- per-image field results for the user-supplied full/thumbnail/GIF/graphic cases, kept evaluation-only;
- a clean build command and an offline Chrome smoke test;
- MIT license and third-party notices.

The private POIDH benchmark remains unknowable. Passing this contract is necessary evidence for submission, not a guarantee of the private score.
