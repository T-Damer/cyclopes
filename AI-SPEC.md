# Cyclopes AI contract

This document freezes the detector and evaluation rules before model training. A result may be reported only when produced by the commands and split rules below.

## Objective

Given pixels from an ordinary webpage image, return one number `p_ai` in `[0, 1]`. The extension labels an image AI-generated exactly when `p_ai >= 0.65`.

The release gate is at least `0.750` balanced accuracy:

```text
balanced_accuracy = (AI recall + real-image specificity) / 2
```

We also report AI recall, real specificity, ROC-AUC, per-source accuracy, model size, and browser latency. The fixed-threshold balanced accuracy is the primary metric.

## Model

Cyclopes uses a browser-local two-CNN ensemble, not Community Forensics:

- FastViT-T8 CNN backbone initialized from ImageNet.
- RGB pixels plus fixed high-pass residual channels generated inside the model.
- An Apache-2.0 ConvNeXt-Small adds a complementary real-image signal; a fixed scalar blend produces the final score.
- One browser-matched 256×256 resize per image followed by one frozen calibration transform.
- Two packaged ONNX graphs, quantized only if quantization loses less than 0.5 balanced-accuracy points on validation.

The fixed residual front end makes local synthesis and resampling artifacts available to a small network while the RGB path retains semantic context. It contains no metadata, filename, hash, watermark lookup, or generator-name feature.

## Data contract

Every sample is recorded in a CSV manifest with:

```text
path,label,source,generator,group,split
```

- `label` is `0` for real and `1` for AI-generated.
- `source` identifies the originating dataset or collection.
- `generator` identifies the synthetic generator when known.
- `group` binds related images, including real/synthetic pairs and derivatives.
- `split` is `train`, `calibration`, `validation`, or `test`.

No group may cross splits. Test sources and synthetic generators must be absent from training where the dataset permits it. Training batches are balanced by label and source.

Before the model sees an image, it is decoded to RGB pixels. EXIF, C2PA, PNG text chunks, filenames, paths, and file sizes are unavailable to the model. Both classes receive the same randomized JPEG/WebP recompression, resizing, blur, and color transforms. This prevents codec or dataset provenance from becoming a class shortcut.

## Splits and gates

1. **Smoke split**: small data only; proves training, export, and browser parity.
2. **Calibration split**: fits a scalar temperature and bias. It maps the selected operating point to the required displayed threshold of `0.65`.
3. **Validation split**: selects checkpoint and quantization. It is never used to fit calibration.
4. **Frozen test split**: opened once after all choices are frozen. It must include untouched images plus JPEG, resize, crop, screenshot, and social-media-style transcode variants.

Required pre-submission gates:

- Frozen-test balanced accuracy at `0.65` is at least `0.75`.
- Both AI recall and real specificity are at least `0.70`.
- No evaluated source is below `0.60` balanced accuracy when it contains both classes.
- Python ONNX and Chrome scores differ by at most `0.01` per test image.
- The unpacked extension works in a fresh Chrome profile after network access is disabled.
- No runtime request is made except fetching an image already displayed by the active webpage.

## Browser behavior

- Native Manifest V3 extension.
- ONNX Runtime Web with WebGPU first and WASM fallback; all executable and inference assets are packaged locally.
- An offscreen extension document fetches displayed image bytes, decodes pixels, runs inference, and returns only the score to the content script.
- Static and dynamically inserted webpage images of at least 64 by 64 pixels are analyzed automatically.
- The only UI is the extension's Filter ON/OFF button; qualifying images are blurred without badges or overlays.
- Results are cached by URL during the browser session and inference concurrency is bounded.

## Reproducibility evidence

The release must contain:

- exact dependency lock files;
- dataset acquisition/provenance instructions and generated manifest hashes;
- training configuration, seed, checkpoint hash, calibration parameters, and ONNX hash;
- machine-readable validation and frozen-test summaries;
- a clean build command and an offline Chrome smoke test;
- MIT license and third-party notices.

The private POIDH benchmark remains unknowable. Passing this contract is necessary evidence for submission, not a guarantee of the private score.
