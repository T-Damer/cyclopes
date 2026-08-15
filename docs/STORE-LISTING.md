# Store listing

## Name

Cyclopes — Local AI Image Detector

## Summary

Detects AI-generated web images locally in your browser — no uploads.

## Description

Cyclopes labels likely AI-generated images directly on webpages. Detection runs through the packaged ONNX model on your device; images are never uploaded to a detector service.

- One-click global and per-site controls
- Local WebGPU inference with WASM fallback
- Smart badges that follow visible images
- Optional local AI/non-AI reports
- Configurable size and confidence thresholds

Results are probabilistic and may be wrong. Cyclopes ignores tiny, hidden, heavily occluded, and video-poster images by default.

## Category

Accessibility

## Permission justifications

- `activeTab`, `scripting`: apply the user-facing detector to the active webpage.
- `<all_urls>`: detect images on sites the user has not excluded.
- `offscreen`: host local ONNX Runtime inference under Manifest V3.
- `storage`: retain settings, exclusions, and optional local reports.
- `contextMenus`: let the user label an image AI or non-AI locally.

## Privacy

Policy: https://github.com/T-Damer/cyclopes/blob/main/docs/PRIVACY.md

Data disclosure: webpage images and source-page context are processed locally. Nothing is transmitted or shared. Optional reports stay in browser-local storage until deleted.

## Reviewer notes

1. Install and pin Cyclopes.
2. Open a page containing images at least 256×256.
3. Open the toolbar popup and enable detection for the current site.
4. A short animated toolbar state appears while the packaged model initializes; image badges then show local scores.
5. Right-click an image to test the optional local report actions.

No account, network service, or test credentials are required.
