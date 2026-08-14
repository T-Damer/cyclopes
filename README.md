<p align="center"><img src="logo.png" width="150" alt="Cyclopes logo"></p>

# Cyclopes

## Demo

![Cyclopes blurs images classified as AI-generated](docs/demo.svg)

Browser-local AI image filtering. One button, no uploads: images scoring at least **65% AI** are blurred.

**91.90% AI precision · 90.65% balanced accuracy** on AI Detector Arena v0.1.

![Cyclopes benchmark metrics](docs/metrics.svg)

## Contents

- [Install](#install)
- [Approach](#approach)
- [Metrics](#metrics)
- [Development](#development)

## Install

```bash
npm ci
npm run build
```

Open `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select `dist/`. Reload already-open pages once, then click the Cyclopes icon to toggle **Filter: ON/OFF**.

## Approach

One project-trained ScalePair CNN compares each displayed image with its own downscale/upscale probe, using a shared MobileNetV3 backbone and early-feature residual statistics. Strong portrait/landscape ratios use one centered square view, not a crop ensemble. The packaged ONNX runs through WebGPU with a WASM fallback; pixels never leave the browser.

## Metrics

| External set | Images | Balanced accuracy | AI precision | AI recall | Real specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI Detector Arena v0.1 | 2,031 | **90.65%** | **91.90%** | 89.19% | 92.10% |
| Held-out replay test | 510 | **82.36%** | 81.74% | 97.72% | 67.00% |

The operating point is fixed at 65%. On 56 frozen user-supplied browser regression files, Cyclopes detects 18/20 AI images and correctly rejects 19/36 real images. This small adversarial set is reported for transparency, not as a representative benchmark.

## Development

```bash
python -m pytest -q
npm test
```

Full data, training, export, and evaluation instructions: [`docs/agents.md`](docs/agents.md). Dataset provenance: [`DATASETS.md`](DATASETS.md). Third-party licenses: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

MIT licensed. Built for [POIDH bounty #323](https://poidh.xyz/arbitrum/bounty/323).
