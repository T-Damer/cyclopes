<p align="center"><img src="logo.png" width="150" alt="Cyclopes logo"></p>

# Cyclopes

## Demo

![Cyclopes labels images classified as AI-generated](docs/demo.svg)

Browser-local AI image detection. One button, no uploads: eligible, substantially visible images receive an anchored AI confidence badge; images below 256×256, video posters, and images over 85% occluded are ignored.

**93.68% AI precision · 94.83% balanced accuracy** on AI Detector Arena v0.1.

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

A frozen ViT-S forensic prior is adapted with project-trained multi-layer and scale-consistency residual heads. Each complete displayed image is compared with its own downscale/upscale probe inside one ONNX graph. The graph runs through WebGPU with a WASM fallback; pixels never leave the browser.

## Metrics

| External set | Images | Balanced accuracy | AI precision | AI recall | Real specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI Detector Arena v0.1 | 2,031 | **94.83%** | **93.68%** | 96.17% | 93.48% |
| Arena web-degraded | 2,031 | **91.67%** | 87.53% | 97.25% | 86.08% |
| Held-out training sources, web-degraded | 2,798 | **95.35%** | 96.11% | 93.82% | 96.87% |

The operating point is fixed at 65%. AI Detector Arena is external to training and calibration; the private bounty distribution remains unknown.

## Development

```bash
python -m pytest -q
npm test
```

Full data, training, export, and evaluation instructions: [`docs/agents.md`](docs/agents.md). Dataset provenance: [`DATASETS.md`](DATASETS.md). Third-party licenses: [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).

MIT licensed. Built for [POIDH bounty #323](https://poidh.xyz/arbitrum/bounty/323).
