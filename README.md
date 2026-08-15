# <img src="extension/icons/on-128.png" width="40" alt="" align="absmiddle"> Cyclopes

Private, browser-local AI image detection: Cyclopes labels visible web images without uploading pixels.

**93.68% AI precision · 94.83% balanced accuracy** on AI Detector Arena v0.1.

## Demo

![Cyclopes demo placeholder](docs/demo.svg)

<details>
<summary><strong>Table of contents</strong></summary>

- [Install and launch](#install-and-launch)
- [Metrics](#metrics)
- [Dev](#dev)

</details>

<details id="install-and-launch">
<summary><strong>Install and launch</strong></summary>

> Chrome Web Store and Edge Add-ons links will appear here after review.

For now, build and load `dist/` as an unpacked extension. Cyclopes includes detection controls, themes, per-site exclusions, smart badge positioning, and private local reports.

Put personal regression images in `personal-tests/`. Its contents are ignored by Git.

</details>

<details id="metrics">
<summary><strong>Metrics</strong></summary>

| External set | Images | Balanced accuracy | AI precision | AI recall | Real specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI Detector Arena v0.1 | 2,031 | **94.83%** | **93.68%** | 96.17% | 93.48% |
| Arena web-degraded | 2,031 | **91.67%** | 87.53% | 97.25% | 86.08% |
| Held-out sources, web-degraded | 2,798 | **95.35%** | 96.11% | 93.82% | 96.87% |

Fixed operating point: 65%. The private bounty distribution is unknown.

</details>

<details id="dev">
<summary><strong>Dev</strong></summary>

### How it was created

Cyclopes v0.2 freezes a ViT-S forensic prior and trains project-specific multi-layer and scale-consistency residual heads. The audited data mixes licensed real photos, art, CGI, UI-like negatives, legacy generators, and modern diffusion models; every source is decoded to RGB, metadata-stripped, deduplicated, and split by related-content group. Full provenance is in [DATASETS.md](DATASETS.md).

### How to install and launch locally

```bash
npm ci
npm run build
```

Open `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select `dist/`. Run checks with `npm test` and `python -m pytest -q`.

### What we tried and what we got

- A small CNN was fast, but transferred poorly to memes, games, anime, and thumbnails.
- A FastViT + ConvNeXt ensemble repeated the same domain biases at a higher browser cost.
- Scale-paired v0.2 produced the strongest external result. A later expert/router experiment was rejected because real-image specificity regressed sharply.

Training, export, reproducibility commands, and the rejected-experiment record live in [docs/agents.md](docs/agents.md) and [docs/TRAINING-PLAN.md](docs/TRAINING-PLAN.md).

### How the frontend works

The Manifest V3 extension schedules only eligible images in the active tab, waits for stable layout, ignores tiny, hidden, heavily occluded, and video-poster images, then runs one local ONNX job at a time through WebGPU with WASM fallback. Badges are anchored by DOM hit-testing; feedback previews, settings, and site exclusions stay in browser storage.

### Comparison with others

Architecture and public-submission notes are kept in [docs/TRAINING-PLAN.md](docs/TRAINING-PLAN.md), away from the short product overview.

</details>

MIT licensed. See [privacy](docs/PRIVACY.md), [dataset terms](DATASETS.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Built for [POIDH bounty #323](https://poidh.xyz/arbitrum/bounty/323).
