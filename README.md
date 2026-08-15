# <img src="extension/icons/on-128.png" width="40" alt="" align="absmiddle" style="filter: brightness(200%)"> Cyclopes

Private, browser-local AI image detection: Cyclopes labels visible web images without uploading pixels.

**93.68% AI precision · 94.83% balanced accuracy**

## Cyclopes

<!-- ![Cyclopes demo placeholder](docs/demo.svg) -->
<!-- ![Og Image](./og-image.png) -->
<img src="og-image.png" alt="Cyclopes og image" width="400"/>

<details id="install-and-launch">
<summary><strong>Install and launch</strong></summary>

> Extensions are currently **in review**, so public store links are not active yet.

<p>
  <a href="https://github.com/T-Damer/cyclopes/releases">
    <img src="https://img.shields.io/badge/GitHub-Releases-181717?logo=github&logoColor=white&label=%F0%9F%93%81%20Releases" alt="GitHub releases"/>
  </a>
  <a href="release/cyclopes-0.2.0.zip">
    <img src="https://img.shields.io/badge/Chrome%20%2F%20Edge-ZIP%20(in%20review)-4f5a57?logo=googlechrome&logoColor=white&label=%F0%9F%93%82%20ZIP" alt="Chrome/Edge ZIP"/>
  </a>
  <a href="release/cyclopes-firefox-0.2.0.zip">
    <img src="https://img.shields.io/badge/Firefox-ZIP%20(in%20review)-f26c34?logo=firefoxbrowser&logoColor=white&label=%F0%9F%93%82%20ZIP" alt="Firefox ZIP"/>
  </a>
</p>

For now, build and load `dist/` as an unpacked extension. Cyclopes includes detection controls, themes, per-site exclusions, smart badge positioning, and private local reports.

Put personal regression images in `personal-tests/`. Its contents are ignored by Git.

</details>

<details id="methodology">
<summary><strong>Methodology</strong></summary>

- Freeze a ViT-S forensic prior and train project-specific residual heads for scale consistency.
- Deduplicate and split by content source groups to reduce leakage, strip EXIF metadata, and keep provenance tracked.
- Evaluate against public held-out and web-degraded sets at the same operating point (fixed threshold), then keep the conservative model choice.
- Maintain private compatibility checks for non-public image streams before any release decision.
- Prefer browser-local inference safety: no image upload, no external scoring APIs, no raw telemetry.

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
