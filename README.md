# <img src="extension/icons/off-128.png" width="40" alt="" align="absmiddle"> Cyclopes

Private, browser-local AI image detection: Cyclopes labels visible web images without uploading pixels.

**93.68% AI precision · 94.83% balanced accuracy**

## Cyclopes

<!-- ![Cyclopes demo placeholder](docs/demo.svg) -->
<!-- ![Og Image](./og-image.png) -->
<img src="og-image.png" alt="Cyclopes og image" width="400"/>

<details id="install-and-launch" open>
<summary><strong>Install and launch</strong></summary>

> Extensions are currently **in review**, so public store links are not active yet.

<p>
  <a href="https://github.com/T-Damer/cyclopes/releases">
    <img src="https://img.shields.io/badge/GitHub-v0.2.0-181717?logo=github&logoColor=white&label=%F0%9F%93%81%20Releases" alt="GitHub releases"/>
  </a>
  <a href="release/cyclopes-0.2.0.zip">
    <img src="https://img.shields.io/badge/Chrome%20%2F%20Edge-v0.2.0%20ZIP-4f5a57?logo=googlechrome&logoColor=white&label=%F0%9F%93%82%20ZIP" alt="Chrome/Edge ZIP"/>
  </a>
  <a href="release/cyclopes-firefox-0.2.0.zip">
    <img src="https://img.shields.io/badge/Firefox-v0.2.0%20ZIP-f26c34?logo=firefoxbrowser&logoColor=white&label=%F0%9F%93%82%20ZIP" alt="Firefox ZIP"/>
  </a>
</p>

For now, build and load `dist/` as an unpacked extension. Cyclopes includes detection controls, themes, per-site exclusions, smart badge positioning, and private local reports.

Put personal regression images in `personal-tests/`. Its contents are ignored by Git.

</details>

<details id="methodology">
<summary><strong>Methodology</strong></summary>

- Use a frozen ViT-S forensic encoder as feature backbone, then train ScalePair-style residual heads that explicitly model scale consistency between resized input variants.
- Build training partitions by source-linked groups, remove duplicates, and normalize/canonicalize images to reduce leakage and split contamination across public, web-degraded, and private subsets.
- Expand negatives with curated hard negatives (UI, game/anime/meme-like content, thumbnails, edited composites) and keep real AI-image boundaries explicit during sample curation.
- Evaluate fixed-threshold operating points on held-out public and web-degraded sets, then keep the conservative checkpoint only if real-image specificity remains stable.
- Run local private compatibility checks on additional non-public streams before changing release direction.
- Keep inference entirely browser-local: no image upload, no remote scoring APIs, no raw telemetry.

</details>

<details id="metrics">
<summary><strong>Metrics</strong></summary>

Running on WebGPU at around **90ms/image** (desktop-class browser GPU), Cyclopes scores at a fixed **65%** threshold on a **36,000-image** held-out benchmark (clean / web / hard-degraded): **91.3% / 87.3% / 84.6% balanced accuracy**.

| External set | Images | Balanced accuracy | AI precision | AI recall | Real specificity |
| --- | ---: | ---: | ---: | ---: | ---: |
| AI Detector Arena v0.1 | 2,031 | **94.83%** | **93.68%** | 96.17% | 93.48% |
| Arena web-degraded | 2,031 | **91.67%** | 87.53% | 97.25% | 86.08% |
| Held-out sources, web-degraded | 2,798 | **95.35%** | 96.11% | 93.82% | 96.87% |

Fixed operating point: 65%.

</details>

<details id="dev">
<summary><strong>Dev</strong></summary>

### How it was created

Cyclopes v0.2 uses a ScalePair pipeline: a frozen ViT-S encoder plus multi-layer residual heads trained for scale consistency. The primary training mix includes licensed real photos, artwork/CGI, legacy generators, and modern diffusion families with explicit negatives for UI-heavy, thumbnail-like, and meme/game content. All images are decoded to RGB, stripped of metadata, deduplicated, and split with source-linked grouping to reduce leakage. Full provenance is in [DATASETS.md](DATASETS.md).

### How to install and launch locally

```bash
npm ci
npm run build
```

Open `chrome://extensions`, enable Developer mode, choose **Load unpacked**, and select `dist/`. Run checks with `npm test` and `python -m pytest -q`.

### What we tried and what we got

#### Additional datasets and baselines

What changed between internal experiments:

- v0.1 baseline (single-head ViT-S fine-tuning): good initial AI-recall but weaker stability on scale shifts and noisy negatives (UI/game-like and thumbnail-heavy streams).
- Additional hard-negative sets: social thumbnails, posters, low-contrast UI captures, text-heavy images, and synthetic art variants were tested to reduce false positives.
- v0.3 expert/router branch: improved some public set numbers but regressed real-image specificity enough to fail our acceptance criteria, so it was rejected.
- Baseline model family variants (small CNN, FastViT+ConvNeXt stack): showed higher latency or repeated domain-specific failures; none surpassed ScalePair v0.2 at the target safety/reuse trade-off.

Cycle artifacts and rejected runs are tracked in [docs/TRAINING-PLAN.md](docs/TRAINING-PLAN.md) and [docs/agents.md](docs/agents.md).

Training, export, reproducibility commands, and the rejected-experiment record live in [docs/agents.md](docs/agents.md) and [docs/TRAINING-PLAN.md](docs/TRAINING-PLAN.md).

### How the frontend works

The Manifest V3 extension schedules only eligible images in the active tab, waits for stable layout, ignores tiny, hidden, heavily occluded, and video-poster images, then runs one local ONNX job at a time through WebGPU with WASM fallback. Badges are anchored by DOM hit-testing; feedback previews, settings, and site exclusions stay in browser storage.

</details>

MIT licensed. See [privacy](docs/PRIVACY.md), [dataset terms](DATASETS.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Built for [POIDH bounty #323](https://poidh.xyz/arbitrum/bounty/323).
