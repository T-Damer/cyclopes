# Dataset provenance

Cyclopes trains only on decoded RGB pixels. Every source is rewritten as JPEG quality 92/subsampling 0, stripping metadata and source-codec shortcuts. `content_group` keeps related frames and derivatives in one split.

## Audited training set

The base-training manifest contains **46,417** images after exact SHA-256 validation, dual pHash+dHash deduplication, and removal of low-information frames: **25,116 real / 21,301 AI**. Manifest SHA-256: `87e0da6bce5fa635883094a567599a2bc19429decced8fc7a57fbb27955f5a4e`.

| Source | Kept | Role | Terms |
| --- | ---: | --- | --- |
| [Google DOCCI](https://huggingface.co/datasets/google/docci), revision `a0a43eaf34676ffd008fb6565dd8c2ba00d09100` | 14,844 | real photos | CC BY 4.0 |
| [The Met Open Access](https://huggingface.co/datasets/metmuseum/openaccess), revision `c65f8d6041aea7b3bc767a54d93772c3c6a365f6` | 4,336 | real art | CC0 |
| [Blender Open Movies](https://video.blender.org/) | 4,068 | real CGI; film-disjoint splits | CC BY / CC BY-SA per film |
| Cyclopes deterministic posters/logos/diagrams | 1,868 | real hard negatives | CC0 |
| [BitMind Nano Banana](https://huggingface.co/datasets/bitmind/nano-banana), revision `9ea8da32a5be03f4946e6cb10c2d2f8e90f0a0a4` | 1,000 | modern AI | MIT dataset |
| [COCOXGEN](https://huggingface.co/datasets/heikeadel/cocoxgen), revision `c336ad187c2ab298ce825df65088bdacbae104f6` | 4,244 | Fooocus/SDXL | CC BY 4.0 |
| [PAMELA](https://huggingface.co/datasets/pamela-dataset/pamela), revision `14ebd68d2a2c34367d41020b62ee60b7504725fb` | 5,076 | FLUX.2/Nano Banana | CC BY 4.0; generator output terms also apply |
| Self-generated BigGAN | 5,000 | legacy GAN | MIT implementation/weights package |
| Self-generated ADM | 2,000 | legacy diffusion | MIT implementation |
| Self-generated GLIDE | 1,981 | legacy diffusion | MIT implementation |
| Self-generated LDM | 2,000 | legacy diffusion | CreativeML Open RAIL-M checkpoint terms |

Training images are not redistributed. The report [`reports/data-preflight.json`](reports/data-preflight.json) records source, family, domain, license, split, removal counts, and the final manifest hash.

## V5 replay fine-tune

The accepted V5 checkpoint received a short frozen-backbone replay fine-tune on **9,206** audited images: **3,707 real OpenGameArt illustrations/CGI** and **5,499 AI images** sampled evenly from COCOXGEN and PAMELA. The manifest is group-disjoint, keeps 510 images for its test split, and has SHA-256 `9e554b2d99159c7fd9d3e237d1486c1b3818892548f1bed7054d23e17e8899fa`.

| Source | Kept | Role | Terms |
| --- | ---: | --- | --- |
| [OpenGameArt](https://opengameart.org/) | 3,707 | real digital-art/CGI hard negatives | CC0 assets only |
| [COCOXGEN](https://huggingface.co/datasets/heikeadel/cocoxgen), revision `c336ad187c2ab298ce825df65088bdacbae104f6` | 2,750 | AI replay | CC BY 4.0 |
| [PAMELA](https://huggingface.co/datasets/pamela-dataset/pamela), revision `14ebd68d2a2c34367d41020b62ee60b7504725fb` | 2,749 | AI replay | CC BY 4.0; generator output terms also apply |

[`reports/data-preflight-replay.json`](reports/data-preflight-replay.json) records its split, source, family, domain, license, and deduplication audit.

## Evaluation-only sets

- [AI Detector Arena v0.1](https://huggingface.co/datasets/aidetectarena/ai-image-detector-benchmark): development OOD benchmark; never training or calibration.
- Optional OpenFake/Synthbuster/RAISE checks remain non-commercial evaluation only and are not part of the released weights.

The vNext release gate is a separate 50,000–100,000-image corpus covering photos, UI, screenshots, illustration, film frames, 3D/game renders, memes, logos/text, and old and new generators. [`docs/EVALUATION.md`](docs/EVALUATION.md) defines its source-disjointness, licensing, deduplication, and scenario quotas. The repository contains the builder and quota file, not the image corpus.

The private POIDH dataset is unavailable, so no public metric is presented as a guarantee of its score.
