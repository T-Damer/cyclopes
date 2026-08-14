# Cyclopes replacement-model plan

**Status:** v0.1 executed; bounded vNext adaptation prepared
**Date:** 2026-08-14  
**Decision owner:** Cyclopes maintainers  
**Submission rule:** do not submit until every release gate in this document passes

## Implemented result

The research below led to a single **ScalePair MobileNetV3-Large** ONNX model, not the rejected FastViT + ConvNeXt ensemble. It compares RGB features with a downscale/upscale probe inside one project-trained graph; no public AI-detector checkpoint, metadata, watermark, or server is used.

V5 was selected after a bounded 100-step frozen-backbone replay fine-tune. No H200 was needed for this final pass. At the fixed 0.65 threshold it reaches **90.65% balanced accuracy and 91.90% AI precision** on AI Detector Arena v0.1. The 56-file user field regression reaches 90% AI recall and 52.78% real specificity; this deliberately difficult, non-representative set exposes the remaining false positives rather than being used for tuning.

The final graph is 14,974,340 bytes with SHA-256 `4936a9ef0988efe9717da24c45da61a213ed09eb39437f1ea7ee0474471fc359`. Reproducibility commands are in [`docs/agents.md`](agents.md); machine-readable results are in `reports/*-v5.json`.

V0.1 remains the rollback baseline. The vNext experiment starts from the MIT-licensed Community Forensics ViT-S checkpoint at pinned revision `ac6ee457bea904a373065754107451793b56db00`, freezes its original detector, and learns only Cyclopes multi-layer and scale-consistency residual heads. This is a direct upstream dependency, not code or data taken from another bounty submission; its required attribution remains in `THIRD_PARTY_NOTICES.md`.

## 1. Executive decision

Replace the former FastViT-T8 + Sentry ConvNeXt-Small ensemble with one project-trained, browser-local RGB + forensic-residual CNN.

The replacement will:

- use no Community Forensics, ViT, Swin, ConvNeXt, FastViT, public AI-detector checkpoint, metadata, filename, watermark, or lookup signal;
- learn from labeled pixels and paired full-size/web-degraded views;
- export as one ONNX graph and run once per image in Chrome;
- preserve the bounty decision rule: AI if and only if `p_ai >= 0.65`;
- be rejected rather than submitted if it lacks a clear margin over the bounty's 75% balanced-accuracy requirement.

The distinctive part is not merely a different backbone. During training, the same image is scored as a clean/full-size view and as a thumbnail/JPEG/web view. A consistency loss explicitly penalizes the score instability observed in the current browser build.

## 2. Bounty inputs and constraints

Primary source: [POIDH bounty #323](https://poidh.xyz/arbitrum/bounty/323).

Required product behavior:

- native Manifest V3 Chrome extension;
- automatic detection on ordinary webpage images;
- all inference in the browser;
- no cloud inference, external API, upload, or local companion server;
- WebGPU with a WASM fallback;
- the extension remains usable offline after its assets are installed;
- confidence threshold fixed at 65%;
- balanced accuracy of at least 75% on the maintainer's private evaluation.

Cyclopes-specific product constraints agreed during development:

- one toolbar action: filter ON/OFF;
- qualifying images are blurred without adding badges or a complex popup;
- a visible warm-up/loading state is allowed;
- static and dynamically inserted images are supported;
- GIF input is decoded to a still bitmap for classification;
- README remains short; reproducibility detail belongs in project documentation.

The private evaluation distribution is unknown. No public benchmark result guarantees success.

## 3. Research performed

### 3.1 Public bounty submissions

The bounty page and public repositories were rechecked on 2026-08-14. The page listed 11 claims. The table records what is verifiable from the public claim or repository, not an endorsement of the reported accuracy.

| Claim | Project | Publicly described detector |
| ---: | --- | --- |
| 1026 | [PixelWitness](https://github.com/ebreen/pixelwitness) | Three ready-made detectors: xRayon ConvNeXtV2-Base, Six-Fingers/Community-Forensics-style ViT-S, and Sentry ConvNeXt-Small; calibrated score fusion. |
| 1025 | [Blur](https://github.com/maddiedreese/blur) | Community Forensics ViT-S; repository also contains thumbnail-head experiments. Its own README reports strong full-resolution results but weak thumbnail performance. |
| 1023 | [Local Lens](https://github.com/dusy4/local-ai-image-detector-bounty) | Quantized ViT plus Community Forensics and modern-generator ConvNeXt, then C2PA, SDXL watermark, metadata, and carrier signals. |
| 1022 | [aidetect](https://github.com/anudit/aidetect) | ViT-Small/16 forward pass implemented as custom WebGPU shaders. |
| 1021 | [AI Image Detector](https://github.com/RajeshRk18/ai-image-detector) | Community Forensics ViT-S plus Organika SDXL Swin, crop aggregation, metadata/provenance scan, and calibrated fusion. |
| 1020 | [Dino-ImageGen-Ext / Proofmark](https://github.com/Dyno-man/Dino-ImageGen-Ext) | Community Forensics ViT backbone with a locally trained small head, plus metadata, provenance, watermark, and pixel-statistic signals. |
| 1019 | [RealGuard](https://github.com/choir94/RealGuard-ai-image-detector) | Community Forensics ViT-S with calibration and metadata/C2PA. DCT/Laplacian analysis exists but is disabled in score fusion. |
| 1018 | [SynthCheck](https://github.com/thedudeb/synthcheck) | Community Forensics ViT-S with a modern, degradation-trained classifier head. |
| 1016 | [Six-Fingers](https://github.com/the-gadget-lab/Six-Fingers) | Community Forensics ViT-S fine-tuned on web degradation and recent generators, then quantized. |
| 1015 | [Caravela local-ai-detector](https://github.com/CaravelaLabs/local-ai-detector) | Community Forensics ViT-S with six-crop maximum aggregation and metadata/C2PA inspection. |
| 1014 | [LocalLens](https://github.com/takhir-iota/locallens-ai-detector) | Public `delpot/steganograph-ia-detector` ViT-B/16 checkpoint. |

Observed pattern:

1. Most submissions package or fine-tune the same Community Forensics ViT family.
2. Several combine public ViT/ConvNeXt/Swin detectors into a calibrated ensemble.
3. Several add metadata, C2PA, generator strings, or watermark evidence.
4. Web degradation is commonly used as augmentation, but the submitted runtime remains primarily a public detector or public-detector ensemble.
5. PixelWitness and the dusy4 Local Lens already claim ConvNeXt experts, so keeping Sentry ConvNeXt would not be a defensible original approach.

Relevant model background:

- [Community Forensics paper](https://arxiv.org/abs/2411.04125)
- [FastViT paper](https://arxiv.org/abs/2303.14189)
- [Spatial Rich Model residual filters](https://ws2.binghamton.edu/fridrich/Research/TIFS2012-SRM.pdf)
- [ONNX Runtime Web documentation](https://onnxruntime.ai/docs/tutorials/web/)

### 3.2 Dataset and evaluation research

The existing acquisition plan pins source revisions and licenses in [`DATASETS.md`](../DATASETS.md). The current usable sources are:

| Source | Intended role | Important coverage |
| --- | --- | --- |
| [Google DOCCI](https://huggingface.co/datasets/google/docci) | real training | camera photographs |
| [FakeClue](https://huggingface.co/datasets/bitmind/FakeClue) | balanced training | real images and six AI-generator families |
| [COCOXGEN](https://huggingface.co/datasets/heikeadel/cocoxgen) | AI training | Fooocus and SDXL |
| [Nano Banana](https://huggingface.co/datasets/bitmind/nano-banana) | AI training | Gemini 2.5 Flash Image |
| [Nano Banana Pro 1K](https://huggingface.co/datasets/ash12321/nano-banana-pro-generated-1k) | AI training | newer Nano Banana Pro |
| [PAMELA](https://huggingface.co/datasets/pamela-dataset/pamela) | AI training | Flux 2 and Nano Banana Pro |
| [The Met Open Access](https://huggingface.co/datasets/metmuseum/openaccess) | real training | public-domain art |
| [AI Detector Arena](https://huggingface.co/datasets/aidetectarena/ai-image-detector-benchmark) | development OOD | recent generators and mixed real sources |
| [Synthbuster](https://doi.org/10.5281/zenodo.10066460) + RAISE-1k | historical external regression | DALL-E 2/3, Firefly, Midjourney v5, and real photos |

OpenFake remains excluded from release training because its pinned dataset is CC BY-NC 4.0. It may not be used to improve weights for a paid bounty.

The present real-art coverage is insufficient for anime, CGI, contemporary digital illustration, logos, typography, posters, icons, diagrams, and rasterized vector art. Before training, add 5,000–10,000 real hard negatives obtained through license-filtered [Openverse](https://openverse.org/), [OpenGameArt](https://opengameart.org/), and Wikimedia Commons sources. Keep original URL, creator, license, license URL, and SHA-256 in provenance. Accept only CC0, CC BY, and CC BY-SA material whose terms can be satisfied; do not redistribute the image corpus in the repository.

## 4. What Cyclopes built

Current branch components:

- Manifest V3 extension with browser-local ONNX Runtime Web inference;
- WebGPU/WASM assets packaged locally;
- toolbar click toggles filtering directly;
- warm-up, ON, OFF, and error badge states;
- bounded inference concurrency and URL cache;
- browser diagnostics via `data-cyclopes-score` and `data-cyclopes-error`;
- training, calibration, evaluation, export, ONNX parity, and manifest-validation commands;
- duplicate path, duplicate byte-content, and split-group leakage checks.

Rejected baseline at the start of this plan:

| Evaluation | Balanced accuracy | AI recall | Real specificity | AI precision |
| --- | ---: | ---: | ---: | ---: |
| Internal validation | 89.63% | 86.52% | 92.74% | 84.89% |
| AI Detector Arena | 75.49% | 71.12% | 79.86% | 78.02% |
| Synthbuster + RAISE-1k | 75.36% | 65.72% | 85.01% | 94.63% |

The large internal-to-OOD drop was evidence of source/generalization shortcuts. Passing 75% by less than half a percentage point was not a safe margin for an unknown private evaluation, so those graphs were removed.

## 5. Browser field audit

The existing unpacked extension was exercised on user-selected Rule34 posts. Labels below come from the site's `ai generated` tag and user assessment; they are not independently verified scientific ground truth. These URLs must remain evaluation-only and must never enter training, calibration, checkpoint selection, or augmentation development.

| Post | Expected | Thumbnail score | Full-page score | Current result |
| --- | --- | ---: | ---: | --- |
| [18447255](https://rule34.xxx/index.php?page=post&s=view&id=18447255) | AI-tagged | 0.6443 | 0.3868 | miss |
| [18447258](https://rule34.xxx/index.php?page=post&s=view&id=18447258) | AI-tagged | 0.1981 | 0.4237 | miss |
| [18447257](https://rule34.xxx/index.php?page=post&s=view&id=18447257) | AI-tagged | 0.6161 | 0.3070 | miss |
| [18447247](https://rule34.xxx/index.php?page=post&s=view&id=18447247) | AI-tagged | 0.3231 | 0.2980 | miss |
| [18447265](https://rule34.xxx/index.php?page=post&s=view&id=18447265) | AI-tagged | 0.5070 | 0.7576 | full only |
| [18447254](https://rule34.xxx/index.php?page=post&s=view&id=18447254) | AI-tagged, visually ambiguous | 0.6488 | 0.4379 | miss |
| [18447260](https://rule34.xxx/index.php?page=post&s=view&id=18447260) | AI-tagged, visually ambiguous | 0.6483 | 0.6487 | miss |
| [18447263](https://rule34.xxx/index.php?page=post&s=view&id=18447263) | AI-tagged, visually ambiguous | 0.6467 | 0.4850 | miss |
| [18447261](https://rule34.xxx/index.php?page=post&s=view&id=18447261) | user-identified real | 0.6477 | 0.6505 | false positive on full image |
| [18447267](https://rule34.xxx/index.php?page=post&s=view&id=18447267) | AI-tagged, obvious | — | 0.0056 | severe miss |
| [18447269](https://rule34.xxx/index.php?page=post&s=view&id=18447269) | AI-tagged GIF | [0.6498](https://wimg.rule34.xxx/thumbnails/703/thumbnail_58712c18ee0cae454e7dfbadad771ff5.jpg?18447269) | 0.3170 | miss |

The GIF path is functional: the browser decoded and scored a still frame. The failure is classification, not missing GIF support.

An additional real-graphics audit exposed a separate threshold-boundary failure:

| Image pair | Expected | Small/cached score | Larger-source score | Current result |
| --- | --- | ---: | ---: | --- |
| [Yabloko poster cached image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQSS2AcirVk3FIc6Hxw_JtAsDcwFuWMsJJxoaJlJrI-zg&s=10) / [1200 px source](https://www.yabloko.ru/files/styles/max_1300x1300/public/for-smi-gallery/2025-08/01.png?itok=EVHk0C7A) | real flat graphic | 0.6504 | 0.6504 | false positive |
| [Yabloko emblem cached image](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRs1S8eMlC1QBIFh8QSVMQ5eUk5PsS9ZZ1FDIhuK17PCA&s) / [250 px Wikimedia source](https://upload.wikimedia.org/wikipedia/ru/thumb/1/19/Yabloko_emblem.svg/250px-Yabloko_emblem.svg.png) | real logo/vector rendering | 0.6503 | 0.6497 | threshold flips across representations |
| [Google cached graphic 2](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTS1azaop7DT5uG32yneNV7kYjuMwwxr_6cHmfuOQhwVw&s=10) | real web graphic | 0.6506 | — | false positive |
| [Google cached graphic 3](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTcwj3FPZUVjv19cFfFTQr3gLyHLtz5R9MXAhNybCZmcw&s=10) | real web graphic | 0.6502 | — | false positive |
| [RFE/RL graphic](https://gdb.rferl.org/fdabfb49-5181-4c57-46eb-08decacf94d0_w408_r1_s.png) | real web graphic | 0.6505 | — | false positive |

The second decision changes across the fixed `0.65` boundary for a score difference of only `0.0006`. The first pair remains slightly above the boundary at both resolutions. These cases show that logos, typography, flat color fields, and rasterized vector art need explicit real hard-negative coverage. They also show why tuning the threshold on individual examples is invalid.

All user-supplied field cases are frozen in [`tests/field-regression.json`](../tests/field-regression.json). The manifest explicitly forbids training use and preserves the current scores for before/after comparison. [`tests/test_field_regression.py`](../tests/test_field_regression.py) validates its schema and uniqueness without making network requests.

## 6. Why the previous approach was rejected

### 6.1 Originality

Sentry ConvNeXt overlaps the public-detector/ConvNeXt ensemble approach already submitted by PixelWitness and the dusy4 Local Lens. Keeping it would contradict the project goal of a materially different detector.

### 6.2 Insufficient private-benchmark margin

Arena and Synthbuster balanced accuracy are only 75.49% and 75.36%. Both are too close to the 75% minimum. Synthbuster AI recall is only 65.72%.

### 6.3 Resolution and codec instability

The same content can cross the threshold after resizing in either direction. Examples include:

- post 18447265: `0.5070` thumbnail versus `0.7576` full-size;
- post 18447269: `0.6498` thumbnail versus `0.3170` GIF still;
- post 18447255: `0.6443` thumbnail versus `0.3868` full-size.

A threshold adjustment cannot solve contradictory scores for the same content and would violate the fixed 0.65 rule.

### 6.4 Runtime cost

Two large graphs increase extension size, warm-up time, memory use, and serialized page latency. A single compact graph is preferable if it meets the accuracy gate.

### 6.5 Training-distribution weakness

The internal validation result is much higher than both OOD results. The current real negatives cover photos and museum art better than contemporary digital art, anime, and CGI. That gap is consistent with the observed false positive.

## 7. Implemented model

Release name: **Cyclopes ScalePair CNN V5**.

### 7.1 Inference graph

Input: one RGB tensor, `1 × 3 × 224 × 224`.

Two branches inside one model:

1. **RGB branch** — MobileNetV3-Large initialized from ordinary ImageNet classification weights. It supplies semantic and shape context but no AI-detector knowledge.
2. **ScalePair signal** — the same image is downscaled and restored; early shared-backbone feature differences and compact residual statistics expose resampling and synthesis texture behavior.

Global-average-pooled features are concatenated and passed through one dropout + linear classifier head. Calibration is represented by one scalar temperature and one scalar bias before the final sigmoid.

Targets:

- one ONNX graph;
- less than 30 MiB FP32, with FP16 or INT8 accepted only if validation balanced accuracy drops by no more than 0.5 percentage points;
- batch-one browser inference;
- no test-time crop ensemble in the first release.

The training topology is Siamese, but the browser topology is not: both paired views share the same model during training, while inference scores one rendered image once.

### 7.2 Why this is different

As of the review date, no examined bounty submission ships a project-trained RGB + fixed-residual CNN whose primary training objective enforces full-size/thumbnail score consistency. Individual ingredients such as CNNs, high-pass filters, and JPEG augmentation are established techniques; the claim of originality is limited to this submitted model and training/runtime combination, not invention of those primitives.

No public AI-detector weights will be included or distilled into the initial candidate. Existing detectors may be used only to inspect disagreements after labels and splits are frozen; their scores cannot become training features or runtime inputs.

## 8. Concrete data plan and result

The base manifest finished at 46,417 images. The V5 replay manifest added 9,206 audited examples: 3,707 CC0 OpenGameArt digital-art/CGI negatives and 5,499 CC BY 4.0 COCOXGEN/PAMELA positives. Exact counts, revisions, and hashes are in [`DATASETS.md`](../DATASETS.md).

### 8.1 Manifest contract

Every row must contain:

```text
path,label,source,generator,group,split,license,sha256
```

Rules:

- label `0` is real and `1` is AI;
- related originals, derivatives, prompt siblings, and near duplicates share a group;
- no group may cross splits;
- exact path and byte duplicates are rejected;
- perceptual-hash near duplicates are audited before the GPU run;
- metadata is retained only in the provenance manifest, never exposed to the model;
- both classes are decoded and normalized through the same image pipeline.

### 8.2 Target composition

| Class/source family | Target unique images | Sampling rule |
| --- | ---: | --- |
| Real camera photographs | 12,000–15,000 | source-balanced from DOCCI and FakeClue real |
| Real traditional/public-domain art | 5,000–7,000 | Met Open Access |
| Real contemporary digital art/anime/CGI | 4,000–7,000 | license-filtered Openverse/OpenGameArt hard negatives |
| Real logos/posters/icons/vector renderings | 1,000–3,000 | license-filtered Openverse/Wikimedia hard negatives |
| AI legacy/open generators | 8,000–10,000 | source- and generator-balanced FakeClue |
| AI SDXL/Fooocus | approximately 2,400 | COCOXGEN |
| AI recent generators | 2,000–3,000 | PAMELA, Nano Banana, Nano Banana Pro |

Epoch sampling is 50% real and 50% AI. Within each label, sources and generators are sampled approximately uniformly so the largest dataset cannot dominate.

### 8.3 Split policy

- `train`: 80% of eligible groups from training sources;
- `calibration`: 10%, disjoint groups, used only for temperature/bias;
- `validation`: 10%, disjoint groups, used for checkpoint and compression selection;
- `Arena`: development OOD only; already observed and therefore not a frozen test;
- `Synthbuster + RAISE-1k`: historical OOD regression only; already observed and therefore no longer honestly blind;
- `Rule34 field set`: one-time qualitative browser regression after the model is frozen.

The private POIDH set is the only truly blind final evaluation. Public results will be described as development or regression evidence, not as a prediction of the private score.

## 9. Paired web-consistency training

For each source image, construct two views on the fly:

### View A: reference

- decode to RGB;
- aspect-preserving resize/crop to the model input;
- light label-symmetric horizontal flip and color jitter only.

### View B: browser/web

Apply a randomized chain, identically distributed for real and AI labels:

- downscale the shorter side to 96–256 px;
- Lanczos, bilinear, or bicubic resampling;
- JPEG quality 35–95 or WebP quality 40–95;
- optional second recompression;
- mild Gaussian blur or sharpening;
- 0–10% crop followed by resize;
- screenshot-like resampling;
- final browser-matched conversion to `256 × 256`.

Do not add fake generator markers, class-dependent codecs, or metadata shortcuts.

Loss for paired logits `z_ref` and `z_web`:

```text
L = 0.5 * BCE(z_ref, y)
  + 0.5 * BCE(z_web, y)
  + 0.25 * SmoothL1(z_ref, z_web)
```

The consistency weight is frozen before the OOD evaluations. It may be changed only using the calibration/validation splits, never the Rule34 cases.

## 10. Training run

The H200 envelope below was the safety ceiling for v0.1, whose accepted V5 replay run completed locally on Apple MPS in 100 steps. VNext uses a separate 90-minute wall-clock limit and is not started until its training manifest and 50,000–100,000-image independent evaluation corpus pass the frozen audits.

### 10.1 Before paid GPU time

1. Acquire and verify all allowed datasets.
2. Materialize equal-codec training images or cached resize sources.
3. Build the final manifest and provenance report.
4. Run duplicate, near-duplicate, group-leak, license, and decode audits.
5. Run a CPU smoke training, ONNX export, and one-image browser inference.
6. Freeze config, seed, split hashes, and commands in the repository.

### 10.2 Frozen hyperparameters

| Parameter | Value |
| --- | --- |
| Input | `256 × 256` RGB |
| Seed | `323` |
| Precision | BF16 mixed precision |
| Optimizer | AdamW |
| RGB-backbone learning rate | `5e-5` |
| Residual branch/head learning rate | `3e-4` |
| Weight decay | `0.05` |
| Schedule | 1 warm-up epoch, cosine decay |
| Batch | 128 paired samples / 256 rendered views, increase only after a measured memory smoke |
| Maximum epochs | 20 |
| Early stopping | patience 4 on validation selection score |

Checkpoint selection score:

```text
min(BA_reference_at_0.65, BA_web_at_0.65)
```

This prevents a high clean score from hiding thumbnail failure.

### 10.3 Ablations within the same run budget

Train only the following three candidates:

1. RGB branch baseline.
2. RGB + residual branch without consistency loss.
3. RGB + residual branch with paired consistency loss.

The ablations establish whether each added component materially improves web validation. Do not branch into additional architectures during the paid run.

### 10.4 H200 time and cost envelope

Use one verified H200 instance around the previously observed `$3.5/hour` rate. Hard stop remains the user's `$25` total project budget.

| Work | Maximum wall time |
| --- | ---: |
| Environment/data smoke | 15 min |
| Three compact ablations | 60 min |
| Calibration, validation, OOD scoring | 25 min |
| ONNX export, parity, compression check | 15 min |
| Contingency | 5 min |

Maximum billed GPU window: 2 hours. Dataset download and exploratory generation must not consume this window. Stop the instance immediately after artifacts and reports are copied locally.

## 11. Evaluation and release gates

All classification metrics use AI if and only if displayed score is at least `0.65`.

### 11.1 Required quantitative gates

| Gate | Required result |
| --- | ---: |
| Validation balanced accuracy, reference views | at least 80% |
| Validation balanced accuracy, web views | at least 78% |
| Arena development OOD balanced accuracy | at least 78% |
| AI recall | at least 75% |
| Real specificity | at least 75% |
| Any mixed-label source balanced accuracy | at least 65% |
| Full/web threshold-decision agreement | at least 92% |
| Mean absolute full/web score difference | at most 0.10 |
| Python vs ONNX per-image score difference | at most 0.01 |
| ONNX vs Chrome per-image score difference | at most 0.01 |
| Compression-induced BA loss | at most 0.5 percentage points |

Synthbuster is reported but cannot serve as a blind release gate because its current-model result has already informed this redesign.

### 11.2 Browser gates

In a clean Chrome profile:

1. Load the unpacked production `dist/`.
2. Confirm toolbar OFF → loading → ON behavior.
3. Confirm an ordinary static-image page and a dynamic/infinite page are analyzed.
4. Confirm the extension works after inference assets are local and external networking is disabled.
5. Confirm no runtime request except fetching the webpage's displayed image URL.
6. Confirm each analyzed element receives a numeric diagnostic score.
7. Confirm AI scores at or above 0.65 blur and scores below 0.65 do not.
8. Run the frozen Rule34 URLs once and record full/thumbnail results without further tuning on them.

Field target, not a statistical accuracy claim: at least 8 of the 10 AI-tagged cases blur, obvious case 18447267 and GIF thumbnail 18447269 blur, and the user-identified real case 18447261 remains unblurred.

### 11.3 Automatic rejection

Do not submit if any of the following is true:

- Arena OOD balanced accuracy is below 78%;
- either class recall is below 75%;
- web-view validation is below 78%;
- full/web agreement is below 92%;
- browser and Python decisions disagree at the fixed threshold;
- model provenance or a training-data license is unresolved;
- the extension requires a runtime model download, server, or API;
- the final graph still contains Sentry, ConvNeXt, Community Forensics, or another public AI-detector weight.

## 12. Expected repository changes

Implementation should stay within the existing project structure:

- `cyclopes/modeling.py`: replace the current model with the two-branch CNN;
- `cyclopes/data.py`: emit paired reference/web tensors;
- `cyclopes/cli.py`: train with paired loss and report view-specific metrics;
- dataset preparation tool: add licensed digital-art hard negatives and provenance;
- `AI-SPEC.md`: freeze the accepted single-model contract;
- `DATASETS.md`: record final sources, revisions, counts, and exclusions;
- `extension/src/inference.js`: load one graph and apply one calibration;
- `extension/models/`: replace both current graphs with one exported ONNX;
- `reports/`: add calibration, validation, OOD, parity, and browser reports;
- `docs/agents.md`: retain detailed reproducibility commands;
- `README.md`: only final demo, measured metrics, install/test, short approach, and development link.

No second extension UI, server component, metadata subsystem, watermark detector, or generalized model framework is planned.

## 13. Deliverables for peer review

Before submission, reviewers should be able to inspect:

- manifest and provenance hashes;
- exact training config and seed;
- ablation table;
- calibration file;
- fixed-threshold confusion matrices;
- per-source and per-generator metrics;
- reference versus web-view stability metrics;
- PyTorch/ONNX/Chrome parity rows;
- final ONNX hash and size;
- clean-profile offline browser evidence;
- Rule34 full/thumbnail regression table;
- concise disclosure of limitations and all failed gates.

## 14. Resolved review questions

Decisions used for V5:

1. Are the proposed Openverse/OpenGameArt digital-art licenses and attribution workflow acceptable for released model weights?
2. Is MobileNetV3 ImageNet initialization sufficiently distinct for the originality claim, or should the RGB branch also be trained from scratch at a likely accuracy cost?
3. Are the 80%/78% internal gates enough margin, or should Arena require 80%?
4. Should the field target require 8/10 or 9/10 AI-tagged Rule34 examples, given that three were explicitly judged visually ambiguous?
5. Does the team accept using no public-detector teacher distillation in the first candidate?

The final choices were: CC0 OpenGameArt only for the added real hard negatives; ordinary ImageNet initialization is acceptable because it is not an AI detector; Arena must exceed 80%; the user field set remains evaluation-only; and no teacher distillation is used.
