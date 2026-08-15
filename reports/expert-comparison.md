# Expert experiment

Frozen field manifest: 78 images from 71 user-provided cases. These images are evaluation-only and were not used for training, calibration, threshold selection, or checkpoint selection.

| Model | Arena clean BA | Arena web BA | Field BA | Field AI recall | Field real specificity | Field FP / FN | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| v0.2 single ViT + ScalePair | 94.88% | 91.67% | 67.29% | 86.36% | 48.21% | 29 / 3 | keep |
| Soft content experts | — | — | 62.82% | 86.36% | 39.29% | 34 / 3 | reject |
| Hard transform experts | — | — | 67.29% | 86.36% | 48.21% | 29 / 3 | reject: no change |
| Memotion composite expert | 92.75% | 86.49% | 68.18% | 86.36% | 50.00% | 28 / 3 | reject: Arena regression |
| Composite-only expert | — | — | 65.50% | 86.36% | 44.64% | 31 / 3 | reject |
| Mixed pixel/composite expert | — | — | 64.61% | 86.36% | 42.86% | 32 / 3 | reject |
| Dedicated pixel-art expert | — | — | 67.29% | 86.36% | 48.21% | 29 / 3 | reject: no change |
| Dedicated retro expert | — | — | 67.29% | 86.36% | 48.21% | 29 / 3 | reject: no change |
| Retro expert + 2 ViT blocks | — | — | 63.72% | 86.36% | 41.07% | 33 / 3 | reject |

| Frozen field slice | v0.2 real specificity | Best field-only expert |
| --- | ---: | ---: |
| Photo memes (10) | 20.00% | 40.00% |
| Game screenshots (3) | 33.33% | 0.00% |
| Photos (5; 1 AI, 4 real) | 0.00% | 50.00% |

The added GPT Image Gen retro/anime regression remained below threshold in every run (26.30% baseline; 33.06% best rejected expert). Public pixel-art data did not match this VHS/composite domain closely enough. A future run needs licensed GPT Image Gen retro/anime positives; further tuning on this single evaluation image would be leakage.

Acceptance requires lower field false positives without a material regression in Arena clean/web balanced accuracy or AI recall. The private bounty set remains unavailable.
