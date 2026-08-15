# Third-party notices

- The Cyclopes ScalePair backbone is initialized from TorchVision's MobileNetV3-Large ImageNet V2 weights. TorchVision is BSD-3-Clause licensed; Cyclopes supplies its own trained heads and fine-tuned weights.
- The experimental multi-layer ViT candidate initializes from the MIT-licensed Community Forensics ViT-S weights adapted by Borderless at pinned Hugging Face revision `ac6ee457bea904a373065754107451793b56db00`. Credit remains with Jeongsoo Park and Andrew Owens (Community Forensics) and the Borderless model adapter. Cyclopes adds its own layer/scale residual heads and paired-degradation training.
- `onnxruntime-web` is distributed under the MIT license. Its notices are preserved in the packaged runtime.

Training data is not redistributed. Dataset licenses and pinned revisions are listed in [`DATASETS.md`](DATASETS.md).

The V5 replay fine-tune uses CC0 assets indexed by OpenGameArt and CC BY 4.0 images from COCOXGEN and PAMELA. Source URLs, revisions, and counts are recorded in [`DATASETS.md`](DATASETS.md); no training images are shipped.
