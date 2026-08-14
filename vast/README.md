# Vast run

Use one verified H200 with the image `pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime`, at least 250 GB disk, 24 CPU cores, and 180 GB free after checkout/data staging.

The account balance and offers must be checked before renting. At the current ~$3.91/hour H200 price, `MAX_SECONDS=5400` caps compute near $5.87 before bandwidth/storage. The controller must destroy the instance after copying `runs/vit-v1.tar.gz`; stopping it is not enough because storage continues billing.

```bash
export HF_READ_ONLY_TOKEN=...
bash vast/bootstrap.sh

export TRAIN_MANIFEST=/workspace/data/training/manifest.csv
export EVAL_MANIFEST=/workspace/data/evaluation/manifest.csv
MAX_SECONDS=5400 bash vast/run.sh
```

Do not put `VAST_API_KEY` on the rented machine. It remains local and is used only to create/destroy the instance. Existing SSH keys are attached through Vast; these scripts never create, overwrite, or delete keys.
