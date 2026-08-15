#!/usr/bin/env python3
"""Generate deterministic 256px samples with OpenAI's filtered GLIDE release."""

from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import product
from pathlib import Path

import torch
from PIL import Image
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import create_model_and_diffusion, model_and_diffusion_defaults, model_and_diffusion_defaults_upsampler

from generate_legacy import ADJECTIVES, SETTINGS, SUBJECTS, FIELDS, save, split_for


def tokens_for(model, prompts, text_ctx):
    encoded = [model.tokenizer.padded_tokens_and_mask(model.tokenizer.encode(prompt), text_ctx) for prompt in prompts]
    return torch.tensor([item[0] for item in encoded], device="cuda"), torch.tensor([item[1] for item in encoded], dtype=torch.bool, device="cuda")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    root = args.output.resolve()
    options = model_and_diffusion_defaults()
    options.update(use_fp16=True, timestep_respacing="50")
    model, diffusion = create_model_and_diffusion(**options)
    model.eval().convert_to_fp16()
    model.to("cuda").load_state_dict(load_checkpoint("base", torch.device("cuda")))
    options_up = model_and_diffusion_defaults_upsampler()
    options_up.update(use_fp16=True, timestep_respacing="fast27")
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.eval().convert_to_fp16()
    model_up.to("cuda").load_state_dict(load_checkpoint("upsample", torch.device("cuda")))
    prompts = [f"a {adjective} {subject} {setting}" for adjective, subject, setting in product(ADJECTIVES, SUBJECTS, SETTINGS)]
    rows = []
    for start in range(0, args.count, args.batch_size):
        batch = prompts[start:start + min(args.batch_size, args.count - start)]
        size = len(batch)
        torch.manual_seed(323 + start)
        cond_tokens, cond_mask = tokens_for(model, batch, options["text_ctx"])
        empty_tokens, empty_mask = model.tokenizer.padded_tokens_and_mask([], options["text_ctx"])
        kwargs = {
            "tokens": torch.cat((cond_tokens, torch.tensor([empty_tokens] * size, device="cuda"))),
            "mask": torch.cat((cond_mask, torch.tensor([empty_mask] * size, dtype=torch.bool, device="cuda"))),
        }
        guidance = (2.0, 3.0, 4.0)[(start // args.batch_size) % 3]

        def guided(value, timesteps, **model_kwargs):
            half = value[: len(value) // 2]
            output = model(torch.cat((half, half)), timesteps, **model_kwargs)
            epsilon, rest = output[:, :3], output[:, 3:]
            conditional, unconditional = torch.chunk(epsilon, 2)
            guided_epsilon = unconditional + guidance * (conditional - unconditional)
            return torch.cat((torch.cat((guided_epsilon, guided_epsilon)), rest), dim=1)

        model.del_cache()
        low = diffusion.p_sample_loop(guided, (size * 2, 3, 64, 64), device="cuda", clip_denoised=True,
                                      progress=False, model_kwargs=kwargs)[:size]
        model.del_cache()
        up_tokens, up_mask = tokens_for(model_up, batch, options_up["text_ctx"])
        up_kwargs = {"low_res": ((low + 1) * 127.5).round() / 127.5 - 1, "tokens": up_tokens, "mask": up_mask}
        shape = (size, 3, 256, 256)
        model_up.del_cache()
        images = diffusion_up.ddim_sample_loop(model_up, shape, noise=torch.randn(shape, device="cuda") * 0.997,
                                               device="cuda", clip_denoised=True, progress=False, model_kwargs=up_kwargs)
        model_up.del_cache()
        for offset, tensor in enumerate(images.add(1).div(2).clamp(0, 1).cpu()):
            index = start + offset
            image = Image.fromarray(tensor.mul(255).byte().permute(1, 2, 0).numpy())
            path = root / "images" / f"{index:06d}.jpg"
            digest = save(image, path)
            group = f"glide:{index:06d}"
            rows.append({"path": path.relative_to(root), "label": 1, "source_dataset": "self-generated/glide",
                         "generator_model": "glide-filtered", "content_group": group, "split": split_for(group),
                         "family": "legacy-diffusion", "domain": "mixed", "license": "MIT", "sha256": digest})
    manifest = root / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)} sha256={hashlib.sha256(manifest.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
