#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tomllib
from collections import Counter
from pathlib import Path

from PIL import Image


EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
REJECTED_LICENSES = {"", "unknown", "source-specific"}
FIELDS = (
    "path",
    "label",
    "source",
    "generator",
    "group",
    "split",
    "generator_family",
    "content_domain",
    "license",
    "sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _training_fingerprints(path: str | None) -> tuple[set[str], set[str]]:
    if not path:
        return set(), set()
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {row.get("source", row.get("source_dataset", "")) for row in rows}, {
        row["sha256"] for row in rows if row.get("sha256")
    }


def build(config_path: Path, output: Path, training_manifest: str | None) -> dict:
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    minimum = int(config.get("minimum_images", 50_000))
    maximum = int(config.get("maximum_images", 100_000))
    training_sources, training_hashes = _training_fingerprints(training_manifest)
    rows: list[dict[str, str | int]] = []
    seen: set[str] = set()

    for source in config["source"]:
        name = str(source["name"])
        license_name = str(source["license"]).strip()
        if license_name.lower() in REJECTED_LICENSES or "non-commercial" in license_name.lower() or "-nc" in license_name.lower():
            raise ValueError(f"evaluation source has an unapproved license: {name}: {license_name}")
        if name in training_sources:
            raise ValueError(f"evaluation source overlaps training source: {name}")
        root = (config_path.parent / source["root"]).resolve()
        candidates = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in EXTENSIONS
        )
        accepted: list[tuple[str, Path]] = []
        for path in candidates:
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception:
                continue
            digest = _sha256(path)
            if digest in training_hashes:
                raise ValueError(f"evaluation image overlaps training bytes: {path}")
            if digest not in seen:
                accepted.append((digest, path))
                seen.add(digest)
        accepted.sort(key=lambda item: item[0])
        limit = int(source["limit"])
        if len(accepted) < limit:
            raise ValueError(f"evaluation source {name} has {len(accepted)} unique images; requires {limit}")
        for digest, path in accepted[:limit]:
            rows.append(
                {
                    "path": str(path),
                    "label": int(source["label"]),
                    "source": name,
                    "generator": str(source.get("generator", "real")),
                    "group": f"{name}:{digest}",
                    "split": "test",
                    "generator_family": str(source.get("generator_family", "unknown")),
                    "content_domain": str(source["scenario"]),
                    "license": license_name,
                    "sha256": digest,
                }
            )

    if not minimum <= len(rows) <= maximum:
        raise ValueError(f"evaluation corpus has {len(rows)} images; expected {minimum}..{maximum}")
    labels = Counter(int(row["label"]) for row in rows)
    if min(labels.values(), default=0) < minimum // 3:
        raise ValueError(f"evaluation labels are imbalanced: {dict(labels)}")
    scenarios = Counter(str(row["content_domain"]) for row in rows)
    missing = sorted(set(config["required_scenarios"]) - scenarios.keys())
    if missing:
        raise ValueError(f"missing scenarios: {', '.join(missing)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "manifest": str(output.resolve()),
        "count": len(rows),
        "labels": dict(sorted(labels.items())),
        "scenarios": dict(sorted(scenarios.items())),
        "sources": dict(sorted(Counter(str(row["source"]) for row in rows).items())),
        "sha256": _sha256(output),
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="evaluation/sources.toml")
    parser.add_argument("--training-manifest")
    parser.add_argument("--output", default="data/evaluation/manifest.csv")
    parser.add_argument("--report", default="reports/evaluation-corpus.json")
    args = parser.parse_args()
    report = build(Path(args.config), Path(args.output), args.training_manifest)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
