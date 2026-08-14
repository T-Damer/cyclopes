#!/usr/bin/env python3
"""Materialize the user-provided, evaluation-only field regression set."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

FIELDS = (
    "path", "label", "source_dataset", "generator_model", "content_group",
    "split", "family", "domain", "license", "sha256",
)
USER_AGENT = "Mozilla/5.0 Cyclopes field regression"


def fetch(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read(), response.headers.get_content_type()


def rule34_media(page: bytes) -> list[str]:
    text = page.decode("utf-8", "replace")
    matches = []
    for pattern in (
        r'<meta\s+property="og:image"[^>]+content="([^"]+)"',
        r'https://wimg\.rule34\.xxx/+thumbnails/+[^"& ]+\.jpg',
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            matches.append(html.unescape(match.group(1) if match.lastindex else match.group(0)))
    return list(dict.fromkeys(matches))


def suffix(url: str, content_type: str) -> str:
    known = {"image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp"}
    return known.get(content_type, Path(urlparse(url).path).suffix.lower() or ".img")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("cases", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    payload = json.loads(args.cases.read_text(encoding="utf-8"))
    if payload.get("training_allowed") is not False:
        raise ValueError("field cases must remain evaluation-only")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    evidence = []
    seen: dict[str, dict[str, object]] = {}

    for case in payload["cases"]:
        media_urls: list[str] = []
        if case.get("local_file"):
            media_urls.append((args.output / case["local_file"]).resolve().as_uri())
        for url in case["urls"]:
            if url.startswith("https://rule34.xxx/index.php"):
                page, _ = fetch(url)
                media_urls.extend(rule34_media(page))
            else:
                media_urls.append(url)
        for index, url in enumerate(dict.fromkeys(media_urls)):
            encoded, content_type = fetch(url)
            digest = hashlib.sha256(encoded).hexdigest()
            label = int(case["expected_label"] == "ai")
            if digest in seen:
                previous = seen[digest]
                if previous["label"] != label:
                    raise ValueError(f"identical field image has conflicting labels: {url}")
                evidence.append({"case": case["id"], "url": url,
                                 "duplicate_of": previous["path"], "sha256": digest})
                continue
            path = args.output / f'{case["id"]}-{index}{suffix(url, content_type)}'
            path.write_bytes(encoded)
            row = {
                "path": path.name,
                "label": label,
                "source_dataset": "field-regression",
                "generator_model": "unknown",
                "content_group": f'field:{case["id"]}',
                "split": "test",
                "family": "modern" if case["expected_label"] == "ai" else "real",
                "domain": case["content_kind"],
                "license": "evaluation-only-not-for-redistribution",
                "sha256": digest,
            }
            rows.append(row)
            seen[digest] = row
            evidence.append({"case": case["id"], "url": url, "path": path.name,
                             "sha256": digest, "bytes": len(encoded), "content_type": content_type})

    manifest = args.output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "training_allowed": False,
        "cases": len(payload["cases"]),
        "images": len(rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "evidence": evidence,
    }
    (args.output / "provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("cases", "images", "manifest_sha256")}, indent=2))


if __name__ == "__main__":
    main()
