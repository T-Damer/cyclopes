import csv
from pathlib import Path

import pytest

from cyclopes.data import load_manifest


FIELDS = ("path", "label", "source", "generator", "group", "split")


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(path: str, group: str = "g", split: str = "train") -> dict[str, object]:
    return {
        "path": path,
        "label": 0,
        "source": "source",
        "generator": "",
        "group": group,
        "split": split,
    }


def test_manifest_rejects_duplicate_paths(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row("missing.png"), _row("./missing.png", group="other")])

    with pytest.raises(ValueError, match="duplicate image path"):
        load_manifest(manifest)


def test_manifest_rejects_duplicate_file_content(tmp_path: Path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"same image bytes")
    second.write_bytes(first.read_bytes())
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [_row(first.name), _row(second.name, group="other", split="test")])

    with pytest.raises(ValueError, match="duplicate image content"):
        load_manifest(manifest)
