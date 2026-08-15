import csv
from pathlib import Path

import pytest
from PIL import Image

from tools.build_evaluation import build


def test_build_evaluation_is_source_disjoint_and_scenario_stratified(tmp_path: Path) -> None:
    scenarios = ("photo", "ui", "illustration", "film", "3d", "meme", "logo-text", "ai-legacy", "ai-modern", "ai-anime-cgi")
    blocks = ["minimum_images = 10", "maximum_images = 20", f"required_scenarios = {list(scenarios)!r}"]
    for index, scenario in enumerate(scenarios):
        root = tmp_path / scenario
        root.mkdir()
        Image.new("RGB", (8, 8), (index, index, index)).save(root / "image.png")
        blocks.extend(
            (
                "[[source]]",
                f'name = "eval-{scenario}"',
                f'root = "{root}"',
                f"label = {int(scenario.startswith('ai-'))}",
                f'scenario = "{scenario}"',
                'license = "test"',
                "limit = 1",
            )
        )
    config = tmp_path / "sources.toml"
    config.write_text("\n".join(blocks).replace("'", '"') + "\n", encoding="utf-8")
    output = tmp_path / "manifest.csv"

    report = build(config, output, None)

    assert report["count"] == 10
    with output.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["content_domain"] for row in rows} == set(scenarios)
    assert {row["split"] for row in rows} == {"test"}


def test_build_evaluation_rejects_license_placeholders(tmp_path: Path) -> None:
    root = tmp_path / "images"
    root.mkdir()
    Image.new("RGB", (8, 8)).save(root / "image.png")
    config = tmp_path / "sources.toml"
    config.write_text(
        f'''minimum_images = 1
maximum_images = 2
required_scenarios = ["photo"]
[[source]]
name = "photos"
root = "{root}"
label = 0
scenario = "photo"
license = "source-specific"
limit = 1
''',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unapproved license"):
        build(config, tmp_path / "manifest.csv", None)
