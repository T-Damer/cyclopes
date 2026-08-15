from tools.prepare_hard_negatives import eligible_oga, eligible_photo
from tools.prepare_ai_replay import cocoxgen_group
from tools.prepare_blender_frames import download_url
from tools.prepare_anime_pairs import art_prompt, author_group
from tools.generate_modern_anime import prompts
from tools.audit_manifest import low_information, near_duplicate, normalized_group_split
from cyclopes.data import Sample
from pathlib import Path


def test_blender_download_prefers_the_largest_web_sized_file() -> None:
    info = {
        "files": [
            {"fileDownloadUrl": "720", "resolution": {"id": 720}},
            {"fileDownloadUrl": "1080", "resolution": {"id": 1080}},
            {"fileDownloadUrl": "2160", "resolution": {"id": 2160}},
        ]
    }
    assert download_url(info) == "1080"


def test_docci_neighbors_share_a_content_group() -> None:
    first = Sample(Path("train_00007.jpg"), 0, "google/docci", "camera", "old-7", "train")
    second = Sample(Path("train_00009.jpg"), 0, "google/docci", "camera", "old-9", "test")
    assert normalized_group_split(first) == normalized_group_split(second)


def test_anime_pair_selection_prefers_art_prompts_and_groups_authors() -> None:
    assert art_prompt("award-winning anime character illustration")
    assert not art_prompt("a documentary photograph of a bridge")
    assert author_group("images/artist-name/work.png", {str(index): 1 for index in range(20)}) == "csip:artist-name"
    assert len(prompts(512)) == 512
    assert len(set(prompts(512))) == 512


def test_perceptual_dedup_requires_structure_to_match() -> None:
    assert near_duplicate((0, 0), (0b111, 0xFF), 3, 8)
    assert not near_duplicate((0, 0), (0b111, 0x1FF), 3, 8)
    assert low_information((0, 0, 0.04), 3.0)


def test_hard_negative_filters_require_pre_ai_human_sources() -> None:
    oga = {"licenses": ["CC0"], "preview_images": ["image"], "post_date": "June 1, 2020", "tags": ["3D"]}
    assert eligible_oga(oga)
    assert not eligible_oga({**oga, "title": "AI-generated pack"})
    photo = {
        "jpg": {"src": "image"}, "capturedevice": "Nikon", "datetaken": "2019-02-01",
        "width": 640, "height": 480, "title": "press conference",
    }
    assert eligible_photo(photo)
    assert not eligible_photo({**photo, "datetaken": "2024-02-01"})
    assert cocoxgen_group("fooocus/123-long-fooocus.png") == "123"
    assert cocoxgen_group("sdxl/123-short-sdxl.jpg") == "123"
