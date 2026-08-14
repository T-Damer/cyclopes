import json
from pathlib import Path


def test_field_regression_manifest_is_evaluation_only() -> None:
    manifest = json.loads(
        Path(__file__).with_name("field-regression.json").read_text(encoding="utf-8")
    )

    assert manifest["training_allowed"] is False
    assert manifest["decision_threshold"] == 0.65
    assert len(manifest["cases"]) == 54
    candidates = manifest["ai_positive_profile_candidates"]
    assert "not licensed for training" in candidates["use"]
    assert len(candidates["urls"]) == len(set(candidates["urls"])) == 11

    case_ids = [case["id"] for case in manifest["cases"]]
    urls = [url for case in manifest["cases"] for url in case["urls"]]
    assert len(case_ids) == len(set(case_ids))
    assert len(urls) == len(set(urls)) == 19

    for case in manifest["cases"]:
        assert case["expected_label"] in {"ai", "real"}
        assert case["label_basis"]
        assert case["content_kind"]
        assert case["urls"] or case.get("local_file")
        if case.get("local_file"):
            assert len(case["sha256"]) == 64
        assert all(0.0 <= score <= 1.0 for score in case["current_scores"].values())
