import json

import pytest

from banking77_experiment.artifacts import (
    append_jsonl,
    canonical_hash,
    choose_checkpoint,
    create_or_validate_unseal_receipt,
    load_and_validate_test_plan,
    require_complete_coverage,
    successful_records_by_id,
)


def test_jsonl_resume_and_coverage(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, {"example_id": "a", "status": "error"})
    append_jsonl(path, {"example_id": "a", "status": "ok", "strict_pred": "x"})
    append_jsonl(path, {"example_id": "b", "status": "ok", "strict_pred": "y"})
    records = successful_records_by_id(path)
    require_complete_coverage(["a", "b"], records)
    assert set(records) == {"a", "b"}
    with pytest.raises(ValueError, match="coverage mismatch"):
        require_complete_coverage(["a", "c"], records)


def test_duplicate_success_is_rejected(tmp_path):
    path = tmp_path / "predictions.jsonl"
    append_jsonl(path, {"example_id": "a", "status": "ok"})
    append_jsonl(path, {"example_id": "a", "status": "ok"})
    with pytest.raises(ValueError, match="Duplicate completed"):
        successful_records_by_id(path)


def test_checkpoint_tie_breaking_is_deterministic():
    selected = choose_checkpoint(
        [
            {"step": 20, "strict_macro_f1": 0.8, "strict_invalid_rate": 0.1, "train_tokens": 5},
            {"step": 10, "strict_macro_f1": 0.8, "strict_invalid_rate": 0.0, "train_tokens": 8},
            {"step": 5, "strict_macro_f1": 0.7, "strict_invalid_rate": 0.0, "train_tokens": 1},
        ]
    )
    assert selected["step"] == 10


def test_test_plan_and_receipt_are_immutable(tmp_path):
    plan = {
        "experiment_id": "exp",
        "config_hash": "cfg",
        "dataset_revision": "rev",
        "test_partition_hash": "test",
        "effort": 0.0,
        "max_tokens": 32,
        "arms": [
            {
                "target_name": "base-full",
                "sampler_path": None,
                "prompt_variant": "full_taxonomy",
            }
        ],
    }
    plan_path = tmp_path / "test-plan.json"
    plan_path.write_text(json.dumps(plan))
    loaded, digest = load_and_validate_test_plan(
        plan_path,
        experiment_id="exp",
        config_hash="cfg",
        dataset_revision="rev",
        test_partition_hash="test",
        target_name="base-full",
        sampler_path=None,
        effort=0.0,
        max_tokens=32,
        prompt_variant="full_taxonomy",
    )
    assert loaded == plan
    assert digest == canonical_hash(plan)

    receipt_path = tmp_path / "receipt.json"
    create_or_validate_unseal_receipt(receipt_path, {"test_plan_hash": digest})
    create_or_validate_unseal_receipt(receipt_path, {"test_plan_hash": digest})
    with pytest.raises(ValueError, match="different receipt"):
        create_or_validate_unseal_receipt(receipt_path, {"test_plan_hash": "changed"})
