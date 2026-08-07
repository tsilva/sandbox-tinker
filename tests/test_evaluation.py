import asyncio

import pytest

from banking77_experiment.artifacts import append_jsonl
from banking77_experiment.data import Example
from banking77_experiment.evaluation import EvaluationSpec, evaluate_rows, is_retryable_error


def test_non_transport_error_is_not_retryable():
    assert not is_retryable_error(ValueError("bad record"))


def test_resume_rejects_changed_contract_before_sampling(tmp_path):
    output = tmp_path / "predictions.jsonl"
    append_jsonl(
        output,
        {
            "status": "ok",
            "example_id": "a",
            "target_name": "old-target",
            "model_name": "model",
            "checkpoint_path": None,
            "effort": 0.0,
            "max_tokens": 32,
            "prompt_variant": "full_taxonomy",
            "config_hash": "cfg",
            "partition_hash": "partition",
        },
    )
    example = Example("a", "dev", 0, "text", 0, "label")
    spec = EvaluationSpec(
        target_name="new-target",
        model_name="model",
        checkpoint_path=None,
        effort=0.0,
        max_tokens=32,
        sample_concurrency=1,
        retry_attempts=1,
        config_hash="cfg",
        partition_hash="partition",
        compact_system_prompt="Return the Banking77 label.",
    )
    with pytest.raises(ValueError, match="violates the evaluation contract"):
        asyncio.run(
            evaluate_rows(
                None,  # type: ignore[arg-type]
                None,
                None,
                (example,),
                ("label",),
                spec,
                output,
            )
        )
