from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker

from .artifacts import (
    append_jsonl,
    atomic_write_json,
    require_complete_coverage,
    successful_records_by_id,
)
from .data import Example
from .metrics import metrics_from_records
from .pricing import ModelPrice, TokenEstimate, estimate_cost
from .rendering import generation_prompt, parse_sample


@dataclass(frozen=True)
class EvaluationSpec:
    target_name: str
    model_name: str
    checkpoint_path: str | None
    effort: float
    max_tokens: int
    sample_concurrency: int
    retry_attempts: int
    config_hash: str
    partition_hash: str
    compact_system_prompt: str
    prompt_variant: str = "full_taxonomy"


def is_retryable_error(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            tinker.APIConnectionError,
            tinker.APITimeoutError,
            tinker.RateLimitError,
            tinker.InternalServerError,
        ),
    ):
        return True
    if isinstance(exc, tinker.APIStatusError):
        status = getattr(exc, "status_code", None)
        return status == 429 or (isinstance(status, int) and 500 <= status < 600)
    return False


def _jitter(example_id: str, attempt: int) -> float:
    digest = hashlib.sha256(f"{example_id}:{attempt}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big")).random() * 0.25


async def create_sampling_client(
    service_client: tinker.ServiceClient, model_name: str, checkpoint_path: str | None
) -> tinker.SamplingClient:
    if checkpoint_path is None:
        return await service_client.create_sampling_client_async(base_model=model_name)
    return await service_client.create_sampling_client_async(
        model_path=checkpoint_path,
        base_model=model_name,
    )


async def evaluate_rows(
    sampling_client: tinker.SamplingClient,
    renderer: Any,
    tokenizer: Any,
    rows: tuple[Example, ...],
    labels: tuple[str, ...],
    spec: EvaluationSpec,
    output_path: Path,
    price: ModelPrice | None = None,
) -> dict[str, Any]:
    completed = successful_records_by_id(output_path)
    resume_contract = {
        "target_name": spec.target_name,
        "model_name": spec.model_name,
        "checkpoint_path": spec.checkpoint_path,
        "effort": spec.effort,
        "max_tokens": spec.max_tokens,
        "prompt_variant": spec.prompt_variant,
        "config_hash": spec.config_hash,
        "partition_hash": spec.partition_hash,
    }
    for example_id, record in completed.items():
        mismatches = {
            key: (record.get(key), expected)
            for key, expected in resume_contract.items()
            if record.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                f"Cannot resume {output_path}: record {example_id} violates "
                f"the evaluation contract: {mismatches}"
            )
    semaphore = asyncio.Semaphore(spec.sample_concurrency)
    write_lock = asyncio.Lock()
    error_path = output_path.with_suffix(".errors.jsonl")

    async def predict(example: Example) -> None:
        if example.example_id in completed:
            return
        prompt = generation_prompt(
            renderer,
            example,
            labels,
            spec.prompt_variant,
            spec.compact_system_prompt,
            spec.effort,
        )
        prompt_tokens = len(prompt.to_ints())
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, spec.retry_attempts + 1):
            started = time.perf_counter()
            try:
                async with semaphore:
                    response = await sampling_client.sample_async(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=tinker.SamplingParams(
                            max_tokens=spec.max_tokens,
                            temperature=0.0,
                            stop=renderer.get_stop_sequences(),
                        ),
                    )
                latency = time.perf_counter() - started
                sequence = response.sequences[0]
                parsed = parse_sample(renderer, tokenizer, sequence.tokens, labels)
                record = {
                    "status": "ok",
                    "example_id": example.example_id,
                    "source_split": example.source_split,
                    "source_index": example.source_index,
                    "text": example.text,
                    "true_label": example.label,
                    "strict_pred": parsed.strict_prediction,
                    "tolerant_pred": parsed.tolerant_prediction,
                    "raw_text": parsed.raw_text,
                    "decoded_tokens": parsed.decoded_tokens,
                    "termination": parsed.termination,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": parsed.generated_tokens,
                    "latency_seconds": latency,
                    "attempt": attempt,
                    "retry_history": attempts,
                    "target_name": spec.target_name,
                    "model_name": spec.model_name,
                    "checkpoint_path": spec.checkpoint_path,
                    "effort": spec.effort,
                    "max_tokens": spec.max_tokens,
                    "prompt_variant": spec.prompt_variant,
                    "config_hash": spec.config_hash,
                    "partition_hash": spec.partition_hash,
                }
                async with write_lock:
                    append_jsonl(output_path, record)
                    completed[example.example_id] = record
                return
            except Exception as exc:  # Tinker exposes multiple transport exception subclasses.
                latency = time.perf_counter() - started
                retryable = is_retryable_error(exc)
                attempts.append(
                    {
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "latency_seconds": latency,
                        "retryable": retryable,
                    }
                )
                if not retryable or attempt >= spec.retry_attempts:
                    async with write_lock:
                        append_jsonl(
                            error_path,
                            {
                                "status": "error",
                                "example_id": example.example_id,
                                "target_name": spec.target_name,
                                "attempts": attempts,
                            },
                        )
                    return
                await asyncio.sleep(
                    min(8.0, 2 ** (attempt - 1)) + _jitter(example.example_id, attempt)
                )

    await asyncio.gather(*(predict(row) for row in rows))
    require_complete_coverage((row.example_id for row in rows), completed)
    ordered_records = [completed[row.example_id] for row in rows]
    metrics = metrics_from_records(labels, ordered_records)
    prompt_tokens = sum(int(row["prompt_tokens"]) for row in ordered_records)
    generated_tokens = sum(int(row["generated_tokens"]) for row in ordered_records)
    summary: dict[str, Any] = {
        "target_name": spec.target_name,
        "model_name": spec.model_name,
        "checkpoint_path": spec.checkpoint_path,
        "effort": spec.effort,
        "max_tokens": spec.max_tokens,
        "prompt_variant": spec.prompt_variant,
        "coverage": 1.0,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "metrics": metrics,
    }
    if price is not None:
        summary["estimated_cost"] = estimate_cost(
            TokenEstimate(prefill_tokens=prompt_tokens, sample_tokens=generated_tokens),
            price,
        )
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary
