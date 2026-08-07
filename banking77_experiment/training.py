from __future__ import annotations

import math
import random
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import tinker

from .artifacts import append_jsonl
from .data import Example
from .rendering import training_datum


@dataclass(frozen=True)
class RenderedDatum:
    example: Example
    datum: tinker.Datum
    tokens: int


@dataclass(frozen=True)
class PlannedBatch:
    epoch: int
    batch_index: int
    items: tuple[RenderedDatum, ...]

    @property
    def tokens(self) -> int:
        return sum(item.tokens for item in self.items)


def render_training_rows(
    rows: Iterable[Example],
    renderer: Any,
    tokenizer: Any,
    labels: tuple[str, ...],
    prompt_variant: str,
    compact_system_prompt: str,
    effort: float,
    max_length: int,
) -> tuple[RenderedDatum, ...]:
    rendered: list[RenderedDatum] = []
    for example in rows:
        datum, tokens = training_datum(
            renderer,
            tokenizer,
            example,
            labels,
            prompt_variant,
            compact_system_prompt,
            effort,
            max_length,
        )
        rendered.append(RenderedDatum(example=example, datum=datum, tokens=tokens))
    return tuple(rendered)


def plan_batches(
    rendered: tuple[RenderedDatum, ...],
    batch_size: int,
    epochs: int,
    seed: int,
    max_steps: int | None = None,
) -> tuple[PlannedBatch, ...]:
    rng = random.Random(seed)
    planned: list[PlannedBatch] = []
    for epoch in range(epochs):
        items = list(rendered)
        rng.shuffle(items)
        for batch_index, start in enumerate(range(0, len(items), batch_size)):
            planned.append(
                PlannedBatch(
                    epoch=epoch,
                    batch_index=batch_index,
                    items=tuple(items[start : start + batch_size]),
                )
            )
            if max_steps is not None and len(planned) >= max_steps:
                return tuple(planned)
    return tuple(planned)


def scheduled_lr(
    learning_rate: float,
    schedule: str,
    warmup_ratio: float,
    step_index: int,
    total_steps: int,
) -> float:
    warmup_steps = max(0, int(total_steps * warmup_ratio))
    if warmup_steps and step_index < warmup_steps:
        return learning_rate * (step_index + 1) / warmup_steps
    if schedule == "constant":
        return learning_rate
    decay_steps = max(1, total_steps - warmup_steps - 1)
    progress = min(1.0, max(0.0, (step_index - warmup_steps) / decay_steps))
    if schedule == "linear":
        return learning_rate * (1.0 - progress)
    if schedule == "cosine":
        return learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported schedule: {schedule}")


def tensor_to_floats(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        return [float(item) for item in value.tolist()]
    if hasattr(value, "data"):
        return [float(item) for item in value.data]
    return [float(item) for item in value]


def mean_batch_nll(batch: PlannedBatch, result: Any) -> float:
    total_nll = 0.0
    total_weight = 0.0
    for item, output in zip(batch.items, result.loss_fn_outputs, strict=True):
        logprobs = tensor_to_floats(output["logprobs"])
        weights = tensor_to_floats(item.datum.loss_fn_inputs["weights"])
        total_nll += -sum(lp * weight for lp, weight in zip(logprobs, weights, strict=True))
        total_weight += sum(weights)
    return total_nll / max(total_weight, 1.0)


async def optimizer_step(
    training_client: tinker.TrainingClient,
    batch: PlannedBatch,
    learning_rate: float,
    metrics_path: Path,
    global_step: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    future = await training_client.forward_backward_async(
        [item.datum for item in batch.items],
        loss_fn="cross_entropy",
    )
    optimizer = await training_client.optim_step_async(
        tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    )
    result = await future.result_async()
    optimizer_result = await optimizer.result_async()
    record: dict[str, Any] = {
        "step": global_step,
        "epoch": batch.epoch,
        "batch_index": batch.batch_index,
        "learning_rate": learning_rate,
        "train_mean_nll": mean_batch_nll(batch, result),
        "batch_tokens": batch.tokens,
        "elapsed_seconds": time.perf_counter() - started,
    }
    if getattr(optimizer_result, "metrics", None):
        record["optimizer_metrics"] = optimizer_result.metrics
    append_jsonl(metrics_path, record)
    return record


async def save_immutable_checkpoint(
    training_client: tinker.TrainingClient,
    name: str,
    ttl_seconds: int,
) -> dict[str, str]:
    state_future = await training_client.save_state_async(name, ttl_seconds=ttl_seconds)
    sampler_future = await training_client.save_weights_for_sampler_async(
        name,
        ttl_seconds=ttl_seconds,
    )
    state = await state_future.result_async()
    sampler = await sampler_future.result_async()
    return {"name": name, "state_path": state.path, "sampler_path": sampler.path}
