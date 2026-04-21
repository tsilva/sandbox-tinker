# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "datasets>=4.0.0",
#   "orjson>=3.10.0",
#   "python-dotenv>=1.0.0",
#   "tinker",
#   "tinker-cookbook",
# ]
# ///
"""Fine-tune Tinker's smallest public model on tsilva/banking77.

The dataset is rendered as chat:
  system_prompt -> system
  text          -> user
  label_text    -> assistant

Default target:
  meta-llama/Llama-3.2-1B, the smallest model in the public Tinker lineup.

Run:
  echo 'TINKER_API_KEY=...' > .env
  uv run train_banking77_tinker.py

Fast local validation without a Tinker API call:
  uv run train_banking77_tinker.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets
import tinker
from dotenv import load_dotenv
from tinker_cookbook import model_info, renderers
from tinker_cookbook.renderers import TrainOnWhat, get_text_content
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.tokenizer_utils import get_tokenizer


DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEFAULT_DATASET_NAME = "tsilva/banking77"


@dataclass(frozen=True)
class TrainArgs:
    dataset_name: str
    train_split: str
    eval_split: str
    model_name: str
    renderer_name: str | None
    log_dir: str
    base_url: str | None
    batch_size: int
    epochs: int
    max_steps: int | None
    max_length: int
    learning_rate: float
    lr_schedule: str
    warmup_ratio: float
    lora_rank: int
    eval_every: int
    eval_examples_per_label: int
    final_eval_examples_per_label: int
    sample_concurrency: int
    max_eval_tokens: int
    save_every: int
    seed: int
    dry_run: bool
    dry_run_examples: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(
        description="LoRA SFT for tsilva/banking77 with macro-F1 evaluation."
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--renderer-name",
        default=None,
        help="Defaults to tinker_cookbook.model_info's recommended renderer.",
    )
    parser.add_argument("--log-dir", default="~/logs/tinker-banking77-llama32-1b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "linear", "cosine"),
        default="linear",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Run sampled macro-F1 every N optimizer steps. Use 0 to disable.",
    )
    parser.add_argument(
        "--eval-examples-per-label",
        type=int,
        default=2,
        help="Balanced periodic eval subset size per label. Use 0 for full eval.",
    )
    parser.add_argument(
        "--final-eval-examples-per-label",
        type=int,
        default=0,
        help="Balanced final eval subset size per label. Default 0 means full test split.",
    )
    parser.add_argument("--sample-concurrency", type=int, default=24)
    parser.add_argument("--max-eval-tokens", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-run-examples", type=int, default=8)
    return TrainArgs(**vars(parser.parse_args()))


def resolve_renderer_name(model_name: str, renderer_name: str | None) -> str:
    if renderer_name:
        return renderer_name
    return model_info.get_recommended_renderer_name(model_name)


def load_banking77(args: TrainArgs) -> tuple[datasets.Dataset, datasets.Dataset, list[str]]:
    train_ds = datasets.load_dataset(args.dataset_name, split=args.train_split)
    eval_ds = datasets.load_dataset(args.dataset_name, split=args.eval_split)
    if not isinstance(train_ds, datasets.Dataset) or not isinstance(eval_ds, datasets.Dataset):
        raise TypeError("Expected Hugging Face Dataset splits, not streaming datasets.")

    required = {"text", "label_text", "system_prompt"}
    for split_name, ds in ((args.train_split, train_ds), (args.eval_split, eval_ds)):
        missing = required - set(ds.column_names)
        if missing:
            raise ValueError(f"{split_name!r} split is missing required columns: {sorted(missing)}")

    label_pairs = {
        int(row["label"]): str(row["label_text"])
        for row in list(train_ds.select_columns(["label", "label_text"]))
    }
    labels = [label_pairs[i] for i in sorted(label_pairs)]
    if len(labels) != 77:
        raise ValueError(f"Expected 77 labels, found {len(labels)}")
    return train_ds, eval_ds, labels


def row_to_messages(row: dict[str, Any], include_answer: bool) -> list[renderers.Message]:
    messages: list[renderers.Message] = [
        {"role": "system", "content": str(row["system_prompt"])},
        {"role": "user", "content": str(row["text"])},
    ]
    if include_answer:
        messages.append({"role": "assistant", "content": str(row["label_text"])})
    return messages


def render_training_data(
    train_rows: list[dict[str, Any]],
    renderer: renderers.Renderer,
    max_length: int,
) -> list[tinker.Datum]:
    datums: list[tinker.Datum] = []
    started = time.time()
    for index, row in enumerate(train_rows, start=1):
        datums.append(
            conversation_to_datum(
                row_to_messages(row, include_answer=True),
                renderer,
                max_length=max_length,
                train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
        )
        if index % 1000 == 0:
            print(f"rendered {index}/{len(train_rows)} training examples")
    print(f"rendered {len(datums)} examples in {time.time() - started:.1f}s")
    return datums


def tensor_to_floats(value: Any) -> list[float]:
    if hasattr(value, "tolist"):
        return [float(x) for x in value.tolist()]
    if hasattr(value, "data"):
        return [float(x) for x in value.data]
    return [float(x) for x in value]


def mean_batch_nll(batch: list[tinker.Datum], fwd_bwd_result: Any) -> float:
    total_nll = 0.0
    total_weight = 0.0
    for datum, output in zip(batch, fwd_bwd_result.loss_fn_outputs, strict=True):
        logprobs = tensor_to_floats(output["logprobs"])
        weights = tensor_to_floats(datum.loss_fn_inputs["weights"])
        total_nll += -sum(lp * w for lp, w in zip(logprobs, weights, strict=True))
        total_weight += sum(weights)
    return total_nll / max(total_weight, 1.0)


def scheduled_lr(args: TrainArgs, step_index: int, total_steps: int) -> float:
    warmup_steps = max(0, int(total_steps * args.warmup_ratio))
    if warmup_steps and step_index < warmup_steps:
        return args.learning_rate * (step_index + 1) / warmup_steps

    if args.lr_schedule == "constant":
        return args.learning_rate

    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step_index - warmup_steps) / decay_steps))
    if args.lr_schedule == "linear":
        return args.learning_rate * (1.0 - progress)
    if args.lr_schedule == "cosine":
        return args.learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported LR schedule: {args.lr_schedule}")


def balanced_eval_rows(
    eval_ds: datasets.Dataset,
    labels: list[str],
    examples_per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = eval_ds.to_list()
    if examples_per_label <= 0:
        return rows

    rng = random.Random(seed)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["label_text"])].append(row)

    selected: list[dict[str, Any]] = []
    for label in labels:
        candidates = grouped[label][:]
        rng.shuffle(candidates)
        selected.extend(candidates[:examples_per_label])
    rng.shuffle(selected)
    return selected


def normalize_label(raw_text: str, labels: list[str]) -> str | None:
    label_set = set(labels)
    text = raw_text.strip()
    text = re.sub(r"(?i)^assistant\s*:\s*", "", text)
    text = text.strip().strip("`'\".,;:()[]{}")
    if text in label_set:
        return text

    compact = re.sub(r"[\s-]+", "_", text.lower()).strip("_")
    label_by_lower = {label.lower(): label for label in labels}
    if compact in label_by_lower:
        return label_by_lower[compact]

    matches = [
        label
        for label in labels
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", raw_text)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def macro_f1(labels: list[str], y_true: list[str], y_pred: list[str | None]) -> dict[str, float]:
    f1s: list[float] = []
    valid_predictions = [pred if pred is not None else "__invalid__" for pred in y_pred]
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(y_true, valid_predictions, strict=True))
        fp = sum(t != label and p == label for t, p in zip(y_true, valid_predictions, strict=True))
        fn = sum(t == label and p != label for t, p in zip(y_true, valid_predictions, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)

    correct = sum(t == p for t, p in zip(y_true, valid_predictions, strict=True))
    invalid = sum(pred is None for pred in y_pred)
    return {
        "macro_f1": sum(f1s) / len(f1s),
        "accuracy": correct / len(y_true),
        "invalid_rate": invalid / len(y_true),
    }


async def predict_one(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    row: dict[str, Any],
    labels: list[str],
    max_tokens: int,
) -> tuple[str, str | None, str]:
    prompt = renderer.build_generation_prompt(row_to_messages(row, include_answer=False))
    result = await sampling_client.sample_async(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,
            stop=renderer.get_stop_sequences(),
        ),
    )
    parsed, _success = renderer.parse_response(result.sequences[0].tokens)
    raw_text = get_text_content(parsed)
    return str(row["label_text"]), normalize_label(raw_text, labels), raw_text


async def evaluate_macro_f1(
    sampling_client: tinker.SamplingClient,
    renderer: renderers.Renderer,
    rows: list[dict[str, Any]],
    labels: list[str],
    args: TrainArgs,
    name: str,
    output_path: Path,
) -> dict[str, float]:
    semaphore = asyncio.Semaphore(args.sample_concurrency)
    completed = 0
    predictions: list[dict[str, str | None]] = []

    async def guarded(row: dict[str, Any]) -> tuple[str, str | None, str]:
        nonlocal completed
        async with semaphore:
            result = await predict_one(
                sampling_client,
                renderer,
                row,
                labels,
                max_tokens=args.max_eval_tokens,
            )
            completed += 1
            if completed % 100 == 0 or completed == len(rows):
                print(f"{name}: sampled {completed}/{len(rows)}")
            return result

    results = await asyncio.gather(*(guarded(row) for row in rows))
    y_true = [true for true, _pred, _raw in results]
    y_pred = [pred for _true, pred, _raw in results]
    metrics = macro_f1(labels, y_true, y_pred)
    metrics["num_examples"] = float(len(rows))

    for row, (true, pred, raw) in zip(rows, results, strict=True):
        predictions.append(
            {
                "text": str(row["text"]),
                "true": true,
                "pred": pred,
                "raw": raw,
            }
        )
    output_path.write_text(json.dumps(predictions, indent=2) + "\n")
    print(
        f"{name}: macro_f1={metrics['macro_f1']:.4f} "
        f"accuracy={metrics['accuracy']:.4f} invalid={metrics['invalid_rate']:.4f}"
    )
    return metrics


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def batches(items: list[tinker.Datum], batch_size: int) -> list[list[tinker.Datum]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


async def save_checkpoint(
    training_client: tinker.TrainingClient,
    log_dir: Path,
    name: str,
    loop_state: dict[str, Any],
    ttl_seconds: int | None,
) -> dict[str, str]:
    state_future = await training_client.save_state_async(name, ttl_seconds=ttl_seconds)
    sampler_future = await training_client.save_weights_for_sampler_async(
        name,
        ttl_seconds=ttl_seconds,
    )
    state_result = await state_future.result_async()
    sampler_result = await sampler_future.result_async()
    record = {
        "name": name,
        **loop_state,
        "state_path": state_result.path,
        "sampler_path": sampler_result.path,
    }
    append_jsonl(log_dir / "checkpoints.jsonl", record)
    print(f"saved checkpoint {name}: {sampler_result.path}")
    return {"state_path": state_result.path, "sampler_path": sampler_result.path}


async def run_training(args: TrainArgs) -> None:
    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Set TINKER_API_KEY before running training.")

    renderer_name = resolve_renderer_name(args.model_name, args.renderer_name)
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.json").write_text(json.dumps(asdict(args), indent=2) + "\n")

    train_ds, eval_ds, labels = load_banking77(args)
    service_client = tinker.ServiceClient(base_url=args.base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=args.model_name,
        rank=args.lora_rank,
        user_metadata={"task": "banking77-macro-f1", "renderer": renderer_name},
    )
    tokenizer = training_client.get_tokenizer()
    renderer = renderers.get_renderer(renderer_name, tokenizer)

    rng = random.Random(args.seed)
    train_rows = train_ds.to_list()
    rng.shuffle(train_rows)
    train_datums = render_training_data(train_rows, renderer, args.max_length)
    train_batches = batches(train_datums, args.batch_size)

    total_steps = len(train_batches) * args.epochs
    if args.max_steps is not None:
        total_steps = min(total_steps, args.max_steps)
    print(
        f"model={args.model_name} renderer={renderer_name} "
        f"train_examples={len(train_datums)} batches={len(train_batches)} total_steps={total_steps}"
    )

    periodic_eval_rows = balanced_eval_rows(
        eval_ds,
        labels,
        examples_per_label=args.eval_examples_per_label,
        seed=args.seed,
    )
    final_eval_rows = balanced_eval_rows(
        eval_ds,
        labels,
        examples_per_label=args.final_eval_examples_per_label,
        seed=args.seed + 1,
    )

    metrics_path = log_dir / "metrics.jsonl"
    best_macro_f1 = -1.0
    step = 0

    for epoch in range(args.epochs):
        epoch_batches = train_batches[:]
        rng.shuffle(epoch_batches)
        for batch_index, batch in enumerate(epoch_batches):
            if step >= total_steps:
                break

            lr = scheduled_lr(args, step, total_steps)
            started = time.time()
            fwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
            optim_future = await training_client.optim_step_async(
                tinker.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
            )
            fwd_result = await fwd_future.result_async()
            optim_result = await optim_future.result_async()
            loss = mean_batch_nll(batch, fwd_result)

            record: dict[str, Any] = {
                "step": step,
                "epoch": epoch,
                "batch_index": batch_index,
                "learning_rate": lr,
                "train_mean_nll": loss,
                "elapsed_seconds": time.time() - started,
            }
            if getattr(optim_result, "metrics", None):
                record.update({f"optim_{k}": v for k, v in optim_result.metrics.items()})

            print(
                f"step={step:05d} epoch={epoch} batch={batch_index:04d} "
                f"lr={lr:.2e} nll={loss:.4f} elapsed={record['elapsed_seconds']:.1f}s"
            )

            should_eval = args.eval_every > 0 and (
                step == 0 or (step + 1) % args.eval_every == 0
            )
            if should_eval:
                sampling_client = await training_client.save_weights_and_get_sampling_client_async()
                eval_metrics = await evaluate_macro_f1(
                    sampling_client,
                    renderer,
                    periodic_eval_rows,
                    labels,
                    args,
                    name=f"eval-step-{step + 1}",
                    output_path=log_dir / f"predictions_step_{step + 1:06d}.json",
                )
                record.update({f"eval/{k}": v for k, v in eval_metrics.items()})
                if eval_metrics["macro_f1"] > best_macro_f1:
                    best_macro_f1 = eval_metrics["macro_f1"]
                    paths = await save_checkpoint(
                        training_client,
                        log_dir,
                        name="best",
                        loop_state={"epoch": epoch, "batch": batch_index, "step": step},
                        ttl_seconds=None,
                    )
                    record["best_sampler_path"] = paths["sampler_path"]

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                await save_checkpoint(
                    training_client,
                    log_dir,
                    name=f"{step:06d}",
                    loop_state={"epoch": epoch, "batch": batch_index, "step": step},
                    ttl_seconds=7 * 24 * 60 * 60,
                )

            append_jsonl(metrics_path, record)
            step += 1

        if step >= total_steps:
            break

    final_paths = await save_checkpoint(
        training_client,
        log_dir,
        name="final",
        loop_state={"epoch": args.epochs, "batch": 0, "step": step},
        ttl_seconds=None,
    )

    final_sampling_client = await training_client.save_weights_and_get_sampling_client_async(
        name="final-eval"
    )
    final_metrics = await evaluate_macro_f1(
        final_sampling_client,
        renderer,
        final_eval_rows,
        labels,
        args,
        name="final",
        output_path=log_dir / "predictions_final.json",
    )
    append_jsonl(
        metrics_path,
        {
            "step": step,
            "final": True,
            **{f"final/{k}": v for k, v in final_metrics.items()},
            **final_paths,
        },
    )
    print(f"final sampler path: {final_paths['sampler_path']}")
    print(f"logs: {log_dir}")


def dry_run(args: TrainArgs) -> None:
    renderer_name = resolve_renderer_name(args.model_name, args.renderer_name)
    train_ds, eval_ds, labels = load_banking77(args)
    tokenizer = get_tokenizer(args.model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    rows = train_ds.select(range(min(args.dry_run_examples, len(train_ds)))).to_list()
    datums = render_training_data(rows, renderer, args.max_length)
    eval_rows = balanced_eval_rows(eval_ds, labels, examples_per_label=1, seed=args.seed)
    print(f"dry_run model={args.model_name} renderer={renderer_name}")
    print(f"train split={len(train_ds)} eval split={len(eval_ds)} labels={len(labels)}")
    print(f"rendered datums={len(datums)} eval_smoke_rows={len(eval_rows)}")
    print(f"first labels={labels[:10]}")


def main() -> None:
    load_dotenv()
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.sample_concurrency <= 0:
        raise ValueError("--sample-concurrency must be positive")
    if args.dry_run:
        dry_run(args)
        return
    asyncio.run(run_training(args))


if __name__ == "__main__":
    main()
