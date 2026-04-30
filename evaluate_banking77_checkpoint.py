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
"""Evaluate a saved Tinker Banking77 LoRA sampler checkpoint.

Examples:
  uv run evaluate_banking77_checkpoint.py
  uv run evaluate_banking77_checkpoint.py --checkpoint-name 001000
  uv run evaluate_banking77_checkpoint.py --checkpoint-name best
  uv run evaluate_banking77_checkpoint.py --sampler-path tinker://.../sampler_weights/best
  uv run evaluate_banking77_checkpoint.py --eval-examples-per-label 1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import tinker
from dotenv import load_dotenv
from tinker_cookbook import renderers

from train_banking77_tinker import (
    DEFAULT_DATASET_NAME,
    DEFAULT_MODEL_NAME,
    append_jsonl,
    balanced_eval_rows,
    evaluate_macro_f1,
    load_banking77,
    resolve_renderer_name,
)


DEFAULT_LOG_DIR = "~/logs/tinker-banking77-llama32-1b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure macro-F1 for a saved Banking77 Tinker sampler checkpoint."
    )
    parser.add_argument(
        "--log-dir",
        default=DEFAULT_LOG_DIR,
        help="Training log directory containing config.json and checkpoints.jsonl.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Optional config JSON path. Defaults to <log-dir>/config.json when present.",
    )
    parser.add_argument(
        "--checkpoint-log",
        default=None,
        help="Optional checkpoints JSONL path. Defaults to <log-dir>/checkpoints.jsonl.",
    )
    checkpoint = parser.add_mutually_exclusive_group()
    checkpoint.add_argument(
        "--checkpoint-name",
        default="latest",
        help=(
            "Checkpoint name to load from checkpoints.jsonl. Uses the latest matching row. "
            "The special value 'latest' uses the newest non-'best' checkpoint."
        ),
    )
    checkpoint.add_argument(
        "--sampler-path",
        default=None,
        help="Explicit tinker:// sampler_weights path. Bypasses checkpoints.jsonl lookup.",
    )
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--eval-split", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--renderer-name", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--eval-examples-per-label",
        type=int,
        default=0,
        help="Balanced eval examples per label. Default 0 evaluates the full split.",
    )
    parser.add_argument("--sample-concurrency", type=int, default=None)
    parser.add_argument("--max-eval-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--output-path",
        default=None,
        help="Prediction JSON output path. Defaults under <log-dir>.",
    )
    parser.add_argument(
        "--metrics-path",
        default=None,
        help="Metrics JSONL output path. Defaults to <log-dir>/checkpoint_eval_metrics.jsonl.",
    )
    parser.add_argument(
        "--list-checkpoints",
        action="store_true",
        help="List checkpoint rows from checkpoints.jsonl and exit without evaluating.",
    )
    return parser.parse_args()


def read_config(args: argparse.Namespace, log_dir: Path) -> dict[str, Any]:
    config_path = Path(args.config_path).expanduser() if args.config_path else log_dir / "config.json"
    defaults: dict[str, Any] = {
        "dataset_name": DEFAULT_DATASET_NAME,
        "train_split": "train",
        "eval_split": "test",
        "model_name": DEFAULT_MODEL_NAME,
        "renderer_name": None,
        "base_url": None,
        "sample_concurrency": 24,
        "max_eval_tokens": 16,
        "seed": 13,
    }
    if config_path.exists():
        defaults.update(json.loads(config_path.read_text()))
    return defaults


def read_checkpoint_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint log does not exist: {path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
    return rows


def resolve_sampler_path(args: argparse.Namespace, checkpoint_log: Path) -> tuple[str, str]:
    if args.sampler_path:
        return args.sampler_path, "explicit"

    rows = read_checkpoint_rows(checkpoint_log)
    if args.checkpoint_name == "latest":
        matches = [row for row in rows if row.get("name") and row.get("name") != "best"]
    else:
        matches = [row for row in rows if row.get("name") == args.checkpoint_name]
    if not matches:
        available = sorted({str(row.get("name")) for row in rows if row.get("name")})
        raise ValueError(
            f"No checkpoint named {args.checkpoint_name!r} found in {checkpoint_log}. "
            f"Available names: {', '.join(available) or '<none>'}"
        )

    row = matches[-1]
    checkpoint_name = str(row.get("name", args.checkpoint_name))
    if args.checkpoint_name == "best":
        print(
            "warning: checkpoint name 'best' is a reused alias in this training log; "
            "prefer a numbered checkpoint such as '001100' for persisted eval."
        )
    sampler_path = row.get("sampler_path")
    if not sampler_path:
        raise ValueError(f"Checkpoint row {args.checkpoint_name!r} has no sampler_path")
    return str(sampler_path), checkpoint_name


def merged_eval_args(cli: argparse.Namespace, config: dict[str, Any]) -> SimpleNamespace:
    values = dict(config)
    for key in (
        "dataset_name",
        "train_split",
        "eval_split",
        "model_name",
        "renderer_name",
        "base_url",
        "sample_concurrency",
        "max_eval_tokens",
        "seed",
    ):
        override = getattr(cli, key)
        if override is not None:
            values[key] = override

    values["eval_split"] = values.get("eval_split") or "test"
    values["sample_concurrency"] = int(values.get("sample_concurrency") or 24)
    values["max_eval_tokens"] = int(values.get("max_eval_tokens") or 16)
    values["seed"] = int(values.get("seed") or 13)
    return SimpleNamespace(**values)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "checkpoint"


def print_checkpoints(checkpoint_log: Path) -> None:
    rows = read_checkpoint_rows(checkpoint_log)
    for row in rows:
        name = row.get("name", "<missing>")
        step = row.get("step", "<missing>")
        sampler_path = row.get("sampler_path", "<missing>")
        print(f"{name}\tstep={step}\t{sampler_path}")


async def run_eval() -> None:
    load_dotenv()
    cli = parse_args()
    log_dir = Path(cli.log_dir).expanduser()
    checkpoint_log = (
        Path(cli.checkpoint_log).expanduser()
        if cli.checkpoint_log
        else log_dir / "checkpoints.jsonl"
    )

    if cli.list_checkpoints:
        print_checkpoints(checkpoint_log)
        return

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Set TINKER_API_KEY before evaluating a Tinker checkpoint.")

    config = read_config(cli, log_dir)
    args = merged_eval_args(cli, config)
    sampler_path, checkpoint_label = resolve_sampler_path(cli, checkpoint_log)

    renderer_name = resolve_renderer_name(args.model_name, args.renderer_name)
    service_client = tinker.ServiceClient(base_url=args.base_url)
    sampling_client = await service_client.create_sampling_client_async(
        model_path=sampler_path,
        base_model=args.model_name,
    )
    renderer = renderers.get_renderer(renderer_name, sampling_client.get_tokenizer())

    _train_ds, eval_ds, labels = load_banking77(args)
    rows = balanced_eval_rows(
        eval_ds,
        labels,
        examples_per_label=cli.eval_examples_per_label,
        seed=args.seed,
    )

    safe_label = safe_filename(checkpoint_label)
    output_path = (
        Path(cli.output_path).expanduser()
        if cli.output_path
        else log_dir / f"predictions_{safe_label}_{args.eval_split}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"evaluating checkpoint={checkpoint_label} split={args.eval_split} "
        f"examples={len(rows)} sampler_path={sampler_path}"
    )
    metrics = await evaluate_macro_f1(
        sampling_client,
        renderer,
        rows,
        labels,
        args,
        name=f"checkpoint-{checkpoint_label}-{args.eval_split}",
        output_path=output_path,
    )

    metrics_record = {
        "checkpoint": checkpoint_label,
        "sampler_path": sampler_path,
        "dataset_name": args.dataset_name,
        "eval_split": args.eval_split,
        "eval_examples_per_label": cli.eval_examples_per_label,
        "predictions_path": str(output_path),
        **metrics,
    }
    metrics_path = (
        Path(cli.metrics_path).expanduser()
        if cli.metrics_path
        else log_dir / "checkpoint_eval_metrics.jsonl"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    append_jsonl(metrics_path, metrics_record)
    print(json.dumps(metrics_record, indent=2, sort_keys=True))
    print(f"predictions: {output_path}")
    print(f"metrics: {metrics_path}")


def main() -> None:
    asyncio.run(run_eval())


if __name__ == "__main__":
    main()
