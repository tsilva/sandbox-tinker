"""Run a sequential, interleaved latency benchmark for base and three adapters."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any

import tinker
from dotenv import load_dotenv
from tinker_cookbook.tokenizer_utils import get_tokenizer

from banking77_experiment.artifacts import append_jsonl, atomic_write_json, read_jsonl
from banking77_experiment.config import load_config
from banking77_experiment.data import balanced_subset, build_partitions
from banking77_experiment.evaluation import create_sampling_client, is_retryable_error
from banking77_experiment.metrics import percentile
from banking77_experiment.pricing import (
    TokenEstimate,
    estimate_cost,
    load_snapshot,
    refresh_snapshot,
    require_budget,
)
from banking77_experiment.rendering import generation_prompt, make_renderer, parse_sample

DEFAULT_CONFIG = "configs/inkling_small_banking77.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--selection", action="append", required=True, metavar="SEED=PATH")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _selections(values: list[str]) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator:
            raise ValueError(f"Expected SEED=PATH, got {value!r}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate seed {seed}")
        selection_path = Path(path_text).expanduser().resolve()
        selection = json.loads(selection_path.read_text())
        result[seed] = {
            "path": str(selection_path),
            "sampler_path": str(selection["sampler_path"]),
        }
    return result


async def _sample_with_retries(
    client: tinker.SamplingClient,
    prompt: Any,
    renderer: Any,
    max_tokens: int,
    attempts: int,
) -> tuple[Any, float, int]:
    for attempt in range(1, attempts + 1):
        started = time.perf_counter()
        try:
            response = await client.sample_async(
                prompt=prompt,
                num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=max_tokens,
                    temperature=0.0,
                    stop=renderer.get_stop_sequences(),
                ),
            )
            return response.sequences[0], time.perf_counter() - started, attempt
        except Exception as exc:
            if not is_retryable_error(exc) or attempt == attempts:
                raise
            await asyncio.sleep(min(8.0, 2 ** (attempt - 1)))
    raise AssertionError("unreachable")


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    selections = _selections(args.selection)
    if set(selections) != set(cfg.train_seeds):
        raise ValueError(f"Selections must exactly match seeds {cfg.train_seeds}")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle = build_partitions(cfg)
    rows = balanced_subset(bundle.dev, bundle.labels, cfg.latency_examples_per_label)
    tokenizer = get_tokenizer(cfg.model_name)
    renderer = make_renderer(cfg.model_name, tokenizer, cfg.renderer_name)
    prompt_tokens = sum(len(generation_prompt(renderer, row, cfg.effort).to_ints()) for row in rows)
    warmup_rows = tuple(rows[index % len(rows)] for index in range(cfg.latency_warmups_per_arm))
    warmup_tokens = sum(
        len(generation_prompt(renderer, row, cfg.effort).to_ints()) for row in warmup_rows
    )
    arm_count = 1 + len(selections)
    tokens = TokenEstimate(
        prefill_tokens=(prompt_tokens + warmup_tokens) * arm_count,
        sample_tokens=(len(rows) + len(warmup_rows)) * arm_count * cfg.max_eval_tokens,
    )
    baseline_price = load_snapshot(cfg.price_snapshot)
    price = baseline_price
    if not args.dry_run:
        price = refresh_snapshot(cfg.pricing_url, cfg.model_name, output_dir / "pricing.json")
    cost = estimate_cost(tokens, price.model)
    if not args.dry_run:
        require_budget(cost, args.budget_usd)

    arms = {"base": None} | {
        f"adapter-seed-{seed}": selection["sampler_path"]
        for seed, selection in sorted(selections.items())
    }
    manifest = {
        "config_hash": cfg.config_hash,
        "dev_partition_hash": bundle.manifest["hashes"]["dev"],
        "rows": len(rows),
        "warmups_per_arm": len(warmup_rows),
        "arms": arms,
        "selection_files": selections,
        "token_estimate": asdict(tokens),
        "cost_estimate": cost,
        "pricing": price.to_dict(),
        "baseline_pricing": baseline_price.to_dict(),
        "pricing_changed_since_snapshot": price.model != baseline_price.model,
        "protocol": "sequential_interleaved_rotating_arm_order",
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), **cost}, indent=2))
    if args.dry_run:
        return
    output_path = output_dir / "latency.jsonl"
    completed = {
        str(record["benchmark_id"]): record
        for record in read_jsonl(output_path)
        if record.get("status") == "ok"
    }
    for benchmark_id, record in completed.items():
        target_name = str(record.get("target_name"))
        expected = {
            "sampler_path": arms.get(target_name),
            "config_hash": cfg.config_hash,
            "partition_hash": bundle.manifest["hashes"]["dev"],
            "effort": cfg.effort,
            "max_tokens": cfg.max_eval_tokens,
        }
        mismatches = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if target_name not in arms or mismatches:
            raise ValueError(
                f"Cannot resume latency record {benchmark_id}: contract mismatch {mismatches}"
            )
    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Set TINKER_API_KEY before paid latency benchmarking")

    service = tinker.ServiceClient(base_url=args.base_url)
    clients = {
        name: await create_sampling_client(service, cfg.model_name, sampler_path)
        for name, sampler_path in arms.items()
    }
    for name, client in clients.items():
        service_tokenizer = client.get_tokenizer()
        if service_tokenizer.encode("Inkling renderer check") != tokenizer.encode(
            "Inkling renderer check"
        ):
            raise ValueError(f"Tokenizer differs for {name}")

    for client in clients.values():
        for row in warmup_rows:
            prompt = generation_prompt(renderer, row, cfg.effort)
            await _sample_with_retries(
                client,
                prompt,
                renderer,
                cfg.max_eval_tokens,
                cfg.retry_attempts,
            )

    arm_names = list(arms)
    for row_index, row in enumerate(rows):
        ordered_arms = (
            arm_names[row_index % len(arm_names) :] + arm_names[: row_index % len(arm_names)]
        )
        for target_name in ordered_arms:
            benchmark_id = f"{row.example_id}:{target_name}"
            if benchmark_id in completed:
                continue
            prompt = generation_prompt(renderer, row, cfg.effort)
            sequence, latency, attempt = await _sample_with_retries(
                clients[target_name],
                prompt,
                renderer,
                cfg.max_eval_tokens,
                cfg.retry_attempts,
            )
            parsed = parse_sample(renderer, tokenizer, sequence.tokens, bundle.labels)
            record = {
                "status": "ok",
                "benchmark_id": benchmark_id,
                "example_id": row.example_id,
                "target_name": target_name,
                "sampler_path": arms[target_name],
                "config_hash": cfg.config_hash,
                "partition_hash": bundle.manifest["hashes"]["dev"],
                "effort": cfg.effort,
                "max_tokens": cfg.max_eval_tokens,
                "latency_seconds": latency,
                "attempt": attempt,
                "generated_tokens": parsed.generated_tokens,
            }
            append_jsonl(output_path, record)
            completed[benchmark_id] = record

    expected = {f"{row.example_id}:{name}" for row in rows for name in arms}
    if set(completed) != expected:
        raise ValueError("Latency benchmark coverage is incomplete or contains extra records")
    by_arm = {
        name: [float(completed[f"{row.example_id}:{name}"]["latency_seconds"]) for row in rows]
        for name in arms
    }
    p95 = {name: percentile(values, 0.95) for name, values in by_arm.items()}
    p95_ratio = max(value for name, value in p95.items() if name != "base") / p95["base"]
    summary = {
        "config_hash": cfg.config_hash,
        "dev_partition_hash": bundle.manifest["hashes"]["dev"],
        "arms": arms,
        "protocol": manifest["protocol"],
        "p50_seconds": {name: percentile(values, 0.5) for name, values in by_arm.items()},
        "p95_seconds": p95,
        "mean_seconds": {name: mean(values) for name, values in by_arm.items()},
        "p95_latency_ratio": p95_ratio,
        "threshold": cfg.max_p95_latency_ratio,
        "passes": p95_ratio <= cfg.max_p95_latency_ratio,
    }
    atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
