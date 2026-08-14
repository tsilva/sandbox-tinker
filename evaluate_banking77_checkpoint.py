"""Evaluate base Inkling-Small or an immutable Banking77 LoRA checkpoint."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook.tokenizer_utils import get_tokenizer

from banking77_experiment.artifacts import (
    atomic_write_json,
    create_or_validate_unseal_receipt,
    git_metadata,
    load_and_validate_test_plan,
)
from banking77_experiment.config import load_config
from banking77_experiment.data import balanced_subset, build_partitions
from banking77_experiment.evaluation import (
    EvaluationSpec,
    create_sampling_client,
    evaluate_rows,
)
from banking77_experiment.pricing import (
    TokenEstimate,
    estimate_cost,
    load_snapshot,
    refresh_snapshot,
    require_budget,
)
from banking77_experiment.rendering import generation_prompt, make_renderer

DEFAULT_CONFIG = "configs/inkling_small_banking77.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--partition", choices=("dev", "test"), default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--test-plan", default=None)
    parser.add_argument("--experiment-id", default=None)
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--sampler-path", default=None)
    target.add_argument("--selection", default=None, help="Path to selection.json")
    parser.add_argument("--target-name", default=None)
    parser.add_argument(
        "--prompt-variant",
        choices=("full_taxonomy", "compact"),
        default=None,
        help="Defaults to full_taxonomy for base and compact for adapters.",
    )
    parser.add_argument("--effort", type=float, default=None)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--examples-per-label", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_." else "_" for character in value
    )


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    effort = cfg.effort if args.effort is None else args.effort
    if not 0.0 <= effort < 1.0:
        raise ValueError("--effort must be in [0, 1)")
    max_tokens = args.max_tokens or cfg.max_eval_tokens
    if max_tokens <= 0:
        raise ValueError("--max-tokens must be positive")

    sampler_path = args.sampler_path
    selection: dict[str, object] | None = None
    if args.selection:
        selection = json.loads(Path(args.selection).expanduser().read_text())
        sampler_path = str(selection["sampler_path"])
    prompt_variant = args.prompt_variant or (
        cfg.base_prompt_variant if sampler_path is None else cfg.adapter_prompt_variant
    )
    if args.target_name:
        target_name = args.target_name
    elif sampler_path is None:
        target_name = "base-full" if prompt_variant == cfg.base_prompt_variant else "base-compact"
    else:
        target_name = "adapter"

    bundle = build_partitions(cfg)
    if args.partition == "test":
        if not args.allow_test:
            raise ValueError("The sealed test split requires --allow-test")
        if not args.test_plan:
            raise ValueError("The sealed test split requires a frozen --test-plan")
        if args.examples_per_label is not None:
            raise ValueError("The sealed test split must be evaluated in full")
        rows = bundle.test
        partition_hash = bundle.manifest["hashes"]["test"]
    else:
        rows = bundle.dev
        partition_hash = bundle.manifest["hashes"]["dev"]
    if args.examples_per_label is not None:
        rows = balanced_subset(rows, bundle.labels, args.examples_per_label)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer = make_renderer(cfg.model_name, tokenizer, cfg.renderer_name)
    prompt_tokens = sum(
        len(
            generation_prompt(
                renderer,
                row,
                bundle.labels,
                prompt_variant,
                cfg.compact_system_prompt,
                effort,
            ).to_ints()
        )
        for row in rows
    )
    tokens = TokenEstimate(
        prefill_tokens=prompt_tokens,
        sample_tokens=len(rows) * max_tokens,
    )

    experiment_id = args.experiment_id or cfg.experiment_id
    test_plan: dict[str, object] | None = None
    test_plan_hash: str | None = None
    if args.partition == "test":
        test_plan, test_plan_hash = load_and_validate_test_plan(
            Path(args.test_plan).expanduser().resolve(),
            experiment_id=experiment_id,
            config_hash=cfg.config_hash,
            dataset_revision=cfg.dataset_revision,
            test_partition_hash=partition_hash,
            target_name=target_name,
            sampler_path=sampler_path,
            effort=effort,
            max_tokens=max_tokens,
            prompt_variant=prompt_variant,
        )
    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = cfg.run_root / experiment_id / "evaluations" / _safe_name(target_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_price = load_snapshot(cfg.price_snapshot)
    price = baseline_price
    if not args.dry_run:
        price = refresh_snapshot(cfg.pricing_url, cfg.model_name, output_dir / "pricing.json")
    cost = estimate_cost(tokens, price.model)
    if not args.dry_run:
        require_budget(cost, args.budget_usd)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": experiment_id,
        "target_name": target_name,
        "model_name": cfg.model_name,
        "sampler_path": sampler_path,
        "selection": selection,
        "partition": args.partition,
        "partition_hash": partition_hash,
        "rows": len(rows),
        "effort": effort,
        "max_tokens": max_tokens,
        "prompt_variant": prompt_variant,
        "config_hash": cfg.config_hash,
        "git": git_metadata(Path(__file__).resolve().parent),
        "token_estimate": {
            "prefill_tokens": tokens.prefill_tokens,
            "sample_tokens": tokens.sample_tokens,
        },
        "cost_estimate": cost,
        "pricing": price.to_dict(),
        "baseline_pricing": baseline_price.to_dict(),
        "pricing_changed_since_snapshot": price.model != baseline_price.model,
        "test_plan_hash": test_plan_hash,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    print(json.dumps({"output_dir": str(output_dir), **cost}, indent=2))
    if args.dry_run:
        atomic_write_json(
            output_dir / "dry_run_deterministic.json",
            {
                key: manifest[key]
                for key in (
                    "experiment_id",
                    "target_name",
                    "model_name",
                    "sampler_path",
                    "partition",
                    "partition_hash",
                    "rows",
                    "effort",
                    "max_tokens",
                    "prompt_variant",
                    "config_hash",
                    "token_estimate",
                    "test_plan_hash",
                )
            },
        )
        return

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Launch paid evaluation through `keyenv run -- ...`")
    service = tinker.ServiceClient(base_url=args.base_url)
    sampling_client = await create_sampling_client(service, cfg.model_name, sampler_path)
    if args.partition == "test":
        receipt = {
            "experiment_id": experiment_id,
            "config_hash": cfg.config_hash,
            "dataset_revision": cfg.dataset_revision,
            "test_partition_hash": partition_hash,
            "test_plan_hash": test_plan_hash,
            "arms": test_plan["arms"],
        }
        create_or_validate_unseal_receipt(
            cfg.run_root / experiment_id / "test_unseal_receipt.json",
            receipt,
        )
    summary = await evaluate_rows(
        sampling_client,
        renderer,
        tokenizer,
        tuple(rows),
        bundle.labels,
        EvaluationSpec(
            target_name=target_name,
            model_name=cfg.model_name,
            checkpoint_path=sampler_path,
            effort=effort,
            max_tokens=max_tokens,
            sample_concurrency=cfg.sample_concurrency,
            retry_attempts=cfg.retry_attempts,
            config_hash=cfg.config_hash,
            partition_hash=partition_hash,
            compact_system_prompt=cfg.compact_system_prompt,
            prompt_variant=prompt_variant,
        ),
        output_dir / "predictions.jsonl",
        price=price.model,
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
