"""Train a preregistered Inkling-Small LoRA on Banking77."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import tinker
from dotenv import load_dotenv
from tinker_cookbook.tokenizer_utils import get_tokenizer

from banking77_experiment.artifacts import (
    atomic_write_json,
    choose_checkpoint,
    git_metadata,
)
from banking77_experiment.config import ExperimentConfig, load_config
from banking77_experiment.data import build_partitions, select_training_rows
from banking77_experiment.evaluation import EvaluationSpec, evaluate_rows
from banking77_experiment.pricing import (
    TokenEstimate,
    estimate_cost,
    load_snapshot,
    refresh_snapshot,
    require_budget,
)
from banking77_experiment.rendering import generation_prompt, make_renderer
from banking77_experiment.training import (
    optimizer_step,
    plan_batches,
    render_training_rows,
    save_immutable_checkpoint,
    scheduled_lr,
)

DEFAULT_CONFIG = "configs/inkling_small_banking77.toml"
CHECKPOINT_TTL_SECONDS = 7 * 24 * 60 * 60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-size", choices=("pilot", "scale", "full"), default="pilot")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-examples-per-label", type=int, default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--budget-usd", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _run_directory(cfg: ExperimentConfig, args: argparse.Namespace, learning_rate: float) -> Path:
    if args.run_dir:
        return Path(args.run_dir).expanduser().resolve()
    lr_slug = f"{learning_rate:.0e}".replace("+", "")
    return cfg.run_root / cfg.experiment_id / f"{args.data_size}-lr-{lr_slug}-seed-{args.seed}"


async def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if args.seed not in cfg.train_seeds and not args.dry_run:
        raise ValueError(f"Seed {args.seed} is not preregistered: {cfg.train_seeds}")
    learning_rate = args.learning_rate or cfg.learning_rates[0]
    if learning_rate not in cfg.learning_rates:
        raise ValueError(
            f"Learning rate {learning_rate} is not preregistered: {cfg.learning_rates}"
        )
    if args.max_steps is not None and args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    run_dir = _run_directory(cfg, args, learning_rate)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        paid_artifacts = [
            run_dir / "training_metrics.jsonl",
            run_dir / "checkpoint_candidates.json",
            run_dir / "selection.json",
            *run_dir.glob("predictions_*.jsonl"),
        ]
        existing_paid_artifacts = [str(path) for path in paid_artifacts if path.exists()]
        if existing_paid_artifacts:
            raise ValueError(
                "Training runs are immutable and cannot resume into a new Tinker client; "
                f"choose a new --run-dir. Existing paid artifacts: {existing_paid_artifacts}"
            )
    bundle = build_partitions(cfg)
    train_rows = select_training_rows(bundle, args.data_size, cfg)
    dev_rows = bundle.dev
    if args.eval_examples_per_label is not None:
        from banking77_experiment.data import balanced_subset

        dev_rows = balanced_subset(dev_rows, bundle.labels, args.eval_examples_per_label)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer = make_renderer(cfg.model_name, tokenizer, cfg.renderer_name)
    rendered = render_training_rows(
        train_rows,
        renderer,
        tokenizer,
        bundle.labels,
        cfg.training_prompt_variant,
        cfg.compact_system_prompt,
        cfg.effort,
        cfg.max_length,
    )
    planned = plan_batches(
        rendered,
        cfg.batch_size,
        cfg.epochs,
        args.seed,
        max_steps=args.max_steps,
    )
    if not planned:
        raise ValueError("Training plan contains no batches")

    eval_events = sum(
        1
        for index, batch in enumerate(planned)
        if index == len(planned) - 1 or planned[index + 1].epoch != batch.epoch
    )
    dev_prompt_tokens = sum(
        len(
            generation_prompt(
                renderer,
                row,
                bundle.labels,
                cfg.adapter_prompt_variant,
                cfg.compact_system_prompt,
                cfg.effort,
            ).to_ints()
        )
        for row in dev_rows
    )
    tokens = TokenEstimate(
        train_tokens=sum(batch.tokens for batch in planned),
        prefill_tokens=dev_prompt_tokens * eval_events,
        sample_tokens=len(dev_rows) * cfg.max_eval_tokens * eval_events,
    )
    baseline_price = load_snapshot(cfg.price_snapshot)
    price = baseline_price
    if not args.dry_run:
        price = refresh_snapshot(
            cfg.pricing_url,
            cfg.model_name,
            run_dir / "pricing.json",
        )
    cost = estimate_cost(tokens, price.model)
    if not args.dry_run:
        require_budget(cost, args.budget_usd)

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": cfg.experiment_id,
        "config": cfg.canonical_dict(),
        "config_hash": cfg.config_hash,
        "git": git_metadata(Path(__file__).resolve().parent),
        "partition_manifest": bundle.manifest,
        "data_size": args.data_size,
        "training_prompt_variant": cfg.training_prompt_variant,
        "evaluation_prompt_variant": cfg.adapter_prompt_variant,
        "learning_rate": learning_rate,
        "seed": args.seed,
        "max_steps": args.max_steps,
        "train_examples": len(train_rows),
        "dev_examples": len(dev_rows),
        "planned_steps": len(planned),
        "planned_eval_events": eval_events,
        "token_estimate": asdict(tokens),
        "cost_estimate": cost,
        "pricing": price.to_dict(),
        "baseline_pricing": baseline_price.to_dict(),
        "pricing_changed_since_snapshot": price.model != baseline_price.model,
    }
    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "partition.json", bundle.manifest)
    print(json.dumps({"run_dir": str(run_dir), **manifest["cost_estimate"]}, indent=2))
    if args.dry_run:
        deterministic = {
            key: manifest[key]
            for key in (
                "config_hash",
                "partition_manifest",
                "data_size",
                "learning_rate",
                "seed",
                "max_steps",
                "train_examples",
                "dev_examples",
                "planned_steps",
                "planned_eval_events",
                "token_estimate",
            )
        }
        atomic_write_json(run_dir / "dry_run_deterministic.json", deterministic)
        return

    if "TINKER_API_KEY" not in os.environ:
        raise RuntimeError("Launch paid training through `keyenv run -- ...`")
    service = tinker.ServiceClient(base_url=args.base_url)
    training_client = await service.create_lora_training_client_async(
        base_model=cfg.model_name,
        rank=cfg.lora_rank,
        seed=args.seed,
        user_metadata={
            "experiment_id": cfg.experiment_id,
            "config_hash": cfg.config_hash,
            "data_size": args.data_size,
        },
    )
    checkpoint_candidates: list[dict[str, object]] = []
    cumulative_tokens = 0
    for step, batch in enumerate(planned, start=1):
        lr = scheduled_lr(
            learning_rate,
            cfg.lr_schedule,
            cfg.warmup_ratio,
            step - 1,
            len(planned),
        )
        record = await optimizer_step(
            training_client,
            batch,
            lr,
            run_dir / "training_metrics.jsonl",
            step,
        )
        cumulative_tokens += batch.tokens
        print(
            f"step={step}/{len(planned)} epoch={batch.epoch} "
            f"lr={lr:.2e} nll={record['train_mean_nll']:.4f}"
        )
        epoch_end = step == len(planned) or planned[step].epoch != batch.epoch
        if not epoch_end:
            continue
        checkpoint_name = (
            f"{cfg.experiment_id}-{args.data_size}-seed-{args.seed}-step-{step:06d}-"
            f"{cfg.config_hash[:8]}"
        )
        paths = await save_immutable_checkpoint(
            training_client,
            checkpoint_name,
            CHECKPOINT_TTL_SECONDS,
        )
        sampling_client = await service.create_sampling_client_async(
            model_path=paths["sampler_path"],
            base_model=cfg.model_name,
        )
        summary = await evaluate_rows(
            sampling_client,
            renderer,
            tokenizer,
            tuple(dev_rows),
            bundle.labels,
            EvaluationSpec(
                target_name=checkpoint_name,
                model_name=cfg.model_name,
                checkpoint_path=paths["sampler_path"],
                effort=cfg.effort,
                max_tokens=cfg.max_eval_tokens,
                sample_concurrency=cfg.sample_concurrency,
                retry_attempts=cfg.retry_attempts,
                config_hash=cfg.config_hash,
                partition_hash=bundle.manifest["hashes"]["dev"],
                compact_system_prompt=cfg.compact_system_prompt,
                prompt_variant=cfg.adapter_prompt_variant,
            ),
            run_dir / f"predictions_{step:06d}.jsonl",
            price=price.model,
        )
        metrics = summary["metrics"]
        checkpoint_candidates.append(
            {
                **paths,
                "step": step,
                "epoch": batch.epoch,
                "strict_macro_f1": metrics["macro_f1"],
                "strict_invalid_rate": metrics["invalid_rate"],
                "train_tokens": cumulative_tokens,
            }
        )
        atomic_write_json(run_dir / "checkpoint_candidates.json", checkpoint_candidates)

    selected = choose_checkpoint(checkpoint_candidates)
    atomic_write_json(run_dir / "selection.json", selected)
    print(json.dumps({"selected": selected}, indent=2))


def main() -> None:
    load_dotenv()
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
