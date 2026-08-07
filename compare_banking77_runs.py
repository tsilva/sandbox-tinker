"""Compare a base run with three adapter seeds and freeze the final test plan."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

from banking77_experiment.artifacts import (
    atomic_write_json,
    canonical_hash,
    git_metadata,
    successful_records_by_id,
)
from banking77_experiment.config import load_config
from banking77_experiment.data import build_partitions
from banking77_experiment.metrics import metrics_from_records, paired_seed_bootstrap

DEFAULT_CONFIG = "configs/inkling_small_banking77.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--partition", choices=("dev", "test"), default="dev")
    parser.add_argument("--test-plan", default=None)
    parser.add_argument("--base-predictions", required=True)
    parser.add_argument(
        "--adapter-predictions",
        action="append",
        required=True,
        metavar="SEED=PATH",
    )
    parser.add_argument(
        "--adapter-selection",
        action="append",
        default=[],
        metavar="SEED=PATH",
    )
    parser.add_argument("--latency-summary", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--freeze-test-plan", default=None)
    parser.add_argument("--replicates", type=int, default=None)
    return parser.parse_args()


def _seed_paths(values: list[str]) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for value in values:
        seed_text, separator, path_text = value.partition("=")
        if not separator or not seed_text or not path_text:
            raise ValueError(f"Expected SEED=PATH, got {value!r}")
        seed = int(seed_text)
        if seed in result:
            raise ValueError(f"Duplicate seed {seed}")
        result[seed] = Path(path_text).expanduser().resolve()
    return result


def _load_records(
    path: Path,
    *,
    config_hash: str,
    partition_hash: str,
    effort: float,
    max_tokens: int,
) -> dict[str, dict[str, Any]]:
    records = successful_records_by_id(path)
    if not records:
        raise ValueError(f"No completed predictions in {path}")
    for record in records.values():
        if record.get("config_hash") != config_hash:
            raise ValueError(f"Config hash mismatch in {path}")
        if record.get("partition_hash") != partition_hash:
            raise ValueError(f"Partition hash mismatch in {path}")
        if record.get("effort") != effort or record.get("max_tokens") != max_tokens:
            raise ValueError(f"Primary decoding contract mismatch in {path}")
    return records


def _prediction_map(records: dict[str, dict[str, Any]]) -> dict[str, str | None]:
    return {example_id: record.get("strict_pred") for example_id, record in records.items()}


def _mean_generated(records: dict[str, dict[str, Any]]) -> float:
    return mean(float(record["generated_tokens"]) for record in records.values())


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    bundle = build_partitions(cfg)
    partition_rows = bundle.dev if args.partition == "dev" else bundle.test
    partition_hash = bundle.manifest["hashes"][args.partition]
    expected_ids = {row.example_id for row in partition_rows}
    adapter_paths = _seed_paths(args.adapter_predictions)
    if set(adapter_paths) != set(cfg.train_seeds):
        raise ValueError(
            f"Adapter seeds must exactly match preregistration {cfg.train_seeds}; "
            f"found {tuple(sorted(adapter_paths))}"
        )

    base_path = Path(args.base_predictions).expanduser().resolve()
    base = _load_records(
        base_path,
        config_hash=cfg.config_hash,
        partition_hash=partition_hash,
        effort=cfg.effort,
        max_tokens=cfg.max_eval_tokens,
    )
    adapters = {
        seed: _load_records(
            path,
            config_hash=cfg.config_hash,
            partition_hash=partition_hash,
            effort=cfg.effort,
            max_tokens=cfg.max_eval_tokens,
        )
        for seed, path in adapter_paths.items()
    }
    for name, records in {"base": base, **{str(k): v for k, v in adapters.items()}}.items():
        if set(records) != expected_ids:
            raise ValueError(f"{name} predictions do not cover the frozen dev partition")

    truth = {row.example_id: row.label for row in partition_rows}

    test_plan: dict[str, Any] | None = None
    if args.partition == "test":
        if not args.test_plan:
            raise ValueError("Test reporting requires the frozen --test-plan")
        test_plan = json.loads(Path(args.test_plan).expanduser().read_text())
        expected_plan = {
            "experiment_id": cfg.experiment_id,
            "config_hash": cfg.config_hash,
            "dataset_revision": cfg.dataset_revision,
            "test_partition_hash": partition_hash,
            "effort": cfg.effort,
            "max_tokens": cfg.max_eval_tokens,
        }
        for key, expected in expected_plan.items():
            if test_plan.get(key) != expected:
                raise ValueError(f"Frozen test plan {key} mismatch")
        plan_arms = {str(arm["target_name"]): arm.get("sampler_path") for arm in test_plan["arms"]}
        expected_arm_names = {"base"} | {f"adapter-seed-{seed}" for seed in cfg.train_seeds}
        if len(plan_arms) != len(test_plan["arms"]) or set(plan_arms) != expected_arm_names:
            raise ValueError("Frozen test plan arms do not exactly match the preregistered arms")
        if {record.get("target_name") for record in base.values()} != {"base"}:
            raise ValueError("Base test predictions have the wrong frozen target name")
        if {record.get("checkpoint_path") for record in base.values()} != {plan_arms.get("base")}:
            raise ValueError("Base test predictions do not match the frozen test arm")
        for seed, records in adapters.items():
            target_name = f"adapter-seed-{seed}"
            if {record.get("target_name") for record in records.values()} != {target_name}:
                raise ValueError(f"Seed {seed} test predictions have the wrong target name")
            if {record.get("checkpoint_path") for record in records.values()} != {
                plan_arms.get(target_name)
            }:
                raise ValueError(f"Seed {seed} test predictions do not match the frozen arm")
    base_metrics = metrics_from_records(bundle.labels, base.values())
    adapter_metrics = {
        seed: metrics_from_records(bundle.labels, records.values())
        for seed, records in adapters.items()
    }
    bootstrap = paired_seed_bootstrap(
        bundle.labels,
        truth,
        _prediction_map(base),
        {seed: _prediction_map(records) for seed, records in adapters.items()},
        args.replicates or cfg.bootstrap_replicates,
        cfg.bootstrap_seed,
    )
    base_generated = _mean_generated(base)
    adapter_generated = mean(_mean_generated(records) for records in adapters.values())
    generated_ratio = adapter_generated / base_generated if base_generated else float("inf")
    adapter_invalid = mean(float(metrics["invalid_rate"]) for metrics in adapter_metrics.values())

    latency: dict[str, Any] | None = None
    latency_pass: bool | None = None
    if args.latency_summary:
        latency = json.loads(Path(args.latency_summary).expanduser().read_text())
        if latency.get("config_hash") != cfg.config_hash:
            raise ValueError("Latency summary config hash mismatch")
        if args.partition != "dev":
            raise ValueError("Latency gates apply only to dev promotion")
        if latency.get("dev_partition_hash") != partition_hash:
            raise ValueError("Latency summary dev partition hash mismatch")
        latency_ratio = float(latency["p95_latency_ratio"])
        latency_pass = latency_ratio <= cfg.max_p95_latency_ratio

    gates: dict[str, bool | None] | None = None
    core_pass: bool | None = None
    promotion_pass: bool | None = None
    if args.partition == "dev":
        gates = {
            "bootstrap_lower_95_above_minimum": bootstrap["lower_95"]
            > cfg.min_bootstrap_macro_f1_delta,
            "invalid_rate_within_limit": adapter_invalid
            <= float(base_metrics["invalid_rate"]) + cfg.max_invalid_rate_increase,
            "generated_tokens_within_ratio": generated_ratio <= cfg.max_generated_tokens_ratio,
            "latency_within_ratio": latency_pass,
        }
        core_pass = all(value for key, value in gates.items() if key != "latency_within_ratio")
        promotion_pass = core_pass and latency_pass is True
    output = {
        "created_at": datetime.now(UTC).isoformat(),
        "experiment_id": cfg.experiment_id,
        "partition": args.partition,
        "config_hash": cfg.config_hash,
        "partition_hash": partition_hash,
        "test_plan": str(Path(args.test_plan).expanduser().resolve()) if test_plan else None,
        "test_plan_hash": canonical_hash(test_plan) if test_plan else None,
        "base_predictions": str(base_path),
        "adapter_predictions": {str(seed): str(path) for seed, path in adapter_paths.items()},
        "base_metrics": base_metrics,
        "adapter_metrics": {str(seed): value for seed, value in adapter_metrics.items()},
        "mean_adapter_macro_f1": mean(
            float(metrics["macro_f1"]) for metrics in adapter_metrics.values()
        ),
        "mean_adapter_invalid_rate": adapter_invalid,
        "bootstrap_macro_f1_delta": bootstrap,
        "mean_generated_tokens_ratio": generated_ratio,
        "latency": latency,
        "gates": gates,
        "core_gates_pass": core_pass,
        "promotion_pass": promotion_pass,
        "git": git_metadata(Path(__file__).resolve().parent),
    }
    output_path = Path(args.output).expanduser().resolve()
    atomic_write_json(output_path, output)

    if args.freeze_test_plan:
        if args.partition != "dev":
            raise ValueError("A frozen test plan can only be created from dev results")
        if not promotion_pass:
            raise ValueError("Cannot freeze the test plan until every promotion gate passes")
        if {record.get("checkpoint_path") for record in base.values()} != {None}:
            raise ValueError("Base dev predictions unexpectedly reference a checkpoint")
        selection_paths = _seed_paths(args.adapter_selection)
        if set(selection_paths) != set(cfg.train_seeds):
            raise ValueError("Freezing requires one --adapter-selection for every seed")
        arms: list[dict[str, Any]] = [{"target_name": "base", "sampler_path": None}]
        for seed in cfg.train_seeds:
            selection = json.loads(selection_paths[seed].read_text())
            sampler_path = str(selection["sampler_path"])
            observed_paths = {record.get("checkpoint_path") for record in adapters[seed].values()}
            if observed_paths != {sampler_path}:
                raise ValueError(
                    f"Seed {seed} dev predictions do not match its selected sampler checkpoint"
                )
            arms.append(
                {
                    "target_name": f"adapter-seed-{seed}",
                    "sampler_path": sampler_path,
                    "seed": seed,
                    "selection_path": str(selection_paths[seed]),
                    "selection_hash": canonical_hash(selection),
                }
            )
        expected_latency_arms = {arm["target_name"]: arm["sampler_path"] for arm in arms}
        if latency is None or latency.get("arms") != expected_latency_arms:
            raise ValueError("Latency benchmark arms do not match the frozen test arms")
        test_plan = {
            "created_at": datetime.now(UTC).isoformat(),
            "experiment_id": cfg.experiment_id,
            "config_hash": cfg.config_hash,
            "dataset_revision": cfg.dataset_revision,
            "test_partition_hash": bundle.manifest["hashes"]["test"],
            "effort": cfg.effort,
            "max_tokens": cfg.max_eval_tokens,
            "source_comparison": str(output_path),
            "source_comparison_hash": canonical_hash(output),
            "arms": arms,
        }
        atomic_write_json(Path(args.freeze_test_plan).expanduser().resolve(), test_plan)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
