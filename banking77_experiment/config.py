from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    experiment_id: str
    model_name: str
    renderer_name: str | None
    effort: float
    dataset_name: str
    dataset_revision: str
    train_split: str
    test_split: str
    expected_train_rows: int
    expected_test_rows: int
    expected_labels: int
    expected_quarantined_test_leaks: int
    dev_per_label: int
    split_seed: int
    pilot_per_label: int
    scale_per_label: int
    lora_rank: int
    batch_size: int
    epochs: int
    max_length: int
    learning_rates: tuple[float, ...]
    lr_schedule: str
    warmup_ratio: float
    train_seeds: tuple[int, ...]
    sample_concurrency: int
    max_eval_tokens: int
    retry_attempts: int
    bootstrap_replicates: int
    bootstrap_seed: int
    min_bootstrap_macro_f1_delta: float
    max_invalid_rate_increase: float
    max_generated_tokens_ratio: float
    max_p95_latency_ratio: float
    latency_examples_per_label: int
    latency_warmups_per_arm: int
    run_root: Path
    price_snapshot: Path
    pricing_url: str

    def canonical_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["run_root"] = str(self.run_root)
        result["price_snapshot"] = str(self.price_snapshot)
        return result

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.canonical_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    def validate(self) -> None:
        if self.model_name != "thinkingmachines/Inkling-Small":
            raise ValueError("The preregistered experiment requires thinkingmachines/Inkling-Small")
        if not 0.0 <= self.effort < 1.0:
            raise ValueError("effort must be in [0, 1)")
        if self.effort != 0.0:
            raise ValueError("The primary experiment is preregistered at effort=0.0")
        for name in (
            "expected_train_rows",
            "expected_test_rows",
            "expected_labels",
            "dev_per_label",
            "pilot_per_label",
            "scale_per_label",
            "lora_rank",
            "batch_size",
            "epochs",
            "max_length",
            "sample_concurrency",
            "max_eval_tokens",
            "retry_attempts",
            "bootstrap_replicates",
            "latency_examples_per_label",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.expected_quarantined_test_leaks < 0:
            raise ValueError("expected_quarantined_test_leaks must be non-negative")
        if self.latency_warmups_per_arm < 0:
            raise ValueError("latency_warmups_per_arm must be non-negative")
        if self.max_invalid_rate_increase < 0:
            raise ValueError("max_invalid_rate_increase must be non-negative")
        if self.max_generated_tokens_ratio <= 0 or self.max_p95_latency_ratio <= 0:
            raise ValueError("ratio gates must be positive")
        if self.pilot_per_label >= self.scale_per_label:
            raise ValueError("pilot_per_label must be smaller than scale_per_label")
        if self.lr_schedule not in {"constant", "linear", "cosine"}:
            raise ValueError(f"Unsupported lr_schedule: {self.lr_schedule}")
        if not self.learning_rates or any(value <= 0 for value in self.learning_rates):
            raise ValueError("learning_rates must contain positive values")
        if len(set(self.train_seeds)) != len(self.train_seeds):
            raise ValueError("train_seeds must be unique")


def _required(section: dict[str, Any], key: str) -> Any:
    if key not in section:
        raise ValueError(f"Missing configuration key: {key}")
    return section[key]


def load_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    experiment = raw["experiment"]
    model = raw["model"]
    data = raw["data"]
    training = raw["training"]
    evaluation = raw["evaluation"]
    paths = raw["paths"]

    cfg = ExperimentConfig(
        experiment_id=str(_required(experiment, "id")),
        model_name=str(_required(model, "name")),
        renderer_name=model.get("renderer"),
        effort=float(_required(model, "effort")),
        dataset_name=str(_required(data, "name")),
        dataset_revision=str(_required(data, "revision")),
        train_split=str(_required(data, "train_split")),
        test_split=str(_required(data, "test_split")),
        expected_train_rows=int(_required(data, "expected_train_rows")),
        expected_test_rows=int(_required(data, "expected_test_rows")),
        expected_labels=int(_required(data, "expected_labels")),
        expected_quarantined_test_leaks=int(_required(data, "expected_quarantined_test_leaks")),
        dev_per_label=int(_required(data, "dev_per_label")),
        split_seed=int(_required(data, "split_seed")),
        pilot_per_label=int(_required(data, "pilot_per_label")),
        scale_per_label=int(_required(data, "scale_per_label")),
        lora_rank=int(_required(training, "lora_rank")),
        batch_size=int(_required(training, "batch_size")),
        epochs=int(_required(training, "epochs")),
        max_length=int(_required(training, "max_length")),
        learning_rates=tuple(float(value) for value in _required(training, "learning_rates")),
        lr_schedule=str(_required(training, "lr_schedule")),
        warmup_ratio=float(_required(training, "warmup_ratio")),
        train_seeds=tuple(int(value) for value in _required(training, "seeds")),
        sample_concurrency=int(_required(evaluation, "sample_concurrency")),
        max_eval_tokens=int(_required(evaluation, "max_tokens")),
        retry_attempts=int(_required(evaluation, "retry_attempts")),
        bootstrap_replicates=int(_required(evaluation, "bootstrap_replicates")),
        bootstrap_seed=int(_required(evaluation, "bootstrap_seed")),
        min_bootstrap_macro_f1_delta=float(_required(evaluation, "min_bootstrap_macro_f1_delta")),
        max_invalid_rate_increase=float(_required(evaluation, "max_invalid_rate_increase")),
        max_generated_tokens_ratio=float(_required(evaluation, "max_generated_tokens_ratio")),
        max_p95_latency_ratio=float(_required(evaluation, "max_p95_latency_ratio")),
        latency_examples_per_label=int(_required(evaluation, "latency_examples_per_label")),
        latency_warmups_per_arm=int(_required(evaluation, "latency_warmups_per_arm")),
        run_root=(config_path.parent / str(_required(paths, "run_root"))).resolve(),
        price_snapshot=(config_path.parent / str(_required(paths, "price_snapshot"))).resolve(),
        pricing_url=str(_required(paths, "pricing_url")),
    )
    cfg.validate()
    return cfg
