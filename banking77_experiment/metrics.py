from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

INVALID_LABEL = "__invalid__"


def strict_label(raw_text: str, labels: Sequence[str]) -> str | None:
    text = raw_text.strip()
    return text if text in set(labels) else None


def tolerant_label(raw_text: str, labels: Sequence[str]) -> str | None:
    strict = strict_label(raw_text, labels)
    if strict is not None:
        return strict
    text = re.sub(r"(?i)^assistant\s*:\s*", "", raw_text.strip())
    text = text.strip().strip("`'\".,;:()[]{}")
    by_lower = {label.lower(): label for label in labels}
    compact = re.sub(r"[\s-]+", "_", text.lower()).strip("_")
    if compact in by_lower:
        return by_lower[compact]
    matches = [
        label
        for label in labels
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(label)}(?![A-Za-z0-9_])", raw_text)
    ]
    return matches[0] if len(matches) == 1 else None


def _safe_prediction(value: str | None) -> str:
    return value if value is not None else INVALID_LABEL


def classification_metrics(
    labels: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str | None]
) -> dict[str, Any]:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have equal length")
    if not y_true:
        raise ValueError("Cannot evaluate an empty prediction set")
    predictions = [_safe_prediction(value) for value in y_pred]
    true_counts = Counter(y_true)
    prediction_counts = Counter(predictions)
    true_positives = Counter(
        truth for truth, prediction in zip(y_true, predictions, strict=True) if truth == prediction
    )
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = true_positives[label]
        fp = prediction_counts[label] - tp
        fn = true_counts[label] - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": true_counts[label],
        }
    correct = sum(t == p for t, p in zip(y_true, predictions, strict=True))
    invalid = sum(value is None for value in y_pred)
    confusions = Counter(
        (truth, prediction)
        for truth, prediction in zip(y_true, predictions, strict=True)
        if truth != prediction
    )
    return {
        "macro_f1": sum(f1_values) / len(f1_values),
        "accuracy": correct / len(y_true),
        "invalid_rate": invalid / len(y_true),
        "num_examples": len(y_true),
        "per_class": per_class,
        "confusions": [
            {"true": truth, "pred": prediction, "count": count}
            for (truth, prediction), count in confusions.most_common()
        ],
    }


def _macro_f1(labels: Sequence[str], y_true: Sequence[str], y_pred: Sequence[str | None]) -> float:
    predictions = [_safe_prediction(value) for value in y_pred]
    true_counts = Counter(y_true)
    prediction_counts = Counter(predictions)
    true_positives = Counter(
        truth for truth, prediction in zip(y_true, predictions, strict=True) if truth == prediction
    )
    total = 0.0
    for label in labels:
        tp = true_positives[label]
        denominator = 2 * tp + (prediction_counts[label] - tp) + (true_counts[label] - tp)
        total += 2 * tp / denominator if denominator else 0.0
    return total / len(labels)


def percentile(values: Sequence[float], quantile: float) -> float:
    if not values:
        raise ValueError("Cannot take percentile of empty values")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_seed_bootstrap(
    labels: Sequence[str],
    truth_by_id: dict[str, str],
    base_by_id: dict[str, str | None],
    adapters_by_seed: dict[int, dict[str, str | None]],
    replicates: int,
    seed: int,
) -> dict[str, float]:
    if not adapters_by_seed:
        raise ValueError("At least one adapter seed is required")
    ids = set(truth_by_id)
    if set(base_by_id) != ids:
        raise ValueError("Base prediction ids do not match truth ids")
    for adapter_seed, predictions in adapters_by_seed.items():
        if set(predictions) != ids:
            raise ValueError(f"Adapter seed {adapter_seed} ids do not match truth ids")

    ids_by_label: dict[str, list[str]] = defaultdict(list)
    for example_id, truth in truth_by_id.items():
        ids_by_label[truth].append(example_id)
    for label in labels:
        if not ids_by_label[label]:
            raise ValueError(f"No examples for label {label!r}")

    rng = random.Random(seed)
    seed_values = sorted(adapters_by_seed)
    deltas: list[float] = []
    for _ in range(replicates):
        sampled_ids: list[str] = []
        for label in labels:
            candidates = ids_by_label[label]
            sampled_ids.extend(rng.choice(candidates) for _ in candidates)
        sampled_seeds = [rng.choice(seed_values) for _ in seed_values]
        truths = [truth_by_id[example_id] for example_id in sampled_ids]
        base_predictions = [base_by_id[example_id] for example_id in sampled_ids]
        base_f1 = _macro_f1(labels, truths, base_predictions)
        adapter_f1s = []
        for adapter_seed in sampled_seeds:
            adapter_predictions = [
                adapters_by_seed[adapter_seed][example_id] for example_id in sampled_ids
            ]
            adapter_f1s.append(_macro_f1(labels, truths, adapter_predictions))
        deltas.append(sum(adapter_f1s) / len(adapter_f1s) - float(base_f1))
    return {
        "lower_95": percentile(deltas, 0.025),
        "median": percentile(deltas, 0.5),
        "upper_95": percentile(deltas, 0.975),
        "replicates": float(replicates),
    }


def metrics_from_records(
    labels: Sequence[str], records: Iterable[dict[str, Any]], prediction_key: str = "strict_pred"
) -> dict[str, Any]:
    rows = list(records)
    return classification_metrics(
        labels,
        [str(row["true_label"]) for row in rows],
        [row.get(prediction_key) for row in rows],
    )
