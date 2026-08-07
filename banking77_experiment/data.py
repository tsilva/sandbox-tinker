from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import datasets

from .config import ExperimentConfig

REQUIRED_COLUMNS = {"text", "label", "label_text", "system_prompt"}


@dataclass(frozen=True)
class Example:
    example_id: str
    source_split: str
    source_index: int
    text: str
    label_id: int
    label: str
    system_prompt: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PartitionBundle:
    train_pool: tuple[Example, ...]
    dev: tuple[Example, ...]
    test: tuple[Example, ...]
    labels: tuple[str, ...]
    manifest: dict[str, Any]


def normalized_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"\s+", " ", normalized).strip()


def _content_digest(row: dict[str, Any]) -> str:
    payload = json.dumps(
        {
            "text": str(row["text"]),
            "label": int(row["label"]),
            "label_text": str(row["label_text"]),
            "system_prompt": str(row["system_prompt"]),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _example_id(revision: str, split: str, index: int, digest: str) -> str:
    payload = f"{revision}\0{split}\0{index}\0{digest}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_examples(ds: datasets.Dataset, split: str, revision: str) -> tuple[Example, ...]:
    missing = REQUIRED_COLUMNS - set(ds.column_names)
    if missing:
        raise ValueError(f"{split!r} is missing required columns: {sorted(missing)}")

    examples: list[Example] = []
    for index, row in enumerate(ds):
        digest = _content_digest(row)
        examples.append(
            Example(
                example_id=_example_id(revision, split, index, digest),
                source_split=split,
                source_index=index,
                text=str(row["text"]),
                label_id=int(row["label"]),
                label=str(row["label_text"]),
                system_prompt=str(row["system_prompt"]),
            )
        )
    return tuple(examples)


def _derive_labels(examples: Iterable[Example], expected_labels: int) -> tuple[str, ...]:
    label_by_id: dict[int, str] = {}
    for example in examples:
        previous = label_by_id.setdefault(example.label_id, example.label)
        if previous != example.label:
            raise ValueError(
                f"Label id {example.label_id} maps to both {previous!r} and {example.label!r}"
            )
    if len(label_by_id) != expected_labels:
        raise ValueError(f"Expected {expected_labels} labels, found {len(label_by_id)}")
    expected_ids = list(range(expected_labels))
    if sorted(label_by_id) != expected_ids:
        raise ValueError(f"Label ids must be contiguous 0..{expected_labels - 1}")
    return tuple(label_by_id[index] for index in expected_ids)


def _stable_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\0{label}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _partition_train(
    examples: tuple[Example, ...], labels: tuple[str, ...], dev_per_label: int, seed: int
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in examples:
        grouped[example.label].append(example)

    pool: list[Example] = []
    dev: list[Example] = []
    for label in labels:
        ordered = grouped[label][:]
        _stable_rng(seed, label).shuffle(ordered)
        if len(ordered) <= dev_per_label:
            raise ValueError(
                f"Label {label!r} has {len(ordered)} examples, not enough for "
                f"dev_per_label={dev_per_label}"
            )
        dev.extend(ordered[:dev_per_label])
        pool.extend(ordered[dev_per_label:])
    return tuple(pool), tuple(dev)


def _partition_hash(examples: Iterable[Example]) -> str:
    payload = "\n".join(example.example_id for example in examples)
    return hashlib.sha256(payload.encode()).hexdigest()


def _duplicate_summary(examples: Iterable[Example]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for example in examples:
        grouped[(normalized_text(example.text), example.label)].append(example.example_id)
    return [
        {"normalized_text": key[0], "label": key[1], "example_ids": ids}
        for key, ids in grouped.items()
        if len(ids) > 1
    ]


def _assert_disjoint(partitions: dict[str, tuple[Example, ...]]) -> None:
    id_sets = {name: {example.example_id for example in rows} for name, rows in partitions.items()}
    content_sets = {
        name: {(normalized_text(example.text), example.label) for example in rows}
        for name, rows in partitions.items()
    }
    names = list(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            id_overlap = id_sets[left] & id_sets[right]
            if id_overlap:
                raise ValueError(f"ID overlap between {left} and {right}: {len(id_overlap)} rows")
            content_overlap = content_sets[left] & content_sets[right]
            if content_overlap:
                preview = sorted(content_overlap)[:3]
                raise ValueError(
                    f"Exact normalized (text, label) overlap between {left} and {right}: "
                    f"{len(content_overlap)} rows; examples={preview}"
                )


def _quarantine_test_leaks(
    source_train: tuple[Example, ...], test: tuple[Example, ...]
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    test_keys = {(normalized_text(example.text), example.label) for example in test}
    clean: list[Example] = []
    quarantined: list[Example] = []
    for example in source_train:
        key = (normalized_text(example.text), example.label)
        (quarantined if key in test_keys else clean).append(example)
    return tuple(clean), tuple(quarantined)


def build_partitions(cfg: ExperimentConfig) -> PartitionBundle:
    train_ds = datasets.load_dataset(
        cfg.dataset_name,
        split=cfg.train_split,
        revision=cfg.dataset_revision,
    )
    test_ds = datasets.load_dataset(
        cfg.dataset_name,
        split=cfg.test_split,
        revision=cfg.dataset_revision,
    )
    if not isinstance(train_ds, datasets.Dataset) or not isinstance(test_ds, datasets.Dataset):
        raise TypeError("Expected non-streaming Hugging Face Dataset splits")
    if len(train_ds) != cfg.expected_train_rows:
        raise ValueError(f"Expected {cfg.expected_train_rows} train rows, found {len(train_ds)}")
    if len(test_ds) != cfg.expected_test_rows:
        raise ValueError(f"Expected {cfg.expected_test_rows} test rows, found {len(test_ds)}")

    source_train = _to_examples(train_ds, cfg.train_split, cfg.dataset_revision)
    test = _to_examples(test_ds, cfg.test_split, cfg.dataset_revision)
    labels = _derive_labels(source_train, cfg.expected_labels)
    test_labels = _derive_labels(test, cfg.expected_labels)
    if labels != test_labels:
        raise ValueError("Train and test label mappings differ")

    clean_source_train, quarantined_test_leaks = _quarantine_test_leaks(source_train, test)
    if len(quarantined_test_leaks) != cfg.expected_quarantined_test_leaks:
        raise ValueError(
            "Expected "
            f"{cfg.expected_quarantined_test_leaks} quarantined train/test duplicates, "
            f"found {len(quarantined_test_leaks)}"
        )
    train_pool, dev = _partition_train(
        clean_source_train,
        labels,
        cfg.dev_per_label,
        cfg.split_seed,
    )
    partitions = {"train_pool": train_pool, "dev": dev, "test": test}
    _assert_disjoint(partitions)

    manifest = {
        "dataset_name": cfg.dataset_name,
        "dataset_revision": cfg.dataset_revision,
        "split_seed": cfg.split_seed,
        "dev_per_label": cfg.dev_per_label,
        "labels": list(labels),
        "quarantined_test_leaks": [example.to_dict() for example in quarantined_test_leaks],
        "quarantined_test_leak_count": len(quarantined_test_leaks),
        "counts": {name: len(rows) for name, rows in partitions.items()},
        "class_counts": {
            name: dict(sorted(Counter(row.label for row in rows).items()))
            for name, rows in partitions.items()
        },
        "hashes": {name: _partition_hash(rows) for name, rows in partitions.items()},
        "within_partition_duplicates": {
            name: _duplicate_summary(rows) for name, rows in partitions.items()
        },
    }
    manifest_payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_hash"] = hashlib.sha256(manifest_payload.encode()).hexdigest()
    return PartitionBundle(
        train_pool=train_pool,
        dev=dev,
        test=test,
        labels=labels,
        manifest=manifest,
    )


def select_per_label(
    pool: Iterable[Example], labels: tuple[str, ...], examples_per_label: int
) -> tuple[Example, ...]:
    if examples_per_label <= 0:
        raise ValueError("examples_per_label must be positive")
    grouped: dict[str, list[Example]] = defaultdict(list)
    for example in pool:
        grouped[example.label].append(example)
    selected: list[Example] = []
    for label in labels:
        if len(grouped[label]) < examples_per_label:
            raise ValueError(
                f"Label {label!r} has only {len(grouped[label])} rows; "
                f"requested {examples_per_label}"
            )
        selected.extend(grouped[label][:examples_per_label])
    return tuple(selected)


def select_training_rows(
    bundle: PartitionBundle, data_size: str, cfg: ExperimentConfig
) -> tuple[Example, ...]:
    if data_size == "pilot":
        return select_per_label(bundle.train_pool, bundle.labels, cfg.pilot_per_label)
    if data_size == "scale":
        return select_per_label(bundle.train_pool, bundle.labels, cfg.scale_per_label)
    if data_size == "full":
        return bundle.train_pool
    raise ValueError(f"Unsupported data size: {data_size}")


def balanced_subset(
    rows: Iterable[Example], labels: tuple[str, ...], examples_per_label: int
) -> tuple[Example, ...]:
    return select_per_label(rows, labels, examples_per_label)
