from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import datasets

from .config import ExperimentConfig

TEXT_COLUMN = "text"


@dataclass(frozen=True)
class Example:
    example_id: str
    source_split: str
    source_index: int
    text: str
    label_id: int
    label: str

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


def _content_digest(text: str, label: str) -> str:
    payload = json.dumps(
        {"text": text, "label": label},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _example_id(revision: str, split: str, index: int, digest: str) -> str:
    payload = f"{revision}\0{split}\0{index}\0{digest}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _to_examples(
    ds: datasets.Dataset,
    split: str,
    revision: str,
    label_column: str,
    label_to_id: dict[str, int],
) -> tuple[Example, ...]:
    missing = {TEXT_COLUMN, label_column} - set(ds.column_names)
    if missing:
        raise ValueError(f"{split!r} is missing required columns: {sorted(missing)}")

    examples: list[Example] = []
    for index, row in enumerate(ds):
        text = str(row[TEXT_COLUMN])
        label = str(row[label_column])
        if label not in label_to_id:
            raise ValueError(f"{split}[{index}] has unknown label {label!r}")
        digest = _content_digest(text, label)
        examples.append(
            Example(
                example_id=_example_id(revision, split, index, digest),
                source_split=split,
                source_index=index,
                text=text,
                label_id=label_to_id[label],
                label=label,
            )
        )
    return tuple(examples)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_labels(cfg: ExperimentConfig) -> tuple[str, ...]:
    observed_sha256 = _sha256_file(cfg.labels_path)
    if observed_sha256 != cfg.labels_sha256:
        raise ValueError(
            f"Label asset checksum mismatch: expected {cfg.labels_sha256}, found {observed_sha256}"
        )
    raw = json.loads(cfg.labels_path.read_text())
    if not isinstance(raw, list) or not all(isinstance(value, str) for value in raw):
        raise ValueError("The label asset must be a JSON list of strings")
    labels = tuple(raw)
    if len(labels) != cfg.expected_labels or len(set(labels)) != len(labels):
        raise ValueError(
            f"Expected {cfg.expected_labels} unique labels, found {len(labels)} rows "
            f"and {len(set(labels))} unique values"
        )
    return labels


def _download_pinned_csvs(cfg: ExperimentConfig) -> dict[str, Path]:
    manager = datasets.DownloadManager(dataset_name=cfg.dataset_name, record_checksums=True)
    downloaded = manager.download(
        {cfg.train_split: cfg.train_data_url, cfg.test_split: cfg.test_data_url}
    )
    paths = {split: Path(value) for split, value in downloaded.items()}
    expected = {
        cfg.train_split: cfg.train_data_sha256,
        cfg.test_split: cfg.test_data_sha256,
    }
    for split, expected_sha256 in expected.items():
        observed_sha256 = _sha256_file(paths[split])
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"{split} checksum mismatch: expected {expected_sha256}, found {observed_sha256}"
            )
    return paths


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


def _deduplicate_training_rows(
    examples: tuple[Example, ...],
) -> tuple[tuple[Example, ...], tuple[Example, ...]]:
    seen: set[tuple[str, str]] = set()
    unique: list[Example] = []
    duplicates: list[Example] = []
    for example in examples:
        key = (normalized_text(example.text), example.label)
        if key in seen:
            duplicates.append(example)
        else:
            seen.add(key)
            unique.append(example)
    return tuple(unique), tuple(duplicates)


def build_partitions(cfg: ExperimentConfig) -> PartitionBundle:
    labels = _load_labels(cfg)
    label_to_id = {label: index for index, label in enumerate(labels)}
    data_paths = _download_pinned_csvs(cfg)
    loaded = datasets.load_dataset(
        "csv",
        data_files={split: str(path) for split, path in data_paths.items()},
    )
    train_ds = loaded[cfg.train_split]
    test_ds = loaded[cfg.test_split]
    if not isinstance(train_ds, datasets.Dataset) or not isinstance(test_ds, datasets.Dataset):
        raise TypeError("Expected non-streaming Hugging Face Dataset splits")
    if len(train_ds) != cfg.expected_train_rows:
        raise ValueError(f"Expected {cfg.expected_train_rows} train rows, found {len(train_ds)}")
    if len(test_ds) != cfg.expected_test_rows:
        raise ValueError(f"Expected {cfg.expected_test_rows} test rows, found {len(test_ds)}")

    source_train = _to_examples(
        train_ds, cfg.train_split, cfg.dataset_revision, cfg.label_column, label_to_id
    )
    test = _to_examples(
        test_ds, cfg.test_split, cfg.dataset_revision, cfg.label_column, label_to_id
    )
    for split, examples in ((cfg.train_split, source_train), (cfg.test_split, test)):
        observed_labels = {example.label for example in examples}
        if observed_labels != set(labels):
            missing = sorted(set(labels) - observed_labels)
            raise ValueError(f"{split} does not cover the frozen taxonomy; missing={missing}")

    clean_source_train, quarantined_test_leaks = _quarantine_test_leaks(source_train, test)
    if len(quarantined_test_leaks) != cfg.expected_quarantined_test_leaks:
        raise ValueError(
            "Expected "
            f"{cfg.expected_quarantined_test_leaks} quarantined train/test duplicates, "
            f"found {len(quarantined_test_leaks)}"
        )
    unique_source_train, deduplicated_train_rows = _deduplicate_training_rows(clean_source_train)
    if len(deduplicated_train_rows) != cfg.expected_deduplicated_train_rows:
        raise ValueError(
            f"Expected {cfg.expected_deduplicated_train_rows} duplicate training rows, "
            f"found {len(deduplicated_train_rows)}"
        )
    train_pool, dev = _partition_train(
        unique_source_train,
        labels,
        cfg.dev_per_label,
        cfg.split_seed,
    )
    partitions = {"train_pool": train_pool, "dev": dev, "test": test}
    _assert_disjoint(partitions)

    manifest = {
        "dataset_name": cfg.dataset_name,
        "dataset_revision": cfg.dataset_revision,
        "source_urls": {
            cfg.train_split: cfg.train_data_url,
            cfg.test_split: cfg.test_data_url,
        },
        "source_sha256": {
            cfg.train_split: cfg.train_data_sha256,
            cfg.test_split: cfg.test_data_sha256,
            "labels": cfg.labels_sha256,
        },
        "label_column": cfg.label_column,
        "split_seed": cfg.split_seed,
        "dev_per_label": cfg.dev_per_label,
        "labels": list(labels),
        "quarantined_test_leaks": [example.to_dict() for example in quarantined_test_leaks],
        "quarantined_test_leak_count": len(quarantined_test_leaks),
        "deduplicated_train_rows": [example.to_dict() for example in deduplicated_train_rows],
        "deduplicated_train_row_count": len(deduplicated_train_rows),
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
