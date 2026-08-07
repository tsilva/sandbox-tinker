from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected JSON object at {path}:{line_number}")
        records.append(record)
    return records


def successful_records_by_id(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for record in read_jsonl(path):
        example_id = str(record.get("example_id", ""))
        if not example_id:
            raise ValueError(f"Prediction record in {path} has no example_id")
        if record.get("status") != "ok":
            continue
        if example_id in records:
            raise ValueError(f"Duplicate completed prediction for {example_id} in {path}")
        records[example_id] = record
    return records


def require_complete_coverage(
    planned_ids: Iterable[str], records: dict[str, dict[str, Any]]
) -> None:
    planned = set(planned_ids)
    completed = set(records)
    missing = planned - completed
    unexpected = completed - planned
    if missing or unexpected:
        raise ValueError(
            f"Prediction coverage mismatch: missing={len(missing)} unexpected={len(unexpected)}"
        )


def create_or_validate_unseal_receipt(path: Path, receipt: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text())
        if existing != receipt:
            raise ValueError(
                f"Test is already unsealed with a different receipt at {path}; "
                "use a new experiment id"
            )
        return
    atomic_write_json(path, receipt)


def load_and_validate_test_plan(
    path: Path,
    *,
    experiment_id: str,
    config_hash: str,
    dataset_revision: str,
    test_partition_hash: str,
    target_name: str,
    sampler_path: str | None,
    effort: float,
    max_tokens: int,
) -> tuple[dict[str, Any], str]:
    plan = json.loads(path.read_text())
    expected = {
        "experiment_id": experiment_id,
        "config_hash": config_hash,
        "dataset_revision": dataset_revision,
        "test_partition_hash": test_partition_hash,
        "effort": effort,
        "max_tokens": max_tokens,
    }
    for key, value in expected.items():
        if plan.get(key) != value:
            raise ValueError(
                f"Test plan {key} mismatch: expected {value!r}, found {plan.get(key)!r}"
            )
    arms = plan.get("arms")
    if not isinstance(arms, list) or not arms:
        raise ValueError("Test plan must contain a non-empty arms list")
    matching = [
        arm
        for arm in arms
        if arm.get("target_name") == target_name and arm.get("sampler_path") == sampler_path
    ]
    if len(matching) != 1:
        raise ValueError(
            f"Target {target_name!r} with sampler {sampler_path!r} is not a unique frozen arm"
        )
    return plan, canonical_hash(plan)


def git_metadata(cwd: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    diff = run("diff", "--binary", "HEAD")
    status = run("status", "--short")
    return {
        "commit": run("rev-parse", "HEAD"),
        "status": status,
        "diff_hash": hashlib.sha256(diff.encode()).hexdigest(),
        "dirty": bool(status),
    }


def choose_checkpoint(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(records)
    if not candidates:
        raise ValueError("No checkpoint candidates")
    required = {"strict_macro_f1", "strict_invalid_rate", "train_tokens", "step"}
    for candidate in candidates:
        missing = required - set(candidate)
        if missing:
            raise ValueError(f"Checkpoint candidate is missing fields: {sorted(missing)}")
    return min(
        candidates,
        key=lambda row: (
            -float(row["strict_macro_f1"]),
            float(row["strict_invalid_rate"]),
            int(row["train_tokens"]),
            int(row["step"]),
        ),
    )
