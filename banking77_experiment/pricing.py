from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json


@dataclass(frozen=True)
class ModelPrice:
    tinker_id: str
    prefill_usd_per_million: float
    cached_prefill_usd_per_million: float
    sample_usd_per_million: float
    train_usd_per_million: float
    context: str
    note: str | None = None


@dataclass(frozen=True)
class PriceSnapshot:
    retrieved_at: str
    source_url: str
    model: ModelPrice

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieved_at": self.retrieved_at,
            "source_url": self.source_url,
            "model": asdict(self.model),
        }


@dataclass(frozen=True)
class TokenEstimate:
    train_tokens: int = 0
    prefill_tokens: int = 0
    sample_tokens: int = 0


def _money(value: str) -> float:
    return float(value.removeprefix("$"))


def _from_model_row(row: dict[str, Any]) -> ModelPrice:
    return ModelPrice(
        tinker_id=str(row["tinker_id"]),
        prefill_usd_per_million=_money(str(row["prefill"])),
        cached_prefill_usd_per_million=_money(str(row["cached_prefill"])),
        sample_usd_per_million=_money(str(row["sample"])),
        train_usd_per_million=_money(str(row["train"])),
        context=str(row["context"]),
        note=str(row["note"]) if row.get("note") else None,
    )


def load_snapshot(path: str | Path) -> PriceSnapshot:
    raw = json.loads(Path(path).read_text())
    return PriceSnapshot(
        retrieved_at=str(raw["retrieved_at"]),
        source_url=str(raw["source_url"]),
        model=ModelPrice(**raw["model"]),
    )


def fetch_snapshot(url: str, model_name: str, timeout: float = 20.0) -> PriceSnapshot:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "sandbox-tinker/0.1 (+https://github.com/tsilva/sandbox-tinker)",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        rows = json.load(response)
    matches = [row for row in rows if row.get("tinker_id") == model_name]
    if len(matches) != 1:
        raise ValueError(f"Expected one pricing row for {model_name}, found {len(matches)}")
    return PriceSnapshot(
        retrieved_at=datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        source_url=url,
        model=_from_model_row(matches[0]),
    )


def refresh_snapshot(url: str, model_name: str, output_path: Path) -> PriceSnapshot:
    snapshot = fetch_snapshot(url, model_name)
    atomic_write_json(output_path, snapshot.to_dict())
    return snapshot


def estimate_cost(tokens: TokenEstimate, price: ModelPrice) -> dict[str, float]:
    train = tokens.train_tokens / 1_000_000 * price.train_usd_per_million
    prefill = tokens.prefill_tokens / 1_000_000 * price.prefill_usd_per_million
    sample = tokens.sample_tokens / 1_000_000 * price.sample_usd_per_million
    return {
        "train_usd": train,
        "prefill_usd_uncached": prefill,
        "sample_usd": sample,
        "total_usd_conservative": train + prefill + sample,
    }


def require_budget(cost: dict[str, float], budget_usd: float | None) -> None:
    if budget_usd is None:
        raise ValueError("Paid execution requires an explicit --budget-usd ceiling")
    total = cost["total_usd_conservative"]
    if total > budget_usd:
        raise ValueError(f"Estimated cost ${total:.4f} exceeds budget ${budget_usd:.4f}")
