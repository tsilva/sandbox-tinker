import io
import json

import pytest

from banking77_experiment import pricing
from banking77_experiment.data import Example
from banking77_experiment.pricing import (
    TokenEstimate,
    estimate_cost,
    load_snapshot,
    require_budget,
)
from banking77_experiment.training import PlannedBatch, RenderedDatum, plan_batches, scheduled_lr


def _rendered(index: int) -> RenderedDatum:
    example = Example(str(index), "train", index, str(index), 0, "label")
    return RenderedDatum(example, None, index + 1)  # type: ignore[arg-type]


def test_cost_estimate_and_budget_gate():
    price = load_snapshot("pricing/inkling_small_2026-08-07.json").model
    cost = estimate_cost(
        TokenEstimate(train_tokens=1_000_000, prefill_tokens=1_000_000, sample_tokens=1_000_000),
        price,
    )
    assert cost["total_usd_conservative"] == pytest.approx(1.73 + 0.58 + 1.44)
    require_budget(cost, 4.0)
    with pytest.raises(ValueError, match="exceeds budget"):
        require_budget(cost, 3.0)
    with pytest.raises(ValueError, match="explicit"):
        require_budget(cost, None)


def test_live_pricing_request_sets_http_headers(monkeypatch):
    row = {
        "tinker_id": "model",
        "prefill": "$1.0",
        "cached_prefill": "$0.5",
        "sample": "$2.0",
        "train": "$3.0",
        "context": "1K",
        "note": "test",
    }

    def fake_urlopen(request, timeout):
        assert timeout == 20.0
        headers = dict(request.header_items())
        assert headers["Accept"] == "application/json"
        assert "sandbox-tinker" in headers["User-agent"]
        return io.BytesIO(json.dumps([row]).encode())

    monkeypatch.setattr(pricing.urllib.request, "urlopen", fake_urlopen)
    snapshot = pricing.fetch_snapshot("https://example.test/models.json", "model")
    assert snapshot.model.train_usd_per_million == 3.0


def test_batch_plan_and_lr_schedule_are_deterministic():
    rows = tuple(_rendered(index) for index in range(5))
    first = plan_batches(rows, batch_size=2, epochs=2, seed=13)
    second = plan_batches(rows, batch_size=2, epochs=2, seed=13)
    assert first == second
    assert len(first) == 6
    assert isinstance(first[0], PlannedBatch)
    assert sum(len(batch.items) for batch in first) == 10
    assert scheduled_lr(1.0, "linear", 0.0, 0, 5) == 1.0
    assert scheduled_lr(1.0, "linear", 0.0, 4, 5) == 0.0
    assert scheduled_lr(1.0, "constant", 0.0, 4, 5) == 1.0
