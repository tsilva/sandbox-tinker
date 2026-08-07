import pytest

from banking77_experiment.metrics import (
    classification_metrics,
    paired_seed_bootstrap,
    strict_label,
    tolerant_label,
)

LABELS = ("card_payment", "cash_withdrawal")


def test_strict_metric_does_not_silently_normalize():
    assert strict_label("card_payment\n", LABELS) == "card_payment"
    assert strict_label("Card payment", LABELS) is None
    assert tolerant_label("Assistant: Card payment.", LABELS) == "card_payment"
    metrics = classification_metrics(
        LABELS,
        ["card_payment", "cash_withdrawal"],
        ["card_payment", None],
    )
    assert metrics["accuracy"] == 0.5
    assert metrics["invalid_rate"] == 0.5
    assert metrics["macro_f1"] == pytest.approx(0.5)


def test_hierarchical_paired_bootstrap_is_deterministic_and_positive():
    truth = {"a": "card_payment", "b": "cash_withdrawal"}
    base = {"a": None, "b": None}
    adapters = {
        13: {"a": "card_payment", "b": "cash_withdrawal"},
        17: {"a": "card_payment", "b": "cash_withdrawal"},
        29: {"a": "card_payment", "b": "cash_withdrawal"},
    }
    first = paired_seed_bootstrap(LABELS, truth, base, adapters, 100, 7)
    second = paired_seed_bootstrap(LABELS, truth, base, adapters, 100, 7)
    assert first == second
    assert first["lower_95"] == 1.0
