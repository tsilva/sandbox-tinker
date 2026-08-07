from dataclasses import replace

import pytest

from banking77_experiment.config import load_config
from banking77_experiment.data import Example, _quarantine_test_leaks, normalized_text


def _example(example_id: str, split: str, text: str, label: str = "card_payment") -> Example:
    return Example(example_id, split, 0, text, 0, label, "Choose a label")


def test_preregistered_config_is_pinned():
    cfg = load_config("configs/inkling_small_banking77.toml")
    assert cfg.model_name == "thinkingmachines/Inkling-Small"
    assert cfg.effort == 0.0
    assert cfg.dataset_revision == "4235e96197daaaf23a9e278d3cbce078de7fee36"
    assert cfg.expected_quarantined_test_leaks == 7
    assert len(cfg.config_hash) == 64
    with pytest.raises(ValueError, match="effort=0.0"):
        replace(cfg, effort=0.5).validate()


def test_normalization_and_training_side_quarantine():
    train = (
        _example("train-leak", "train", "  CARD   charged  "),
        _example("train-clean", "train", "cash withdrawal"),
    )
    test = (_example("test", "test", "card charged"),)
    clean, quarantined = _quarantine_test_leaks(train, test)
    assert [row.example_id for row in clean] == ["train-clean"]
    assert [row.example_id for row in quarantined] == ["train-leak"]
    assert normalized_text("  CARD   charged  ") == "card charged"
