from dataclasses import replace

import pytest

from banking77_experiment.config import load_config
from banking77_experiment.data import Example, _quarantine_test_leaks, normalized_text


def _example(example_id: str, split: str, text: str, label: str = "card_payment") -> Example:
    return Example(example_id, split, 0, text, 0, label)


def test_preregistered_config_is_pinned():
    cfg = load_config("configs/inkling_small_banking77.toml")
    assert cfg.model_name == "thinkingmachines/Inkling-Small"
    assert cfg.effort == 0.0
    assert cfg.dataset_revision == "57ec275d8078af65b7731c2a98be812d844a6d6b"
    assert cfg.expected_train_rows == 10003
    assert cfg.expected_test_rows == 3080
    assert cfg.expected_quarantined_test_leaks == 7
    assert cfg.expected_deduplicated_train_rows == 4
    assert cfg.training_prompt_variant == "compact"
    assert cfg.base_prompt_variant == "full_taxonomy"
    assert cfg.adapter_prompt_variant == "compact"
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
