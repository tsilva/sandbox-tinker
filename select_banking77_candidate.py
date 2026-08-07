"""Select one training run by frozen dev metrics and deterministic tie-breakers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from banking77_experiment.artifacts import atomic_write_json, choose_checkpoint
from banking77_experiment.config import load_config

DEFAULT_CONFIG = "configs/inkling_small_banking77.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    candidates: list[dict[str, Any]] = []
    expected_data_size: str | None = None
    expected_seed: int | None = None
    seen_learning_rates: set[float] = set()
    for value in args.candidate:
        selection_path = Path(value).expanduser().resolve()
        selection = json.loads(selection_path.read_text())
        manifest = json.loads((selection_path.parent / "manifest.json").read_text())
        if manifest["config_hash"] != cfg.config_hash:
            raise ValueError(f"Config hash mismatch for {selection_path}")
        data_size = str(manifest["data_size"])
        seed = int(manifest["seed"])
        learning_rate = float(manifest["learning_rate"])
        if expected_data_size is None:
            expected_data_size = data_size
            expected_seed = seed
        if data_size != expected_data_size or seed != expected_seed:
            raise ValueError("Candidate runs must use the same data size and seed")
        if learning_rate in seen_learning_rates:
            raise ValueError(f"Duplicate learning rate {learning_rate}")
        seen_learning_rates.add(learning_rate)
        candidates.append(
            {
                **selection,
                "learning_rate": learning_rate,
                "seed": seed,
                "data_size": data_size,
                "selection_path": str(selection_path),
            }
        )
    selected = choose_checkpoint(candidates)
    atomic_write_json(Path(args.output).expanduser().resolve(), selected)
    print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
