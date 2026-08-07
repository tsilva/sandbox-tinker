# Inkling-Small × Banking77

This repository is a reproducible, budget-gated experiment for supervised LoRA fine-tuning of
`thinkingmachines/Inkling-Small` on Banking77. It keeps the official test split sealed until the
learning rate, checkpoint, seeds, evaluation settings, and latency result are frozen.

The implementation follows Tinker's documented SFT sequence: render masked `Datum` objects,
`forward_backward`, `optim_step`, save immutable training and sampler checkpoints, then evaluate
through a `SamplingClient`. See the [Tinker quick start](https://tinker-docs.thinkingmachines.ai/tinker/quickstart/)
and [weights guide](https://tinker-docs.thinkingmachines.ai/tutorials/core-concepts/weights/).

## Local setup

Requirements: `uv`, Python 3.11+, and a Tinker account only when a paid command is run.

```bash
uv sync --group dev
uv run ruff check .
uv run pytest -q
```

Put `TINKER_API_KEY` in the environment or the ignored `.env` file. Never commit it.

The dependency graph is locked in `uv.lock`. New Python releases are held back for seven days,
and known-bad package releases remain constrained in `pyproject.toml`.

## Safe preflight

This command downloads the pinned dataset revision, exercises the real Inkling renderer, builds
one deterministic batch, estimates the worst-case uncached cost, and makes no Tinker API calls:

```bash
uv run python train_banking77_tinker.py \
  --dry-run \
  --max-steps 1 \
  --eval-examples-per-label 1 \
  --run-dir runs/preflight
```

The checked preflight estimate on 2026-08-07 is **$0.0461**. Repeating the preflight in two output
directories produced byte-identical `dry_run_deterministic.json` files.

Paid commands require an explicit `--budget-usd` and refresh official model pricing before they
can construct a service client. If the refreshed conservative estimate exceeds the ceiling, the
command exits without training or sampling.

## One-step paid smoke test

Run this only after approving a **$0.06 maximum**:

```bash
uv run python train_banking77_tinker.py \
  --max-steps 1 \
  --eval-examples-per-label 1 \
  --run-dir runs/smoke \
  --budget-usd 0.06
```

This validates authentication, the local/service tokenizer identity, one optimizer step,
checkpoint persistence, sampling, strict parsing, and artifact writing. Stop if it does not end
with a complete 77-example evaluation.

## Experiment workflow

The exact hypotheses, fixed splits, selection rules, gates, and cost stages are in
[`EXPERIMENT.md`](EXPERIMENT.md). The short sequence is:

1. Run the base model on dev at efforts 0.0, 0.5, and 0.9; effort 0.0 remains primary.
2. Pilot three learning rates on seed 13 and select by strict dev macro-F1.
3. Scale the selected learning rate and confirm it on seeds 13, 17, and 29.
4. Re-evaluate the base and selected adapters on the full frozen dev partition.
5. Run the sequential interleaved latency benchmark.
6. Compare all seeds; freeze a test plan only if every promotion gate passes.
7. Evaluate every frozen arm exactly once on the full official test split.

Every paid stage can be dry-run first. Current conservative dry-run ceilings are:

| Stage | Per run | Planned runs | Conservative total |
|---|---:|---:|---:|
| One-step smoke + 77 dev examples | $0.0461 | 1 | $0.0461 |
| Base full-dev evaluation | $0.2958 | 3 efforts | $0.8874 |
| Pilot, 8 examples/label | $2.7748 | 3 learning rates | $8.3244 |
| Scale, 24 examples/label | $6.5489 | 3 seeds | $19.6467 |

These are ceilings based on maximum output tokens, uncached prefill, and the pinned 2026-08-07
price snapshot—not invoices. Re-run dry mode immediately before each stage.

## Data integrity

The experiment pins [`tsilva/banking77`](https://huggingface.co/datasets/tsilva/banking77)
at revision `4235e96197daaaf23a9e278d3cbce078de7fee36`. The dataset is CC BY 4.0 and contains
9,993 source-train rows, 3,076 official-test rows, and 77 labels.

The source revision contains seven exact normalized `(text, label)` duplicates across train and
test. The harness preserves the official test split, quarantines those seven training-side rows
before deriving dev, records their stable IDs and contents in `partition.json`, and aborts if the
count changes. Train, dev, and test are then checked for both ID and normalized-content overlap.

## Main files

- `train_banking77_tinker.py` — deterministic SFT, epoch checkpoints, and dev selection.
- `evaluate_banking77_checkpoint.py` — resumable base/adapter evaluation and sealed-test guard.
- `benchmark_banking77_latency.py` — paired sequential latency protocol.
- `compare_banking77_runs.py` — hierarchical bootstrap, promotion gates, and test-plan freeze.
- `select_banking77_candidate.py` — deterministic pilot learning-rate selection.
- `configs/inkling_small_banking77.toml` — preregistered configuration.
- `pricing/inkling_small_2026-08-07.json` — pinned official price snapshot.
- `banking77_experiment/` — data, rendering, training, metrics, pricing, and artifact contracts.

Generated runs live under ignored `runs/`. Preserve completed run directories: JSONL evaluation is
resumable, but duplicate successful records, partial coverage, changed test plans, and incompatible
config or partition hashes are rejected.

