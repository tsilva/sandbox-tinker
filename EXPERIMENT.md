# Preregistered experiment protocol

## Question and estimand

Does supervised LoRA fine-tuning improve strict intent classification by Inkling-Small on
Banking77 when inference is constrained to `effort=0.0`?

The primary estimand is the mean adapter-seed minus base-model difference in strict macro-F1 on
the frozen official test examples. The uncertainty interval resamples examples within each label
and adapter seeds, preserving the paired base/adapter comparison.

Primary model and rendering settings:

- Base model: `thinkingmachines/Inkling-Small`
- Renderer: the cookbook-recommended `tml_v0`, asserted at runtime
- Training and primary inference effort: `0.0`
- Decoding: temperature 0, maximum 32 tokens, renderer stop sequences
- Output contract: one exact label; normalization is diagnostic only

## Frozen data

- Dataset: `tsilva/banking77`
- Revision: `4235e96197daaaf23a9e278d3cbce078de7fee36`
- Source counts: 9,993 train; 3,076 official test; 77 labels
- Training-side train/test duplicate quarantine: exactly 7 rows
- Dev: 10 examples per label selected deterministically from clean source train with seed 13
- Train pool: all remaining clean source-train examples
- Pilot: first 8 train-pool examples per label
- Scale: first 24 train-pool examples per label

The partition manifest contains stable content-derived example IDs, per-class counts, duplicate
reports, partition hashes, and a full quarantine record. Any upstream drift is fatal.

## Training and selection

Fixed settings are LoRA rank 32, batch size 16, three epochs, maximum length 2,048, linear decay,
3% warmup, and Adam `(beta1=0.9, beta2=0.95, eps=1e-8)`.

Candidate learning rates are `5e-5`, `2e-4`, and `8e-4`. Pilot all three on seed 13. For each run,
select the epoch checkpoint by:

1. highest strict dev macro-F1;
2. lowest strict invalid-output rate;
3. fewest cumulative training tokens;
4. earliest optimizer step.

Select the pilot learning rate with the same ordered rule, then run scale training at seeds 13,
17, and 29. No official-test result may influence any selection.

Example pilot commands (each checked ceiling is $2.7748; use a $2.90 guard):

```bash
uv run python train_banking77_tinker.py --data-size pilot --learning-rate 5e-5 \
  --seed 13 --run-dir runs/pilot/lr-5e-5 --budget-usd 2.90
uv run python train_banking77_tinker.py --data-size pilot --learning-rate 2e-4 \
  --seed 13 --run-dir runs/pilot/lr-2e-4 --budget-usd 2.90
uv run python train_banking77_tinker.py --data-size pilot --learning-rate 8e-4 \
  --seed 13 --run-dir runs/pilot/lr-8e-4 --budget-usd 2.90

uv run python select_banking77_candidate.py \
  --candidate runs/pilot/lr-5e-5/selection.json \
  --candidate runs/pilot/lr-2e-4/selection.json \
  --candidate runs/pilot/lr-8e-4/selection.json \
  --output runs/pilot/best.json
```

Replace `<BEST_LR>` below with `learning_rate` from `runs/pilot/best.json`. Dry-run each command,
then run with a current ceiling (the checked scale estimate is $6.5489; $6.75 leaves little price
drift and intentionally fails closed if pricing moves farther):

```bash
uv run python train_banking77_tinker.py --dry-run --data-size scale \
  --learning-rate <BEST_LR> --seed 13 --run-dir runs/dry-scale-13

uv run python train_banking77_tinker.py --data-size scale --learning-rate <BEST_LR> \
  --seed 13 --run-dir runs/scale/seed-13 --budget-usd 6.75
uv run python train_banking77_tinker.py --data-size scale --learning-rate <BEST_LR> \
  --seed 17 --run-dir runs/scale/seed-17 --budget-usd 6.75
uv run python train_banking77_tinker.py --data-size scale --learning-rate <BEST_LR> \
  --seed 29 --run-dir runs/scale/seed-29 --budget-usd 6.75
```

## Dev evaluation and effort profile

Characterize the unchanged base model at efforts 0.0, 0.5, and 0.9, without selecting the primary
effort from results:

```bash
uv run python evaluate_banking77_checkpoint.py --partition dev --effort 0.0 \
  --target-name base-effort-0 --output-dir runs/dev/base-effort-0 --budget-usd 0.35
uv run python evaluate_banking77_checkpoint.py --partition dev --effort 0.5 \
  --target-name base-effort-0.5 --output-dir runs/dev/base-effort-0.5 --budget-usd 0.35
uv run python evaluate_banking77_checkpoint.py --partition dev --effort 0.9 \
  --target-name base-effort-0.9 --output-dir runs/dev/base-effort-0.9 --budget-usd 0.35
```

Create standardized full-dev records for each selected adapter (one example shown):

```bash
uv run python evaluate_banking77_checkpoint.py --partition dev \
  --selection runs/scale/seed-13/selection.json --target-name adapter-seed-13 \
  --output-dir runs/dev/adapter-seed-13 --budget-usd 0.35
```

Repeat for seeds 17 and 29. JSONL writes are append-only and resumable. An evaluation is invalid
unless every planned example has exactly one successful record; failed attempts are stored in a
separate errors file, and only retryable transport, 429, and 5xx errors are retried.

## Metrics and promotion gates

Primary metrics use exact, whitespace-trimmed labels:

- strict macro-F1 (primary);
- accuracy;
- invalid-output rate;
- per-class precision, recall, F1, and support;
- confusion counts.

Case/punctuation/format normalization is reported only as a tolerant diagnostic and never replaces
the primary prediction. The hierarchical paired bootstrap uses 10,000 replicates and seed
20260807.

Promotion to the sealed test split requires all of:

1. the 95% bootstrap lower bound for mean adapter macro-F1 minus base macro-F1 is greater than 0;
2. mean adapter invalid rate is no more than one percentage point above base;
3. mean generated tokens per adapter are no more than 1.25× base;
4. worst adapter p95 latency is no more than 1.25× base in the separate sequential benchmark.

The latency benchmark uses two fixed dev examples per label, ten warmups per arm, sequential calls,
and a rotating interleaved arm order. It includes one base arm and all three selected adapters:

```bash
uv run python benchmark_banking77_latency.py --dry-run \
  --selection 13=runs/scale/seed-13/selection.json \
  --selection 17=runs/scale/seed-17/selection.json \
  --selection 29=runs/scale/seed-29/selection.json \
  --output-dir runs/latency
```

Inspect its estimate, then repeat without `--dry-run` and with a separately approved
`--budget-usd` ceiling.

Compare dev results and freeze the exact test arms only after latency passes:

```bash
uv run python compare_banking77_runs.py \
  --base-predictions runs/dev/base-effort-0/predictions.jsonl \
  --adapter-predictions 13=runs/dev/adapter-seed-13/predictions.jsonl \
  --adapter-predictions 17=runs/dev/adapter-seed-17/predictions.jsonl \
  --adapter-predictions 29=runs/dev/adapter-seed-29/predictions.jsonl \
  --adapter-selection 13=runs/scale/seed-13/selection.json \
  --adapter-selection 17=runs/scale/seed-17/selection.json \
  --adapter-selection 29=runs/scale/seed-29/selection.json \
  --latency-summary runs/latency/summary.json \
  --output runs/dev/comparison.json \
  --freeze-test-plan runs/test-plan.json
```

## Sealed test execution

`evaluate_banking77_checkpoint.py` refuses test access without all of:

- `--allow-test`;
- the full official test partition (subsets are forbidden);
- a frozen plan whose config, revision, partition, effort, token limit, target, and sampler match;
- an explicit budget ceiling.

The first test call writes a receipt containing the plan hash and arms. Later calls must match the
same receipt. A different test plan requires a new experiment ID.

```bash
uv run python evaluate_banking77_checkpoint.py --partition test --allow-test \
  --test-plan runs/test-plan.json --target-name base \
  --output-dir runs/test/base --budget-usd <DRY_RUN_CEILING>

uv run python evaluate_banking77_checkpoint.py --partition test --allow-test \
  --test-plan runs/test-plan.json --selection runs/scale/seed-13/selection.json \
  --target-name adapter-seed-13 --output-dir runs/test/adapter-seed-13 \
  --budget-usd <DRY_RUN_CEILING>
```

Repeat the adapter command for seeds 17 and 29. Run every command with `--dry-run` first and use
the printed estimate to choose the explicit ceiling. Report all frozen arms regardless of outcome;
do not reopen hyperparameter selection after seeing test labels.

Create the final paired, seed-aware test report from the four completed prediction files:

```bash
uv run python compare_banking77_runs.py --partition test \
  --test-plan runs/test-plan.json \
  --base-predictions runs/test/base/predictions.jsonl \
  --adapter-predictions 13=runs/test/adapter-seed-13/predictions.jsonl \
  --adapter-predictions 17=runs/test/adapter-seed-17/predictions.jsonl \
  --adapter-predictions 29=runs/test/adapter-seed-29/predictions.jsonl \
  --output runs/test/comparison.json
```

## Artifact contract

Each run records the complete config and hash, Git commit/dirty/diff hash, dataset manifest and
partition hashes, price snapshot and drift flag, token/cost estimate, retry history, raw text,
decoded tokens, termination reason, latency, strict and tolerant parses, and aggregate metrics.

Training checkpoints include both resumable state and sampler weights with a seven-day TTL. Copy or
extend the TTL of any checkpoint selected for longer-lived use; Tinker's documentation distinguishes
sampler-only weights from resumable optimizer state.

## Interpretation limits

This is a single English intent dataset, a LoRA-only intervention, and a fixed prompt. The official
test set is nearly balanced but not a deployment distribution. The bootstrap quantifies example and
training-seed variation; it does not cover prompt, dataset, model-version, or provider-infrastructure
uncertainty. The 2026-08-07 price snapshot includes a limited-time discount and can change.
