# Preregistered schema-distillation experiment

## Question

Can supervised LoRA training teach Inkling-Small the Banking77 intent taxonomy well enough to
replace a prompt containing all 77 labels with a short instruction?

All arms use `thinkingmachines/Inkling-Small`, the cookbook `tml_v0` renderer, effort `0.0`,
temperature `0`, a 32-token output limit, and strict exact-label scoring.

The three comparisons are:

1. `base-full`: unchanged model plus the full ordered label taxonomy;
2. `base-compact`: unchanged model plus the compact instruction, with no candidate labels;
3. `adapter-seed-*`: trained model plus that exact same compact instruction.

The compact prompt is:

> Classify with the exact Banking77 intent label. Output only the label.

The base model is not expected to infer arbitrary label spellings reliably. That is the point of
the compact base control: it separates knowledge transferred by training from help supplied by the
full prompt.

## Frozen data

- Source: `PolyAI-LDN/task-specific-datasets`
- Commit: `57ec275d8078af65b7731c2a98be812d844a6d6b`
- Source counts: 10,003 train; 3,080 official test; 77 labels
- Cross-split quarantine: exactly 7 training rows
- Within-training deduplication: exactly 4 repeated rows
- Dev: 10 unique examples per label selected deterministically with seed 13
- Train pool: all remaining 9,222 unique training examples
- Pilot: first 8 train-pool examples per label, 616 total
- Scale: first 24 train-pool examples per label, 1,848 total

The source URLs and CSV SHA-256 digests are frozen in the config. The vendored label asset is also
checksum-verified. Any count, label, checksum, duplicate, or partition drift aborts the run.

## Training and staging

Training always uses the compact prompt. Fixed settings are LoRA rank 32, batch size 16, three
epochs, maximum length 2,048, linear decay, 3% warmup, Adam
`(beta1=0.9, beta2=0.95, eps=1e-8)`, and learning rate `2e-4`.

Start with one seed-13 pilot. Do not launch the three scale seeds unless its dev comparison is
promising. Dry-run first:

```bash
uv run python train_banking77_tinker.py --dry-run --data-size pilot \
  --learning-rate 2e-4 --seed 13 --run-dir runs/v2/pilot/seed-13
```

The checked estimate is $0.3248. A paid rerun requires a newly approved ceiling:

```bash
uv run python train_banking77_tinker.py --data-size pilot \
  --learning-rate 2e-4 --seed 13 --run-dir runs/v2/pilot/seed-13 \
  --budget-usd <APPROVED_CEILING>
```

Each epoch checkpoint is selected by highest strict dev macro-F1, then lowest invalid rate, fewest
training tokens, and earliest step. If the pilot passes the decision check, dry-run and then train
the scale subset for seeds 13, 17, and 29 in separate immutable directories. The checked scale
estimate is $0.6465 per seed.

## Dev evaluation

Dry-run the two unchanged-model controls:

```bash
uv run python evaluate_banking77_checkpoint.py --dry-run --partition dev \
  --prompt-variant full_taxonomy --target-name base-full \
  --output-dir runs/v2/dev/base-full

uv run python evaluate_banking77_checkpoint.py --dry-run --partition dev \
  --prompt-variant compact --target-name base-compact \
  --output-dir runs/v2/dev/base-compact
```

Checked estimates are $0.2333 for `base-full` and $0.0547 for `base-compact`. After approval,
repeat without `--dry-run` and add an explicit `--budget-usd` ceiling.

Evaluate each selected adapter using the same compact prompt:

```bash
uv run python evaluate_banking77_checkpoint.py --dry-run --partition dev \
  --selection runs/v2/scale/seed-13/selection.json \
  --prompt-variant compact --target-name adapter-seed-13 \
  --output-dir runs/v2/dev/adapter-seed-13
```

Repeat for seeds 17 and 29. Evaluation JSONL is append-only and resumable under a contract that
includes the config, partition, checkpoint, decoding settings, and prompt variant.

## Metrics and gates

Primary metrics are strict macro-F1, accuracy, invalid-output rate, per-class statistics, and
confusions. Tolerant parsing is diagnostic only.

Two paired, label-stratified, seed-aware bootstraps use 10,000 replicates and seed 20260807:

- Training benefit: mean adapter minus `base-compact`; lower 95% bound must exceed `0.00`.
- Prompt compression: mean adapter minus `base-full`; lower 95% bound must exceed `-0.02`.

Promotion also requires:

- mean adapter prompt tokens no more than 10% of `base-full`;
- mean adapter invalid rate no more than one percentage point above `base-full`;
- mean adapter generated tokens no more than 1.25× `base-full`;
- worst adapter p95 latency no more than 1.25× `base-full`.

The latency benchmark uses two fixed dev examples per label, ten warmups per arm, sequential calls,
and rotating arm order. It includes `base-full`, `base-compact`, and all three adapters:

```bash
uv run python benchmark_banking77_latency.py --dry-run \
  --selection 13=runs/v2/scale/seed-13/selection.json \
  --selection 17=runs/v2/scale/seed-17/selection.json \
  --selection 29=runs/v2/scale/seed-29/selection.json \
  --output-dir runs/v2/latency
```

The checked five-arm estimate is $0.0960. Run it only after separate approval.

Compare dev results and freeze the exact test plan only after all gates pass:

```bash
uv run python compare_banking77_runs.py \
  --base-full-predictions runs/v2/dev/base-full/predictions.jsonl \
  --base-compact-predictions runs/v2/dev/base-compact/predictions.jsonl \
  --adapter-predictions 13=runs/v2/dev/adapter-seed-13/predictions.jsonl \
  --adapter-predictions 17=runs/v2/dev/adapter-seed-17/predictions.jsonl \
  --adapter-predictions 29=runs/v2/dev/adapter-seed-29/predictions.jsonl \
  --adapter-selection 13=runs/v2/scale/seed-13/selection.json \
  --adapter-selection 17=runs/v2/scale/seed-17/selection.json \
  --adapter-selection 29=runs/v2/scale/seed-29/selection.json \
  --latency-summary runs/v2/latency/summary.json \
  --output runs/v2/dev/comparison.json \
  --freeze-test-plan runs/v2/test-plan.json
```

## Sealed test

Test evaluation requires `--allow-test`, the full official test partition, a matching frozen plan,
and an explicit paid ceiling. The plan freezes five arms, including prompt variant as well as model
checkpoint. The first test call writes an immutable unseal receipt.

Example base calls:

```bash
uv run python evaluate_banking77_checkpoint.py --dry-run --partition test --allow-test \
  --test-plan runs/v2/test-plan.json --prompt-variant full_taxonomy \
  --target-name base-full --output-dir runs/v2/test/base-full

uv run python evaluate_banking77_checkpoint.py --dry-run --partition test --allow-test \
  --test-plan runs/v2/test-plan.json --prompt-variant compact \
  --target-name base-compact --output-dir runs/v2/test/base-compact
```

After inspecting estimates and obtaining approval, repeat with paid ceilings. Evaluate each adapter
similarly with its selection file, compact prompt, and frozen target name. Report all five arms
regardless of outcome; do not reopen selection after observing test results.

Create the final report:

```bash
uv run python compare_banking77_runs.py --partition test \
  --test-plan runs/v2/test-plan.json \
  --base-full-predictions runs/v2/test/base-full/predictions.jsonl \
  --base-compact-predictions runs/v2/test/base-compact/predictions.jsonl \
  --adapter-predictions 13=runs/v2/test/adapter-seed-13/predictions.jsonl \
  --adapter-predictions 17=runs/v2/test/adapter-seed-17/predictions.jsonl \
  --adapter-predictions 29=runs/v2/test/adapter-seed-29/predictions.jsonl \
  --output runs/v2/test/comparison.json
```

## Interpretation limits

This is one English intent dataset, one model, one LoRA recipe, and one compact prompt. The official
test distribution is not a deployment distribution. Bootstrap intervals cover example and training
seed variation, not prompt choice, model-version drift, provider infrastructure, or label noise.
