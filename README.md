# Inkling-Small × Banking77 schema distillation

This repository tests whether a LoRA can internalize Banking77's 77-label output schema so that
`thinkingmachines/Inkling-Small` can classify requests with a short production prompt.

The experiment has three essential arms:

| Arm | Model | Prompt | What it measures |
|---|---|---|---|
| `base-full` | unchanged Inkling-Small | all 77 labels | Best prompt-only reference with a cheat sheet |
| `base-compact` | unchanged Inkling-Small | no label list | What the base model already knows about the exact schema |
| `adapter-seed-*` | trained LoRA | same compact prompt | Whether training learned the schema and task |

The compact control is intentionally difficult. It tells the model to return an exact Banking77
intent name but does not reveal the possible names. That makes `adapter - base-compact` the training
effect, while `adapter - base-full` measures the quality cost of removing the long label list.

## Local setup

Requirements are `uv`, Python 3.11+, and a Tinker account only for paid commands.

```bash
uv sync --group dev
uv run ruff check .
uv run pytest -q
```

Private credentials are declared in the committed, value-free `.keyenv.toml` and stored in the
macOS Keychain. Store and verify the Tinker key once:

```bash
keyenv set TINKER_API_KEY
keyenv doctor
```

Run paid commands through `keyenv run -- ...`; Python receives the key through `os.environ` without
writing it to `.env` or a command argument. Dependencies are locked in `uv.lock`; releases newer
than seven days and known-bad package versions remain constrained.

## Safe preflight

This makes no Tinker API calls. It downloads and checksum-verifies the pinned source data, renders
the real Inkling format, plans one compact-prompt batch, and estimates uncached cost:

```bash
uv run python train_banking77_tinker.py \
  --dry-run \
  --max-steps 1 \
  --eval-examples-per-label 1 \
  --run-dir runs/v2-preflight/smoke
```

The checked v2 estimate on 2026-08-07 is **$0.0069**. This is only an estimate and not permission
to make a paid call. Every paid command refreshes official pricing, requires an explicit
`--budget-usd`, and stops before constructing a client if the ceiling is absent or too low.

## Workflow

The full preregistration and commands are in [`EXPERIMENT.md`](EXPERIMENT.md). In short:

1. Evaluate the unchanged model on dev with both full and compact prompts.
2. Run one seed-13 pilot at the fixed `2e-4` learning rate using compact prompts.
3. Stop if it does not clearly beat `base-compact` or roughly retain `base-full` quality.
4. If promising, train the scale subset at seeds 13, 17, and 29.
5. Evaluate all three adapters with the compact prompt, benchmark latency, and freeze a five-arm
   test plan only if every gate passes.
6. Unseal the official test set once and report all frozen arms.

Current conservative dry-run estimates are:

| Stage | Per run | Planned runs | Conservative total |
|---|---:|---:|---:|
| One-step v2 smoke + 77 dev examples | $0.0069 | 1 | $0.0069 |
| Full-prompt base dev evaluation | $0.2333 | 1 | $0.2333 |
| Compact-prompt dev evaluation | $0.0547 | 4 | $0.2188 |
| Pilot, 8 examples/label | $0.3248 | 1 | $0.3248 |
| Scale, 24 examples/label | $0.6465 | 3 | $1.9395 |
| Five-arm latency benchmark | $0.0960 | 1 | $0.0960 |

These ceilings assume maximum output tokens, uncached prefill, and the pinned 2026-08-07 price
snapshot. Re-run dry mode before requesting approval for any paid stage.

The earlier artifacts in `runs/smoke` and `runs/base-smoke-control` belong to the v1 plumbing smoke
test. They remain preserved, but they are not evidence for this v2 comparison because the source
data and prompt contract changed.

## Data integrity

The experiment reads the original PolyAI CSV files from immutable commit
`57ec275d8078af65b7731c2a98be812d844a6d6b` of
[`PolyAI-LDN/task-specific-datasets`](https://github.com/PolyAI-LDN/task-specific-datasets/tree/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data).
SHA-256 checks and expected row counts make source drift fatal.

- Source train: 10,003 rows
- Official test: 3,080 rows
- Frozen taxonomy: 77 labels from the canonical `categories.json`
- Cross-split leakage: 7 train rows quarantined because their normalized `(text, label)` occurs in
  test
- Within-train duplication: 4 repeated rows removed before the dev split
- Frozen dev: 10 unique examples per label; 770 total
- Train pool: 9,222 unique, test-disjoint examples

The official test remains unchanged, including one duplicate pair. The manifest records every
removed training row, source checksum, stable example ID, class count, and partition hash.

## Main files

- `train_banking77_tinker.py` — compact-prompt SFT, epoch checkpoints, and dev selection
- `evaluate_banking77_checkpoint.py` — explicit prompt arms, resumable evaluation, sealed-test guard
- `benchmark_banking77_latency.py` — five-arm sequential interleaved latency protocol
- `compare_banking77_runs.py` — paired bootstrap gates and frozen test-plan creation
- `configs/inkling_small_banking77.toml` — pinned v2 experiment contract
- `configs/banking77_categories.json` — vendored canonical label order
- `banking77_experiment/` — data, rendering, training, metrics, pricing, and artifact contracts

Generated runs live under ignored `runs/`. Completed paid run directories are immutable.
