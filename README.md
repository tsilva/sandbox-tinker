<p align="center">
  <!-- repo-tagline:start -->
  <strong>🧪 Test whether LoRA can internalize Banking77 classification labels 🏷️</strong>
  <!-- repo-tagline:end -->
</p>

This experiment compares Banking77 intent classification with a full-label
prompt, a compact prompt, and LoRA adapters using the compact prompt. It measures
whether the adaptation learns the task and label schema.

## Get started

Use Python 3.11+, uv, and the locked environment:

```bash
uv sync --frozen --group dev
```

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

[Banking77 experiment protocol](docs/sandbox-tinker-guide.md).
