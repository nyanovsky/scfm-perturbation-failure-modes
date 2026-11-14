# Environments

Run 5 loads the same three checkpoints runs 1–2 loaded, through run 2's own adapters, so it
runs in run 2's environments rather than a new one. Nothing here pins a fourth torch: a
version no adapter was verified against would change the numbers silently.

Verified on this machine, 2026-08-29:

| model | env | python | torch | package |
|---|---|---|---|---|
| scGPT | `~/miniconda3/envs/scgpt` | 3.10.19 | 2.9.1+cu128 (CUDA 12.8) | `scgpt 0.2.4` |
| TranscriptFormer | `~/miniconda3/envs/transcriptformer_env` | 3.11.15 | 2.5.1+cu124 (CUDA 12.4) | `transcriptformer 0.6.1` |
| scPRINT | `~/miniconda3/envs/scprint_env` | 3.11.16 | 2.5.1+cu124 (CUDA 12.4) | `scprint 2.3.8` (`--no-deps`) |

Built here on 2026-08-29. All three envs import run 2's `models/queries.py` and its
`BUILDERS` table unmodified, which is the surface run 5 subclasses.

## Building `scprint_env`, and the one trap in checking it

Runs 1–2 ran scPRINT on the Silico fabric; nothing on this machine carried the package until
this run built it.
`docs/INDEX.md` §8 has the recipe, and every clause of it is load-bearing:

- install `scprint==2.3.8` with `--no-deps` — its metadata pins `torch==2.2.0` and would
  downgrade the CUDA build out from under the other adapters;
- set `precpt_gene_emb=None` — both checkpoints record stale authors'-cluster paths to a
  `gene_embeddings.parquet` that neither exists nor is needed, since the embeddings are in
  the weights;
- keep the `gene_encoder.0.embedding.weight → …embeddings.weight` rename. **Skipping it
  leaves a randomly initialised embedding and every response is noise** — and it is noise
  that looks like a result, which is exactly the failure this run exists to avoid;
- override `transformer="flash"` to the pure-torch path.

The checkpoint that runs 1–2 actually used is `medium-v1.5` (`nlayers 8`, `d_model 256`), not
the `(large)` its manifest label says — a documentation error resolved in `docs/INDEX.md` §8.
`large-v1` is on disk but nothing has used it.

⚠️ **`import scprint` is the wrong check and will send you installing packages the run does
not need.** The top-level package pulls `scprint.tasks.grn`, which drags in `bengrn`, `gget`,
`scib_metrics` and the whole lamindb/bionty ontology stack. Run 2 never imports it: its
`scprint_model.py:48 import_scprint_model()` stubs the heavy chain out and imports
`scprint.model.model` alone. Check *that*, not the package:

```bash
cd experiments/experiment-2-nygwcv
PYTHONPATH=src ~/miniconda3/envs/scprint_env/bin/python -c \
  "from scfm_pfm.models.scprint_model import import_scprint_model; import_scprint_model()"
```

`No module named 'adasplash'` and `FlashAttention is not installed` on that path are expected
and harmless — run 2 overrides `transformer="flash"` to the pure-torch path regardless, so the
kernel was never going to be used.

Only the import is verified here. Whether the checkpoint *loads* correctly is checked by the
smoke run, not asserted from this table: a scPRINT that silently kept a randomly initialised
gene embedding imports exactly as cleanly as one that did not.

## Figures

`report/figures.py` needs `plotly` and `kaleido` (for the PDF export) and `silico_figures`,
which is **not** installed: it is the local reconstruction under `tools/silico-figures-local`,
so put it on the path rather than pip-installing anything:

```bash
PYTHONPATH=src:../../tools/silico-figures-local/src \
  ~/miniconda3/envs/scgpt/bin/python -m scfm_probe.report.figures
```

`plotly 7.0.0` and `kaleido` were added to the `scgpt` env on 2026-08-29; neither is a model
dependency, and the figure build is the only thing that needs them.

## Hardware

One RTX 3090, 24 GB — not the nine H100s runs 1–4 had. Run 2's measured per-anchor costs are
the scaling baseline (`docs/INDEX.md` §8); divide by the throughput ratio before promising
coverage.
