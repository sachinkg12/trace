# TRACE

TRACE (Target-aware Retrieval and Attested Citation Evidence) is the system
developed by team **gabby** for the GroundLM 2026 LitTraceQA shared task. It
plans evidence targets, retrieves candidate papers from metadata and parsed PDF
indexes, localizes source evidence, and emits multiple-choice or structured
table answers in the official format.

This repository is a clean source release. It contains no benchmark inputs,
source PDFs, generated predictions, API credentials, per-question rules, or
leaderboard-derived overrides.

## Reproducibility scope

Team gabby's historical submission scored **0.705669**. A later provenance
audit found that its preserved table component descended from an earlier
test-conditioned runtime. The new source-owner retrieval and repeat-agreement
multiple-choice changes were generic, but the complete historical JSONL should
not be presented as a clean end-to-end regeneration.

Accordingly, this repository releases the generic full-generation architecture
and reports the historical result with that limitation. See
[`docs/results.md`](docs/results.md) and [`docs/limitations.md`](docs/limitations.md).

## Architecture

1. A question planner extracts named methods, requested properties, source
   types, cardinality, and table schema targets.
2. Exact, alias, relation, passage, and dense routes retrieve candidates.
3. Target-aware coverage and answer-bearing evidence select up to five papers.
4. PDFs are parsed with PyMuPDF; the evidence localizer emits scorer-compatible
   table, figure, equation, citation, or text locators.
5. Answer strategies produce multiple-choice and schema-planned table answers.
6. The official validator runs before predictions are exposed for submission.

## Installation

Python 3.11--3.13 is supported. Python 3.12.1 was used for the preserved run
environment.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
cp .env.example .env
```

Set `GEMINI_API_KEY` in the shell or `.env`. Do not commit `.env`.

## Data and indexes

Benchmark annotations are not redistributed. Download the public release:

```bash
python scripts/fetch_data.py
python -m littraceqa.corpus.download data/paper_metadata.jsonl data/pdfs
python -m littraceqa.corpus.parse \
  data/paper_metadata.jsonl data/pdfs --parsed-dir data/parsed
python -m littraceqa.corpus.build_indexes \
  data/parsed data/indexes --pool data/paper_metadata.jsonl
```

The first dense run creates `data/cache/pool_emb.npy` using
`BAAI/bge-small-en-v1.5`. Some OpenReview-hosted PDFs may require
`OPENREVIEW_TOKEN`. See [`docs/data-and-assets.md`](docs/data-and-assets.md).

## Full-generation command

```bash
export GEMINI_API_KEY='<your-key>'
export LITTRACEQA_SOURCE_REVISION="$(git rev-parse HEAD)"
python -m littraceqa.experiments.submit \
  --config configs/trace-littraceqa.yaml \
  --index-dir data/indexes \
  --inputs data/test.jsonl \
  --pool-path data/paper_metadata.jsonl \
  --pool-emb-cache data/cache/pool_emb.npy \
  --output outputs/predictions.jsonl \
  --trace-output outputs/traces.jsonl \
  --workers 8
```

The command performs full generation only. It does not accept a previous
prediction file as a parent. It verifies the released input/pool/index profile,
validates all output rows, and writes a hash-bound provenance manifest.

## Models and checkpoints

TRACE has no task-specific trained checkpoint.

- Planning, text grounding, and answer generation:
  `gemini-2.5-flash` through the Google GenAI API.
- Visual table and figure reading: `gemini-2.5-pro` through the same API.
- Dense metadata embeddings:
  [`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5).

For a submission form that requires a checkpoint URL, use this section's URL.
Hosted API behavior can change over time even with a fixed model identifier.

## Tests

```bash
pytest -q
python -m compileall -q src
python -m littraceqa.experiments.submit --help
```

## Licenses

The original TRACE source code is MIT licensed. The vendored official
submission validator and LitTraceQA benchmark files are CC BY-NC 4.0. Paper
metadata and PDFs remain subject to their publishers' terms; PDFs are
downloaded from original source URLs and are not redistributed here.
