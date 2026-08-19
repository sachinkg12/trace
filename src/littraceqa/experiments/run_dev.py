"""CLI + composition root for the dev-55 experiment harness.

    python -m littraceqa.experiments.run_dev --config configs/dev-old.yaml

This is the ONLY place concrete runners are named. `make_runner` maps
`config.path` -> a `DevRunner`:
- "old"   -> `build_pipeline(...)` wrapped in `OldPathRunner` (rung 1).
- "new"   -> planner -> router -> config-selected precision-cascade over the
  corpus indexes (`NewPathRunner`); `params.cascade_stages` selects the rung.
- "union" -> NotImplementedError (a separate follow-up).

Adding the planner/cascade path was a one-branch change here plus a new runner
class; the driver, manifest, config schema, and run-dir layout are untouched.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
from typing import Any

from littraceqa.experiments.cohort_consensus import prepare_batch_runner
from littraceqa.experiments.config import RunConfig, load_config
from littraceqa.experiments.driver import DEFAULT_PER_RECORD_TIMEOUT, run_experiment
from littraceqa.experiments.manifest import build_manifest
from littraceqa.experiments.rundir import DEFAULT_BASE, prepare_run_dir
from littraceqa.experiments.runner import DevRunner, OldPathRunner
from littraceqa.pipeline.input import parse_input_record

DEFAULT_INPUTS = "data/validation_inputs.jsonl"
DEFAULT_GOLD = "data/validation.jsonl"

# `build_pipeline` kwargs the config `params` may set. Anything else in params
# is provenance-only (passed through to the manifest) and ignored by the OLD
# builder -- keeping the config permissive for Build B.
_OLD_PIPELINE_KEYS = frozenset(
    {
        "llm_name",
        "top_n_seeds",
        "use_expander",
        "expand_k",
        "evidence_confidence_floor",
        "vision_model",
    }
)


def _seed_everything(seed: int) -> None:
    """Deterministic startup: seed `random` and (if present) numpy."""
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:  # noqa: BLE001 -- numpy always present here, but stay defensive
        pass


def make_runner(
    config: RunConfig,
    *,
    pool_path: str | pathlib.Path | None = None,
    index_dir: str | pathlib.Path | None = None,
) -> DevRunner:
    """Composition root: `config.path` -> a concrete `DevRunner`.

    - "old"   -> `build_pipeline(...)` wrapped in `OldPathRunner` (rung 1).
    - "new"   -> planner -> router -> config-selected precision-cascade over the
      corpus indexes (`NewPathRunner`); `params.cascade_stages` picks the rung.
    - "union" -> NotImplementedError (a separate follow-up).

    This is the ONLY place concretes are named for either path.
    """
    if config.path == "old":
        from littraceqa.pipeline.build import build_pipeline

        kwargs: dict[str, Any] = {
            k: v for k, v in config.params.items() if k in _OLD_PIPELINE_KEYS
        }
        if pool_path is not None:
            kwargs["pool_path"] = pool_path
        pipeline = build_pipeline(**kwargs)
        return OldPathRunner(pipeline)

    if config.path == "new":
        return _make_new_runner(config, pool_path=pool_path, index_dir=index_dir)

    if config.path == "union":
        raise NotImplementedError(
            "path='union' (old + new fusion) is a separate follow-up build; "
            "the harness implements 'old' and 'new'. Register it in make_runner()."
        )

    raise ValueError(f"unknown config path {config.path!r}")


def _make_new_runner(
    config: RunConfig,
    *,
    pool_path: str | pathlib.Path | None,
    index_dir: str | pathlib.Path | None,
) -> DevRunner:
    """Wire the planner -> router -> cascade `NewPathRunner`. The composition root
    reads these `config.params` keys:

      - `llm_name` (default "gemini"): the shared LLM backend.
      - `llm_model` (optional): concrete model for the shared planner/text-answer
        client. Omitted preserves the backend default exactly. This is separate
        from `evidence_model` and `vision_model` so model gates can isolate one
        component instead of silently changing the whole pipeline.
      - `vision_model` (default "gemini-2.5-pro"; null to disable): stronger
        multimodal client for the figure/table vision path.
      - `cascade_stages` (default `["filter","rerank","attest","llm"]`): the
        ablation RUNG -- an empty list `[]` => bypass (rung "new-raw", paper_ids
        straight from candidates).
      - `reranker` (optional, e.g. "bge") + `rerank_top_k` (optional): the rerank
        stage is a no-op pass-through when no reranker is configured.
      - `use_dense` (default True): build a `DenseRetriever` over the cached
        title+abstract embeddings and inject it so the "knn" (semantic) strategy
        runs. False disables it; a build failure degrades to None (knn no-ops).
      - `selection` (default "gate"): CascadeResult -> paper_ids policy. "gate"
        emits the accepted set (or survivors if no accept-gate) -- today's honest,
        can-be-empty precision result. "rank" is NEVER-empty: accepted first (by
        confidence), then the remaining survivors appended. "property_first"
        BYPASSES the cascade and orders the RAW router candidates via
        `provenance_rerank.property_first` (property-provenance first) -- the exact
        offline ordering that scored 0.531 paper-F1. "target_coverage" performs
        group-aware requested-target coverage. "answer_bearing" preserves that
        exact floor and permits only a high-confidence, constraint-attested
        single-paper rescue from real passage/object evidence.
      - `max_papers` (optional): hard cap on the multi-multiplicity list (None =>
        keep all). Single-multiplicity still keeps only the top-1.
      - `select_only` (default False): stop after paper_ids -- skip evidence
        localization AND answer generation (empty evidence/answer) for a cheap
        (~1 planner LLM call/question) paper-selection reproduction check.
      - `use_object_route` (default False): opt-in BM25 retrieval over the
        existing table/figure-caption index for table-answer or explicitly
        object-grounded questions. Omitted by production configs until gated.
      - `property_search_k` (default 30): per-query passage-chunk depth for the
        property and target-property BM25 routes. Experimental configs may raise
        it to distinguish ranking-depth misses from corpus-representation holes.
      - `evidence_confidence_floor` (default 0.5): the answer evidence gate.
      - CORPUS PDF SOURCE (one required): `corpus_pdf_dir` (local dir of
        `{paper_id}.pdf`) OR `corpus_gcs_bucket` (+ `corpus_pdf_prefix`, default
        "pdfs") for the GCS snapshot.
    """
    from littraceqa.corpus.indexes.build import load_indexes
    from littraceqa.experiments.new_runner import (
        assemble_new_runner,
        build_dense_retriever,
    )
    from littraceqa.llm.interfaces import build_llm
    from littraceqa.retrieval.pool import PoolIndex, load_pool

    params = config.params
    if index_dir is None:
        raise ValueError(
            "path='new' requires --index-dir (the persisted Level-2 corpus indexes)"
        )

    # ---- Build the FRESH per-call resources (this function's ONE job) ----
    indexes = load_indexes(index_dir)

    papers = load_pool(pool_path) if pool_path is not None else load_pool()
    pool = PoolIndex(papers)

    llm_name = params.get("llm_name", "gemini")
    llm_model = params.get("llm_model")
    llm = (
        build_llm(llm_name, model=llm_model)
        if llm_model
        else build_llm(llm_name)
    )

    # A role-specific evidence model leaves planner and answer behavior
    # untouched. Omitted => the historical shared client remains exact.
    evidence_llm = llm
    evidence_model = params.get("evidence_model")
    if evidence_model:
        evidence_llm = build_llm(
            llm_name, model=evidence_model
        )

    vision_model = params.get("vision_model", "gemini-2.5-pro")
    vision_llm = None
    if vision_model:
        from littraceqa.llm.gemini import GeminiClient

        vision_llm = GeminiClient(model=vision_model)

    reranker = None
    reranker_name = params.get("reranker")
    if reranker_name:
        from littraceqa.paperset.cascade import build_reranker

        reranker = build_reranker(reranker_name)

    # Dense/kNN semantic retriever: REUSE the old path's cached title+abstract
    # embeddings (`data/cache/pool_emb.npy`). Built here only when `use_dense`
    # (default ON); `build_dense_retriever` degrades to None on any failure. The
    # shared assembler applies the same `use_dense` gate as a single source.
    dense = (
        build_dense_retriever(pool) if params.get("use_dense", True) else None
    )

    # ---- Delegate the pure per-config ASSEMBLY to the shared constructor so
    # run_dev and run_ladder can never drift (selection/dense/cascade/etc.). ----
    return assemble_new_runner(
        config,
        indexes=indexes,
        pool=pool,
        llm=llm,
        evidence_llm=evidence_llm,
        vision_llm=vision_llm,
        reranker=reranker,
        dense=dense,
    )


def _load_inputs(path: str | pathlib.Path) -> list:
    """Load the input JSONL ONCE into parsed `InputRecord`s."""
    records = []
    with pathlib.Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(parse_input_record(json.loads(line)))
    return records


def _load_jsonl(path: str | pathlib.Path) -> list[dict]:
    with pathlib.Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m littraceqa.experiments.run_dev",
        description="Run a config against the 55 dev questions as a reproducible experiment.",
    )
    parser.add_argument("--config", required=True, help="Path to the run-config YAML.")
    parser.add_argument("--inputs", default=DEFAULT_INPUTS, help="Input JSONL (55 dev records).")
    parser.add_argument("--gold", default=DEFAULT_GOLD, help="Gold JSONL for scoring.")
    parser.add_argument(
        "--index-dir", default=None, help="Optional local index directory (Build-B hook)."
    )
    parser.add_argument(
        "--out-base", default=str(DEFAULT_BASE), help="Base dir for run directories."
    )
    parser.add_argument(
        "--pool-path", default=None, help="Override the paper-metadata pool path."
    )
    parser.add_argument(
        "--on-exist",
        choices=("suffix", "raise"),
        default="suffix",
        help="Behaviour when the run dir already exists non-empty.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent question workers (thread pool). 1 = serial (default); "
             ">1 overlaps the per-question Gemini/network waits. Artifacts stay "
             "byte-identical to serial (results reassembled in input order).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_PER_RECORD_TIMEOUT,
        help="Per-record wall-clock budget in seconds (parallel path only). A "
             "record still running past it becomes an isolated 'timeout' failure "
             "so one wedged question can never hang the whole run. Default "
             f"{DEFAULT_PER_RECORD_TIMEOUT:g}s (normal questions are ~90s).",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    _seed_everything(config.seed)

    # Load inputs ONCE (composition root).
    records = _load_inputs(args.inputs)
    gold_records = _load_jsonl(args.gold)

    manifest = build_manifest(config, pool_path=args.pool_path)
    run_dir = prepare_run_dir(
        args.out_base, manifest.git_commit, config.name, on_exist=args.on_exist
    )

    runner = make_runner(config, pool_path=args.pool_path, index_dir=args.index_dir)
    runner, batch_summary = prepare_batch_runner(
        runner,
        records,
        batch_selection=config.params.get("batch_selection"),
        min_records=config.params.get("batch_cohort_min_records", 6),
        min_votes=config.params.get("batch_cohort_min_votes", 2),
        max_papers=(
            config.params.get("batch_cohort_max_papers")
            or config.params.get("max_papers")
            or 5
        ),
        workers=args.workers,
        per_record_timeout=args.timeout,
    )
    if batch_summary is not None:
        print("Batch selection prepared:")
        print(json.dumps(batch_summary, indent=2, ensure_ascii=False))
    summary = run_experiment(
        runner, records, gold_records, run_dir, manifest,
        workers=args.workers, per_record_timeout=args.timeout,
    )

    print(f"Run dir: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
