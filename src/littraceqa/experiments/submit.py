"""Input-only production submitter for the ``path: new`` architecture.

Example::

    python -m littraceqa.experiments.submit \
      --config configs/test-candidate-target-aware.yaml \
      --index-dir data/indexes \
      --inputs data/test.jsonl \
      --output artifacts/test_predictions.jsonl \
      --source-revision <deployed-git-sha> \
      --trace-output artifacts/test_traces.jsonl \
      --workers 8

Unlike ``run_dev``, this command never asks for hidden gold.  It shares the
same runner factory/execution boundary, validates against the pinned official
validator, and only then writes the upload file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any

from littraceqa.corpus.indexes.build import (
    ALIASES_STORE,
    OBJECTS_STORE,
    PASSAGES_STORE,
    RELATIONS_STORE,
)
from littraceqa.experiments.cohort_consensus import prepare_batch_runner
from littraceqa.experiments.config import load_config
from littraceqa.experiments.driver import (
    DEFAULT_PER_RECORD_TIMEOUT,
    compute_records,
)
from littraceqa.experiments.manifest import git_commit
from littraceqa.experiments.run_dev import _seed_everything, make_runner
from littraceqa.pipeline.input import parse_input_record
from littraceqa.retrieval.embedder_local import DEFAULT_POOL_EMB_CACHE
from littraceqa.retrieval.pool import DEFAULT_POOL_PATH
from littraceqa.submission import write_submission
from littraceqa.validator import assert_valid_submission


DEFAULT_INPUTS = "data/test.jsonl"
MAIN_TEST_ROWS = 71
MAIN_TEST_MC_ROWS = 50
MAIN_TEST_TABLE_ROWS = 21
MAIN_POOL_IDS = 27_487
MAIN_TEST_SHA256 = "0fef8024b90360978b14a33f7103c8b5ea926a6572a20acc142d7c78b8f28196"
MAIN_POOL_SHA256 = "b498186449179daf4dfe6cb37d911010888b8d5a14fb3a2b38c1838fd1b62e5c"
_MAIN_QUERY_ID = re.compile(r"^ltqa_[0-9a-f]{16}$")
_SOURCE_REVISION = re.compile(r"^[0-9a-f]{7,40}$")
_INDEX_STORES = (PASSAGES_STORE, OBJECTS_STORE, ALIASES_STORE, RELATIONS_STORE)
MAIN_INDEX_RECORDS = {
    PASSAGES_STORE: 1_409_382,
    OBJECTS_STORE: 500_928,
    ALIASES_STORE: 3_711_590,
    RELATIONS_STORE: 492_892,
}
MAIN_INDEX_BYTES = {
    PASSAGES_STORE: 2_379_550_058,
    OBJECTS_STORE: 133_280_484,
    ALIASES_STORE: 1_005_318_119,
    RELATIONS_STORE: 274_310_431,
}


def _read_jsonl(path: str | pathlib.Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _load_paper_ids(path: str | pathlib.Path) -> set[str]:
    paper_ids: set[str] = set()
    for row in _read_jsonl(path):
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            paper_ids.add(paper_id)
    return paper_ids


def _sha256_file(path: str | pathlib.Path) -> str:
    digest = hashlib.sha256()
    with pathlib.Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_record_count(path: pathlib.Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def resolve_source_revision(value: str | None, *, required: bool) -> str:
    """Resolve an explicit/deployed source revision and validate git-SHA shape."""
    candidate = str(value or "").strip().lower()
    if not candidate:
        if required:
            raise ValueError(
                "strict submission requires --source-revision (7-40 lowercase hex); "
                "pass the commit used to package the VM source"
            )
        detected = git_commit().strip().lower()
        candidate = detected if _SOURCE_REVISION.fullmatch(detected) else ""
    if not candidate:
        return "unknown"
    if not _SOURCE_REVISION.fullmatch(candidate):
        raise ValueError("source revision must be a 7-40 character lowercase git SHA")
    return candidate


def enforce_strict_release_safety(
    *,
    strict_release_profile: bool,
    config_params: dict[str, Any],
    frozen_source: str | pathlib.Path | None,
) -> None:
    """Reject release paths whose clean ancestry cannot be proven.

    The public command supports full generation only. The pinned main-test
    profile therefore fails closed if a caller supplies an external parent.
    """
    if not strict_release_profile:
        return
    if frozen_source is not None:
        raise ValueError(
            "strict main-test release refuses component replay parents because "
            "no clean Git-bound ancestor-manifest verifier is available; "
            "use a full-generation release artifact"
        )


def write_submission_manifest(
    destination: str | pathlib.Path,
    *,
    source_revision: str,
    config,
    config_path: str | pathlib.Path,
    input_path: str | pathlib.Path,
    pool_path: str | pathlib.Path,
    output_path: str | pathlib.Path,
    trace_path: str | pathlib.Path | None,
    preflight: dict[str, Any],
    summary: dict[str, Any],
    frozen_predictions_path: str | pathlib.Path | None = None,
    frozen_trace_path: str | pathlib.Path | None = None,
    generation_mode: str | None = None,
) -> pathlib.Path:
    """Atomically write provenance for the exact validator-approved artifact."""
    source_revision = resolve_source_revision(source_revision, required=True)
    destination = pathlib.Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trace = pathlib.Path(trace_path) if trace_path is not None else None
    frozen = (
        pathlib.Path(frozen_predictions_path)
        if frozen_predictions_path is not None else None
    )
    frozen_trace = (
        pathlib.Path(frozen_trace_path)
        if frozen_trace_path is not None else None
    )
    active_keys = {
        "selection", "target_aware", "planner_max_attempts", "table_strategy",
        "evidence_localizer",
        "evidence_passage_k", "evidence_object_k", "evidence_emit_cap",
        "evidence_shortlist_k", "evidence_shortlist_max_pages",
        "evidence_rebalance_explicit_table",
        "evidence_contract_normalization",
        "evidence_cohort_coverage",
        "evidence_target_query",
        "evidence_target_query_max_chars",
        "evidence_target_unrelated_confidence",
        "evidence_query_mode",
        "evidence_model",
        "evidence_max_chars_per_page",
        "max_papers", "use_dense", "use_object_route",
        "table_container_selection", "cascade_stages",
        "llm_name", "llm_model", "property_search_k",
        "vision_model", "evidence_confidence_floor",
        "table_retain_concise_missing_expected_rows",
        "table_route_expected_rows_to_papers",
        "table_open_ended_one_row_per_paper",
        "table_prefer_owned_cells",
        "table_retain_source_attested_expected_rows",
        "table_visual_extraction_mode",
        "table_visual_retry_owned_rows",
        "table_visual_consensus_repeats",
        "table_extraction_sources",
        "table_text_context_mode",
        "table_text_page_k",
        "table_review_frozen_deltas",
        "table_verify_frozen_source_cells",
        "table_native_verify_max_pages",
        "table_native_verify_add_rows",
        "table_native_verify_substitute_rows",
        "table_native_verify_cell_updates",
        "table_native_verify_source_key_canonicalization",
        "table_fill_nulls_from_scalar_evidence",
        "table_trace_reassembly_inputs",
        "table_fact_producer", "table_fact_max_pages",
        "table_fact_allow_row_additions",
        "table_fact_preserve_unmatched_frozen",
        "table_fact_allow_cell_replacements",
        "table_fact_canonicalize_attestation_only_rows",
        "table_fact_cell_value_policy",
        "batch_selection", "batch_cohort_min_records",
        "batch_cohort_min_votes", "batch_cohort_max_papers",
    }
    created_at = datetime.now(timezone.utc).isoformat()
    output_sha256 = _sha256_file(output_path)
    trace_sha256 = (
        _sha256_file(trace) if trace is not None and trace.is_file() else None
    )
    run_id = hashlib.sha256(
        (
            f"{created_at}|{source_revision}|{output_sha256}|"
            f"{trace_sha256 or ''}"
        ).encode("utf-8")
    ).hexdigest()[:24]
    manifest = {
        "format_version": 1,
        "created_at": created_at,
        "run_id": run_id,
        "source_revision": source_revision,
        "config_name": config.name,
        "config_path": str(config_path),
        "config_sha256": _sha256_file(config_path),
        "active_params": {
            key: config.params[key] for key in sorted(active_keys)
            if key in config.params
        },
        "input_path": str(input_path),
        "input_sha256": _sha256_file(input_path),
        "pool_path": str(pool_path),
        "pool_sha256": _sha256_file(pool_path),
        "output_path": str(output_path),
        "output_sha256": output_sha256,
        "trace_path": str(trace) if trace is not None else None,
        "trace_sha256": trace_sha256,
        "generation_mode": generation_mode or "full",
        "frozen_predictions_path": str(frozen) if frozen is not None else None,
        "frozen_predictions_sha256": (
            _sha256_file(frozen)
            if frozen is not None and frozen.is_file() else None
        ),
        "frozen_trace_path": (
            str(frozen_trace) if frozen_trace is not None else None
        ),
        "frozen_trace_sha256": (
            _sha256_file(frozen_trace)
            if frozen_trace is not None and frozen_trace.is_file() else None
        ),
        "preflight": preflight,
        "summary": summary,
    }
    body = json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    temp_path: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=destination.parent,
            prefix=f".{destination.name}.", suffix=".tmp", delete=False,
        ) as handle:
            handle.write(body)
            temp_path = pathlib.Path(handle.name)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    return destination


def preflight_release_assets(
    raw_inputs: list[dict[str, Any]],
    paper_ids: set[str],
    index_dir: str | pathlib.Path,
    *,
    strict_release_profile: bool = True,
    input_path: str | pathlib.Path | None = None,
    pool_path: str | pathlib.Path | None = None,
    require_dense_cache: bool = False,
    pool_emb_cache: str | pathlib.Path = DEFAULT_POOL_EMB_CACHE,
) -> dict[str, Any]:
    """Fail fast on incomplete release assets before any model call.

    The pinned main split has a stable public profile.  Strict mode protects
    the expensive production command from an empty/wrong sync; relaxed mode is
    available for synthetic or future splits but still requires non-vacuous
    inputs, a paper pool, and all four non-empty local index stores.
    """
    if not raw_inputs:
        raise ValueError("no input records loaded (refusing a vacuous submission)")
    if not paper_ids:
        raise ValueError("paper pool is empty")

    query_ids = [str(row.get("query_id") or "").strip() for row in raw_inputs]
    if any(not query_id for query_id in query_ids):
        raise ValueError("input contains a missing query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("input contains duplicate query_ids")

    index_root = pathlib.Path(index_dir)
    store_sizes = {
        name: (index_root / name).stat().st_size
        for name in _INDEX_STORES
        if (index_root / name).is_file()
    }
    bad_stores = [name for name in _INDEX_STORES if store_sizes.get(name, 0) <= 0]
    if bad_stores:
        raise ValueError(
            "missing or empty index stores in "
            f"{index_root}: {', '.join(bad_stores)}"
        )

    dense_summary: dict[str, Any] | None = None
    if require_dense_cache:
        embedding_path = pathlib.Path(pool_emb_cache)
        ids_path = embedding_path.with_name(embedding_path.stem + ".ids.json")
        if not embedding_path.is_file() or not ids_path.is_file():
            raise ValueError(
                "dense embedding cache is incomplete; expected "
                f"{embedding_path} and {ids_path}"
            )
        try:
            import numpy as np

            embeddings = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            cache_meta = json.loads(ids_path.read_text(encoding="utf-8"))
            cached_ids = cache_meta.get("ids")
            valid_dense = (
                embeddings.ndim == 2
                and embeddings.shape[0] == len(paper_ids)
                and isinstance(cached_ids, list)
                and len(cached_ids) == len(paper_ids)
                and set(cached_ids) == paper_ids
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"dense embedding cache is unreadable: {exc}") from exc
        if not valid_dense:
            raise ValueError("dense embedding cache is not aligned to the released paper pool")
        dense_summary = {
            "path": str(embedding_path),
            "shape": list(embeddings.shape),
            "model_name": cache_meta.get("model_name"),
        }

    if strict_release_profile:
        if len(raw_inputs) != MAIN_TEST_ROWS:
            raise ValueError(
                f"expected {MAIN_TEST_ROWS} input rows for pinned main split; "
                f"found {len(raw_inputs)}"
            )
        if len(paper_ids) != MAIN_POOL_IDS:
            raise ValueError(
                f"expected {MAIN_POOL_IDS} distinct pool paper IDs; found {len(paper_ids)}"
            )
        bad_query_ids = [query_id for query_id in query_ids if not _MAIN_QUERY_ID.fullmatch(query_id)]
        if bad_query_ids:
            raise ValueError(
                "main-split query_id shape mismatch (first: "
                f"{bad_query_ids[0]!r})"
            )

        for row in raw_inputs:
            answer_types = row.get("answer_types")
            if answer_types not in (["multiple_choice"], ["table"]):
                raise ValueError(
                    f"{row.get('query_id')}: answer_types must be exactly "
                    "['multiple_choice'] or ['table']"
                )

        mc_rows = [
            row for row in raw_inputs
            if "multiple_choice" in (row.get("answer_types") or [])
        ]
        table_rows = [
            row for row in raw_inputs
            if "table" in (row.get("answer_types") or [])
        ]
        freeform_rows = [
            row for row in raw_inputs
            if "freeform" in (row.get("answer_types") or [])
        ]
        if (
            len(mc_rows) != MAIN_TEST_MC_ROWS
            or len(table_rows) != MAIN_TEST_TABLE_ROWS
            or freeform_rows
        ):
            raise ValueError(
                "main-split answer profile mismatch: expected "
                f"MC={MAIN_TEST_MC_ROWS}, table={MAIN_TEST_TABLE_ROWS}, freeform=0; "
                f"found MC={len(mc_rows)}, table={len(table_rows)}, "
                f"freeform={len(freeform_rows)}"
            )

        for row in mc_rows:
            options = row.get("multiple_choice_options")
            labels = {
                str(option.get("label") or "").strip().upper()
                for option in options
                if isinstance(option, dict)
            } if isinstance(options, list) else set()
            valid_options = (
                isinstance(options, list)
                and len(options) == 4
                and all(
                    isinstance(option, dict)
                    and set(option) >= {"label", "text"}
                    and isinstance(option.get("text"), str)
                    and bool(option["text"].strip())
                    for option in options
                )
            )
            if labels != {"A", "B", "C", "D"} or not valid_options:
                raise ValueError(
                    f"{row.get('query_id')}: expected four non-empty list-form MC options A-D"
                )

        for row in table_rows:
            schema = row.get("table_schema")
            valid_schema = (
                isinstance(schema, list)
                and bool(schema)
                and all(
                    isinstance(column, dict)
                    and isinstance(column.get("name"), str)
                    and bool(column["name"].strip())
                    and column.get("type") in {"string", "number", "boolean"}
                    for column in schema
                )
                and len({column["name"] for column in schema}) == len(schema)
                and sum(column.get("is_row_key") is True for column in schema) == 1
            )
            if not valid_schema:
                raise ValueError(f"{row.get('query_id')}: malformed table_schema")

        store_counts = {
            name: _jsonl_record_count(index_root / name) for name in _INDEX_STORES
        }
        if store_counts != MAIN_INDEX_RECORDS:
            raise ValueError(
                "index record counts do not match the pinned full-corpus build: "
                f"expected {MAIN_INDEX_RECORDS}, found {store_counts}"
            )
        if store_sizes != MAIN_INDEX_BYTES:
            raise ValueError(
                "index byte sizes do not match the pinned full-corpus build: "
                f"expected {MAIN_INDEX_BYTES}, found {store_sizes}"
            )
        if input_path is None or pool_path is None:
            raise ValueError(
                "strict release preflight requires input_path and pool_path for checksum attestation"
            )
        input_digest = _sha256_file(input_path)
        pool_digest = _sha256_file(pool_path)
        if input_digest != MAIN_TEST_SHA256:
            raise ValueError(
                f"released test input checksum mismatch: {input_digest}"
            )
        if pool_digest != MAIN_POOL_SHA256:
            raise ValueError(
                f"paper metadata checksum mismatch: {pool_digest}"
            )
    else:
        store_counts = None
        input_digest = None
        pool_digest = None

    return {
        "n_inputs": len(raw_inputs),
        "n_pool_papers": len(paper_ids),
        "index_bytes": sum(store_sizes.values()),
        "index_stores": store_sizes,
        "index_records": store_counts,
        "input_sha256": input_digest,
        "pool_sha256": pool_digest,
        "dense_cache": dense_summary,
        "strict_release_profile": strict_release_profile,
    }


def generate_submission(
    runner,
    raw_inputs: list[dict[str, Any]],
    output_path: str | pathlib.Path,
    *,
    paper_ids: set[str],
    workers: int = 1,
    per_record_timeout: float = DEFAULT_PER_RECORD_TIMEOUT,
    trace_path: str | pathlib.Path | None = None,
    require_evidence: bool = True,
) -> dict[str, Any]:
    """Run, officially validate, then atomically expose one upload file."""
    if not raw_inputs:
        raise ValueError("no input records loaded (refusing a vacuous submission)")
    if not paper_ids:
        raise ValueError("paper pool is empty")
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer >= 1")
    if not math.isfinite(per_record_timeout) or per_record_timeout <= 0:
        raise ValueError("per-record timeout must be a positive finite number")
    query_ids = [str(row.get("query_id") or "").strip() for row in raw_inputs]
    if any(not query_id for query_id in query_ids):
        raise ValueError("input contains a missing query_id")
    if len(set(query_ids)) != len(query_ids):
        raise ValueError("input contains duplicate query_ids")
    records = [parse_input_record(row) for row in raw_inputs]
    records, computed = compute_records(
        runner,
        records,
        workers=workers,
        per_record_timeout=per_record_timeout,
    )
    predictions = [line for line, _trace, _failure in computed]
    traces = [trace for _line, trace, _failure in computed]
    failures = [record.query_id for record, result in zip(records, computed) if result[2]]

    # This is the load-bearing order: the official nested/pool/coverage checks
    # run before write_submission makes the upload path visible.
    assert_valid_submission(
        raw_inputs,
        predictions,
        require_evidence=require_evidence,
        paper_ids=paper_ids,
    )
    if trace_path is not None:
        trace_destination = pathlib.Path(trace_path)
        trace_destination.parent.mkdir(parents=True, exist_ok=True)
        trace_destination.write_text(
            "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
            encoding="utf-8",
        )
    write_submission(
        predictions,
        output_path,
        input_query_ids=[row.get("query_id") for row in raw_inputs],
    )
    return {"n_records": len(records), "n_failures": len(failures)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m littraceqa.experiments.submit",
        description="Generate and officially validate a path:new LitTraceQA submission.",
    )
    parser.add_argument("--config", required=True, help="Path:new run-config YAML.")
    parser.add_argument("--index-dir", required=True, help="Persisted corpus index directory.")
    parser.add_argument("--inputs", default=DEFAULT_INPUTS, help="Released input JSONL.")
    parser.add_argument("--output", required=True, help="Upload JSONL to create.")
    parser.add_argument(
        "--pool-path", default=str(DEFAULT_POOL_PATH), help="Released paper metadata JSONL."
    )
    parser.add_argument(
        "--pool-emb-cache",
        default=str(DEFAULT_POOL_EMB_CACHE),
        help="Cached pool embedding .npy used when config use_dense is true.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_PER_RECORD_TIMEOUT,
        help="Per-record wall-clock budget in seconds (parallel execution).",
    )
    parser.add_argument("--trace-output", default=None, help="Optional trace JSONL path.")
    parser.add_argument(
        "--manifest-output",
        default=None,
        help="Provenance JSON path (default: <output>.manifest.json).",
    )
    parser.add_argument(
        "--source-revision",
        default=os.getenv("LITTRACEQA_SOURCE_REVISION"),
        help=(
            "Git revision used to package the deployed source. Required by the "
            "strict release profile (or set LITTRACEQA_SOURCE_REVISION)."
        ),
    )
    parser.add_argument(
        "--require-evidence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require the evidence field (true for the main test split).",
    )
    parser.add_argument(
        "--strict-release-profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Require the pinned 71-row/27,487-paper main-split profile. "
            "Disable only for a different released or synthetic split."
        ),
    )
    args = parser.parse_args(argv)

    manifest_destination = args.manifest_output or f"{args.output}.manifest.json"
    protected_paths = {
        "config": args.config,
        "inputs": args.inputs,
        "pool-path": args.pool_path,
        "pool-emb-cache": args.pool_emb_cache,
        "output": args.output,
        "trace-output": args.trace_output,
        "manifest-output": manifest_destination,
    }
    resolved_paths: dict[pathlib.Path, str] = {}
    for label, value in protected_paths.items():
        if value is None:
            continue
        resolved = pathlib.Path(value).resolve()
        previous = resolved_paths.get(resolved)
        if previous is not None:
            parser.error(
                f"path collision: {label} must differ from {previous} ({resolved})"
            )
        resolved_paths[resolved] = label

    config = load_config(args.config)
    if config.path != "new":
        parser.error("production submit requires config path: new")
    if config.params.get("select_only"):
        parser.error("select_only is a paper-ablation mode and cannot produce a submission")
    try:
        enforce_strict_release_safety(
            strict_release_profile=args.strict_release_profile,
            config_params=config.params,
            frozen_source=None,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be a positive finite number")
    if (
        args.strict_release_profile
        and config.params.get("table_strategy") in {
            "planned_native", "planned_visual_fill"
        }
        and args.timeout < 1200
    ):
        table_strategy = config.params["table_strategy"]
        parser.error(
            f"strict {table_strategy} release requires --timeout >= 1200 so "
            "the internal source-consensus stage cannot erase a completed "
            "base answer at the driver timeout boundary"
        )
    try:
        source_revision = resolve_source_revision(
            args.source_revision, required=args.strict_release_profile
        )
    except ValueError as exc:
        parser.error(str(exc))

    _seed_everything(config.seed)
    raw_inputs = _read_jsonl(args.inputs)
    paper_ids = _load_paper_ids(args.pool_path)
    try:
        preflight = preflight_release_assets(
            raw_inputs,
            paper_ids,
            args.index_dir,
            strict_release_profile=args.strict_release_profile,
            input_path=args.inputs,
            pool_path=args.pool_path,
            require_dense_cache=config.params.get("use_dense", True),
            pool_emb_cache=args.pool_emb_cache,
        )
    except ValueError as exc:
        parser.error(str(exc))
    print("Preflight passed:")
    print(json.dumps(preflight, indent=2))
    batch_summary: dict[str, Any] | None = None
    runner = make_runner(
        config, pool_path=args.pool_path, index_dir=args.index_dir
    )
    runner, batch_summary = prepare_batch_runner(
        runner,
        [parse_input_record(row) for row in raw_inputs],
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
    summary = generate_submission(
        runner,
        raw_inputs,
        args.output,
        paper_ids=paper_ids,
        workers=args.workers,
        per_record_timeout=args.timeout,
        trace_path=args.trace_output,
        require_evidence=args.require_evidence,
    )
    if batch_summary is not None:
        summary["batch_selection"] = batch_summary
    manifest_output = manifest_destination
    write_submission_manifest(
        manifest_output,
        source_revision=source_revision,
        config=config,
        config_path=args.config,
        input_path=args.inputs,
        pool_path=args.pool_path,
        output_path=args.output,
        trace_path=args.trace_output,
        preflight=preflight,
        summary=summary,
        frozen_predictions_path=None,
        frozen_trace_path=None,
        generation_mode="full",
    )
    print(f"Wrote validator-approved submission to {args.output}")
    print(f"Wrote submission provenance to {manifest_output}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
