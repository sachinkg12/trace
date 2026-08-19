"""`run_experiment`: the harness driver.

Given a `DevRunner`, the input records, gold records, a prepared run dir, and a
captured `RunManifest`, it writes the FULL run: manifest.json, env-lock.txt,
predictions.jsonl (streamed, one line per query_id -- NONE dropped),
traces.jsonl (1:1 with predictions), scores.json (the in-process evaluator
output), summary.json (macro metrics + totals), and run.log.

Failure isolation: if a runner raises on one record, that query_id still gets a
VALID blank submission line with `failure_reason` set, and the run continues.
"""
from __future__ import annotations

import json
import logging
import pathlib
import threading
import time
import weakref
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from concurrent.futures.thread import _worker
from typing import Any

from littraceqa.experiments.manifest import (
    RunManifest,
    capture_env_lock,
    write_manifest,
)
from littraceqa.experiments.runner import DevRunner, make_trace
from littraceqa.experiments.rundir import (
    LOG_NAME,
    PREDICTIONS_NAME,
    SCORES_NAME,
    SUMMARY_NAME,
    TRACES_NAME,
    _is_nonempty_dir,
)
from littraceqa.pipeline.input import InputRecord
from littraceqa.scorer import evaluate_full
from littraceqa.submission import build_fallback_line, repair_line

# Default per-record wall-clock budget (seconds) for the parallel path. A normal
# question is ~90s; 300s is generous headroom while still guaranteeing that ONE
# wedged record can never hold the whole run hostage (FIX B).
DEFAULT_PER_RECORD_TIMEOUT = 300.0


class _DaemonThreadPoolExecutor(ThreadPoolExecutor):
    """A `ThreadPoolExecutor` whose worker threads are daemon AND are NOT
    registered in `concurrent.futures`' global atexit join table.

    Why this exists (FIX B): a runner that hangs forever leaves its worker
    thread stuck. The stdlib pool creates NON-daemon threads and registers each
    in `concurrent.futures.thread._threads_queues`, whose atexit handler
    (`_python_exit`) joins EVERY such thread at interpreter shutdown -- so a
    single hung worker makes the whole process un-exitable (produces no output
    and never dies). We override thread creation to (a) mark threads `daemon`
    and (b) skip that global registration, so:

      * our own `shutdown(wait=False)` returns immediately (never joins), and
      * at interpreter exit the hung worker is a daemon thread the runtime
        reclaims instead of joining -- the process can actually terminate.

    `self._threads` is still populated, so an ordinary `shutdown(wait=True)`
    would still work; we simply never rely on it here. This mirrors CPython
    3.12's `_adjust_thread_count`; it is pinned to this interpreter's internals
    by design and documented as such.
    """

    def _adjust_thread_count(self) -> None:  # pragma: no cover - exercised via the pool
        if self._idle_semaphore.acquire(timeout=0):
            return

        def weakref_cb(_, q=self._work_queue):
            q.put(None)

        if len(self._threads) < self._max_workers:
            thread_name = "%s_%d" % (self._thread_name_prefix or self, len(self._threads))
            t = threading.Thread(
                name=thread_name,
                target=_worker,
                args=(weakref.ref(self, weakref_cb), self._work_queue,
                      self._initializer, self._initargs),
                daemon=True,  # a hung worker must never block interpreter exit
            )
            t.start()
            self._threads.add(t)
            # Deliberately NOT added to concurrent.futures.thread._threads_queues:
            # that skips the global atexit join so a lingering hung daemon worker
            # cannot wedge process shutdown.


# The macro metrics surfaced in summary.json (the full 11 live in scores.json).
_SUMMARY_METRICS = (
    "paper_f1_macro",
    "evidence_f1_macro",
    "multiple_choice_accuracy",
    "freeform_exact_match",
    "table_row_f1_macro",
    "table_cell_accuracy_micro",
)


def _make_logger(run_dir: pathlib.Path) -> tuple[logging.Logger, logging.FileHandler]:
    logger = logging.getLogger(f"littraceqa.experiments.run.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
    handler = logging.FileHandler(run_dir / LOG_NAME, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger, handler


def _as_float(x: Any) -> float:
    """Coerce a trace numeric to float, treating a missing/non-numeric field as
    0.0. A single bad `cost`/`latency_s` value must NEVER abort the run or drop
    a query_id, so summation is done through this guard everywhere."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def _dedup_records(records: list[InputRecord], logger: logging.Logger) -> list[InputRecord]:
    """Enforce one-distinct-query_id-per-run at the driver boundary.

    Duplicate input query_ids would otherwise emit duplicate prediction lines
    (which `write_submission`'s guard rejects and the scorer double-counts). We
    de-dup here rather than at load so the invariant holds no matter who calls
    the driver; the FIRST occurrence is kept and input order is preserved, so no
    DISTINCT query_id is ever dropped."""
    seen: set[str] = set()
    out: list[InputRecord] = []
    for record in records:
        qid = record.query_id
        if qid in seen:
            logger.warning("duplicate input query_id %s: dropping repeat (keeping first)", qid)
            continue
        seen.add(qid)
        out.append(record)
    return out


def _safe_line(line: Any, record: InputRecord) -> dict:
    """Coerce a runner's line into a valid submission line for ``record``.

    Guards the never-drop and nested answer contract at the write boundary;
    malformed components degrade only that query rather than corrupting or
    dropping the whole submission."""
    return repair_line(record, line)


def _line_changed(original: Any, repaired: dict) -> bool:
    """Best-effort equality check used only for repair observability."""
    try:
        return bool(original != repaired)
    except Exception:  # e.g. array-valued dict equality is ambiguous
        return True


def _failure_line(runner: DevRunner, record: InputRecord) -> dict:
    """Return the safest runner-aware line for an exception or timeout.

    Replay runners promise that components outside their replay scope remain
    frozen. A generic placeholder would violate that promise on the exact rows
    most likely to time out. Runners may therefore expose ``failure_line``;
    this boundary still repairs/validates it and fails closed to the historical
    placeholder if the hook is absent, raises, or returns malformed data.
    """
    factory = getattr(runner, "failure_line", None)
    if callable(factory):
        try:
            candidate = factory(record)
            repaired = _safe_line(candidate, record)
            if not _line_changed(candidate, repaired):
                return repaired
        except KeyboardInterrupt:
            raise
        except BaseException:  # noqa: BLE001 -- fallback construction is total
            pass
    return build_fallback_line(record)


def _compute_one(
    runner: DevRunner, record: InputRecord
) -> tuple[dict, dict, str | None]:
    """Run ONE record through `runner`, returning `(line, trace, failure_reason)`
    with the FULL never-drop / line-repair contract applied.

    This is the exact body the serial loop used, extracted so the thread pool can
    call it per record: a raising `run_one` degrades to a valid fallback line + an
    `error` trace for that query_id (never propagates, never drops the qid). It
    builds only per-call state, so it is safe to invoke concurrently for distinct
    records against the same read-only runner (indexes/pool/llm are call-stateless;
    the one shared local-inference component -- the cross-encoder reranker -- is
    lock-guarded inside `CrossEncoderRerankStage`)."""
    qid = record.query_id
    started = time.perf_counter()
    try:
        outcome = runner.run_one(record)
        line = _safe_line(outcome.line, record)
        trace = dict(outcome.trace or make_trace(qid, "unknown", 0.0))
        failure_reason = trace.get("failure_reason")
        if not failure_reason and _line_changed(outcome.line, line):
            failure_reason = "submission contract repaired"
            trace["failure_reason"] = failure_reason
    except KeyboardInterrupt:
        # A deliberate operator abort must still propagate and stop the batch --
        # never swallow Ctrl-C into a single record's failure line.
        raise
    except BaseException as exc:  # noqa: BLE001 -- isolate ANY per-record failure
        # Catch BaseException (not just Exception): a runner raising e.g.
        # SystemExit must degrade to THIS qid's fallback line, never abort the batch
        # and drop the not-yet-written qids (FIX A).
        line = _failure_line(runner, record)
        failure_reason = f"{type(exc).__name__}: {exc}"
        trace = make_trace(
            qid, "error", time.perf_counter() - started, failure_reason=failure_reason
        )
    return line, trace, failure_reason


def _log_progress(
    logger: logging.Logger,
    qid: str,
    trace: dict,
    failure_reason: str | None,
    lock: threading.Lock | None = None,
) -> None:
    """Emit the one progress line for a finished record. `lock` (used only by the
    thread pool) serializes the log call so concurrent progress lines never
    interleave-corrupt in run.log."""
    def _emit() -> None:
        if failure_reason:
            logger.warning("query %s FAILED: %s", qid, failure_reason)
        else:
            logger.info("query %s ok (%.3fs)", qid, _as_float(trace.get("latency_s")))

    if lock is not None:
        with lock:
            _emit()
    else:
        _emit()


def _compute_one_logged(
    runner: DevRunner, record: InputRecord, logger: logging.Logger
) -> tuple[dict, dict, str | None]:
    """Serial helper: compute one record and log its progress inline (preserving
    the historical interleaved logging of the serial path)."""
    line, trace, failure_reason = _compute_one(runner, record)
    _log_progress(logger, record.query_id, trace, failure_reason)
    return line, trace, failure_reason


def _compute_parallel(
    runner: DevRunner,
    records: list[InputRecord],
    workers: int,
    logger: logging.Logger,
    per_record_timeout: float,
) -> list[tuple[dict, dict, str | None]]:
    """Run `run_one` for every record concurrently on a daemon thread pool,
    returning `(line, trace, failure_reason)` REASSEMBLED IN INPUT ORDER.

    The parallelism win is the overlapping Gemini/network waits; local work stays
    correct because `_compute_one` builds only per-call state.

    Isolation of a HUNG record (FIX B): every worker records its own start time;
    the coordinator watches all futures together and expires each at one fixed
    deadline.  Timeouts therefore neither reset while earlier input-order
    futures are collected nor accumulate N times. Results are still stored and
    returned in input order. If every worker is irrecoverably hung, queued work
    is failed explicitly as pool-starved instead of waiting forever.

    Process-exit safety: the pool is a `_DaemonThreadPoolExecutor` (daemon workers,
    not in the global atexit join table) and we call `shutdown(wait=False)` in a
    `finally`, so a worker still wedged in a hung `run_one` can neither block this
    function's return nor block interpreter shutdown -- the run produces its output
    and the process can actually exit.

    A future raising some other `BaseException` (it shouldn't -- `_compute_one`
    catches all but KeyboardInterrupt) still degrades to a fallback line for that qid.
    Completion logs may arrive out of input order; returned artifacts never do."""
    results: list[tuple[dict, dict, str | None] | None] = [None] * len(records)
    timings: list[dict[str, float | None]] = [
        {"started_at": None, "finished_at": None} for _ in records
    ]
    log_lock = threading.Lock()
    pool = _DaemonThreadPoolExecutor(max_workers=workers)
    pools = [pool]

    def timed_compute(index: int) -> tuple[dict, dict, str | None]:
        timings[index]["started_at"] = time.perf_counter()
        try:
            return _compute_one(runner, records[index])
        finally:
            timings[index]["finished_at"] = time.perf_counter()

    def timeout_result(index: int, reason: str) -> tuple[dict, dict, str]:
        qid = records[index].query_id
        line = _failure_line(runner, records[index])
        trace = make_trace(
            qid, "error", per_record_timeout, failure_reason=reason
        )
        return line, trace, reason

    def record_result(index: int, result: tuple[dict, dict, str | None]) -> None:
        results[index] = result
        _line, trace, failure_reason = result
        _log_progress(
            logger,
            records[index].query_id,
            trace,
            failure_reason,
            lock=log_lock,
        )

    try:
        futures = [pool.submit(timed_compute, i) for i in range(len(records))]
        pending = set(range(len(records)))
        timed_out_running: set[int] = set()

        while pending:
            now = time.perf_counter()
            progressed = False

            for i in list(pending):
                fut = futures[i]
                started_at = timings[i]["started_at"]
                finished_at = timings[i]["finished_at"]

                if fut.done():
                    try:
                        result = fut.result()
                        elapsed = (
                            finished_at - started_at
                            if started_at is not None and finished_at is not None
                            else 0.0
                        )
                        if elapsed > per_record_timeout:
                            result = timeout_result(
                                i, f"timeout after {per_record_timeout:g}s"
                            )
                    except KeyboardInterrupt:
                        raise
                    except BaseException as exc:  # noqa: BLE001 -- defensive
                        reason = f"{type(exc).__name__}: {exc}"
                        line = _failure_line(runner, records[i])
                        trace = make_trace(
                            records[i].query_id, "error", 0.0,
                            failure_reason=reason,
                        )
                        result = (line, trace, reason)
                    record_result(i, result)
                    pending.remove(i)
                    progressed = True
                    continue

                if (
                    started_at is not None
                    and now - started_at >= per_record_timeout
                ):
                    fut.cancel()  # best-effort; running calls remain daemonized
                    reason = f"timeout after {per_record_timeout:g}s"
                    record_result(i, timeout_result(i, reason))
                    timed_out_running.add(i)
                    pending.remove(i)
                    progressed = True

            if not pending:
                break

            # A timed-out thread cannot be killed. If every slot in this pool is
            # occupied by one, recycle the still-queued rows onto a fresh daemon
            # pool.  The timed-out calls may linger, but they must not cause
            # runnable questions to be falsely discarded as "starved".
            active_timed_out = {
                i for i in timed_out_running if not futures[i].done()
            }
            if len(active_timed_out) >= workers:
                queued = [i for i in pending if timings[i]["started_at"] is None]
                if queued:
                    replacement = _DaemonThreadPoolExecutor(max_workers=workers)
                    pools.append(replacement)
                    for i in queued:
                        futures[i].cancel()
                        timings[i] = {"started_at": None, "finished_at": None}
                        futures[i] = replacement.submit(timed_compute, i)
                    timed_out_running.clear()
                    progressed = True

            if pending and not progressed:
                deadlines = [
                    started + per_record_timeout
                    for i in pending
                    if (started := timings[i]["started_at"]) is not None
                ]
                wait_for = max(0.0, min(deadlines) - time.perf_counter()) if deadlines else 0.01
                wait(
                    [futures[i] for i in pending],
                    timeout=min(wait_for, 0.05),
                    return_when=FIRST_COMPLETED,
                )
    finally:
        # NEVER join: a hung worker must not wedge our return or process exit.
        for active_pool in pools:
            active_pool.shutdown(wait=False)
    # No None survives: every index was assigned exactly once.
    return [r for r in results if r is not None]


def compute_records(
    runner: DevRunner,
    records: list[InputRecord],
    *,
    workers: int = 1,
    per_record_timeout: float = DEFAULT_PER_RECORD_TIMEOUT,
    logger: logging.Logger | None = None,
) -> tuple[list[InputRecord], list[tuple[dict, dict, str | None]]]:
    """Execute records with the shared never-drop/timeout/repair contract.

    Both the gold-scored dev harness and the input-only production submission
    command call this function, preventing the two paths from drifting at the
    most dangerous boundary.  Returned results are in deduplicated input order.
    """
    active_logger = logger or logging.getLogger(__name__)
    deduped = _dedup_records(records, active_logger)
    if workers is not None and workers > 1:
        computed = _compute_parallel(
            runner, deduped, workers, active_logger, per_record_timeout
        )
    else:
        computed = [
            _compute_one_logged(runner, record, active_logger)
            for record in deduped
        ]
    return deduped, computed


def run_experiment(
    runner: DevRunner,
    records: list[InputRecord],
    gold_records: list[dict],
    run_dir: str | pathlib.Path,
    manifest: RunManifest,
    workers: int = 1,
    per_record_timeout: float = DEFAULT_PER_RECORD_TIMEOUT,
) -> dict[str, Any]:
    """Drive `runner` over `records`, persist all artifacts into `run_dir`, and
    score in-process against `gold_records`. Returns the summary dict.

    Deterministic order: records are processed in the order given.

    `workers`: with `workers <= 1` the run is serial and byte-identical to the
    historical behaviour. With `workers > 1` each record's `run_one` is dispatched
    to a `ThreadPoolExecutor(max_workers=workers)` so the per-question Gemini /
    network waits overlap; results are REASSEMBLED IN INPUT ORDER before any
    aggregation or write, so predictions.jsonl / traces.jsonl (and the float
    cost/latency summation) are byte-identical to the serial run for a
    deterministic, order-preserving runner. Dedup of duplicate input query_ids,
    per-record failure isolation, and the never-drop-a-qid guarantee all hold
    unchanged under the pool.

    `per_record_timeout` (parallel path only): the wall-clock budget for a single
    record. A record still running past it becomes an ISOLATED `timeout` failure
    (fallback line for that qid) while every other qid is written; a hung worker can
    neither wedge the run nor block process exit (daemon pool + shutdown(wait=False)).
    """
    run_dir = pathlib.Path(run_dir)
    # No-overwrite guarantee enforced HERE too, not only by the CLI's
    # prepare_run_dir: a reproducible artifact must never be silently clobbered.
    if _is_nonempty_dir(run_dir):
        raise FileExistsError(
            f"run_dir already exists and is non-empty: {run_dir} (refusing to overwrite)"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    logger, handler = _make_logger(run_dir)

    records = _dedup_records(records, logger)

    write_manifest(manifest, run_dir)
    capture_env_lock(run_dir)
    logger.info("run start: %d records, commit=%s", len(records), manifest.git_commit)

    predictions: list[dict] = []
    total_cost = 0.0
    total_latency = 0.0
    failures: list[str] = []

    # Compute every record's (line, trace, failure_reason). Serial stays LAZY
    # so a long dev run streams each completed record to disk; parallel uses
    # the shared bounded executor and reassembles in input order.
    if workers is not None and workers > 1:
        computed = _compute_parallel(
            runner, records, workers, logger, per_record_timeout
        )
        iterator: Any = zip(records, computed)
    else:
        iterator = (
            (record, _compute_one_logged(runner, record, logger))
            for record in records
        )

    pred_path = run_dir / PREDICTIONS_NAME
    trace_path = run_dir / TRACES_NAME
    try:
        with pred_path.open("w", encoding="utf-8") as pf, trace_path.open(
            "w", encoding="utf-8"
        ) as tf:
            for record, (line, trace, failure_reason) in iterator:
                qid = record.query_id
                if failure_reason:
                    failures.append(qid)

                total_cost += _as_float(trace.get("cost"))
                total_latency += _as_float(trace.get("latency_s"))

                predictions.append(line)
                pf.write(json.dumps(line, ensure_ascii=False) + "\n")
                tf.write(json.dumps(trace, ensure_ascii=False) + "\n")

        # In-process scoring via the vendored evaluator (metrics + details).
        scores = evaluate_full(gold_records, predictions)
        (run_dir / SCORES_NAME).write_text(
            json.dumps(scores, indent=2, ensure_ascii=False)
        )

        metrics = scores.get("metrics", {})
        summary = {
            "name": run_dir.name,
            "n_records": len(records),
            "n_failures": len(failures),
            "failures": failures,
            "total_cost": total_cost,
            "total_latency_s": total_latency,
            "macro_metrics": {k: metrics.get(k) for k in _SUMMARY_METRICS},
        }
        (run_dir / SUMMARY_NAME).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        logger.info("run complete: %s", json.dumps(summary["macro_metrics"]))
        logger.info(
            "totals: cost=%.4f latency_s=%.2f failures=%d",
            total_cost,
            total_latency,
            len(failures),
        )
        return summary
    finally:
        handler.close()
        logger.removeHandler(handler)
