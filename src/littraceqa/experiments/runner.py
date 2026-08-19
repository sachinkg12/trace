"""The `DevRunner` seam + the rung-1 `OldPathRunner` adapter.

`DevRunner` is the single Protocol the driver depends on: given one parsed
`InputRecord`, produce a `RunOutcome` = (submission `line`, per-question
`trace`). `OldPathRunner` wraps the OLD `Pipeline` (from `build_pipeline`).
The follow-up planner/cascade path (Build B) implements the SAME Protocol and
plugs into the SAME driver + factory -- nothing in `driver.py` changes.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from littraceqa.pipeline.input import InputRecord
from littraceqa.submission import build_fallback_line


@dataclass
class RunOutcome:
    """One question's result.

    - line: the submission line (official sample shape, via `build_line`) --
      exactly one per query_id, NEVER dropped.
    - trace: per-question provenance -- at minimum `query_id`, `path`,
      `candidates`, `cost`, `latency_s`, `failure_reason` (None on success,
      else the caught error string). Build B adds cascade stage reasons /
      confidences under the same dict.
    """

    line: dict[str, Any]
    trace: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DevRunner(Protocol):
    """Produce a `RunOutcome` for one input record. The ONLY seam the driver
    knows about; every path (old/new/union) implements exactly this."""

    def run_one(self, record: InputRecord) -> RunOutcome: ...


def make_trace(
    query_id: str,
    path: str,
    latency_s: float,
    *,
    failure_reason: str | None = None,
    candidates: list[Any] | None = None,
    cost: float = 0.0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Canonical per-question trace shape, shared by every runner path so the
    `traces.jsonl` schema stays uniform as Build B adds cascade fields."""
    trace: dict[str, Any] = {
        "query_id": query_id,
        "path": path,
        "candidates": list(candidates) if candidates else [],
        "cost": float(cost),
        "latency_s": float(latency_s),
        "failure_reason": failure_reason,
    }
    if extra:
        trace.update(extra)
    return trace


class OldPathRunner:
    """Rung-1 adapter: runs a record through the OLD `Pipeline`.

    `pipeline` is anything exposing `run(InputRecord) -> dict` (the real
    orchestrator, or a fake in tests) -- this class never names a concrete
    pipeline. It times each call and, defensively, catches any exception the
    pipeline itself failed to swallow, degrading to a validator-safe fallback
    line for that query_id (a dropped query_id scores 0 precision) with
    `failure_reason` set.
    """

    def __init__(self, pipeline: Any):
        self._pipeline = pipeline

    def run_one(self, record: InputRecord) -> RunOutcome:
        start = time.perf_counter()
        failure_reason: str | None = None
        try:
            line = self._pipeline.run(record)
        except Exception as exc:  # noqa: BLE001 -- never drop a query_id
            line = build_fallback_line(record)
            failure_reason = f"{type(exc).__name__}: {exc}"
        latency_s = time.perf_counter() - start
        trace = make_trace(
            record.query_id, "old", latency_s, failure_reason=failure_reason
        )
        return RunOutcome(line=line, trace=trace)
