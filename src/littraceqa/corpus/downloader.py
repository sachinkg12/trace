"""Resumable, reproducibility-first bulk corpus downloader.

Iterates `paper_metadata.jsonl`, downloads each paper's PDF (`pdf_url`) into a
configurable `dest_dir`, and records a checksummed row per paper in
`manifest.jsonl`. The manifest is the resume source of truth: on restart,
papers already `downloaded` are skipped (the fetcher is not even called),
`failed`/`pending` papers are retried. One bad URL is recorded and the run
continues -- a failure here just falls back to on-demand fetch at answer time.

Seams (all injectable, all OCP):
  * `StorageBackend`  -- where bytes land (LocalDirBackend now; GCS later).
  * `PdfFetcher`      -- how bytes are fetched (reuse LocalPdfFetcher /
                         OpenReviewFetcher from the localize layer). Resolved
                         PER HOST via `host_fetchers`, so openreview.net routes
                         through the injected OpenReviewFetcher path while other
                         hosts use the default fetcher.
  * `HostPolicy`      -- per-host politeness (min_interval spacing) + retry
                         with exponential backoff.
  * `sleep` / `now`   -- injected clock so tests run with no real delay.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol, runtime_checkable
from urllib.parse import urlparse

from littraceqa.localize.interfaces import PdfFetcher

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DEST_DIR = REPO_ROOT / "data" / "corpus"
MANIFEST_NAME = "manifest.jsonl"

# Canonical manifest column order -- stable so a diff of the frozen snapshot's
# manifest is human-readable and reproducible.
MANIFEST_FIELDS = (
    "paper_id", "url", "host", "status",
    "sha256", "bytes", "http_status", "attempts", "error",
)

STATUS_DOWNLOADED = "downloaded"
STATUS_FAILED = "failed"
STATUS_PENDING = "pending"
# A paper is skipped on resume ONLY when already downloaded; everything else
# (failed / pending / absent) is retried.
_RESUME_SKIP = frozenset({STATUS_DOWNLOADED})


def host_of(url: str | None) -> str:
    """Lowercased network host of a URL (`""` if missing/unparseable). Single
    host-extraction point for the whole subsystem, so grouping/routing/policy
    lookups can never disagree on what host a URL belongs to."""
    if not url:
        return ""
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:  # noqa: BLE001 -- a malformed URL must never crash the run
        return ""


# --------------------------------------------------------------------------- #
# Storage seam (OCP): swap LocalDirBackend for a GcsBackend later, no edits to
# CorpusDownloader.
# --------------------------------------------------------------------------- #
@runtime_checkable
class StorageBackend(Protocol):
    def exists(self, paper_id: str) -> bool: ...
    def write(self, paper_id: str, data: bytes) -> None: ...


class LocalDirBackend:
    """Writes each PDF to `<dest_dir>/{paper_id}.pdf` (same on-disk layout as
    the localize-layer cache, so the corpus doubles as an answer-time cache)."""

    def __init__(self, dest_dir: str | pathlib.Path):
        self._dest = pathlib.Path(dest_dir)

    def _path(self, paper_id: str) -> pathlib.Path:
        return self._dest / f"{paper_id}.pdf"

    def exists(self, paper_id: str) -> bool:
        return self._path(paper_id).exists()

    def write(self, paper_id: str, data: bytes) -> None:
        self._dest.mkdir(parents=True, exist_ok=True)
        # Atomic within the dir: write to a temp sibling, then rename, so a kill
        # mid-write never leaves a truncated `.pdf` masquerading as complete.
        tmp = self._path(paper_id).with_suffix(".pdf.part")
        tmp.write_bytes(data)
        os.replace(tmp, self._path(paper_id))


# --------------------------------------------------------------------------- #
# Per-host politeness + retry policy.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HostPolicy:
    """Politeness + retry knobs for one host. `min_interval` spaces successive
    requests to the SAME host (serial processing already caps concurrency at 1
    per host by construction). `backoff_base * 2**(attempt-1)` seconds are slept
    between retries."""
    max_attempts: int = 3
    backoff_base: float = 1.0
    min_interval: float = 1.0


DEFAULT_POLICY = HostPolicy()


# --------------------------------------------------------------------------- #
# Manifest: append-only JSONL log; latest row per paper_id wins on reload.
# --------------------------------------------------------------------------- #
def _manifest_row(paper_id, url, host, status, sha256, nbytes, http_status,
                  attempts, error) -> dict:
    return {
        "paper_id": paper_id, "url": url, "host": host, "status": status,
        "sha256": sha256, "bytes": nbytes, "http_status": http_status,
        "attempts": attempts, "error": error,
    }


class Manifest:
    """Append-safe JSONL manifest. Each finalized paper appends one complete
    line (flush + fsync), so a kill mid-run can at worst drop the last line --
    never corrupt earlier ones. On reload, corrupt/partial trailing lines are
    skipped and the LAST row per paper_id wins (append-log semantics)."""

    def __init__(self, path: str | pathlib.Path):
        self.path = pathlib.Path(path)

    def load(self) -> dict[str, dict]:
        rows: dict[str, dict] = {}
        if not self.path.exists():
            return rows
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn trailing line from a hard kill
                pid = rec.get("paper_id")
                if pid is not None:
                    rows[pid] = rec
        return rows

    def append(self, row: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())


# --------------------------------------------------------------------------- #
# Run summary (returned + printed by the CLI).
# --------------------------------------------------------------------------- #
@dataclass
class RunSummary:
    by_status: Counter = field(default_factory=Counter)
    by_host: Counter = field(default_factory=Counter)   # downloaded count per host
    total_bytes: int = 0
    skipped: int = 0                                     # resumed (already done)
    processed: int = 0                                   # actually attempted

    def render(self) -> str:
        lines = ["Corpus download summary:"]
        lines.append(f"  processed={self.processed} skipped(resumed)={self.skipped}")
        for status in (STATUS_DOWNLOADED, STATUS_FAILED, STATUS_PENDING):
            if self.by_status.get(status):
                lines.append(f"  status {status}: {self.by_status[status]}")
        lines.append(f"  total_bytes={self.total_bytes}")
        if self.by_host:
            lines.append("  downloaded by host:")
            for host, n in self.by_host.most_common():
                lines.append(f"    {host or '(none)'}: {n}")
        return "\n".join(lines)


class CorpusDownloader:
    def __init__(
        self,
        dest_dir: str | pathlib.Path = DEFAULT_DEST_DIR,
        *,
        fetcher: PdfFetcher,
        host_fetchers: dict[str, PdfFetcher] | None = None,
        backend: StorageBackend | None = None,
        manifest: Manifest | None = None,
        policies: dict[str, HostPolicy] | None = None,
        default_policy: HostPolicy = DEFAULT_POLICY,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ):
        self._dest = pathlib.Path(dest_dir)
        self._fetcher = fetcher
        self._host_fetchers = dict(host_fetchers or {})
        self._backend = backend if backend is not None else LocalDirBackend(dest_dir)
        self._manifest = manifest if manifest is not None else Manifest(self._dest / MANIFEST_NAME)
        self._policies = dict(policies or {})
        self._default_policy = default_policy
        self._sleep = sleep
        self._now = now
        self._last_request: dict[str, float] = {}   # host -> monotonic ts

    # -- seams -------------------------------------------------------------- #
    def _fetcher_for(self, host: str) -> PdfFetcher:
        """Route by host: openreview.net (and any registered host) goes through
        its dedicated fetcher (the OpenReview API path); everything else uses
        the default fetcher."""
        return self._host_fetchers.get(host, self._fetcher)

    def _policy_for(self, host: str) -> HostPolicy:
        return self._policies.get(host, self._default_policy)

    def _respect_interval(self, host: str, policy: HostPolicy) -> None:
        """Sleep just enough to keep >= min_interval between requests to `host`."""
        if policy.min_interval <= 0:
            return
        last = self._last_request.get(host)
        if last is not None:
            wait = policy.min_interval - (self._now() - last)
            if wait > 0:
                self._sleep(wait)
        self._last_request[host] = self._now()

    # -- core --------------------------------------------------------------- #
    def _download_one(self, paper_id: str, url: str | None) -> dict:
        """Attempt one paper with per-host politeness + retry/backoff. Never
        raises: any exception is captured into a `failed` manifest row."""
        host = host_of(url)
        policy = self._policy_for(host)
        fetcher = self._fetcher_for(host)

        data: bytes | None = None
        error: str | None = None
        attempts = 0
        for attempt in range(1, policy.max_attempts + 1):
            attempts = attempt
            self._respect_interval(host, policy)
            try:
                data = fetcher.fetch(paper_id, url)
            except Exception as exc:  # noqa: BLE001 -- one bad URL must not kill the run
                data, error = None, repr(exc)
            else:
                error = None if data else "fetch returned no bytes"
            if data:
                break
            if attempt < policy.max_attempts:
                self._sleep(policy.backoff_base * (2 ** (attempt - 1)))

        if data:
            self._backend.write(paper_id, data)
            return _manifest_row(
                paper_id, url, host, STATUS_DOWNLOADED,
                hashlib.sha256(data).hexdigest(), len(data), 200, attempts, None,
            )
        return _manifest_row(
            paper_id, url, host, STATUS_FAILED,
            None, None, None, attempts, error or "fetch returned no bytes",
        )

    def run(self, papers: Iterable, *, max_papers: int | None = None,
            hosts: set[str] | None = None) -> RunSummary:
        """Download PDFs for `papers` (any objects exposing `.paper_id` and
        `.pdf_url`). Resumes from the existing manifest. `hosts`, if given,
        restricts the run to those hosts; `max_papers` caps how many are
        *attempted* this run (skips don't count against it)."""
        done = self._manifest.load()
        summary = RunSummary()
        for paper in papers:
            paper_id = paper.paper_id
            url = paper.pdf_url
            host = host_of(url)
            if hosts is not None and host not in hosts:
                continue
            prior = done.get(paper_id)
            if prior is not None and prior.get("status") in _RESUME_SKIP:
                summary.skipped += 1
                continue
            if max_papers is not None and summary.processed >= max_papers:
                break
            row = self._download_one(paper_id, url)
            self._manifest.append(row)
            summary.processed += 1
            summary.by_status[row["status"]] += 1
            if row["status"] == STATUS_DOWNLOADED:
                summary.by_host[host] += 1
                summary.total_bytes += row["bytes"] or 0
        return summary
