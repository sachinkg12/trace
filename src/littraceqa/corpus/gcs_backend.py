"""Google Cloud Storage storage backend for the bulk corpus downloader.

Lets a throwaway cloud VM stream all ~27k PDFs straight into a GCS bucket with
no local disk hoarding: the VM's `/tmp` is never the source of truth, the
bucket is. `GcsBackend` implements the SAME `StorageBackend` Protocol as
`LocalDirBackend` (`exists` / `write`), so `CorpusDownloader` swaps one for the
other with zero edits (OCP) -- the only wiring change is in the CLI.

Design points:
  * Lazy import (OCP + optional dependency): `google-cloud-storage` is imported
    INSIDE `GcsBackend.__init__`, and only when no `client` is injected. So
    importing this module never requires the lib (local dev / the fast test
    suite don't install it); only *constructing* a real GcsBackend does. Tests
    inject a fake client and never touch the network.
  * Default auth: on the VM the default `storage.Client()` auto-authenticates
    via the attached service account (`--scopes=cloud-platform`) -- no key
    files, no `GOOGLE_APPLICATION_CREDENTIALS`.
  * Self-contained snapshot: the run's `manifest.jsonl` is written through this
    same backend to `gs://<bucket>/manifest.jsonl`, so the bucket alone is a
    fully reproducible snapshot (PDFs under `<prefix>/` + the manifest beside
    them). GCS objects aren't appendable, so `GcsManifest` keeps rows in memory
    and rewrites the whole object on flush (reload+rewrite) -- the simplest
    correct approach, resume-safe: a crash loses at most the unflushed tail,
    and those papers are just re-downloaded on the next run.
"""
from __future__ import annotations

import json

from littraceqa.corpus.downloader import MANIFEST_NAME


class GcsBackend:
    """Writes each PDF to `gs://<bucket>/<prefix>/{paper_id}.pdf`, mirroring
    `LocalDirBackend`'s existence-check/skip so resume works against the bucket.

    Constructor accepts an injected `client` (a `google.cloud.storage.Client`
    or a test fake); when omitted, a real client is lazily imported + built.
    """

    def __init__(self, bucket: str, prefix: str = "pdfs", client=None):
        if client is None:
            # Lazy import: importing this module must not require the lib; only
            # constructing a *real* backend does. On the VM this default client
            # auto-authenticates via the attached service account.
            from google.cloud import storage  # noqa: PLC0415 -- intentional lazy import

            client = storage.Client()
        self._client = client
        self._bucket = client.bucket(bucket)
        self._prefix = prefix.strip("/")

    def _blob_name(self, paper_id: str) -> str:
        return f"{self._prefix}/{paper_id}.pdf" if self._prefix else f"{paper_id}.pdf"

    # -- StorageBackend Protocol ------------------------------------------- #
    def exists(self, paper_id: str) -> bool:
        return self._bucket.blob(self._blob_name(paper_id)).exists()

    def write(self, paper_id: str, data: bytes) -> None:
        # GCS uploads are atomic: the blob is not visible until the upload
        # completes, so a kill mid-write never leaves a truncated object.
        self._bucket.blob(self._blob_name(paper_id)).upload_from_string(
            data, content_type="application/pdf"
        )

    # -- generic blob access (used by GcsManifest so the manifest is a bucket
    #    object beside the PDFs; keeps ALL GCS access funnelled through here) -- #
    def read_blob(self, name: str) -> bytes | None:
        blob = self._bucket.blob(name)
        if not blob.exists():
            return None
        return blob.download_as_bytes()

    def write_blob(self, name: str, data: bytes) -> None:
        self._bucket.blob(name).upload_from_string(
            data, content_type="application/x-ndjson"
        )

    def list_blob_names(self, prefix: str = "") -> list[str]:
        """Names of every blob under `prefix` (used to enumerate the parsed
        artifacts for the index build). Keeps bucket listing funnelled through
        this backend so the read side has one place that touches GCS."""
        return [b.name for b in self._bucket.list_blobs(prefix=prefix or None)]


class GcsManifest:
    """`manifest.jsonl` persisted as ONE bucket object (`gs://<bucket>/{name}`),
    written through a `GcsBackend`. Drop-in for the local `Manifest` (`load` +
    `append`), so `CorpusDownloader` uses it unchanged.

    GCS blobs can't be appended to, so rows are held in memory (latest row per
    `paper_id` wins, append-log semantics) and the whole object is rewritten on
    flush. `flush_every` batches rewrites so a 27k run doesn't re-upload the
    object on every paper; `flush()` is also called once at run end by the CLI.
    A hard kill loses at most the last `<flush_every` rows -- those papers are
    simply re-fetched on resume (idempotent), and every flushed row is skipped.
    """

    def __init__(self, backend: GcsBackend, name: str = MANIFEST_NAME,
                 flush_every: int = 25):
        self._backend = backend
        self._name = name
        self._flush_every = max(1, flush_every)
        self._rows: dict[str, dict] = {}
        self._pending = 0

    def load(self) -> dict[str, dict]:
        """Reload the manifest object from the bucket. Tolerates a torn line
        from a partial upload and keeps the LAST row per paper_id."""
        rows: dict[str, dict] = {}
        data = self._backend.read_blob(self._name)
        if data:
            for line in data.decode("utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = rec.get("paper_id")
                if pid is not None:
                    rows[pid] = rec
        self._rows = rows
        return dict(rows)

    def append(self, row: dict) -> None:
        pid = row.get("paper_id")
        if pid is not None:
            self._rows[pid] = row
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self) -> None:
        """Rewrite the whole manifest object from the in-memory rows (insertion
        order preserved for a stable, human-readable diff)."""
        body = "".join(json.dumps(self._rows[pid]) + "\n" for pid in self._rows)
        self._backend.write_blob(self._name, body.encode("utf-8"))
        self._pending = 0
