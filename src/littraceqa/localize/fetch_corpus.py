"""Corpus PDF fetcher: reads a paper's PDF bytes from the pre-downloaded corpus
SNAPSHOT (a local dir or a GCS bucket), degrading to None -- NO live fetch.

This is the fetcher the NEW (planner->router->cascade) path uses in place of the
token-dependent `OpenReviewFetcher`: on the reproducible dev-55 harness (and the
27k VM run) every PDF has already been downloaded to a corpus source, so evidence
localization reads bytes straight from there rather than hitting OpenReview/arXiv.

SOURCE seam (DIP, mirrors the parse/index layers' `PdfSource`/`ArtifactSource`):
`CorpusPdfSource.read(paper_id) -> bytes | None`. `LocalDirPdfSource` reads
`{paper_id}.pdf` from a directory (tests / a synced VM disk); `GcsPdfSource`
streams `gs://<bucket>/<prefix>/{paper_id}.pdf` through the existing `GcsBackend`
(lazy google-cloud-storage import lives in `GcsBackend`, so importing THIS module
never needs the lib). Registered as `"corpus"` for symmetry with the fetcher
family, though the composition root constructs it directly.

Total degradation: a missing paper_id, an unreadable file, or a backend error
returns None (the `PdfFetcher` Protocol's miss contract -- exactly what
`LocalPdfFetcher` does), so a fetch miss contributes no evidence and the run
continues.
"""
from __future__ import annotations

import pathlib
from typing import Protocol, runtime_checkable

from littraceqa.localize.interfaces import register_fetcher


@runtime_checkable
class CorpusPdfSource(Protocol):
    """Reads a paper's raw PDF bytes from the corpus snapshot; None on miss."""

    def read(self, paper_id: str) -> bytes | None: ...


class LocalDirPdfSource:
    """Reads `{paper_id}.pdf` from a local directory. A missing/unreadable file
    is a MISS (None), never an error -- so the fetcher degrades cleanly."""

    def __init__(self, pdf_dir: str | pathlib.Path):
        self._dir = pathlib.Path(pdf_dir)

    def read(self, paper_id: str) -> bytes | None:
        path = self._dir / f"{paper_id}.pdf"
        try:
            if not path.exists():
                return None
            return path.read_bytes()
        except OSError:
            return None


class GcsPdfSource:
    """Reads `gs://<bucket>/<prefix>/{paper_id}.pdf` through a `GcsBackend`
    (reused from the corpus downloader). A missing blob or backend error is a
    MISS (None). The lazy google-cloud-storage import lives in `GcsBackend`, so
    tests inject a fake backend and never touch the network."""

    def __init__(self, backend, prefix: str = "pdfs"):
        self._backend = backend
        self._prefix = prefix.strip("/")

    def _blob_name(self, paper_id: str) -> str:
        return f"{self._prefix}/{paper_id}.pdf" if self._prefix else f"{paper_id}.pdf"

    def read(self, paper_id: str) -> bytes | None:
        try:
            return self._backend.read_blob(self._blob_name(paper_id))
        except Exception:  # noqa: BLE001 -- a backend error is a miss, not fatal
            return None


@register_fetcher("corpus")
class CorpusPdfFetcher:
    """`PdfFetcher` over the corpus snapshot. Construct from an injected `source`,
    a local `pdf_dir`, or a GCS `backend` (+`prefix`). `fetch` ignores `pdf_url`
    (there is no live fetch) and returns the source's bytes, or None on any miss/
    error -- honouring the Protocol's `bytes | None` contract."""

    def __init__(self, source: CorpusPdfSource | None = None, *,
                 pdf_dir: str | pathlib.Path | None = None,
                 backend=None, prefix: str = "pdfs"):
        if source is not None:
            self._source: CorpusPdfSource = source
        elif pdf_dir is not None:
            self._source = LocalDirPdfSource(pdf_dir)
        elif backend is not None:
            self._source = GcsPdfSource(backend, prefix=prefix)
        else:
            raise ValueError(
                "CorpusPdfFetcher needs a `source`, a local `pdf_dir`, or a GCS `backend`"
            )

    def fetch(self, paper_id: str, pdf_url: str | None = None) -> bytes | None:
        try:
            return self._source.read(paper_id)
        except Exception:  # noqa: BLE001 -- never breach the total-degradation contract
            return None
