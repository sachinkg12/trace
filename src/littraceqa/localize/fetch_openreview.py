"""OpenReview API fetcher.

OpenReview-hosted PDFs (~12k of the pool) 403 on direct HTTP (bot protection),
so this fetcher downloads the published PDF through the authenticated OpenReview
API using the paper's forum id. It WRAPS an inner `PdfFetcher`: the inner runs
first (serving cache hits and non-OpenReview hosts); only on a miss does this
fall back to the API for papers that have a known OpenReview id.

Credentials come from an OpenReview token (env `OPENREVIEW_TOKEN`), so the
pipeline stays reproducible — whoever runs it supplies their own free OpenReview
account token, exactly like the Gemini key. Downloaded PDFs are cached under
`{paper_id}.pdf` so subsequent runs are served by the inner fetcher's cache with
no API call. Degrades to None at every failure; never raises.
"""
from __future__ import annotations

import pathlib

from littraceqa.localize.fetch_local import DEFAULT_CACHE_DIR


class OpenReviewFetcher:
    def __init__(self, inner, id_map: dict[str, str], *, token: str | None = None,
                 client=None, cache_dir=None, baseurl: str = "https://api2.openreview.net"):
        # `inner` is a PdfFetcher; `id_map` maps paper_id -> OpenReview forum id.
        self._inner = inner
        self._id_map = id_map
        self._token = token
        self._client = client            # injectable for tests
        self._baseurl = baseurl
        self._cache_dir = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    def _get_client(self):
        if self._client is None:
            from openreview.api import OpenReviewClient  # lazy: only when an API call is needed
            self._client = OpenReviewClient(baseurl=self._baseurl, token=self._token)
        return self._client

    def fetch(self, paper_id: str, pdf_url: str | None) -> bytes | None:
        # 1) inner fetcher first: serves cache hits and non-OpenReview hosts.
        got = self._inner.fetch(paper_id, pdf_url)
        if got:
            return got
        # 2) OpenReview API fallback for papers with a known forum id.
        forum_id = self._id_map.get(paper_id)
        if not forum_id:
            return None
        try:
            pdf = self._get_client().get_pdf(forum_id)
        except Exception:  # noqa: BLE001 -- any API failure degrades to None
            return None
        if not (isinstance(pdf, (bytes, bytearray)) and pdf[:5] == b"%PDF-"):
            return None
        # cache under paper_id so the inner fetcher serves it next time (no re-fetch).
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"{paper_id}.pdf").write_bytes(pdf)
        except Exception:  # noqa: BLE001 -- caching is best-effort
            pass
        return bytes(pdf)
