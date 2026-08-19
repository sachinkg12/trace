"""Direct authenticated PDF fetcher.

OpenReview PDFs (~12k of the pool) 403 on *anonymous* HTTP, but a plain
`GET <pdf_url>` with an `Authorization: Bearer <token>` header returns the
published PDF directly (verified 200 / ~0.3s). This is far simpler and faster
than routing through the openreview-py client, whose urllib3 retry/backoff
stalls under rapid bulk requests. `AuthPdfFetcher` wraps an inner `PdfFetcher`
(serving cache hits + non-matching hosts first) and, only on a miss for a URL
on `auth_host`, does the authenticated GET. Total-degradation contract: any
failure returns None, never raises.
"""
from __future__ import annotations

import pathlib
from urllib.parse import urlsplit

from littraceqa.localize.fetch_local import DEFAULT_CACHE_DIR


class AuthPdfFetcher:
    def __init__(self, inner, *, token: str | None,
                 auth_host: str = "openreview.net",
                 cache_dir=None, timeout: float = 30.0):
        self._inner = inner
        self._token = token
        self._auth_host = auth_host
        self._timeout = timeout
        self._cache_dir = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR

    def fetch(self, paper_id: str, pdf_url: str | None) -> bytes | None:
        # 1) inner first: cache hits + any non-auth host.
        got = self._inner.fetch(paper_id, pdf_url)
        if got:
            return got
        # 2) authenticated direct GET, only for the auth host.
        if not pdf_url or not self._token:
            return None
        try:
            host = urlsplit(pdf_url).hostname or ""
        except Exception:  # noqa: BLE001
            return None
        if not (host == self._auth_host or host.endswith("." + self._auth_host)):
            return None
        try:
            import requests  # noqa: PLC0415 -- lazy; matches LocalPdfFetcher
            resp = requests.get(
                pdf_url,
                headers={"Authorization": f"Bearer {self._token}",
                         "User-Agent": "Mozilla/5.0 (litetraceqa corpus fetcher)"},
                timeout=self._timeout,
            )
        except Exception:  # noqa: BLE001 -- network failure degrades to None
            return None
        if resp.status_code != 200:
            return None
        data = resp.content
        if not (data[:5] == b"%PDF-"):
            return None
        try:  # cache under paper_id so the inner fetcher serves it next time.
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            (self._cache_dir / f"{paper_id}.pdf").write_bytes(data)
        except Exception:  # noqa: BLE001 -- caching best-effort
            pass
        return data
