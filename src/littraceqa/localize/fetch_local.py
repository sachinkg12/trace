"""Local PDF fetcher: HTTP download + disk cache, degrading to None.

Registered as `"local"`. Google Document AI's online-processing fetch (or
a GCS-backed fetcher) swaps in later behind the same `PdfFetcher` Protocol
by registration, with no changes to the localization service.
"""
from __future__ import annotations

import pathlib

from littraceqa.localize.interfaces import register_fetcher

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache" / "pdfs"


@register_fetcher("local")
class LocalPdfFetcher:
    def __init__(self, cache_dir=None, *, session=None, timeout: float = 30.0):
        self._cache_dir = pathlib.Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self._timeout = timeout
        if session is not None:
            self._session = session
        else:                                   # lazy: only import requests when really downloading
            import requests
            self._session = requests.Session()

    def _cache_path(self, paper_id: str) -> pathlib.Path:
        return self._cache_dir / f"{paper_id}.pdf"

    def fetch(self, paper_id: str, pdf_url: str | None) -> bytes | None:
        path = self._cache_path(paper_id)
        if path.exists():
            return path.read_bytes()
        if not pdf_url:
            return None
        try:
            resp = self._session.get(pdf_url, timeout=self._timeout)
            if getattr(resp, "status_code", None) != 200:
                return None
            body = resp.content
        except Exception:
            return None
        if not (isinstance(body, (bytes, bytearray)) and body[:5] == b"%PDF-"):
            return None
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        return bytes(body)
