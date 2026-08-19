"""CLI entry for the real bulk download:

    python -m littraceqa.corpus.download <pool.jsonl> <dest_dir> \
        [--max N] [--hosts h1 h2 ...] [--gcs-bucket B [--gcs-prefix P]]

Wires the real fetchers (composition root, mirrors `pipeline/build.py`):
openreview.net routes through `OpenReviewFetcher` (those PDFs 403 on direct
HTTP; the token comes from env `OPENREVIEW_TOKEN`), every other host uses the
plain `LocalPdfFetcher`. Prints a status/host/bytes summary at the end.

Storage is a seam (OCP): by default PDFs + manifest land in `dest_dir` on the
local disk; pass `--gcs-bucket` to stream them straight to
`gs://<bucket>/<prefix>/{paper_id}.pdf` (manifest at `gs://<bucket>/manifest.jsonl`)
instead -- ideal on a throwaway VM where `dest_dir` (e.g. `/tmp/corpus`) is just
a scratch path the local fetchers still cache through.

No real network is touched by importing this module; only `main()` fetches.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

from littraceqa.corpus.downloader import MANIFEST_NAME, CorpusDownloader
from littraceqa.localize.fetch_auth import AuthPdfFetcher
from littraceqa.localize.fetch_local import LocalPdfFetcher
from littraceqa.retrieval.pool import DEFAULT_POOL_PATH, load_pool


def _gcs_client():
    """Build the default GCS client (lazy import: only reached when
    `--gcs-bucket` is used). On the VM it auto-authenticates via the attached
    service account. Kept as a seam so tests inject a fake and skip network."""
    from google.cloud import storage  # noqa: PLC0415 -- intentional lazy import

    return storage.Client()


def _build_downloader(dest_dir: str | pathlib.Path, pool_path, *,
                      token: str | None,
                      gcs_bucket: str | None = None,
                      gcs_prefix: str = "pdfs",
                      gcs_manifest: str | None = None) -> CorpusDownloader:
    """Composition root for the download run. Default fetcher = direct HTTP;
    openreview.net gets a dedicated OpenReviewFetcher (API path) when a token
    is available. Both cache into `dest_dir` so the corpus doubles as the
    answer-time PDF cache.

    When `gcs_bucket` is given, storage is swapped to `GcsBackend` +
    `GcsManifest` (PDFs + manifest stream to the bucket, no local hoarding);
    otherwise the default local dir backend/manifest is used."""
    local = LocalPdfFetcher(cache_dir=dest_dir)
    host_fetchers = {}
    if token:
        # Direct authenticated GET (fast, reliable) instead of the openreview-py
        # API path, whose urllib3 retry/backoff stalls under rapid bulk requests.
        host_fetchers["openreview.net"] = AuthPdfFetcher(local, token=token,
                                                         auth_host="openreview.net",
                                                         cache_dir=dest_dir)
    backend = None
    manifest = None
    if gcs_bucket:
        from littraceqa.corpus.gcs_backend import GcsBackend, GcsManifest

        backend = GcsBackend(gcs_bucket, prefix=gcs_prefix, client=_gcs_client())
        manifest = GcsManifest(backend, name=gcs_manifest or MANIFEST_NAME)
    return CorpusDownloader(dest_dir, fetcher=local, host_fetchers=host_fetchers,
                            backend=backend, manifest=manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littraceqa.corpus.download")
    parser.add_argument("pool", nargs="?", default=str(DEFAULT_POOL_PATH),
                        help="paper_metadata.jsonl (default: data/paper_metadata.jsonl)")
    parser.add_argument("dest_dir", help="destination directory for the corpus PDFs + manifest")
    parser.add_argument("--max", type=int, default=None, dest="max_papers",
                        help="cap papers ATTEMPTED this run (skips don't count)")
    parser.add_argument("--hosts", nargs="+", default=None,
                        help="restrict to these URL hosts (e.g. aclanthology.org)")
    parser.add_argument("--gcs-bucket", default=None, dest="gcs_bucket",
                        help="stream PDFs + manifest to this GCS bucket instead of local disk")
    parser.add_argument("--gcs-prefix", default="pdfs", dest="gcs_prefix",
                        help="key prefix for PDFs within the bucket (default: pdfs)")
    parser.add_argument("--gcs-manifest", default=None, dest="gcs_manifest",
                        help="manifest object name in the bucket (default: manifest.jsonl). "
                             "Give each parallel per-host worker its own, e.g. "
                             "manifest-acl.jsonl, so they don't overwrite each other.")
    args = parser.parse_args(argv)

    papers = load_pool(args.pool)
    token = os.getenv("OPENREVIEW_TOKEN")
    downloader = _build_downloader(args.dest_dir, args.pool, token=token,
                                   gcs_bucket=args.gcs_bucket, gcs_prefix=args.gcs_prefix,
                                   gcs_manifest=args.gcs_manifest)
    summary = downloader.run(
        papers,
        max_papers=args.max_papers,
        hosts=set(args.hosts) if args.hosts else None,
    )
    # GcsManifest batches its uploads; flush the tail so the bucket object is
    # complete. (The local Manifest has no flush and needs none -- it appends.)
    flush = getattr(downloader._manifest, "flush", None)
    if callable(flush):
        flush()
    print(summary.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
