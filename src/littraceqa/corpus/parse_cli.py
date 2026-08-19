"""CLI entry for the Level-1 corpus parse:

    python -m littraceqa.corpus.parse <pool.jsonl> <pdf_source> \
        [--parsed-dir DIR] [--max N] \
        [--gcs-bucket B [--gcs-pdf-prefix pdfs] [--gcs-parsed-prefix parsed]]

Composition root (mirrors `corpus/download.py`). Storage is a seam (OCP):
  * Default (local): reads PDFs from `<pdf_source>/{paper_id}.pdf`, writes
    artifacts to `<parsed-dir>/{paper_id}.json`, manifest beside them.
  * `--gcs-bucket`: reads PDFs from `gs://<bucket>/<pdf_prefix>/{paper_id}.pdf`
    and writes artifacts to `gs://<bucket>/<parsed_prefix>/{paper_id}.json`,
    manifest at `gs://<bucket>/<gcs-manifest>` -- ideal on a throwaway VM whose
    local disk is just scratch. `pdf_source` is still required positionally
    (used as the local scratch/default parsed-dir root) but PDFs come from GCS.

No real network is touched by importing this module; only `main()` does I/O.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

from littraceqa.corpus.parse import (
    PARSE_MANIFEST_NAME,
    CorpusParser,
    LocalArtifactSink,
    LocalDirPdfSource,
)
from littraceqa.corpus.downloader import Manifest
from littraceqa.retrieval.pool import DEFAULT_POOL_PATH, load_pool


def _gcs_client():
    """Build the default GCS client (lazy import: only reached with
    `--gcs-bucket`). On the VM it auto-authenticates via the attached service
    account. A seam so tests inject a fake and skip the network."""
    from google.cloud import storage  # noqa: PLC0415 -- intentional lazy import

    return storage.Client()


def _build_parser(pdf_source: str, parsed_dir: str | pathlib.Path, *,
                  gcs_bucket: str | None = None,
                  gcs_pdf_prefix: str = "pdfs",
                  gcs_parsed_prefix: str = "parsed",
                  gcs_manifest: str | None = None,
                  client=None) -> CorpusParser:
    """Composition root: wire the PDF source, artifact sink, and manifest either
    locally or against a GCS bucket. `client` is injectable for tests."""
    if gcs_bucket:
        from littraceqa.corpus.gcs_backend import GcsBackend, GcsManifest
        from littraceqa.corpus.parse import GcsArtifactSink, GcsPdfSource

        backend = GcsBackend(gcs_bucket, prefix=gcs_pdf_prefix,
                             client=client or _gcs_client())
        source = GcsPdfSource(backend, prefix=gcs_pdf_prefix)
        sink = GcsArtifactSink(backend, prefix=gcs_parsed_prefix)
        manifest = GcsManifest(backend, name=gcs_manifest or PARSE_MANIFEST_NAME)
    else:
        source = LocalDirPdfSource(pdf_source)
        sink = LocalArtifactSink(parsed_dir)
        manifest = Manifest(pathlib.Path(parsed_dir) / PARSE_MANIFEST_NAME)
    return CorpusParser(source=source, sink=sink, manifest=manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m littraceqa.corpus.parse")
    parser.add_argument("pool", nargs="?", default=str(DEFAULT_POOL_PATH),
                        help="paper_metadata.jsonl (default: data/paper_metadata.jsonl)")
    parser.add_argument("pdf_source",
                        help="local dir of {paper_id}.pdf (also the parsed-dir root "
                             "default); with --gcs-bucket, PDFs are read from the bucket")
    parser.add_argument("--parsed-dir", default=None, dest="parsed_dir",
                        help="local dir for artifact JSON + manifest "
                             "(default: <pdf_source>/parsed)")
    parser.add_argument("--max", type=int, default=None, dest="max_papers",
                        help="cap papers PARSED this run (skips don't count)")
    parser.add_argument("--gcs-bucket", default=None, dest="gcs_bucket",
                        help="read PDFs from + write artifacts to this GCS bucket")
    parser.add_argument("--gcs-pdf-prefix", default="pdfs", dest="gcs_pdf_prefix",
                        help="prefix of the PDFs within the bucket (default: pdfs)")
    parser.add_argument("--gcs-parsed-prefix", default="parsed", dest="gcs_parsed_prefix",
                        help="prefix for artifact JSON within the bucket (default: parsed)")
    parser.add_argument("--gcs-manifest", default=None, dest="gcs_manifest",
                        help=f"manifest object name in the bucket "
                             f"(default: {PARSE_MANIFEST_NAME})")
    args = parser.parse_args(argv)

    parsed_dir = args.parsed_dir or str(pathlib.Path(args.pdf_source) / "parsed")
    corpus_parser = _build_parser(
        args.pdf_source, parsed_dir,
        gcs_bucket=args.gcs_bucket, gcs_pdf_prefix=args.gcs_pdf_prefix,
        gcs_parsed_prefix=args.gcs_parsed_prefix, gcs_manifest=args.gcs_manifest,
    )
    papers = load_pool(args.pool)
    summary = corpus_parser.run(papers, max_papers=args.max_papers)

    # GcsManifest batches uploads; flush the tail so the bucket object is
    # complete. (The local Manifest appends and needs no flush.)
    flush = getattr(corpus_parser._manifest, "flush", None)
    if callable(flush):
        flush()
    print(summary.render())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
