"""Bulk corpus downloader: fetch every pool PDF into a verifiable, frozen,
resumable local snapshot with a checksummed manifest.

Kept OCP: the write path is behind a `StorageBackend` Protocol (local now,
GCS/remote later by adding an impl -- zero edits here), and PDF fetching is
injected as a `PdfFetcher` (reusing the localize-layer fetchers), so a bad
URL never crashes the run and a new host policy slots in without touching
`CorpusDownloader`.
"""
