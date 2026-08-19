"""`python -m littraceqa.corpus.build_indexes` entry point.

Thin shim so the index build has the same top-level `python -m
littraceqa.corpus.<x>` invocation shape as the downloader/parser CLIs; the real
logic lives in `littraceqa.corpus.indexes.build`.
"""
from __future__ import annotations

import sys

from littraceqa.corpus.indexes.build import main

if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
