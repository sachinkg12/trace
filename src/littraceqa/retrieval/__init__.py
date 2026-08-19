"""Retrieval core package.

Retriever implementations register themselves with `register_retriever`
(interfaces.py) as a side effect of being imported. Importing them here
means callers can do `build_retriever("dense", ...)` right after
`import littraceqa.retrieval` (or after importing any submodule, since
that always initializes this package first) without needing to know
which module a given retriever backend lives in.
"""

from __future__ import annotations

from littraceqa.retrieval import bm25 as _bm25  # noqa: F401 -- registers "bm25"
from littraceqa.retrieval import dense as _dense  # noqa: F401 -- registers "dense"
from littraceqa.retrieval import hybrid as _hybrid  # noqa: F401 -- registers "rrf"
