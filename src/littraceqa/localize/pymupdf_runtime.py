"""Process-wide serialization boundary for PyMuPDF calls.

PyMuPDF explicitly does not support multithreaded use. LitTraceQA overlaps
record-level LLM work with a thread pool, so every PyMuPDF entry point must use
the same lock even when each thread opens independent PDF bytes. The lock is
re-entrant because a higher-level PDF helper may call another guarded helper.
"""
from __future__ import annotations

from functools import wraps
import threading
from typing import Callable, ParamSpec, TypeVar


_P = ParamSpec("_P")
_R = TypeVar("_R")
PYMUPDF_LOCK = threading.RLock()


def serialized_pymupdf(
    function: Callable[_P, _R],
) -> Callable[_P, _R]:
    """Run one PyMuPDF operation under the shared process-wide lock."""

    @wraps(function)
    def guarded(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        with PYMUPDF_LOCK:
            return function(*args, **kwargs)

    return guarded
