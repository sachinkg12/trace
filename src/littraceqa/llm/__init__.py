"""LLM client package — the "AI brain" interface used by seed-finding (#4)
and the answer pipeline (#7).

LLM backends register themselves with `register_llm` (interfaces.py) as a
side effect of being imported. Importing them here means callers can do
`build_llm("gemini", ...)` right after `import littraceqa.llm` (or after
importing any submodule, since that always initializes this package first)
without needing to know which module a given backend lives in.
"""

from __future__ import annotations

from littraceqa.llm import fake as _fake  # noqa: F401 -- registers "fake"
from littraceqa.llm import gemini as _gemini  # noqa: F401 -- registers "gemini"
