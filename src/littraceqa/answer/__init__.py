"""Answer package.

Strategy implementations register themselves with `register_strategy`
(interfaces.py) as a side effect of being imported. Importing them here
(mirroring `littraceqa.retrieval.__init__`) means `build_strategy(...)` --
and `AnswerPipeline()`'s default `strategies=None` path -- works right
after `import littraceqa.answer` (or after importing any submodule, since
that always initializes this package first), without callers needing to
know which module a given strategy lives in.
"""

from __future__ import annotations

from littraceqa.answer import freeform as _freeform  # noqa: F401 -- registers "freeform"
from littraceqa.answer import multiple_choice as _multiple_choice  # noqa: F401 -- registers "multiple_choice"
from littraceqa.answer import table as _table  # noqa: F401 -- registers "table"
