"""Seed-finding: identifies which paper(s) a question is about via a
recognition-first cascade (parametric -> exact/acronym -> dense), producing
ranked, confidence-scored `SeedCandidate`s for downstream evidence
localization to confirm against. See `docs/superpowers/plans/
2026-08-02-seed-finding.md` for the full architecture.

`SeedCandidate`/`Anchor` live in `interfaces.py`; `extract_anchor`
(anchor.py) is the first stage of the cascade.
"""

from __future__ import annotations
