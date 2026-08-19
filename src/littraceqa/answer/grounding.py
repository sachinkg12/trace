"""Attestation primitive: confirm a proposed answer value appears VERBATIM
in the cited evidence (its quote, or the parsed page it points to). Reuses
the same artifact-tolerant normalization the localizer's quote check uses.
Returns the supporting evidence items -- the basis for keeping evidence for
precision (a value we cannot ground yields no attested evidence)."""
from __future__ import annotations

import re
import unicodedata

from littraceqa.localize.interfaces import LocatedEvidence, ParsedPdf


def _normalize(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"-\s*\n\s*", "", s)   # line-wrap hyphenation
    s = s.replace("­", "")        # soft hyphen (mirror localizer)
    s = re.sub(r"\s+", " ", s)
    return s.strip().casefold()


def _contains(haystack: str, needle: str) -> bool:
    """Boundary-aware containment: `needle` must appear NOT embedded inside a
    larger alphanumeric token. This stops a short value from spuriously
    grounding -- e.g. "1" must not match inside "15.66", and "14.70" must not
    match inside "114.700" -- which would otherwise inflate attestation and
    let junk evidence through the precision gate. Both args are pre-normalized."""
    if not needle:
        return False
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def ground_value(value: str, evidence: list[LocatedEvidence],
                 parsed_by_id: dict[str, ParsedPdf]) -> list[LocatedEvidence]:
    nval = _normalize(value or "")
    if not nval:
        return []
    hits: list[LocatedEvidence] = []
    for ev in evidence:
        if ev.quote and _contains(_normalize(ev.quote), nval):
            hits.append(ev)
            continue
        parsed = parsed_by_id.get(ev.paper_id)
        page = parsed.page(ev.page) if parsed else None
        if page and _contains(_normalize(page.text), nval):
            hits.append(ev)
    return hits
