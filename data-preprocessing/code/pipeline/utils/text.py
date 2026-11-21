"""Text normalization shared by pipeline stages."""

from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
# Space before close punct/brackets/quotes: "technical ." / "(1999 )" / 'elevation '
_SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([.,;:!?)\]}”\"'’])")
# Space after open brackets/quotes: "( PIN" / "“ (You Gotta)"
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{“\"'‘])\s+")
# Normalize "word.Word" / "word.  Word" -> "word. Word"
_SPACE_AFTER_PUNCT_RE = re.compile(r"([.,;:!?])\s*")


def fix_punct_spacing(text: Any) -> str:
    """
    Rewrite scrape spacing artifacts in stored text; keep case.

    "as is ." -> "as is."
    "tricky ," -> "tricky,"
    "(1999 )" -> "(1999)"
    Does not strip space before '(' after a word ("see (note)" stays).
    Does not insert space after '.' (so "1.5" stays "1.5").
    """
    t = str(text or "").strip()
    if not t:
        return ""
    t = _WHITESPACE_RE.sub(" ", t)
    t = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN_RE.sub(r"\1", t)
    return _WHITESPACE_RE.sub(" ", t).strip()


def norm_key(text: Any) -> str:
    """Lookup key: stored-text spacing rewrite, then word.Word collapse, then casefold."""
    t = fix_punct_spacing(text)
    if not t:
        return ""
    t = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", t)
    t = _WHITESPACE_RE.sub(" ", t).strip()
    return t.casefold()
