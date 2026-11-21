"""WikiHow MediaWiki diff URL helpers.

Stored pair/group JSON uses a clickable `diff_url`. Vandalism filtering and
grouping still need MediaWiki IDs internally; parse them from the URL (with
a fallback to leftover `diff_id` / `old_id` fields on mixed files).
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse

WIKIHOW_DIFF_BASE = "https://www.wikihow.com/index.php"


def make_diff_url(
    article_title: str | None,
    diff_id: Any,
    old_id: Any,
) -> str | None:
    """Build a WikiHow MediaWiki diff URL, or None if ids are missing."""
    if diff_id is None or old_id is None:
        return None
    diff_s = str(diff_id).strip()
    old_s = str(old_id).strip()
    if not diff_s or not old_s:
        return None
    title = (article_title or "").strip().replace(" ", "-")
    query = urlencode(
        {"title": title, "diff": diff_s, "oldid": old_s},
        quote_via=quote,
    )
    return f"{WIKIHOW_DIFF_BASE}?{query}"


def parse_diff_ids(url: Any) -> tuple[str | None, str | None]:
    """Return (diff_id, old_id) from a MediaWiki diff URL."""
    if url is None:
        return None, None
    raw = str(url).strip()
    if not raw:
        return None, None
    qs = parse_qs(urlparse(raw).query)
    diff = (qs.get("diff") or [None])[0]
    oldid = (qs.get("oldid") or [None])[0]
    diff_s = str(diff).strip() if diff is not None else ""
    old_s = str(oldid).strip() if oldid is not None else ""
    return (diff_s or None, old_s or None)


def _id_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def pair_revision_ids(pair: dict[str, Any] | None) -> tuple[str | None, str | None]:
    """Prefer parsing URL; fall back to leftover diff_id/old_id fields."""
    if not pair:
        return None, None
    diff_id, old_id = parse_diff_ids(pair.get("diff_url"))
    if diff_id is None:
        diff_id = _id_str(pair.get("diff_id"))
    if old_id is None:
        old_id = _id_str(pair.get("old_id"))
    return diff_id, old_id


def pair_diff_id(pair: dict[str, Any] | None) -> str | None:
    return pair_revision_ids(pair)[0]


def pair_old_id(pair: dict[str, Any] | None) -> str | None:
    return pair_revision_ids(pair)[1]


def pair_diff_url(article_title: str | None, pair: dict[str, Any] | None) -> str | None:
    """Prefer an existing diff_url; else build from leftover ids."""
    if not pair:
        return None
    existing = pair.get("diff_url")
    if existing is not None and str(existing).strip():
        return str(existing).strip()
    return make_diff_url(article_title, pair.get("diff_id"), pair.get("old_id"))


def revision_time_key(pair: dict[str, Any] | None) -> tuple[int, int]:
    """Sortable key for chronological order (older -> newer)."""
    diff_id, old_id = pair_revision_ids(pair)
    try:
        diff_n = int(diff_id or 0)
    except (TypeError, ValueError):
        diff_n = 0
    try:
        old_n = int(old_id or 0)
    except (TypeError, ValueError):
        old_n = 0
    return (diff_n, old_n)
