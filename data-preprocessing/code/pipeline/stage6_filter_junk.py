"""
Drop junk pairs: URL-heavy text, quiz/JSON schema keys, empty quiz fields,
affiliate/query leftovers, and leftover HTML tags. Instructional > / -> stay.

Reads:  data/05_bleu_passed/*.json
Writes:
  survived  -> data/06_junk_free/<article>.json
  discarded -> data/discarded/06_junk/<article>.json  (only if any dropped)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any
from pipeline.utils.io import create_output_folder, save_article, article_envelope, resolve_article_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "05_bleu_passed")
SURVIVED_FOLDER = os.path.join(BASE_DIR, "data", "06_junk_free")
DISCARDED_FOLDER = os.path.join(BASE_DIR, "data", "discarded", "06_junk")

# URL-heavy: normal and spaced variants left by markup cleaning.
_URL_RE = re.compile(
    r"(?:https?\s*:\s*//|www\.wikihow\.com|www\.\w+)",
    re.IGNORECASE,
)

# Affiliate / Amazon query / broken-HTML leftovers after markup cleaning.
# Do not treat instructional > / -> / → as junk.
_AFFILIATE_JUNK_RE = re.compile(
    r"(?:#{1,2}\s*)?(?:affili?ate[\s_]*article[\s_]*id|article[\s_]*affili?ate[\s_]*id)|tag=wikihow[\w-]+",
    re.IGNORECASE,
)
_QUERY_JUNK_RE = re.compile(
    r"&(?:crid|sprefix|ref|k|i|tag)=|\b(?:crid|sprefix)=|\bnb_sb_|\bp_\d+_\d+&?",
    re.IGNORECASE,
)
_HTML_TAG_JUNK_RE = re.compile(
    r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*\s*/?\s*>|\{end[a-z]*\}",
    re.IGNORECASE,
)

# Quiz / JSON schema key chrome, e.g. meaning": "...", 5": "...", question": "
_JSON_KEY_RE = re.compile(
    r'(?:^|[\s{,])(?:"?)([A-Za-z_][\w]*|\d+)(?:"?)\s*"\s*:\s*"?',
)

# Known quiz / article-schema field names (case-insensitive).
_QUIZ_KEYS = frozenset(
    {
        "incorrect_message",
        "correct_message",
        "question",
        "meaning",
        "title",
        "intra_article_intro",
        "text",
        "answer",
        "answers",
        "options",
        "option",
        "quiz",
        "result",
        "results",
        "message",
        "prompt",
        "label",
        "value",
        "url",
        "href",
        "link",
        "image",
        "img",
    }
)

# After stripping JSON key wrappers, treat leftover as empty/trivial.
_TRIVIAL_VALUES = frozenset(
    {
        "",
        "nope",
        "nope!",
        "yes",
        "yes!",
        "ok",
        "okay",
        "n/a",
        "na",
        "none",
        "null",
        "true",
        "false",
    }
)

_WHITESPACE_RE = re.compile(r"\s+")
# Strip leading `key": "` / trailing `"` wrappers for emptiness checks.
_STRIP_KEY_PREFIX_RE = re.compile(
    r'^(?:"?)([A-Za-z_][\w]*|\d+)(?:"?)\s*"\s*:\s*"?',
)
_STRIP_TRAILING_QUOTE_RE = re.compile(r'"\s*$')


def _norm(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _payload_after_json_key(text: str) -> str | None:
    """
    If text looks like `key": "value..."` (possibly truncated), return value
    body; otherwise None.
    """
    s = _norm(text)
    m = _STRIP_KEY_PREFIX_RE.match(s)
    if not m:
        return None
    body = s[m.end() :]
    body = _STRIP_TRAILING_QUOTE_RE.sub("", body).strip()
    return body


def _looks_like_json_quiz_schema(text: str) -> bool:
    """True if text is dominated by quiz/JSON key chrome."""
    s = _norm(text)
    if not s:
        return False

    m = _STRIP_KEY_PREFIX_RE.match(s)
    if m:
        key = m.group(1).lower()
        # Numeric option slots: 5": "https://... or 1": ""
        if key.isdigit():
            return True
        if key in _QUIZ_KEYS:
            return True

    # Fallback: any quiz key pattern anywhere near the start.
    for match in _JSON_KEY_RE.finditer(s):
        key = match.group(1).lower()
        if key.isdigit() or key in _QUIZ_KEYS:
            # Schema-like if the key appears in the first ~40 chars
            if match.start() <= 40:
                return True
    return False


def _is_empty_or_trivial_quiz_field(text: str) -> bool:
    """Empty / near-empty quiz field after stripping key chrome."""
    payload = _payload_after_json_key(text)
    if payload is None:
        return False
    compact = _WHITESPACE_RE.sub("", payload).lower().strip(".,!;:")
    return compact in _TRIVIAL_VALUES


def is_junk_text(text: Any) -> bool:
    """Return True if a single base/revised side should be treated as junk."""
    s = str(text or "")
    if not s.strip():
        return True
    if _URL_RE.search(s):
        return True
    if _AFFILIATE_JUNK_RE.search(s):
        return True
    if _QUERY_JUNK_RE.search(s):
        return True
    if _HTML_TAG_JUNK_RE.search(s):
        return True
    if _is_empty_or_trivial_quiz_field(s):
        return True
    if _looks_like_json_quiz_schema(s):
        return True
    return False


def is_junk_pair(pair: dict[str, Any]) -> bool:
    """Drop if either side is junk."""
    return is_junk_text(pair.get("base")) or is_junk_text(pair.get("revised"))


def split_junk_pairs(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []
    for pair in pairs:
        if is_junk_pair(pair):
            discarded.append(pair)
        else:
            kept.append(pair)
    return kept, discarded


def process_folder(
    input_folder: str,
    survived_folder: str,
    discarded_folder: str,
    article: str | None = None,
) -> None:
    files = resolve_article_files(input_folder, article)
    if article and not files:
        print(f"Article not found in {input_folder}: {article}")
        return

    written = 0
    skipped = 0
    pairs_in = 0
    pairs_kept = 0
    pairs_discarded = 0

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                article_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skipping {file_name}: could not read JSON")
            skipped += 1
            continue

        raw_pairs = list(article_data.get("pairs") or [])
        kept, discarded = split_junk_pairs(raw_pairs)
        pairs_in += len(raw_pairs)
        pairs_kept += len(kept)
        pairs_discarded += len(discarded)

        if not kept:
            skipped += 1
            if discarded and save_article(
                article_envelope(article_data, discarded), discarded_folder
            ):
                pass
            continue

        if not save_article(article_envelope(article_data, kept), survived_folder):
            skipped += 1
            continue

        written += 1
        if discarded:
            save_article(article_envelope(article_data, discarded), discarded_folder)

        if written % 100 == 0:
            print(f"Wrote {written} articles...")

    print(f"Done. Survived articles: {written} -> {survived_folder}")
    print(f"Skipped (no survived pairs / bad JSON): {skipped}")
    print(
        f"Pairs: {pairs_in} -> kept {pairs_kept}, "
        f"discarded {pairs_discarded} -> {discarded_folder}"
    )
    if pairs_in:
        print(f"Retention: {100.0 * pairs_kept / pairs_in:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop URL / JSON-quiz / affiliate / leftover-HTML junk pairs."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "00s-Music-Quiz" or "Hack.json"',
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FOLDER,
        help=f"Input folder (default: {INPUT_FOLDER})",
    )
    parser.add_argument(
        "--survived",
        type=str,
        default=SURVIVED_FOLDER,
        help=f"Survived output folder (default: {SURVIVED_FOLDER})",
    )
    parser.add_argument(
        "--discarded",
        type=str,
        default=DISCARDED_FOLDER,
        help=f"Discarded output folder (default: {DISCARDED_FOLDER})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        return

    create_output_folder(args.survived)
    create_output_folder(args.discarded)
    print(f"Input:     {args.input}")
    print(f"Survived:  {args.survived}")
    print(f"Discarded: {args.discarded}")
    process_folder(
        args.input,
        args.survived,
        args.discarded,
        article=args.article,
    )


if __name__ == "__main__":
    main()
