"""
Modular WikiHow markup cleaner.

Reads:  data/01_articles/*.json
Writes: data/02_markup_cleaned/<article>.json  (one file per article)

Each markup type is handled by its own filter method; clean_text() calls
them in order. Pairs that become empty on both sides are dropped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from pipeline.utils.diff_url import pair_diff_url
from pipeline.utils.io import create_output_folder, save_article, resolve_article_files
from pipeline.utils.text import fix_punct_spacing

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "01_articles")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "02_markup_cleaned")

# Media / namespace prefixes inside [[...]] that should be removed entirely.
_MEDIA_PREFIXES = (
    "image:",
    "file:",
    "media:",
    "category:",
    "special:",
    "user:",
    "user_talk:",
    "wikipedia:",
)

_MAGIC_WORD_RE = re.compile(r"__[A-Z]+__")
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_REF_BLOCK_RE = re.compile(r"<ref\b[^>]*>.*?</ref>", re.IGNORECASE | re.DOTALL)
_REF_SELF_RE = re.compile(r"<ref\b[^/]*/>", re.IGNORECASE)
_REFERENCES_RE = re.compile(r"<references\b[^>]*/?>", re.IGNORECASE)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WIKI_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")
# Allow spaces inside the URL so Amazon affiliate [https://… label] still matches.
_EXTERNAL_LINK_RE = re.compile(r"\[(https?\s*:\s*//[^\]]*)\]", re.IGNORECASE)
_TEMPLATE_RE = re.compile(r"\{\{([^{}]+)\}\}")
_HEADER_RE = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")
_BOLD_ITALIC_RE = re.compile(r"'''''(.*?)'''''")
_BOLD_RE = re.compile(r"'''(.*?)'''")
_ITALIC_RE = re.compile(r"''(.*?)''")
_LIST_MARKER_RE = re.compile(r"(?m)^[#*;:=]+")
_HTML_ENTITY_RE = re.compile(
    r"&(?:nbsp|amp|lt|gt|quot|#\d+|[a-zA-Z]+);",
    re.IGNORECASE,
)
_WHITESPACE_RE = re.compile(r"\s+")

# Affiliate / query leftovers after a spaced [url label] split.
_AFFILIATE_PLACEHOLDER_RE = re.compile(
    r"(?:#{1,2}\s*)?(?:affili?ate[\s_]*article[\s_]*id|article[\s_]*affili?ate[\s_]*id)(?:\s*#*)?",
    re.IGNORECASE,
)
_TAG_WIKIHOW_RE = re.compile(r"tag=wikihow[\w-]+", re.IGNORECASE)
_QUERY_PARAM_RE = re.compile(
    r"[?&](?:crid|sprefix|ref|k|i|tag|qid|keywords)=[^\s&\]]*",
    re.IGNORECASE,
)
_QUERY_FRAG_RE = re.compile(
    r"&(?:crid|sprefix|ref|k|i|tag|qid|keywords)=[^\s&]*",
    re.IGNORECASE,
)
_LOOSE_AMAZON_QUERY_RE = re.compile(
    r"\b(?:crid|sprefix)=[^\s&]+|\bref=nb_sb_[^\s&]*|\bnb_sb_[^\s]*|\bp_\d+_\d+&?",
    re.IGNORECASE,
)
_ENDBOLD_RE = re.compile(r"\{end[a-z]*\}", re.IGNORECASE)
# Spaced/broken tags like < /div> or < br />; does not match File > Save.
_BROKEN_HTML_TAG_RE = re.compile(
    r"<\s*/?\s*[a-zA-Z][a-zA-Z0-9]*(?:\s[^>]*)?\s*/?\s*>",
)
_STRAY_HTML_NAME_RE = re.compile(
    r"(?:^|(?<=\s))(?:/?\s*)(?:div|br|li|ul|ol|p|span|em)\s*>",
    re.IGNORECASE,
)
_BRACKET_WORDS_RE = re.compile(r"\[([^\[\]]+)\]")
_BRACKET_WORDS_OK_RE = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[\s.'/-][A-Za-z0-9][A-Za-z0-9]*)*",
)


# ---------------------------------------------------------------------------
# Individual filters (each handles one markup family)
# ---------------------------------------------------------------------------

def remove_html_comments(text: str) -> str:
    """Strip <!-- ... --> comments."""
    return _HTML_COMMENT_RE.sub(" ", text)


def remove_magic_words(text: str) -> str:
    """Strip wiki magic words like __TOC__, __SECTIONS__, __NOTOC__."""
    return _MAGIC_WORD_RE.sub(" ", text)


def remove_refs(text: str) -> str:
    """Strip citation tags and their contents (<ref>...</ref>, <references/>)."""
    text = _REF_BLOCK_RE.sub(" ", text)
    text = _REF_SELF_RE.sub(" ", text)
    text = _REFERENCES_RE.sub(" ", text)
    return text


def remove_images_and_media(text: str) -> str:
    """
    Remove media / namespace wiki links entirely.
    Handles [[Image:...]], [[File:...]], [[Media:...]], [[Category:...]], etc.
    """

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        head = inner.split("|", 1)[0].strip().lower()
        for prefix in _MEDIA_PREFIXES:
            if head.startswith(prefix):
                return " "
        # leave non-media links for expand_internal_links
        return match.group(0)

    return _WIKI_LINK_RE.sub(_replace, text)


def expand_internal_links(text: str) -> str:
    """
    Keep readable link text only.
      [[Page|label]] -> label
      [[Page]]       -> Page
      [[#anchor|x]]  -> x
    """

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if "|" in inner:
            display = inner.split("|", 1)[1].strip()
            return display if display else " "
        # bare target; drop leading # for section anchors
        target = inner.lstrip("#").strip()
        return target if target else " "

    return _WIKI_LINK_RE.sub(_replace, text)


def expand_external_links(text: str) -> str:
    """
    Keep the human label; drop the URL.
      [https://example.com label] -> label
      [https://example.com]       -> https://example.com
    Tolerates spaces inside Amazon affiliate URLs so the label is still kept.
    """

    def _replace(match: re.Match[str]) -> str:
        return _human_label_from_external(match.group(1))

    return _EXTERNAL_LINK_RE.sub(_replace, text)


def _human_label_from_external(inner: str) -> str:
    """Take trailing words after the last query param; else URL + space + label."""
    raw = (inner or "").strip()
    if not raw:
        return " "

    last = None
    for match in _QUERY_PARAM_RE.finditer(raw):
        last = match
    if last is not None:
        label = _clean_link_label(raw[last.end() :])
        if label:
            return label

    spaced = re.match(r"(https?\s*:\s*//\S+)\s+(.*)$", raw, re.IGNORECASE)
    if spaced:
        label = _clean_link_label(spaced.group(2))
        if label:
            return label

    leftover = _clean_link_label(
        re.sub(r"https?\s*:\s*//\S*", " ", raw, flags=re.IGNORECASE)
    )
    if leftover:
        return leftover
    return raw


def _clean_link_label(text: str) -> str:
    text = _AFFILIATE_PLACEHOLDER_RE.sub(" ", text or "")
    text = _TAG_WIKIHOW_RE.sub(" ", text)
    text = _QUERY_FRAG_RE.sub(" ", text)
    text = _LOOSE_AMAZON_QUERY_RE.sub(" ", text)
    text = text.strip(" &")
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_markup_leftovers(text: str) -> str:
    """
    Drop affiliate tokens, query fragments, {endbold}, and broken HTML tags.
    Keeps instructional arrows such as File > Save or Account → Help.
    """
    text = _AFFILIATE_PLACEHOLDER_RE.sub(" ", text)
    text = _TAG_WIKIHOW_RE.sub(" ", text)
    text = _QUERY_FRAG_RE.sub(" ", text)
    text = _LOOSE_AMAZON_QUERY_RE.sub(" ", text)
    text = _ENDBOLD_RE.sub(" ", text)
    text = _BROKEN_HTML_TAG_RE.sub(" ", text)
    text = _STRAY_HTML_NAME_RE.sub(" ", text)
    return text


def unwrap_bracket_words(text: str) -> str:
    """[AdSense] / [HTML code] -> the words; leave URLs and other brackets."""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        if not inner:
            return " "
        if re.match(r"https?\s*:", inner, re.IGNORECASE):
            return match.group(0)
        if "|" in inner:
            return match.group(0)
        if _BRACKET_WORDS_OK_RE.fullmatch(inner):
            return inner
        return match.group(0)

    return _BRACKET_WORDS_RE.sub(_replace, text)


def _expand_template(inner: str) -> str:
    """Expand a few contentful templates; drop structural ones."""
    parts = [p.strip() for p in inner.split("|")]
    if not parts or not parts[0]:
        return " "

    name = parts[0].lower().replace(" ", "")

    # {{keypress|Alt|Z}} -> Alt + Z
    if name in {"keypress", "kbd", "key", "button", "macbutton"}:
        keys = [p for p in parts[1:] if p]
        if not keys:
            return " "
        return " + ".join(keys)

    # {{convert|3|in|cm|abbr=on}} -> 3 in
    if name == "convert" and len(parts) >= 3:
        value, unit = parts[1], parts[2]
        return f"{value} {unit}".strip()

    # {{nowrap|text}} -> text
    if name == "nowrap" and len(parts) >= 2:
        return parts[1]

    # Structural / media / editorial templates: drop
    return " "


def remove_templates(text: str) -> str:
    """
    Expand contentful templates (keypress, convert, nowrap);
    remove structural ones (TYSK, whvid, stub, etc.).
    Runs repeatedly to clear nested / adjacent templates.
    """
    prev = None
    while prev != text:
        prev = text
        text = _TEMPLATE_RE.sub(lambda m: _expand_template(m.group(1)), text)
    return text


def remove_html_tags(text: str) -> str:
    """
    Turn <br> into a space, then strip remaining HTML tags while keeping
    inner text (e.g. <div class='subtitle'>X</div> -> X).
    """
    text = _BR_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    return text


def remove_html_entities(text: str) -> str:
    """Replace common HTML entities with spaces / plain chars."""
    replacements = {
        "&nbsp;": " ",
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
    }
    lower = text  # apply case-insensitive known entities first
    for ent, repl in replacements.items():
        lower = re.sub(re.escape(ent), repl, lower, flags=re.IGNORECASE)
    # drop any remaining entities
    lower = _HTML_ENTITY_RE.sub(" ", lower)
    return lower


def remove_bold_italic(text: str) -> str:
    """Keep inner text; drop ''' / '' / ''''' markers."""
    text = _BOLD_ITALIC_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    # leftover stray markers
    text = text.replace("'''", " ")
    text = text.replace("''", " ")
    return text


def remove_headers(text: str) -> str:
    """Keep header title text; drop == / === wrappers."""
    return _HEADER_RE.sub(r"\1", text)


def remove_list_markers(text: str) -> str:
    """
    Strip leading wiki list/indent markers from each line:
    #, #*, *, **, ##, :, ;, and mixes.
    """
    return _LIST_MARKER_RE.sub("", text)


def normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to a single space and trim."""
    return _WHITESPACE_RE.sub(" ", text).strip()


def strip_special_edges(text: str) -> str:
    """Remove leading/trailing leftover punctuation junk."""
    text = text.strip()
    text = re.sub(r"^[^A-Za-z0-9]+", "", text)
    text = re.sub(r"[^A-Za-z0-9.!?\"')\]]+$", "", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def clean_text(text: str | None) -> str:
    """Apply every markup filter, one by one, in a fixed order."""
    if text is None:
        return ""

    text = str(text)

    # 1. comments / magic words (noise first)
    text = remove_html_comments(text)
    text = remove_magic_words(text)

    # 2. citations
    text = remove_refs(text)

    # 3. media / categories (must run before general link expand)
    text = remove_images_and_media(text)

    # 4. links
    text = expand_internal_links(text)
    text = expand_external_links(text)

    # 5. templates
    text = remove_templates(text)

    # 6. HTML chrome
    text = remove_html_tags(text)
    text = remove_html_entities(text)

    # 7. bold / italic
    text = remove_bold_italic(text)

    # 8. headers and list markers
    text = remove_headers(text)
    text = remove_list_markers(text)

    # 9. leftover affiliate / HTML chrome (keep instructional > -> →)
    text = strip_markup_leftovers(text)
    text = unwrap_bracket_words(text)

    # 10. normalize
    text = normalize_whitespace(text)
    text = fix_punct_spacing(text)
    text = strip_special_edges(text)

    return text


def filter_pairs(
    pairs: list[dict[str, Any]],
    article_title: str = "",
) -> list[dict[str, Any]]:
    """Clean base/revised for each pair; drop pairs with no text left."""
    cleaned_pairs: list[dict[str, Any]] = []

    for pair in pairs:
        base = clean_text(pair.get("base", ""))
        revised = clean_text(pair.get("revised", ""))

        if base == "" or revised == "": # if any of the two is empty, skip the pair
            continue

        rec: dict[str, Any] = {
            "base": base,
            "revised": revised,
        }
        url = pair_diff_url(article_title, pair)
        if url:
            rec["diff_url"] = url
        cleaned_pairs.append(rec)

    return cleaned_pairs


def filter_article(article: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of the article with markup-cleaned pairs."""
    title = article.get("article_title") or ""
    return {
        "article_title": article.get("article_title"),
        "start_year": article.get("start_year"),
        "end_year": article.get("end_year"),
        "pairs": filter_pairs(article.get("pairs", []), title),
    }


def process_folder(
    input_folder: str,
    output_folder: str,
    *,
    article: str | None = None,
) -> None:
    """Read each stage-1 article JSON and write one cleaned file per article."""
    files = resolve_article_files(input_folder, article)
    if article and not files:
        print(f"Article not found in {input_folder}: {article}")
        return
    written = 0
    skipped = 0

    for file_name in files:
        input_path = os.path.join(input_folder, file_name)

        try:
            with open(input_path, "r", encoding="utf-8") as f:
                article = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skipping {file_name}: could not read JSON")
            skipped += 1
            continue

        filtered = filter_article(article)

        if len(filtered["pairs"]) == 0:
            skipped += 1
            continue

        if save_article(filtered, output_folder):
            written += 1
            if written % 100 == 0:
                print(f"Wrote {written} articles...")
        else:
            skipped += 1

    print(f"Done. Wrote {written} files to {output_folder}")
    print(f"Skipped {skipped} files.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strip wiki/HTML markup from article edit pairs."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "Activate-Almonds" or "Hack.json"',
    )
    args = parser.parse_args()

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    create_output_folder(OUTPUT_FOLDER)

    # Filters are applied inside clean_text(), called per pair / per article:
    #   remove_html_comments
    #   remove_magic_words
    #   remove_refs
    #   remove_images_and_media
    #   expand_internal_links
    #   expand_external_links
    #   remove_templates
    #   remove_html_tags
    #   remove_html_entities
    #   remove_bold_italic
    #   remove_headers
    #   remove_list_markers
    #   strip_markup_leftovers
    #   unwrap_bracket_words
    #   normalize_whitespace
    #   fix_punct_spacing
    #   strip_special_edges
    process_folder(INPUT_FOLDER, OUTPUT_FOLDER, article=args.article)


if __name__ == "__main__":
    main()
