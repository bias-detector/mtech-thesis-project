"""
Build per-article revision groups with keep-all competing edges.

Walks pairs from the bottom of each article file upward, consumes each edge
at most once, and when several unused edges share the same base (matched via
punctuation-spacing-normalized keys), extends with the chronologically next
successor. Leftover competing edges become their own groups instead of being
dropped.

No vandalism/reversion collapse is applied here, so competing edges are not
silently removed after being consumed.

Input pairs (from 06_junk_free) look like:
    {base, revised, diff_url}

Output one JSON per article in data/07_revision_groups/:
    {
      "article_title": "...",
      "start_year": ...,
      "end_year": ...,
      "groups": [
        {
          "versions": ["base", "r1", "r2", ...],
          "diff_urls": [null, "<url for r1>", "<url for r2>", ...]
        }
      ]
    }
"""

import argparse
import json
import os
import re
from collections import defaultdict

from pipeline.utils.diff_url import pair_diff_url, revision_time_key
from pipeline.utils.io import create_output_folder, save_article, resolve_article_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "06_junk_free")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "07_revision_groups")

_WHITESPACE_RE = re.compile(r"\s+")
# Align with utils.text.norm_key spacing rules (lookup key only).
_SPACE_BEFORE_CLOSE_RE = re.compile(r"\s+([.,;:!?)\]}”\"'’])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([(\[{“\"'‘])\s+")


def chain_key(text):
    """
    Normalize tip/base for chaining so spacing/case variants link.

    Handles: space before .,;:!? / )]} / closing quotes; space after ([{ /
    opening quotes; casefold. Does not strip space before '(' after a word.

    Keeps original version strings in output; only used as a lookup key.
    """
    t = _WHITESPACE_RE.sub(" ", str(text or "")).strip()
    if not t:
        return ""
    t = _SPACE_BEFORE_CLOSE_RE.sub(r"\1", t)
    t = _SPACE_AFTER_OPEN_RE.sub(r"\1", t)
    return t.casefold()


def filter_pairs_for_grouping(pairs):
    """Drop empty or case/whitespace-only identical pairs before chaining."""
    kept = []
    for pair in pairs:
        base = pair.get("base", "")
        revised = pair.get("revised", "")
        if not str(base).strip() or not str(revised).strip():
            continue
        if str(base).strip().lower() == str(revised).strip().lower():
            continue
        kept.append(pair)
    return kept


def pick_chronological_successor(candidate_indices, pairs, current_index):
    """
    Among unused edges with the same base, pick the chronologically next one
    after the current edge. Returns the pair index, or None if none later.
    """
    current_key = revision_time_key(pairs[current_index])
    later = [
        j
        for j in candidate_indices
        if revision_time_key(pairs[j]) > current_key
    ]
    if not later:
        return None
    return min(later, key=lambda j: revision_time_key(pairs[j]))


def build_groups_for_article(pairs, article_title=""):
    """
    Chain pairs bottom-up with keep-all competing edges.

    - Walk unused pairs from the end of the list toward the start.
    - Each edge is consumed at most once (so B->C is not emitted again once
      absorbed into A->B->C).
    - When several unused edges share a tip base, extend with the
      chronologically next successor; leftover edges form their own groups.
    """
    pairs = filter_pairs_for_grouping(pairs)
    if not pairs:
        return []

    by_base = defaultdict(list)
    for index, pair in enumerate(pairs):
        by_base[chain_key(pair["base"])].append(index)

    used = set()
    groups = []

    for start in range(len(pairs) - 1, -1, -1):
        if start in used:
            continue

        edge = pairs[start]
        versions = [edge["base"], edge["revised"]]
        diff_urls = [None, pair_diff_url(article_title, edge)]
        used.add(start)

        current_index = start
        visited = {chain_key(versions[0]), chain_key(versions[1])}

        while True:
            tip = versions[-1]
            # Skip edges that would cycle back into this chain; leave them
            # unused so they can still form their own groups later.
            candidates = [
                j
                for j in by_base.get(chain_key(tip), [])
                if j not in used
                and chain_key(pairs[j].get("revised", "")) not in visited
            ]
            if not candidates:
                break

            nxt_index = pick_chronological_successor(
                candidates, pairs, current_index
            )
            if nxt_index is None:
                break

            nxt = pairs[nxt_index]
            revised = nxt["revised"]
            used.add(nxt_index)
            visited.add(chain_key(revised))
            versions.append(revised)
            diff_urls.append(pair_diff_url(article_title, nxt))
            current_index = nxt_index

        if len(versions) <= 1:
            continue

        groups.append(
            {
                "versions": versions,
                "diff_urls": diff_urls,
            }
        )

    return groups


def construct_article(article):
    """Return article dict with revision groups."""
    title = article.get("article_title") or ""
    return {
        "article_title": article.get("article_title"),
        "start_year": article.get("start_year"),
        "end_year": article.get("end_year"),
        "groups": build_groups_for_article(article.get("pairs", []), title),
    }


def process_folder(input_folder, output_folder, article=None):
    """Read each filtered article and write revision groups."""
    files = resolve_article_files(input_folder, article)
    if article and not files:
        print(f"Article not found in {input_folder}: {article}")
        return
    written = 0
    skipped = 0
    groups_total = 0

    for index, file_name in enumerate(files, start=1):
        input_path = os.path.join(input_folder, file_name)
        try:
            with open(input_path, "r", encoding="utf-8") as f:
                article = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skipping {file_name}: could not read JSON")
            skipped += 1
            continue

        result = construct_article(article)
        if len(result["groups"]) == 0:
            skipped += 1
            continue

        if save_article(result, output_folder):
            written += 1
            groups_total += len(result["groups"])
            if written % 50 == 0:
                print(f"Wrote {written} articles... ({index}/{len(files)} processed)")
        else:
            skipped += 1

    print(f"Done. Wrote {written} files to {output_folder}")
    print(f"Skipped {skipped} files.")
    print(f"Total groups: {groups_total}")


def main():
    parser = argparse.ArgumentParser(
        description="Chain edit pairs into per-article revision groups."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "Activate-Almonds" or "Hack.json"',
    )
    parser.add_argument(
        "--input",
        type=str,
        default=INPUT_FOLDER,
        help=f"Input folder (default: {INPUT_FOLDER})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=OUTPUT_FOLDER,
        help=f"Output folder (default: {OUTPUT_FOLDER})",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"Input folder not found: {args.input}")
        print("Run the clean-pair filters first, or pass another --input directory.")
        return

    create_output_folder(args.output)
    process_folder(args.input, args.output, article=args.article)


if __name__ == "__main__":
    main()
