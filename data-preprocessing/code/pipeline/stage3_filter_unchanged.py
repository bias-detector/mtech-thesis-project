"""
Drop unchanged pairs where base and revised are the same text.

Reads:  data/02_markup_cleaned/*.json
Writes:
  survived  -> data/03_changed/<article>.json
  discarded -> data/discarded/03_unchanged/<article>.json  (only if any dropped)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any
from pipeline.utils.io import create_output_folder, save_article, article_envelope, resolve_article_files
from pipeline.utils.text import norm_key

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "02_markup_cleaned")
SURVIVED_FOLDER = os.path.join(BASE_DIR, "data", "03_changed")
DISCARDED_FOLDER = os.path.join(BASE_DIR, "data", "discarded", "03_unchanged")


def split_unchanged_pairs(
    pairs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep pairs that change text; discard base == revised (and empty both)."""
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []

    for pair in pairs:
        base = norm_key(pair.get("base"))
        revised = norm_key(pair.get("revised"))
        if not base and not revised:
            discarded.append(pair)
            continue
        if base == revised:
            discarded.append(pair)
            continue
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
                article = json.load(f)
        except (json.JSONDecodeError, OSError):
            print(f"Skipping {file_name}: could not read JSON")
            skipped += 1
            continue

        raw_pairs = list(article.get("pairs") or [])
        kept, discarded = split_unchanged_pairs(raw_pairs)
        pairs_in += len(raw_pairs)
        pairs_kept += len(kept)
        pairs_discarded += len(discarded)

        if not kept:
            skipped += 1
            if discarded and save_article(
                article_envelope(article, discarded), discarded_folder
            ):
                pass
            continue

        if not save_article(article_envelope(article, kept), survived_folder):
            skipped += 1
            continue

        written += 1
        if discarded:
            save_article(article_envelope(article, discarded), discarded_folder)

        if written % 100 == 0:
            print(f"Wrote {written} articles...")

    print(f"Done. Survived articles: {written} -> {survived_folder}")
    print(f"Skipped (no survived pairs / bad JSON): {skipped}")
    print(
        f"Pairs: {pairs_in} -> kept {pairs_kept}, "
        f"discarded {pairs_discarded} -> {discarded_folder}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop unchanged (base == revised) pairs."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "Hack" or "Hack.json"',
    )
    args = parser.parse_args()

    if not os.path.isdir(INPUT_FOLDER):
        print(f"Input folder not found: {INPUT_FOLDER}")
        return

    create_output_folder(SURVIVED_FOLDER)
    create_output_folder(DISCARDED_FOLDER)
    process_folder(
        INPUT_FOLDER, SURVIVED_FOLDER, DISCARDED_FOLDER, article=args.article
    )


if __name__ == "__main__":
    main()
