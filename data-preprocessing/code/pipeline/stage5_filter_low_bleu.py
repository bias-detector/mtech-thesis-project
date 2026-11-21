"""
Drop pairs whose base→revised BLEU is at or below a lower-bound threshold.

Keeps edges where:
    BLEU > SIMILARITY_THRESHOLD_LB  (default 0.06)

Uses the same BLEU definition as utils/bleu.py (case-insensitive,
NLTK sentence_bleu, method1 smoothing).

Reads:  data/04_vandal_free/*.json
Writes:
  survived  -> data/05_bleu_passed/<article>.json
  discarded -> data/discarded/05_low_bleu/<article>.json  (only if any dropped)
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

from pipeline.utils.bleu import bleu_similarity
from pipeline.utils.io import create_output_folder, save_article, article_envelope, resolve_article_files

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "04_vandal_free")
SURVIVED_FOLDER = os.path.join(BASE_DIR, "data", "05_bleu_passed")
DISCARDED_FOLDER = os.path.join(BASE_DIR, "data", "discarded", "05_low_bleu")

# Manual inspection of pair_bleu reports: pairs at/below this are too unrelated.
SIMILARITY_THRESHOLD_LB = 0.06


def split_low_bleu_pairs(
    pairs: list[dict[str, Any]],
    threshold_lb: float = SIMILARITY_THRESHOLD_LB,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep pairs with BLEU > threshold_lb; discard the rest (incl. empty)."""
    kept: list[dict[str, Any]] = []
    discarded: list[dict[str, Any]] = []

    for pair in pairs:
        base = pair.get("base", "")
        revised = pair.get("revised", "")
        if not str(base).strip() or not str(revised).strip():
            discarded.append(pair)
            continue
        score = bleu_similarity(base, revised)
        if score > threshold_lb:
            kept.append(pair)
        else:
            discarded.append(pair)

    return kept, discarded


def process_folder(
    input_folder: str,
    survived_folder: str,
    discarded_folder: str,
    article: str | None = None,
    threshold_lb: float = SIMILARITY_THRESHOLD_LB,
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
        kept, discarded = split_low_bleu_pairs(raw_pairs, threshold_lb=threshold_lb)
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
        print(
            f"Retention (BLEU > {threshold_lb}): "
            f"{100.0 * pairs_kept / pairs_in:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop pairs with BLEU at or below a lower-bound threshold."
    )
    parser.add_argument(
        "--article",
        type=str,
        default=None,
        help='Process one article only, e.g. "Hack" or "Hack.json"',
    )
    parser.add_argument(
        "--threshold-lb",
        type=float,
        default=SIMILARITY_THRESHOLD_LB,
        help=f"Keep pairs with BLEU > this value (default: {SIMILARITY_THRESHOLD_LB})",
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
    print(f"Filtering with BLEU > {args.threshold_lb}")
    print(f"Input:     {args.input}")
    print(f"Survived:  {args.survived}")
    print(f"Discarded: {args.discarded}")
    process_folder(
        args.input,
        args.survived,
        args.discarded,
        article=args.article,
        threshold_lb=args.threshold_lb,
    )


if __name__ == "__main__":
    main()
