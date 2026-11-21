"""
Split the input bundle into one JSON file per article.

Reads:  data/input/edit_history.jsonl.gz   (one article per line)
Writes: data/01_articles/<article>.json     (per-pair diff_id/old_id replaced by a diff_url)
"""

import argparse
import gzip
import json
import os

from pipeline.utils.diff_url import pair_diff_url
from pipeline.utils.io import create_output_folder, save_article

# paths relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FILE = os.path.join(BASE_DIR, "data", "input", "edit_history.jsonl.gz")
INPUT_FILE_FALLBACK = os.path.join(BASE_DIR, "data", "input", "edit_history.jsonl")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "data", "01_articles")


def convert_pair_urls(article):
    """Replace per-pair diff_id/old_id with a clickable diff_url."""
    title = article.get("article_title") or ""
    converted = []
    for pair in article.get("pairs") or []:
        rec = {
            "base": pair.get("base"),
            "revised": pair.get("revised"),
        }
        url = pair_diff_url(title, pair)
        if url:
            rec["diff_url"] = url
        converted.append(rec)
    article["pairs"] = converted
    return article


def _open_text(path):
    """Open a plain or gzip-compressed text file for reading."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def process_jsonl(input_file, output_folder):
    """Read the JSONL file and dump each line as its own JSON file."""
    written = 0
    skipped = 0

    with _open_text(input_file) as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if line == "":
                continue

            try:
                article = json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping line {line_number}: could not parse JSON")
                skipped += 1
                continue

            saved = save_article(convert_pair_urls(article), output_folder)
            if saved:
                written += 1
                if written % 100 == 0:
                    print(f"Wrote {written} articles...")
            else:
                print(f"Skipping line {line_number}: missing article_title")
                skipped += 1

    print(f"Done. Wrote {written} files to {output_folder}")
    print(f"Skipped {skipped} lines.")


def main():
    argparse.ArgumentParser(
        description="Split the input bundle into one JSON file per article (data/01_articles/)."
    ).parse_args()
    input_file = INPUT_FILE if os.path.isfile(INPUT_FILE) else INPUT_FILE_FALLBACK
    if not os.path.isfile(input_file):
        print(f"Input file not found: {INPUT_FILE} (or {INPUT_FILE_FALLBACK})")
        return

    create_output_folder(OUTPUT_FOLDER)
    process_jsonl(input_file, OUTPUT_FOLDER)


if __name__ == "__main__":
    main()
