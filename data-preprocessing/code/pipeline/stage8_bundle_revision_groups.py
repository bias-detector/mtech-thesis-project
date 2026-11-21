"""
Bundle the per-article revision groups into one gzipped JSONL file.

Reads:  data/07_revision_groups/*.json
Writes: data/output/revision_groups.jsonl.gz   (one article per line, sorted by file name)

This file is the hand-off to experiments. The gzip header has no timestamp or
file name, so the same input always produces a byte-identical file.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT_FOLDER = os.path.join(BASE_DIR, "data", "07_revision_groups")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "output", "revision_groups.jsonl.gz")


def bundle(input_folder: str, output_file: str) -> int:
    files = sorted(name for name in os.listdir(input_folder) if name.endswith(".json"))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as gz:
        for name in files:
            with open(os.path.join(input_folder, name), encoding="utf-8") as f:
                article = json.load(f)
            line = json.dumps(article, ensure_ascii=False, separators=(",", ":")) + "\n"
            gz.write(line.encode("utf-8"))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bundle revision groups into JSONL.gz.")
    parser.add_argument("--input", default=INPUT_FOLDER)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    n = bundle(args.input, args.output)
    size_mb = os.path.getsize(args.output) / 1024 / 1024
    print(f"Bundled {n} articles -> {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
