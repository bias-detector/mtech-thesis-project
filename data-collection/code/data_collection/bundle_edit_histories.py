"""Bundle per-article edit histories into one gzipped JSONL file.

Reads:  data/edit_histories/*_edit_history.json   (written by the history scraper)
Writes: data/output/edit_history.jsonl.gz         (hand-off to data-preprocessing)

Each output line is one article object, same schema as the per-article files.
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDIT_HISTORY_DIR = ROOT / "data" / "edit_histories"
EDIT_HISTORY_JSONL_GZ = ROOT / "data" / "output" / "edit_history.jsonl.gz"
SUFFIX = "_edit_history.json"


def bundle(edit_history_dir: Path, output_path: Path) -> int:
    files = sorted(edit_history_dir.glob(f"*{SUFFIX}"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # no timestamp / file name in the gzip header -> same input gives identical bytes
    with open(output_path, "wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as gz:
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            data.setdefault("article_title", path.name[: -len(SUFFIX)])
            line = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
            gz.write(line.encode("utf-8"))
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--articles-dir", type=Path, default=EDIT_HISTORY_DIR)
    parser.add_argument("-o", "--output", type=Path, default=EDIT_HISTORY_JSONL_GZ)
    args = parser.parse_args()

    n = bundle(args.articles_dir, args.output)
    print(f"Bundled {n} articles -> {args.output}")


if __name__ == "__main__":
    main()
