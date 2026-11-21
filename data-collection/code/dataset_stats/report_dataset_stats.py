"""Summarize per-article edit-history JSON under data/edit_histories/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EDIT_HISTORY_DIR = ROOT / "data" / "edit_histories"
STATS_SUMMARY_JSON = ROOT / "data" / "wikihow_stats_summary.json"
DATASET_STATS_JSON = ROOT / "data" / "dataset_stats.json"


def load_target_article_count() -> int:
    if STATS_SUMMARY_JSON.exists():
        return int(json.loads(STATS_SUMMARY_JSON.read_text())["total_articles"])
    return 0


def summarize_edit_histories(edit_history_dir: Path = EDIT_HISTORY_DIR) -> dict:
    files = (
        sorted(edit_history_dir.glob("*_edit_history.json"))
        if edit_history_dir.exists()
        else []
    )
    pairs_per_article: list[int] = []

    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        pairs_per_article.append(len(data.get("pairs") or []))

    total_pairs = sum(pairs_per_article)
    fetched = len(files)
    target = load_target_article_count()
    coverage_pct = (100.0 * fetched / target) if target else 0.0

    return {
        "articles_dir": str(edit_history_dir),
        "target_articles": target,
        "articles_saved": fetched,
        "coverage_pct": round(coverage_pct, 2),
        "total_pairs": total_pairs,
        "mean_pairs_per_article": round(
            (total_pairs / fetched) if fetched else 0.0, 1
        ),
        "median_pairs_per_article": sorted(pairs_per_article)[len(pairs_per_article) // 2]
        if pairs_per_article
        else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize per-article edit-history JSON files."
    )
    parser.add_argument(
        "--articles-dir",
        type=Path,
        default=EDIT_HISTORY_DIR,
        help=f"Directory of *_edit_history.json (default: {EDIT_HISTORY_DIR})",
    )
    args = parser.parse_args()

    stats = summarize_edit_histories(args.articles_dir)
    DATASET_STATS_JSON.parent.mkdir(parents=True, exist_ok=True)
    DATASET_STATS_JSON.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("WikiHow edit-history dataset stats")
    print("-" * 42)
    for key, value in stats.items():
        print(f"{key}: {value}")
    print(f"\nSaved {DATASET_STATS_JSON}")


if __name__ == "__main__":
    main()
