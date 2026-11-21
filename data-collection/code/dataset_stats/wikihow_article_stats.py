"""Count categories, subcategories and articles in the sitemap catalog."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CATEGORIES_JSON = ROOT / "data" / "wikihow_categories.json"
STATS_SUMMARY_JSON = ROOT / "data" / "wikihow_stats_summary.json"


def calculate_stats(filename=CATEGORIES_JSON):
    """Calculate statistics from wikihow categories JSON file"""

    with open(filename, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = data.get("categories", [])

    # Total number of categories
    total_categories = len(categories)

    # Overall stats
    total_articles = 0
    total_subcategories = 0
    category_stats = []

    # Process each category
    for category in categories:
        cat_name = category.get("name")
        cat_id = category.get("id")
        subcategories = category.get("subcategories", [])

        # Number of subcategories in this category
        num_subcats = len(subcategories)
        total_subcategories += num_subcats

        # Number of articles in this category
        num_articles = 0
        for subcat in subcategories:
            articles = subcat.get("articles", [])
            num_articles += len(articles)

        total_articles += num_articles

        category_stats.append(
            {
                "category": cat_name,
                "id": cat_id,
                "subcategories": num_subcats,
                "articles": num_articles,
            }
        )

    # Print results
    print("=" * 70)
    print("WIKIHOW STATISTICS")
    print("=" * 70)

    print("\nOVERALL STATS:")
    print(f"   Total Categories: {total_categories}")
    print(f"   Total Subcategories: {total_subcategories}")
    print(f"   Total Articles: {total_articles}")
    print(f"   Average Articles per Category: {total_articles / total_categories:.2f}")
    print(f"   Average Subcategories per Category: {total_subcategories / total_categories:.2f}")

    print("\nCATEGORY BREAKDOWN:")
    print(f"{'-' * 70}")
    print(f"{'Category':<40} {'Subcats':<10} {'Articles':<10}")
    print(f"{'-' * 70}")

    for stat in category_stats:
        cat_name = stat["category"][:37] + "..." if len(stat["category"]) > 40 else stat["category"]
        print(
            f"{cat_name:<40} {stat['subcategories']:<10} {stat['articles']:<10}"
        )

    print(f"{'-' * 70}")
    print(f"{'TOTAL':<40} {total_subcategories:<10} {total_articles:<10}")
    print("=" * 70)

    # Return stats as dict for further processing
    return {
        "total_categories": total_categories,
        "total_subcategories": total_subcategories,
        "total_articles": total_articles,
        "category_stats": category_stats,
    }


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=f"Summarize {CATEGORIES_JSON.name} into {STATS_SUMMARY_JSON.name}."
    ).parse_args()
    stats = calculate_stats()

    STATS_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with STATS_SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=4)
    print(f"\nStats saved to {STATS_SUMMARY_JSON}")
