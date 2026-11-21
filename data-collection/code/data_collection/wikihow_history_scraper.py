"""Scrape edit pairs (base / revised) from each article's wikiHow history diffs."""

import argparse
import os
import json
import time
import random
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.wikihow.com"
CATEGORIES_JSON = ROOT / "data" / "wikihow_categories.json"
EDIT_HISTORY_DIR = ROOT / "data" / "edit_histories"

load_dotenv(ROOT / ".env")

base_url = BASE_URL


class WikiHowHistoryScraper:

    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        cookie = os.getenv("WIKIHOW_COOKIE", "")
        if cookie:
            cookies = {
                k.strip(): v.strip()
                for k, v in [pair.split("=", 1) for pair in cookie.split(";") if "=" in pair]
            }
            self.session.cookies.update(cookies)
        self.input_json = CATEGORIES_JSON
        self.output_dir = EDIT_HISTORY_DIR
        self.checkpoint_file = EDIT_HISTORY_DIR / "edit_history.done.txt"
        self.start_year = 2005
        self.end_year = 2026
        self.limit = 100

    def _get(self, url, retries=3):
        for attempt in range(retries):
            try:
                resp = self.session.get(url, timeout=30)
                if resp.status_code in (429, 503):
                    time.sleep(2 ** (attempt + 1) + random.uniform(0.1, 0.5))
                    continue
                resp.raise_for_status()
                return resp
            except requests.RequestException:
                if attempt == retries - 1:
                    return None
                time.sleep(2 ** (attempt + 1) + random.uniform(0.1, 0.5))
        return None

    def _slug_from_record(self, record):
        url = record.get("url", "")
        if url:
            path = urlparse(url).path.strip("/")
            if path:
                return path
        title = record.get("title", "")
        return title.replace(" ", "-")

    def load_article_slugs(self):
        slugs = []
        seen = set()

        with open(self.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        categories = data.get("categories", []) if isinstance(data, dict) else []
        for category in categories:
            subcategories = category.get("subcategories", [])
            for subcategory in subcategories:
                articles = subcategory.get("articles", [])
                for article in articles:
                    record = {
                        "url": article.get("url", ""),
                        "title": article.get("name", ""),
                    }
                    slug = self._slug_from_record(record)
                    if slug and slug not in seen:
                        slugs.append(slug)
                        seen.add(slug)
                    if len(slugs) >= self.limit:
                        return slugs
        return slugs

    def load_done_articles(self):
        if not self.checkpoint_file.exists():
            return set()
        with open(self.checkpoint_file, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def extract_revision_ids(self, article_slug, year):
        url = (
            f"{self.base_url}/index.php?title={article_slug}&action=history"
            f"&year={year}&month=-1&tagfilter="
        )
        resp = self._get(url)
        if not resp:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        revision_ids = []
        for tag in soup.find_all("input", {"name": "diff", "type": "radio"}):
            value = tag.get("value")
            if value:
                revision_ids.append(value)
        return revision_ids

    def extract_pairs_for_diff(self, article_slug, diff_id, old_id):
        diff_url = (
            f"{self.base_url}/index.php?title={article_slug}&type=revision"
            f"&diff={diff_id}&oldid={old_id}"
        )
        resp = self._get(diff_url)
        if not resp:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        pairs = []
        for row in soup.find_all("tr"):
            deleted = row.find("td", class_="diff-deletedline")
            added = row.find("td", class_="diff-addedline")
            if deleted and added:
                pairs.append(
                    {
                        "base": deleted.get_text(" ", strip=True),
                        "revised": added.get_text(" ", strip=True),
                        "diff_id": diff_id,
                        "old_id": old_id,
                    }
                )
        return pairs

    def crawl_article(self, article_slug):
        all_pairs = []
        for year in range(self.start_year, self.end_year + 1):
            revision_ids = self.extract_revision_ids(article_slug, year)
            print(f"{article_slug} | Year {year}: Found {len(revision_ids)} revisions")

            for index in range(0, len(revision_ids) - 1):
                diff_id = revision_ids[index]
                old_id = revision_ids[index + 1]
                pairs = self.extract_pairs_for_diff(article_slug, diff_id, old_id)
                if pairs:
                    all_pairs.extend(pairs)
                time.sleep(random.uniform(0.8, 1.8))

            time.sleep(random.uniform(1.0, 2.0))
        return all_pairs

    def append_result(self, article_slug, pairs):
        row = {
            "article_title": article_slug,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "pairs": pairs,
        }
        safe = (
            article_slug.replace("%", "%25")
            .replace("/", "%2F")
            .replace("\\", "%5C")
            .replace(":", "%3A")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / f"{safe}_edit_history.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(row, f, ensure_ascii=False, indent=4)

    def mark_done(self, article_slug):
        self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        with self.checkpoint_file.open("a", encoding="utf-8") as fp:
            fp.write(article_slug + "\n")
            fp.flush()

    def run(self):
        article_slugs = self.load_article_slugs()
        done = self.load_done_articles()
        print(f"Loaded {len(article_slugs)} articles, already done {len(done)}")

        for idx, article_slug in enumerate(article_slugs, start=1):
            if article_slug in done:
                print(
                    f"[{idx}/{len(article_slugs)}] Skipping done article: {article_slug}"
                )
                continue

            print(f"[{idx}/{len(article_slugs)}] Crawling article: {article_slug}")
            pairs = self.crawl_article(article_slug)
            self.append_result(article_slug, pairs)
            self.mark_done(article_slug)
            print(
                f"[{idx}/{len(article_slugs)}] Saved {len(pairs)} pairs: {article_slug}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Scrape edit histories into {EDIT_HISTORY_DIR.name}/ (makes network requests)."
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Maximum number of articles to scrape (default: 100)",
    )
    args = parser.parse_args()
    scraper = WikiHowHistoryScraper(base_url)
    scraper.limit = args.limit
    scraper.run()
