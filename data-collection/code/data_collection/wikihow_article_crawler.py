"""Crawl the wikiHow sitemap: categories -> subcategories -> article URLs."""

import argparse
import json
from pathlib import Path
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import requests

ROOT = Path(__file__).resolve().parents[2]
BASE_URL = "https://www.wikihow.com"
CATEGORIES_JSON = ROOT / "data" / "wikihow_categories.json"

base_url = BASE_URL


class WikiHowArticleCrawler:

    def __init__(self, base_url):
        self.base_url = base_url
        self.sitemap_url = f"{base_url}/Special:Sitemap"

    def get_all_category(self, soup):
        category_links = soup.select("div[class='cat_list'] h3 a")
        categories = {"categories": []}
        for idx, cat_link in enumerate(category_links):
            cat_url = self.base_url + cat_link["href"]
            cat_name = cat_link.text.strip()
            print(f"Category {idx + 1}: {cat_name} - {cat_url}")
            category = {
                "id": f"cat_{idx + 1}",
                "name": cat_name,
                "url": cat_url,
                "subcategories": [],
            }
            category["subcategories"] = self.get_subcategories(soup, cat_link["href"])
            categories["categories"].append(category)
        return categories

    def get_subcategories(self, soup, cat_href):
        cat_anchor = soup.select_one(f"div[class='cat_list'] h3 a[href='{cat_href}']")
        subcat_links = (
            cat_anchor.find_parent("h3").find_next_sibling("ul").select("li a")
        )
        subcategories = []
        for idx, subcat_link in enumerate(subcat_links):
            subcat_url = urljoin(self.base_url, subcat_link["href"])
            subcat_url = f"{subcat_url}?viewMode=text"
            subcat_name = subcat_link.text.strip()
            print(f"  Subcategory {idx + 1}: {subcat_name} - {subcat_url}")
            subcategory = {
                "id": f"subcat_{idx + 1}",
                "name": subcat_name,
                "url": subcat_url,
                "articles": [],
            }
            subcategory["articles"] = self.get_articles(subcat_url)
            subcategories.append(subcategory)
        return subcategories

    def get_articles(self, subcat_url):
        response = requests.get(subcat_url)
        soup = BeautifulSoup(response.text, "html.parser")
        article_links = soup.select("div[class='section minor_section'] ul li a")
        articles = []
        for idx, article_link in enumerate(article_links):
            article_url = self.base_url + article_link["href"]
            article_name = article_link.text.strip()
            print(f"    Article {idx + 1}: {article_name} - {article_url}")
            articles.append(
                {"id": f"article_{idx + 1}", "name": article_name, "url": article_url}
            )
        return articles


if __name__ == "__main__":
    argparse.ArgumentParser(
        description=f"Crawl the wikiHow sitemap into {CATEGORIES_JSON.name} (makes network requests)."
    ).parse_args()
    scraper = WikiHowArticleCrawler(base_url)
    response = requests.get(scraper.sitemap_url)
    soup = BeautifulSoup(response.text, "html.parser")
    categories = scraper.get_all_category(soup)
    CATEGORIES_JSON.parent.mkdir(parents=True, exist_ok=True)
    with CATEGORIES_JSON.open("w", encoding="utf-8") as f:
        json.dump(categories, f, ensure_ascii=False, indent=4)
    print(f"Saved categories to {CATEGORIES_JSON}")
