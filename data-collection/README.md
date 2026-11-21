# 1. Data collection

Scrapes wikiHow article edit histories and bundles them for preprocessing.
Run every script from this folder (`data-collection/`).

| Step | Script | Output |
| ---: | --- | --- |
| 1 | `code/data_collection/wikihow_article_crawler.py` | `data/wikihow_categories.json` (sitemap catalog) |
| 2 | `code/data_collection/wikihow_history_scraper.py` | `data/edit_histories/<Article>_edit_history.json` (`base` / `revised` pairs) |
| 3 | `code/data_collection/bundle_edit_histories.py` | `data/output/edit_history.jsonl.gz` (one article per line) |

Dataset statistics (optional):

| Script | Output |
| --- | --- |
| `code/dataset_stats/wikihow_article_stats.py` | `data/wikihow_stats_summary.json` (articles per category) |
| `code/dataset_stats/report_dataset_stats.py` | `data/dataset_stats.json` (coverage, pairs per article) |

Optional cookie for some history pages, in `data-collection/.env`:

```
WIKIHOW_COOKIE='...'
```

## Hand-off to data-preprocessing

Copy the bundle into the next stage's input folder (run from the repo root):

```bash
mkdir -p data-preprocessing/data/input
cp data-collection/data/output/edit_history.jsonl.gz data-preprocessing/data/input/
```
