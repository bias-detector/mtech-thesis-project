# 2. Data preprocessing

Turns raw edit pairs into clean per-article revision groups.

**Input:** `data/input/edit_history.jsonl.gz` (copied from `data-collection`).

## Run

From `data-preprocessing/code/`:

```bash
cd data-preprocessing/code
python -m pipeline.run_all                          # stages 1 -> 8, full corpus
python -m pipeline.run_all --from 4                 # resume at stage 4
python -m pipeline.run_all --article 00s-Music-Quiz # one article, stages 2 -> 7
python -m pipeline.stage3_filter_unchanged          # or run a single stage
```

## Stages

| Stage | Module | Input | Output |
| ---: | --- | --- | --- |
| 1 | `stage1_split_articles` | `data/input/edit_history.jsonl.gz` | `data/01_articles/` |
| 2 | `stage2_clean_markup` | `data/01_articles/` | `data/02_markup_cleaned/` |
| 3 | `stage3_filter_unchanged` | `data/02_markup_cleaned/` | `data/03_changed/` (+ `data/discarded/03_unchanged/`) |
| 4 | `stage4_filter_vandalism` | `data/03_changed/` | `data/04_vandal_free/` (+ `data/discarded/04_vandalised/`) |
| 5 | `stage5_filter_low_bleu` | `data/04_vandal_free/` | `data/05_bleu_passed/` (+ `data/discarded/05_low_bleu/`) |
| 6 | `stage6_filter_junk` | `data/05_bleu_passed/` | `data/06_junk_free/` (+ `data/discarded/06_junk/`) |
| 7 | `stage7_build_revision_groups` | `data/06_junk_free/` | `data/07_revision_groups/` |
| 8 | `stage8_bundle_revision_groups` | `data/07_revision_groups/` | `data/output/revision_groups.jsonl.gz` (hand-off file) |

Stages 2–7 accept `--article <slug>`. Stages 1 and 8 always process the whole corpus.

## Code layout

```
code/pipeline/
├── run_all.py          runs the stages in order
├── stage1_… stage8_…   one file per stage (numbers match the data/0N_ folders)
└── utils/
    ├── io.py           read/write one JSON file per article
    ├── text.py         punctuation-spacing fix and comparison key (stages 2, 3)
    ├── diff_url.py     wikiHow diff URLs and revision ordering (stages 1, 2, 4, 7)
    └── bleu.py         sentence BLEU (stage 5)
```

## Hand-off to experiments

Copy the bundle into the next stage's input folder (run from the repo root):

```bash
mkdir -p experiments/data/input
cp data-preprocessing/data/output/revision_groups.jsonl.gz experiments/data/input/
```
