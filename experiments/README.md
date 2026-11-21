# 3. Experiments

**Input:** `data/input/revision_groups.jsonl.gz` (one article per line, copied from `data-preprocessing`).
The notebooks in `1-data-preparation/` read it directly, no unpacking needed.

Run the folders in order (numbers = run order):

| Folder | Notebooks | Reads | Writes to |
| --- | --- | --- | --- |
| `code/1-data-preparation/` | `revision-groups-to-csv`, `revision-groups-to-sentence-json`, then `sentence-json-to-sentence-csv` | `data/input/`, `data/sentence/sentence-json/` | `data/paragraph/raw-csv/`, `data/sentence/sentence-json/`, `data/sentence/raw-csv/` (+ `data/sentence/sentence-csv-summary.csv`) |
| `code/2-feature-extraction/` | `kazmi-*`, `mean-mdl`, `mean-surprisal-*` (SRILM, GPT-2) | `data/<LEVEL>/raw-csv/` | `data/<LEVEL>/features/*-csv/` |
| `code/3-dataset-building/` | `build-master-sheet`, then `feature-engineering` | `data/<LEVEL>/features/` | `data/<LEVEL>/master/master-*.csv`, `data/<LEVEL>/master/feature-engineering/` |
| `code/4-analysis/` | `delta-mdl-surprisal` | `data/<LEVEL>/features/` | `data/<LEVEL>/delta/`, `output/<LEVEL>/tables/delta-per-depth-stats.csv`, `output/<LEVEL>/figures/` (`delta-by-depth/`, `delta-distribution.png`) |
| `code/5-models/` | `exp-1` … `exp-4`, `exp-5-paragraph-vs-sentence`, `results-walkthrough`, `inference/` app | `data/<LEVEL>/master/` | `output/<LEVEL>/tables/`, `output/comparison/` (exp-5), `data/paragraph/walkthrough/`, `models/inference/` |

## Folder layout

`data/` holds everything a later notebook reads; `output/` holds only final results for the thesis.

```
data/
├── input/revision_groups.jsonl.gz
├── test-articles.csv            shared 20% test split (paragraph + sentence)
├── paragraph/                   (sentence/ has the same layout, plus sentence-json/ and sentence-csv-summary.csv)
│   ├── raw-csv/                 step 1
│   ├── features/                step 2: one folder per metric, one CSV per article
│   ├── master/                  step 3: master-absolute/-pairs-step/-pairs-cumulative + feature-engineering/
│   ├── delta/                   step 4: delta-mdl-surprisal-csv/ + delta-mdl-surprisal-long.csv
│   └── walkthrough/             dummy CSVs from results-walkthrough
output/
├── <LEVEL>/
│   ├── tables/                  experiment results + delta stats (CSV)
│   └── figures/                 plots (PNG)
└── comparison/                  paragraph vs sentence (exp-5): tables/ + figures/
```

Old runs (pooled `en_core_web_sm` snapshot, whole-paragraph syntactic parse), retired notebooks
(`delta-direction-stats`, `mean-surprisal-gpt2-word`, the old `archive.zip` scripts) and the SRILM n-gram counts
(only needed to retrain the LMs) are kept outside the repo in `../../experiments-archive/` (the `mtech-thesis/` folder).

**Sentence level:** `data/sentence/raw-csv/` has the same core columns as `data/paragraph/raw-csv/`
(`revision_group_no`, `revision_depth`, `text`, `diff_url` last), where a revision group is one sentence and its
revisions, plus `paragraph_group_no`, `paragraph_depth`, `prior`, `post` and `bleu_to_previous`.

## Paragraph or sentence level

Every notebook in folders 2-5 has `LEVEL = "paragraph"` at the top of its paths cell. Set it to `"sentence"` in
each notebook to run the same steps on `data/sentence/raw-csv/` and write to `data/sentence/` and `output/sentence/`. What differs:

- **Features:** all metrics (kazmi raw / lexical / POS / syntactic, MDL, GPT-2 and SRILM surprisal) are computed per
  sentence, so the sentence master sheets have the same feature columns as the paragraph ones and all of
  experiments 1-5 run at sentence level. `feature-engineering` decides the final feature list separately per level
  (short sentences make more features undefined, so more are dropped there).
- **BLEU threshold:** `build-master-sheet` splits a sentence group wherever `bleu_to_previous <= 0.06` (the same
  threshold data-preprocessing stage 5 applies to paragraph pairs), so no pair crosses a weak link.
  `delta-mdl-surprisal` applies the same split, so the delta figures cover the same pairs (191,359 sentence pairs).
- **GPT-2 surprisal:** each sentence is scored on its own; `mean-surprisal-gpt2` puts `<|endoftext|>` in front at
  sentence level so the first token also gets a surprisal.
- **mean-mdl:** at paragraph level it writes to `data/paragraph/demo/mdl-csv` (kept for demos); at sentence level
  it writes to `data/sentence/features/mdl-csv`.
- **Same test articles:** all experiments read `data/test-articles.csv` (created on the first run from the paragraph
  split, seed 42), so paragraph and sentence results are measured on the same articles.
- `results-walkthrough` and the Streamlit UI (`inference/app.py`) are paragraph only. `train_models.py` and
  `inference.py` take the level as an argument (`python train_models.py sentence`, `--level sentence`), but only the
  paragraph model has been trained.

SRILM language models live in `models/srilm/` (`trained/*.lm` + the `ngram` / `ngram-count` binaries; the scoring
notebook recreates `tmp/` when it runs), trained pair models in `models/inference/`.
`data/`, `output/` and the large model files are gitignored.
