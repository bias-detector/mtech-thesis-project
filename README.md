# M.Tech thesis: cognitive load in wikiHow edits

Three stages, run in order. Each folder has its own `code/`, a gitignored `data/`, and a README.

Each stage is self-contained: it reads only from its own `data/input/` and writes only inside its own folder.
What a stage hands on is in its `data/output/`; copy it into the next stage's `data/input/` (the command is at the end of each stage's README).

| Stage | Folder | Input | Output |
| --- | --- | --- | --- |
| 1 | [`data-collection/`](data-collection/) | wikiHow site | `data/output/edit_history.jsonl.gz` |
| 2 | [`data-preprocessing/`](data-preprocessing/) | `data/input/edit_history.jsonl.gz` | `data/output/revision_groups.jsonl.gz` |
| 3 | [`experiments/`](experiments/) | `data/input/revision_groups.jsonl.gz` | metrics, models, results in `output/` |

Other folders: `paper/` (reference papers).

## Setup

```bash
python3.12 -m venv thesis-venv
source thesis-venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
```

## History

Stages 1 and 2 were previously separate repos, now archived:
`bias-detector/wikihow-edit` (collection) and `bias-detector/wikihow-text-analysis` (preprocessing).
Classification and training code from `wikihow-text-analysis` was intentionally not carried over.
