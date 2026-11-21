"""Run the preprocessing stages in order.

Run from data-preprocessing/code/:

    python -m pipeline.run_all                       # stages 1 -> 8, full corpus
    python -m pipeline.run_all --from 4              # resume at stage 4
    python -m pipeline.run_all --article 00s-Music-Quiz   # one article, stages 2 -> 7

Stages 1 (split the input bundle) and 8 (bundle the output) always work on
the whole corpus, so they are skipped when --article is given.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

STAGES = [
    "stage1_split_articles",
    "stage2_clean_markup",
    "stage3_filter_unchanged",
    "stage4_filter_vandalism",
    "stage5_filter_low_bleu",
    "stage6_filter_junk",
    "stage7_build_revision_groups",
    "stage8_bundle_revision_groups",
]
WHOLE_CORPUS_ONLY = {1, 8}


def main() -> None:
    n = len(STAGES)
    parser = argparse.ArgumentParser(description="Run preprocessing stages in order.")
    parser.add_argument("--from", dest="start", type=int, default=1, choices=range(1, n + 1))
    parser.add_argument("--to", dest="end", type=int, default=n, choices=range(1, n + 1))
    parser.add_argument("--article", default=None, help="Process one article (stages 2-7).")
    args = parser.parse_args()

    for number in range(args.start, args.end + 1):
        module = STAGES[number - 1]
        cmd = [sys.executable, "-m", f"pipeline.{module}"]
        if args.article:
            if number in WHOLE_CORPUS_ONLY:
                print(f"--- stage {number} skipped (always processes the whole corpus)")
                continue
            cmd += ["--article", args.article]

        print(f"--- stage {number}: {module}", flush=True)
        started = time.time()
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(f"stage {number} failed (exit code {result.returncode})")
        print(f"--- stage {number} done in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
