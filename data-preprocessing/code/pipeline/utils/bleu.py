"""
Sentence-level BLEU used by stage 5 (filter_low_bleu).

Matches MTP1 / wikihow-edit sentence BLEU (case-insensitive, method1 smoothing).
"""

from __future__ import annotations

import os
import tempfile
from typing import cast


def _import_nltk_bleu():
    # Same cwd workaround as other scripts for project-local .venv / NLTK.
    previous = os.getcwd()
    os.chdir(tempfile.gettempdir())
    try:
        from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
    finally:
        os.chdir(previous)
    return sentence_bleu, SmoothingFunction


sentence_bleu, SmoothingFunction = _import_nltk_bleu()
_SMOOTH = SmoothingFunction().method1

# Historical band used by analyze_pair_bleu retention reports (old group filter).
SIMILARITY_THRESHOLD_LB = 0.3
SIMILARITY_THRESHOLD_UB = 0.9


def bleu_similarity(base: str, revised: str) -> float:
    """Sentence-level BLEU between base and revised (case-insensitive)."""
    reference = str(base).lower().split()
    hypothesis = str(revised).lower().split()
    if not reference or not hypothesis:
        return 0.0
    return cast(
        float,
        sentence_bleu([reference], hypothesis, smoothing_function=_SMOOTH),
    )
