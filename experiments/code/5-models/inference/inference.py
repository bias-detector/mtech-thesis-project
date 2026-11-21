"""Classify two versions of a text: which one is the base and which one is the edit?

Needs the models from train_models.py (run that once per level first).

    python inference.py          # runs a small demo
"""
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from features import analyse_text

BASE = Path(__file__).resolve().parents[3]          # .../experiments
MODELS_DIR = BASE / "models" / "inference"
LEVELS = ["paragraph", "sentence"]


def bundle_path(level):
    return MODELS_DIR / f"pair-models-{level}.joblib"


def available_levels():
    return [level for level in LEVELS if bundle_path(level).exists()]


@lru_cache(maxsize=None)
def load_bundle(level):
    bundle = joblib.load(bundle_path(level))
    bundle["model_names"] = sorted(bundle["models"], key=lambda name: -bundle["test_accuracy"][name])  # best first
    return bundle


def model_input(bundle, model_name, T, delta):
    """Build the input row for one model, exactly like the training rows in train_models.py."""
    features = bundle["features"]
    delta_scaled = bundle["scaler"].transform(delta)
    s, m = features.index("mean_surprisal"), features.index("mean_mdl")
    if model_name == "TF-IDF (logistic)":
        return T
    if model_name == "TF-IDF + surprisal (logistic)":
        return hstack([T, csr_matrix(delta_scaled[:, [s]])])
    if model_name == "TF-IDF + surprisal + MDL (logistic)":
        return hstack([T, csr_matrix(delta_scaled[:, [s, m]])])
    if model_name.startswith("TF-IDF +"):            # TF-IDF + all features
        return hstack([T, csr_matrix(delta_scaled)])
    if model_name.startswith("Tree"):
        return delta
    if model_name.startswith("Stacked"):
        text_score = bundle["models"]["TF-IDF (logistic)"].decision_function(T)
        return np.column_stack([delta, text_score])
    raise ValueError(model_name)


def prob_b_is_edit(bundle, model_name, text_a, text_b, feats_a, feats_b):
    vectorizer = bundle["vectorizer"]
    T = vectorizer.transform([text_b]) - vectorizer.transform([text_a])                   # TF-IDF(B) - TF-IDF(A)
    delta = np.array([[feats_b[f] - feats_a[f] for f in bundle["features"]]])             # features(B) - features(A)
    return bundle["models"][model_name].predict_proba(model_input(bundle, model_name, T, delta))[0, 1]


def explain(bundle, text_1, text_2, feats_1, feats_2, top=8):
    """Why the TF-IDF + all-features logistic model leans one way.

    Contribution of a feature towards "text 2 is the edit" = coefficient x (delta / training std).
    Averaging both orderings cancels the scaler mean and intercept, so that is exactly what the
    order-averaged score adds up. Same for words: coefficient x (TF-IDF(text 2) - TF-IDF(text 1)).
    """
    features = bundle["features"]
    name = next(n for n in bundle["models"] if n.startswith("TF-IDF +") and "features" in n)
    coef = bundle["models"][name].coef_[0]
    vectorizer = bundle["vectorizer"]
    n_words = len(vectorizer.vocabulary_)
    word_coef, feat_coef = coef[:n_words], coef[n_words:]

    delta = np.array([feats_2[f] - feats_1[f] for f in features])
    feature_contrib = feat_coef * delta / bundle["scaler"].scale_
    order = np.argsort(-np.abs(feature_contrib))[:top]
    feature_rows = [{"feature": features[i], "text_1": feats_1[features[i]], "text_2": feats_2[features[i]],
                     "contribution": float(feature_contrib[i])} for i in order if feature_contrib[i] != 0]

    T = (vectorizer.transform([text_2]) - vectorizer.transform([text_1])).toarray()[0]
    word_contrib = word_coef * T
    vocab = vectorizer.get_feature_names_out()
    order = np.argsort(-np.abs(word_contrib))[:top]
    word_rows = [{"term": vocab[i], "in": "text 2" if T[i] > 0 else "text 1", "contribution": float(word_contrib[i])}
                 for i in order if word_contrib[i] != 0]
    return {"model": name, "features": feature_rows, "words": word_rows,
            "feature_total": float(feature_contrib.sum()), "word_total": float(word_contrib.sum())}


def analyse_pair(text_1, text_2, level="paragraph"):
    """Features of both texts, every model's verdict and an explanation."""
    bundle = load_bundle(level)
    a1, a2 = analyse_text(text_1, level), analyse_text(text_2, level)
    f1, f2 = a1["features"], a2["features"]
    results = []
    for name in bundle["model_names"]:
        # ask both ways round and average, so swapping the two boxes gives the same answer
        p = (prob_b_is_edit(bundle, name, text_1, text_2, f1, f2)
             + 1 - prob_b_is_edit(bundle, name, text_2, text_1, f2, f1)) / 2
        results.append({"model": name, "test_accuracy": bundle["test_accuracy"][name], "p_text_2_is_edit": float(p)})
    return {"level": level, "text_1": a1, "text_2": a2, "results": results,
            "explanation": explain(bundle, text_1, text_2, f1, f2)}


def classify(text_1, text_2, model_name=None, level="paragraph"):
    """Short answer for one model (kept for scripts)."""
    out = analyse_pair(text_1, text_2, level)
    r = next(r for r in out["results"] if model_name in (None, r["model"]))
    p = r["p_text_2_is_edit"]
    return {"model": r["model"], "text_1": "edit" if p < 0.5 else "base", "text_2": "edit" if p >= 0.5 else "base",
            "score_1": 1 - p, "score_2": p}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Which of two texts is the edit? (no texts = run the pasta demo)")
    parser.add_argument("text_1", nargs="?", help="first version")
    parser.add_argument("text_2", nargs="?", help="second version")
    parser.add_argument("--level", default="paragraph", choices=LEVELS)
    parser.add_argument("--arcs", action="store_true", help="also print every dependency arc behind the MDL")
    args = parser.parse_args()
    if (args.text_1 is None) != (args.text_2 is None):
        parser.error("give both texts, or none for the demo")
    text_1 = args.text_1 or "Fill a pot with water and put it on the stove. Wait until it boils, then add the pasta."
    text_2 = args.text_2 or ("Fill a large pot with 4 quarts of water and place it on the stove over high heat. "
                             "Once the water reaches a rolling boil, add 1 pound of pasta and stir.")

    out = analyse_pair(text_1, text_2, args.level)
    best = out["results"][0]
    p = best["p_text_2_is_edit"]
    print(f"\nVerdict ({best['model']}): Text {2 if p >= 0.5 else 1} is the edit  (P(text 2 is edit) = {p:.2f})\n")
    for r in out["results"]:
        print(f"  {r['model']:42s} test acc={r['test_accuracy']:.3f}  P(text 2 is edit)={r['p_text_2_is_edit']:.2f}")

    for name in ["text_1", "text_2"]:
        a = out[name]
        print(f"\n{name.replace('_', ' ').title()}: mean surprisal {a['features']['mean_surprisal']:.2f} bits, "
              f"MDL {a['features']['mean_mdl']:.2f}")
        for i, s in enumerate(a["mdl_breakdown"], start=1):
            mdl = "no arc, skipped" if s["mdl"] is None else f"{s['mdl']:.2f}"
            print(f"  sentence {i}: root = {s['root']} ({s['root_pos']}), {len(s['arcs'])} arcs, MDL {mdl}  | {s['sentence']}")
            if args.arcs:
                for arc in s["arcs"]:
                    print(f"      {arc['word']:>15} --{arc['relation']}--> {arc['head']:<15} length {arc['length']}")

    print("\nTop feature contributions (+ = towards text 2 is the edit):")
    for row in out["explanation"]["features"][:5]:
        print(f"  {row['feature']:28s} {row['contribution']:+.3f}")
