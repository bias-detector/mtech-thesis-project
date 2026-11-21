"""Train the pair models once and save them for inference.py / app.py.

Run from anywhere:   python train_models.py              (paragraph level)
                     python train_models.py sentence     (sentence level, once its master sheets exist)
Takes a few minutes. Saves models/inference/pair-models-<LEVEL>.joblib and demo-examples-<LEVEL>.csv

All models are trained on master-pairs-step.csv (each version vs the one just before it).
The shared 20% test articles (data/test-articles.csv) are held out, and test accuracy is printed as a sanity check.
The demo examples are picked ONLY from those held-out test articles.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

LEVEL = sys.argv[1] if len(sys.argv) > 1 else "paragraph"
assert LEVEL in ("paragraph", "sentence"), LEVEL

BASE = Path(__file__).resolve().parents[3]          # .../experiments
master_dir = BASE / "data" / LEVEL / "master"
models_dir = BASE / "models" / "inference"
models_dir.mkdir(parents=True, exist_ok=True)
SEED = 42

FEATURES = pd.read_csv(master_dir / "feature-engineering" / "final_features.csv")["feature"].tolist()

# ---------- load step pairs ----------
delta_cols = ["delta_" + f for f in FEATURES]
df = pd.read_csv(master_dir / "master-pairs-step.csv",
                 usecols=["article", "diff_url", "base_text", "edit_text", *delta_cols], low_memory=False)
print("level:", LEVEL, " pairs:", len(df), " features:", len(FEATURES))

# same random A/B swap as the experiments; label 1 = B is the edit
swap = np.random.default_rng(SEED).integers(0, 2, size=len(df)).astype(bool)
base_text = df["base_text"].fillna("").astype(str).to_numpy(dtype=object)
edit_text = df["edit_text"].fillna("").astype(str).to_numpy(dtype=object)
A = np.where(swap, edit_text, base_text)
B = np.where(swap, base_text, edit_text)
y = (~swap).astype(int)
sign = np.where(swap, -1.0, 1.0)
X_feat = df[delta_cols].fillna(0).to_numpy(dtype=float) * sign[:, None]     # features(B) - features(A)

# shared 20% test articles (same file the experiment notebooks read)
articles = df["article"].astype(str).to_numpy()
test_articles = set(pd.read_csv(BASE / "data" / "test-articles.csv")["article"].astype(str))
is_test = np.isin(articles, list(test_articles))
is_train = ~is_test

# ---------- shared pieces ----------
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000)
vectorizer.fit(list(A[is_train]) + list(B[is_train]))
T = (vectorizer.transform(B) - vectorizer.transform(A)).tocsr()               # TF-IDF(B) - TF-IDF(A)

scaler = StandardScaler().fit(X_feat[is_train])
X_scaled = scaler.transform(X_feat)
s = FEATURES.index("mean_surprisal")
m = FEATURES.index("mean_mdl")


def tree():
    return HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.05, max_leaf_nodes=63,
                                          min_samples_leaf=40, l2_regularization=1.0,
                                          early_stopping=True, n_iter_no_change=50, random_state=SEED)


def logistic():
    return LogisticRegression(max_iter=2000, random_state=SEED)


models = {}
test_acc = {}

# 1. TF-IDF only
X = T
models["TF-IDF (logistic)"] = logistic().fit(X[is_train], y[is_train])
test_acc["TF-IDF (logistic)"] = models["TF-IDF (logistic)"].score(X[is_test], y[is_test])

# 2. TF-IDF + surprisal
X = hstack([T, csr_matrix(X_scaled[:, [s]])]).tocsr()
models["TF-IDF + surprisal (logistic)"] = logistic().fit(X[is_train], y[is_train])
test_acc["TF-IDF + surprisal (logistic)"] = models["TF-IDF + surprisal (logistic)"].score(X[is_test], y[is_test])

# 3. TF-IDF + surprisal + MDL
X = hstack([T, csr_matrix(X_scaled[:, [s, m]])]).tocsr()
models["TF-IDF + surprisal + MDL (logistic)"] = logistic().fit(X[is_train], y[is_train])
test_acc["TF-IDF + surprisal + MDL (logistic)"] = models["TF-IDF + surprisal + MDL (logistic)"].score(X[is_test], y[is_test])

# 4. TF-IDF + all features
all_name = f"TF-IDF + {len(FEATURES)} features (logistic)"
X = hstack([T, csr_matrix(X_scaled)]).tocsr()
models[all_name] = logistic().fit(X[is_train], y[is_train])
test_acc[all_name] = models[all_name].score(X[is_test], y[is_test])

# 5. Tree on all features
tree_name = f"Tree ({len(FEATURES)} features)"
models[tree_name] = tree().fit(X_feat[is_train], y[is_train])
test_acc[tree_name] = models[tree_name].score(X_feat[is_test], y[is_test])

# 6. Stacked: tree on all features + TF-IDF text score
#    For training rows the text score comes from 5 folds (grouped by article), so the tree never
#    sees a score from a model that was trained on that same pair.
text_score = np.zeros(len(y))
train_idx = np.where(is_train)[0]
for fit_part, score_part in GroupKFold(n_splits=5).split(train_idx, groups=articles[train_idx]):
    lr = logistic().fit(T[train_idx[fit_part]], y[train_idx[fit_part]])
    text_score[train_idx[score_part]] = lr.decision_function(T[train_idx[score_part]])
text_score[is_test] = models["TF-IDF (logistic)"].decision_function(T[is_test])
X = np.column_stack([X_feat, text_score])
stacked_name = "Stacked: tree + TF-IDF score"
models[stacked_name] = tree().fit(X[is_train], y[is_train])
test_acc[stacked_name] = models[stacked_name].score(X[is_test], y[is_test])

for name, acc in test_acc.items():
    print(f"{name:42s} test accuracy on step pairs = {acc:.4f}")

joblib.dump({"level": LEVEL, "features": FEATURES, "vectorizer": vectorizer, "scaler": scaler, "models": models,
             "test_accuracy": test_acc, "n_train": int(is_train.sum()), "n_test": int(is_test.sum())},
            models_dir / f"pair-models-{LEVEL}.joblib")
print("saved", models_dir / f"pair-models-{LEVEL}.joblib")

# ---------- demo examples: held-out test pairs only ----------
# readable length, a real change (not just whitespace), not a near-total rewrite
lo, hi = (150, 700) if LEVEL == "paragraph" else (40, 300)
test = df[is_test].copy()
test["swap"] = swap[is_test]
test["stacked_prob_B_is_edit"] = models[stacked_name].predict_proba(X[is_test])[:, 1]
length_ok = test["base_text"].fillna("").str.len().between(lo, hi) & test["edit_text"].fillna("").str.len().between(lo, hi)
changed = test["base_text"].fillna("").str.split().str.join(" ") != test["edit_text"].fillna("").str.split().str.join(" ")
ratio = test["edit_text"].fillna("").str.len() / test["base_text"].fillna("").str.len().clip(lower=1)
test = test[length_ok & changed & ratio.between(0.6, 1.8)]
examples = test.groupby("article", group_keys=False).head(1).sample(n=min(40, len(test)), random_state=SEED)
# predicted correctly by the stacked model? (for preparing the talk, not shown in the app)
examples["stacked_correct"] = (examples["stacked_prob_B_is_edit"] >= 0.5) == ~examples["swap"]
examples[["article", "diff_url", "base_text", "edit_text", "stacked_correct"]].to_csv(
    models_dir / f"demo-examples-{LEVEL}.csv", index=False)
print("saved", len(examples), "demo examples,", int(examples["stacked_correct"].sum()), "predicted correctly")
