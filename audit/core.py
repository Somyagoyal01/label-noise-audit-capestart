"""Label-noise audit for annotated datasets.

Idea (a light version of "confident learning", Northcutt et al. 2021): train a model with K-fold
cross-validation so every item gets an out-of-fold predicted probability. Items whose given label gets a
low probability, while another class gets a high one, are likely annotation errors. Rank them so a
reviewer checks the most suspicious items first instead of re-reviewing a random sample.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def default_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.5))


def out_of_fold_proba(X, y, model_fn=default_model, k: int = 5, seed: int = 0):
    """Out-of-fold class probabilities; columns follow np.unique(y)."""
    classes = np.unique(y)
    proba = np.zeros((len(y), len(classes)))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for tr, te in skf.split(X, y):
        m = model_fn().fit(X[tr], y[tr])
        cols = np.searchsorted(classes, m.classes_)
        proba[np.ix_(te, cols)] = m.predict_proba(X[te])
    return proba, classes


@dataclass
class AuditResult:
    order: np.ndarray            # indices sorted most-suspicious first
    score: np.ndarray            # suspicion score per item (higher = more likely mislabeled)
    suggested: np.ndarray        # model's suggested label per item
    flagged: np.ndarray          # boolean mask: passes the confident-learning threshold


def audit(X, y, model_fn=default_model, k: int = 5, seed: int = 0) -> AuditResult:
    y = np.asarray(y)
    proba, classes = out_of_fold_proba(X, y, model_fn, k, seed)
    idx = np.searchsorted(classes, y)
    p_given = proba[np.arange(len(y)), idx]
    # per-class threshold: mean self-confidence of items labelled with that class
    thresh = np.array([p_given[idx == c].mean() for c in range(len(classes))])
    other = proba.copy()
    other[np.arange(len(y)), idx] = -1
    best_other = other.argmax(1)
    p_other = other.max(1)
    flagged = (p_given < thresh[idx]) & (p_other >= thresh[best_other])
    score = p_other - p_given  # margin by which another class beats the given label
    return AuditResult(np.argsort(-score), score, classes[best_other], flagged)


def inject_noise(y, rate: float, seed: int = 0, mode: str = "uniform", pairs=None):
    """Return (noisy_y, noisy_mask). uniform: random wrong class. pairs: systematic swaps like 1->7."""
    rnd = np.random.default_rng(seed)
    y = np.asarray(y).copy()
    classes = np.unique(y)
    mask = np.zeros(len(y), bool)
    if mode == "uniform":
        pick = rnd.choice(len(y), int(rate * len(y)), replace=False)
        for i in pick:
            y[i] = rnd.choice(classes[classes != y[i]])
        mask[pick] = True
    elif mode == "pairs":
        for a, b in pairs:
            cand = np.where(y == a)[0]
            pick = rnd.choice(cand, int(rate * len(cand)), replace=False)
            y[pick] = b
            mask[pick] = True
    return y, mask


def detection_metrics(res: AuditResult, noisy_mask, review_budget: float = 0.1):
    n = len(noisy_mask)
    k = int(review_budget * n)
    top = res.order[:k]
    found_topk = noisy_mask[top].sum()
    random_expect = noisy_mask.mean() * k
    f = res.flagged
    prec = (f & noisy_mask).sum() / max(f.sum(), 1)
    rec = (f & noisy_mask).sum() / max(noisy_mask.sum(), 1)
    return {
        "n_items": int(n), "n_noisy": int(noisy_mask.sum()),
        "review_budget_items": k,
        "noisy_found_in_budget": int(found_topk),
        "recall_at_budget": round(found_topk / max(noisy_mask.sum(), 1), 3),
        "lift_vs_random_review": round(found_topk / max(random_expect, 1e-9), 2),
        "flagged": int(f.sum()), "flag_precision": round(float(prec), 3), "flag_recall": round(float(rec), 3),
    }


def gbm_model():
    return HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1)
