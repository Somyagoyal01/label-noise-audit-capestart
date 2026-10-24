import io

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.datasets import load_digits, load_wine

from audit.api import app
from audit.core import audit, detection_metrics, inject_noise


def test_inject_noise_rate_and_labels_change():
    y = load_digits().target
    yn, mask = inject_noise(y, 0.1, seed=0)
    assert abs(mask.mean() - 0.1) < 0.01
    assert np.all(yn[mask] != y[mask]) and np.all(yn[~mask] == y[~mask])


def test_pairs_mode_only_touches_listed_classes():
    y = load_digits().target
    yn, mask = inject_noise(y, 0.2, seed=0, mode="pairs", pairs=[(1, 7)])
    assert set(y[mask]) == {1} and set(yn[mask]) == {7}


def test_ranked_audit_beats_random_review():
    d = load_digits()
    yn, mask = inject_noise(d.target, 0.08, seed=1)
    m = detection_metrics(audit(d.data, yn), mask, 0.1)
    assert m["lift_vs_random_review"] > 5 and m["flag_precision"] > 0.8


def test_clean_data_flags_little():
    w = load_wine()
    res = audit(w.data, w.target)
    assert res.flagged.mean() < 0.08


def test_api_returns_queue():
    w = load_wine()
    yn, _ = inject_noise(w.target, 0.1, seed=3)
    df = pd.DataFrame(w.data, columns=w.feature_names).assign(label=yn)
    buf = io.BytesIO(df.to_csv(index=False).encode())
    r = TestClient(app).post("/audit?top=10", files={"file": ("b.csv", buf, "text/csv")})
    assert r.status_code == 200 and len(r.json()["review_queue"]) == 10


def test_api_rejects_tiny_batch():
    df = pd.DataFrame({"a": range(10), "label": [0, 1] * 5})
    r = TestClient(app).post("/audit", files={"file": ("b.csv", io.BytesIO(df.to_csv(index=False).encode()), "text/csv")})
    assert r.status_code == 400
