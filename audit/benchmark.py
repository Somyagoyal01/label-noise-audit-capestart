"""Benchmark the audit.

DESIGN setting (used while building): sklearn digits, 8% uniform random label noise, logistic regression.
HELD-OUT settings (run after the design was frozen, never used to pick anything):
  A. digits with SYSTEMATIC confusions (annotators mixing 1<->7, 3<->8, 4<->9) at 15% of those classes
  B. a different dataset (sklearn wine, tabular chemistry features) with 10% uniform noise
  C. digits, 8% uniform noise, but a different model (gradient boosting)
Reviewer-time estimate assumes 30 s per item re-check (an assumption, labelled as such).

Usage: python -m audit.benchmark
"""
import json

from sklearn.datasets import load_digits, load_wine

from .core import audit, detection_metrics, gbm_model, inject_noise

SECONDS_PER_REVIEW = 30  # assumption


def run(name, X, y, noise_kwargs, model_fn=None, budget=0.1):
    y_noisy, mask = inject_noise(y, **noise_kwargs)
    res = audit(X, y_noisy, **({"model_fn": model_fn} if model_fn else {}))
    m = detection_metrics(res, mask, budget)
    # reviewer hours to find the same number of errors by random sampling vs. ranked audit
    rate = mask.mean()
    random_items = m["noisy_found_in_budget"] / rate if rate else 0
    m["est_review_hours_random"] = round(random_items * SECONDS_PER_REVIEW / 3600, 2)
    m["est_review_hours_ranked"] = round(m["review_budget_items"] * SECONDS_PER_REVIEW / 3600, 2)
    return {"setting": name, **m}


def main():
    d, w = load_digits(), load_wine()
    out = [
        run("DESIGN digits uniform 8%", d.data, d.target, {"rate": 0.08, "seed": 1}),
        run("HELD-OUT A digits systematic swaps 15%", d.data, d.target,
            {"rate": 0.15, "seed": 2, "mode": "pairs", "pairs": [(1, 7), (7, 1), (3, 8), (8, 3), (4, 9), (9, 4)]}),
        run("HELD-OUT B wine uniform 10%", w.data, w.target, {"rate": 0.10, "seed": 3}, budget=0.15),
        run("HELD-OUT C digits uniform 8% (GBM model)", d.data, d.target, {"rate": 0.08, "seed": 1}, model_fn=gbm_model),
    ]
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    main()
