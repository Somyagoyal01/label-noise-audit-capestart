# Label-Noise Audit: find likely annotation errors before they reach a client's model

> Personal project by Somya Goyal, built while applying to CapeStart. Not affiliated with or endorsed by CapeStart; uses public/synthetic data.

## Why this matters for CapeStart
- **Problem:** CapeStart's public site describes complex data annotation and data preparation for ML, alongside model development, for 100+ clients. In annotation work, a few wrong labels quietly lower the model trained on them. Random-sample QA finds those errors slowly.
- **Who feels it:** annotation QA leads (re-review hours), ML engineers (noisy training data), and the client (model accuracy, turnaround).
- **What this shows:** a model-assisted QA step. Out-of-fold predictions rank every labelled item by how likely its label is wrong, so reviewers check the most suspicious ~10% first. It is served as a FastAPI endpoint that takes a CSV batch and returns a review queue.
- **Expected impact (estimate, from this repo's benchmark):** in the design setting, the ranked queue surfaced 137 of 143 injected errors by re-checking 179 items. Random review would need about 1,720 items for the same 137 errors: ~1.5 h vs ~14 h at an *assumed* 30 s per item.

Sources: capestart.com and CapeStart careers page (Associate Machine Learning Engineer JD, 23 Oct 2026: "develop, train, evaluate, and deploy ML models … build and maintain APIs … monitor model performance"). Method: a simplified version of confident learning (Northcutt, Jiang & Chuang, 2021).

## Demo
```bash
pip install -r requirements.txt
python -m audit.benchmark            # design + 3 held-out settings
pytest -q                            # 6 tests
uvicorn audit.api:app --reload       # POST /audit (multipart CSV with a 'label' column)
```

## How it works
1. 5-fold stratified CV gives every item an out-of-fold class probability, so the model never scores an item it was trained on.
2. Each class gets a threshold: the mean confidence of items carrying that label.
3. An item is **flagged** when its given label falls below its class threshold and another class reaches that class's threshold.
4. All items are **ranked** by `p(best other class) − p(given label)`. Reviewers work from the top.

Files: `audit/core.py` (CV probabilities, audit, noise injection, metrics) · `audit/benchmark.py` · `audit/api.py` · `tests/`.

## Results (measured; `python -m audit.benchmark`)
| Setting | Noisy items | Found in review budget | Recall @ budget | Lift vs random | Flag precision |
|---|---|---|---|---|---|
| **Design:** digits, 8% uniform noise, logistic regression | 143 | 137 / 179 reviewed | **95.8%** | 9.6× | 93.6% |
| **Held-out A:** digits, *systematic* annotator confusions (1↔7, 3↔8, 4↔9) at 15% | 163 | 130 / 179 | **79.8%** | 8.0× | 78.9% |
| **Held-out B:** different dataset (wine chemistry, tabular), 10% uniform | 17 | 15 / 26 | 88.2% | 6.0× | 91.7% |
| **Held-out C:** digits, 8% uniform, gradient-boosting model | 143 | 143 / 179 | 100% | 10.0× | 92.9% |

The design was frozen before the held-out settings were run. The useful lesson is held-out A: when annotators make **the same** mistake consistently, the model partly learns it, and recall drops by ~16 points. Real annotation errors look more like A than like uniform noise.

## Limitations / next steps
- Noise is injected into clean public datasets. Real annotation errors also come from ambiguous items, not only mistakes.
- For systematic confusions: train on a gold-verified subset, or use per-annotator agreement as a feature.
- Text and image annotation would use embeddings (e.g. a sentence-transformer or a CNN backbone) as features. The audit logic stays the same.
- Add drift monitoring per batch (share of flagged items over time) to catch a new annotator or guideline problem early.
