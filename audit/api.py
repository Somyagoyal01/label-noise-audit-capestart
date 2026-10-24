"""FastAPI: POST /audit with a CSV (numeric feature columns + a 'label' column) -> review queue.

curl -F file=@batch.csv "http://localhost:8000/audit?top=50"
"""
import io

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile

from .core import audit

app = FastAPI(title="Label-noise audit (demo)")


@app.post("/audit")
async def run_audit(file: UploadFile = File(...), top: int = 50, label_col: str = "label"):
    df = pd.read_csv(io.BytesIO(await file.read()))
    if label_col not in df:
        raise HTTPException(400, f"missing '{label_col}' column")
    X = df.drop(columns=[label_col]).select_dtypes("number").to_numpy(float)
    y = df[label_col].to_numpy()
    counts = pd.Series(y).value_counts()
    if len(df) < 50 or counts.min() < 5:
        raise HTTPException(400, "need >= 50 rows and >= 5 rows per class for 5-fold CV")
    res = audit(X, y)
    rows = [{"row": int(i), "given_label": str(y[i]), "suggested_label": str(res.suggested[i]),
             "suspicion": round(float(res.score[i]), 3), "flagged": bool(res.flagged[i])}
            for i in res.order[:top]]
    return {"n_rows": len(df), "n_flagged": int(np.sum(res.flagged)), "review_queue": rows}
