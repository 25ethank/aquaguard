from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pickle
import numpy as np
import datetime

app = FastAPI(title="AquaGuard API", version="1.0")

bundle = pickle.load(open('data/models/aquaguard_rf.pkl', 'rb'))
model = bundle['model']
encoder = bundle['encoder']
feature_names = bundle['feature_names']

incidents = []

class Sample(BaseModel):
    person_id: str
    features: List[Optional[float]]
    timestamp: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(encoder.classes_),
        "n_features": len(feature_names)
    }

@app.get("/features")
def features():
    """The exact feature order the model expects."""
    return {"feature_names": feature_names}


@app.post("/predict")
def predict(s: Sample):
    if len(s.features) != len(feature_names):
        raise HTTPException(400,
            f"Expected {len(feature_names)} features, got {len(s.features)}")
    try:
        x = np.array([np.nan if v is None else v for v in s.features], dtype=float).reshape(1, -1)
        proba = model.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        return {
            "person_id": s.person_id,
            "label": encoder.inverse_transform([idx])[0],
            "confidence": round(float(proba[idx]), 3),
            "proba": {c: round(float(p), 3)
                      for c, p in zip(encoder.classes_, proba)}
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/incidents")
def log_incident(inc: dict):
    inc['logged_at'] = datetime.datetime.now().isoformat()
    incidents.append(inc)
    return {"total": len(incidents)}


@app.get("/incidents")
def get_incidents():
    return {"incidents": incidents, "total": len(incidents)}
