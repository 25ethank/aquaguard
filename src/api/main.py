"""
AquaGuard model-serving API.

  uvicorn src.api.main:app --reload

GET  /health      model metadata
GET  /features    exact feature order the model expects
POST /predict     feature vector -> label, confidence, per-class probabilities
POST /incidents   log an alert (SQLite) and send SMS in the background
GET  /incidents   recent incidents, newest first
"""
import datetime
import os
import pickle
import sys
from contextlib import asynccontextmanager
from typing import List, Optional

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.append('.')
try:
    from dotenv import load_dotenv
    load_dotenv()                      # picks up Twilio creds from .env
except ImportError:
    pass
from src import config
from src.alerts.notifier import SMSNotifier
from src.api.db import IncidentDB

state = {}


def load_bundle(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Model not found at {path}. Run: python src/models/train_baseline.py")
    with open(path, 'rb') as f:
        return pickle.load(f)


@asynccontextmanager
async def lifespan(app):
    b = load_bundle(os.getenv('AQUAGUARD_MODEL', config.MODEL_PATH))
    state.update(model=b['model'], encoder=b['encoder'], feature_names=b['feature_names'],
                 train_info=b.get('metrics', {}),
                 db=IncidentDB(os.getenv('AQUAGUARD_DB', config.DB_PATH)),
                 sms=SMSNotifier())
    yield
    state['db'].conn.close()


app = FastAPI(title="AquaGuard API", version="1.1", lifespan=lifespan)


class Sample(BaseModel):
    person_id: str
    features: List[Optional[float]]     # None = feature unavailable (e.g. ankles submerged)
    timestamp: str


class Incident(BaseModel):
    person_id: str
    label: str
    confidence: float = Field(ge=0, le=1)
    alert_type: Optional[str] = None
    sustained_s: Optional[float] = None
    source: Optional[str] = None
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok",
            "classes": list(state['encoder'].classes_),
            "n_features": len(state['feature_names']),
            "alert_threshold": config.ALERT_THRESHOLD,
            "sms_enabled": state['sms'].enabled,
            "incidents_logged": state['db'].count(),
            "training": state['train_info']}


@app.get("/features")
def features():
    return {"feature_names": state['feature_names']}


@app.post("/predict")
def predict(s: Sample):
    names = state['feature_names']
    if len(s.features) != len(names):
        raise HTTPException(400, f"Expected {len(names)} features, got {len(s.features)}")
    # None -> NaN; the model pipeline's median imputer fills it, exactly as in training
    x = np.array([np.nan if v is None else v for v in s.features], dtype=float).reshape(1, -1)
    if np.isinf(x).any():
        raise HTTPException(400, "features contain inf")
    proba = state['model'].predict_proba(x)[0]
    idx = int(np.argmax(proba))
    label = str(state['encoder'].classes_[idx])
    conf = float(proba[idx])
    return {
        "person_id": s.person_id,
        "label": label,
        "confidence": round(conf, 3),
        "proba": {str(c): round(float(p), 3) for c, p in zip(state['encoder'].classes_, proba)},
        # single-window flag only; the pipeline's AlertPolicy requires it to be sustained
        "danger": label in ('distress', 'drowning') and conf >= config.ALERT_THRESHOLD,
    }


def _notify(incident_id, inc):
    state['db'].set_sms_status(incident_id, state['sms'].send(inc))


@app.post("/incidents")
def log_incident(inc: Incident, bg: BackgroundTasks):
    row = inc.model_dump()
    row['logged_at'] = datetime.datetime.now().isoformat(timespec='seconds')
    row['sms_status'] = 'pending'
    incident_id = state['db'].add(row)
    bg.add_task(_notify, incident_id, row)      # respond immediately; SMS goes out after
    return {"id": incident_id, "total": state['db'].count()}


@app.get("/incidents")
def get_incidents(limit: int = 100):
    rows = state['db'].recent(min(max(limit, 1), 1000))
    return {"incidents": rows, "total": state['db'].count()}
