from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
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
    features: List[float]
    timestamp: str

@app.get("/health")
def health():
    return {
        "status": "ok",
        "classes": list(encoder.classes_),
        "n_features": len(feature_names)
    }
