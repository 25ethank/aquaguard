"""
Single source of truth for every constant that must match between
dataset extraction, training, and the live pipeline.

If extraction and the live pipeline disagree on any of these, the model sees
a different feature distribution at inference than it was trained on.
"""
import os

# --- Video / windowing ---
FPS = 30                 # assumed frame rate when the source doesn't report one
WINDOW = 90              # frames per feature window (3 s at 30 fps)
STRIDE = 15              # step between training windows
MIN_COVERAGE = 0.7       # fraction of a window's frames that must contain a detection

# --- Detection / pose (same model + threshold for extraction AND live) ---
POSE_MODEL = os.getenv("AQUAGUARD_POSE_MODEL", "yolo26m-pose.pt")
CONF_THRESHOLD = 0.25

# --- Live pipeline ---
MIN_FILL = int(WINDOW * MIN_COVERAGE)   # 63 frames buffered before first prediction
PREDICT_EVERY = 10       # frames between predictions per track (~3 Hz at 30 fps)
MIN_TRACK_AGE = 10       # ignore tracks younger than this (kills 1-5 frame junk IDs)
STALE_AFTER = 90         # drop a track's state after this many frames unseen
ALERT_THRESHOLD = 0.85
SUSTAIN_SECONDS = {"distress": 3.0, "drowning": 2.0}
ALERT_COOLDOWN_S = 30.0  # don't re-alert on the same track within this window

# --- Services ---
API_URL = os.getenv("AQUAGUARD_API", "http://localhost:8000")
MODEL_PATH = os.getenv("AQUAGUARD_MODEL", "data/models/aquaguard_rf.pkl")
DB_PATH = os.getenv("AQUAGUARD_DB", "data/incidents.db")
