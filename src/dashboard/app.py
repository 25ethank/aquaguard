"""
Lifeguard-facing incident dashboard.

  streamlit run src/dashboard/app.py

Polls the API every few seconds; needs the API running.
"""
import os
import sys

import pandas as pd
import requests
import streamlit as st

sys.path.append('.')
from src import config

API = os.getenv('AQUAGUARD_API', config.API_URL)
REFRESH_S = 3

st.set_page_config(page_title="AquaGuard", page_icon="🌊", layout="wide")
st.title("AquaGuard — Pool Monitor")


def fetch(path, **params):
    try:
        r = requests.get(f"{API}{path}", params=params, timeout=2)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"error": str(e)}


with st.sidebar:
    st.header("System")
    h = fetch("/health")
    if "error" in h:
        st.error(f"API unreachable at {API}")
        st.caption("Start it with `uvicorn src.api.main:app`")
    else:
        st.success("API online")
        st.write(f"**Classes:** {', '.join(h['classes'])}")
        st.write(f"**Alert threshold:** {h['alert_threshold']:.0%}")
        st.write(f"**SMS:** {'enabled' if h['sms_enabled'] else 'dry-run'}")
        if h.get('training'):
            t = h['training']
            st.caption(f"Model trained on {t.get('n_windows')} windows "
                       f"from {t.get('n_videos')} videos")
    limit = st.slider("Incidents to show", 10, 500, 100, step=10)


@st.fragment(run_every=REFRESH_S)
def live():
    data = fetch("/incidents", limit=limit)
    if "error" in data:
        st.warning("Waiting for API…")
        return

    df = pd.DataFrame(data["incidents"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total incidents", data["total"])
    if df.empty:
        c2.metric("Drowning", 0)
        c3.metric("Distress", 0)
        c4.metric("Last alert", "—")
        st.info("No incidents logged. All swimmers nominal.")
        return

    df["logged_at"] = pd.to_datetime(df["logged_at"])
    c2.metric("Drowning", int((df["label"] == "drowning").sum()))
    c3.metric("Distress", int((df["label"] == "distress").sum()))
    c4.metric("Last alert", df["logged_at"].max().strftime("%H:%M:%S"))

    latest = df.iloc[0]
    box = st.error if latest["label"] == "drowning" else st.warning
    box(f"**Latest:** {latest['alert_type'] or latest['label']} — swimmer "
        f"{latest['person_id']} at {latest['confidence']:.0%} "
        f"({latest['logged_at']:%H:%M:%S}) · SMS: {latest['sms_status']}")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Incident log")
        show = df[["logged_at", "person_id", "label", "confidence",
                   "sustained_s", "sms_status", "source"]].copy()
        show["confidence"] = (show["confidence"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(show, hide_index=True, width='stretch')
    with right:
        st.subheader("Alerts per minute")
        per_min = (df.set_index("logged_at").groupby("label")
                     .resample("1min").size().unstack(0, fill_value=0))
        st.bar_chart(per_min)


live()
