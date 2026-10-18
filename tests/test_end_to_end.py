"""
Synthetic data -> features.csv -> train_baseline -> API -> incidents.
Verifies the whole software path without video, YOLO, or Twilio.
"""
import json
import os
import sys

import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.features.extract_dataset import windows_from_frames
from src.features.feature_engineer import FeatureEngineer
from tests.synthetic import swimmer


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    work = tmp_path_factory.mktemp('work')
    os.makedirs(work / 'data/extracted_features')
    fe = FeatureEngineer(fps=30)
    rows = []
    for b in ('normal', 'distress', 'drowning'):
        for v in range(4):                       # 4 "videos" per class for grouped CV
            kp = swimmer(b, 240, seed=hash((b, v)) % 1000)
            rows += windows_from_frames([(i, kp[i]) for i in range(240)], b, f'{b}_{v}.mp4', fe)
    pd.DataFrame(rows).to_csv(work / 'data/extracted_features/features.csv', index=False)

    cwd = os.getcwd()
    os.chdir(work)
    from src import config
    config.MODEL_PATH = 'data/models/aquaguard_rf.pkl'
    from src.models import train_baseline
    train_baseline.main()
    os.chdir(cwd)
    return work


def test_training_outputs(trained):
    m = json.load(open(trained / 'reports/metrics.json'))
    assert m['cv_folds'] >= 2 and m['n_videos'] == 12
    assert m['report']['accuracy'] > 0.9      # synthetic classes are easy; checks wiring
    assert (trained / 'reports/confusion_matrix.png').exists()


def test_api(trained, monkeypatch):
    for k in ('TWILIO_ACCOUNT_SID', 'TWILIO_AUTH_TOKEN', 'TWILIO_FROM', 'ALERT_TO'):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv('AQUAGUARD_MODEL', str(trained / 'data/models/aquaguard_rf.pkl'))
    monkeypatch.setenv('AQUAGUARD_DB', str(trained / 'incidents.db'))
    from fastapi.testclient import TestClient
    from src.api.main import app

    fe = FeatureEngineer(fps=30)
    with TestClient(app) as c:
        h = c.get('/health').json()
        assert h['classes'] == ['distress', 'drowning', 'normal'] and not h['sms_enabled']

        # predict from a 63-frame buffer (live MIN_FILL), model trained on 90-frame windows
        for b in ('normal', 'distress', 'drowning'):
            vec = FeatureEngineer.to_vector(fe.extract_features(swimmer(b, 63, seed=777)))
            r = c.post('/predict', json={'person_id': 'P1', 'features': vec, 'timestamp': 'now'})
            assert r.status_code == 200, r.text
            assert r.json()['label'] == b, (b, r.json())

        bad = c.post('/predict', json={'person_id': 'P1', 'features': [1.0], 'timestamp': 'x'})
        assert bad.status_code == 400

        inc = {'person_id': 'P7', 'label': 'drowning', 'confidence': 0.93,
               'alert_type': 'MOTIONLESS SWIMMER', 'sustained_s': 2.1, 'timestamp': 't'}
        assert c.post('/incidents', json=inc).json()['total'] == 1
        got = c.get('/incidents').json()
        assert got['total'] == 1
        assert got['incidents'][0]['sms_status'].startswith('skipped')

    # persists across restart
    with TestClient(app) as c:
        assert c.get('/incidents').json()['total'] == 1
