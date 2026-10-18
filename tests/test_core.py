import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.alerts.policy import AlertPolicy
from src.features.feature_engineer import FEATURE_NAMES, FeatureEngineer
from tests.synthetic import swimmer

fe = FeatureEngineer(fps=30)


def test_feature_order_and_keys():
    f = fe.extract_features(swimmer('normal', 90))
    assert list(f) == FEATURE_NAMES


def test_features_are_window_length_invariant():
    """A 63-frame live buffer must look like a 90-frame training window of the same behavior."""
    for b in ('normal', 'distress', 'drowning'):
        long = fe.extract_features(swimmer(b, 90, seed=1))
        short = fe.extract_features(swimmer(b, 63, seed=1))
        for k in ('left_wrist_freq_hz', 'vertical_velocity', 'body_angle_mean'):
            assert short[k] == pytest.approx(long[k], rel=0.35, abs=0.6), (b, k)


def test_behaviors_separate_on_key_features():
    n, d, w = (fe.extract_features(swimmer(b, 90)) for b in ('normal', 'distress', 'drowning'))
    assert n['body_angle_mean'] < 30 < d['body_angle_mean']
    assert d['left_wrist_freq_hz'] > w['left_wrist_freq_hz']
    assert np.isnan(w['leg_activity'])          # submerged ankles gated out


def test_to_vector_is_json_safe():
    v = FeatureEngineer.to_vector(fe.extract_features(swimmer('drowning', 90)))
    assert len(v) == len(FEATURE_NAMES)
    assert v[FEATURE_NAMES.index('leg_activity')] is None


def test_windows_from_frames_coverage_gate():
    from src.features.extract_dataset import windows_from_frames
    kp = swimmer('normal', 200)
    frames = [(i, kp[i]) for i in range(200)]
    full = windows_from_frames(frames, 'normal', 'v.mp4', fe)
    assert len(full) == (200 - 90) // 15 + 1
    # drop detection in 40% of frames -> every window fails 70% coverage
    holey = [(i, None if i % 5 < 2 else k) for i, k in frames]
    assert windows_from_frames(holey, 'normal', 'v.mp4', fe) == []


def P(distress=0.0, drowning=0.0):
    return {'distress': distress, 'drowning': drowning, 'normal': 1 - distress - drowning}


def test_alert_policy_requires_sustain_and_respects_cooldown():
    p = AlertPolicy(threshold=0.85, sustain={'drowning': 2.0, 'distress': 3.0}, cooldown=30)
    assert p.update('P1', P(drowning=0.9), 0.0) is None
    assert p.update('P1', P(drowning=0.9), 1.9) is None
    a = p.update('P1', P(drowning=0.9), 2.0)
    assert a and a['label'] == 'drowning'
    assert p.update('P1', P(drowning=0.9), 5.0) is None          # cooldown
    assert p.update('P1', P(drowning=0.9), 32.5) is not None     # cooldown over


def test_alert_policy_resets_when_danger_drops():
    p = AlertPolicy(threshold=0.85, sustain={'drowning': 2.0, 'distress': 3.0}, cooldown=30)
    p.update('P1', P(drowning=0.9), 0.0)
    p.update('P1', P(drowning=0.05), 1.0)              # looks normal -> streak broken
    assert p.update('P1', P(drowning=0.9), 2.5) is None
    assert p.update('P3', P(distress=0.5), 99.0) is None


def test_alert_policy_survives_distress_drowning_flipping():
    """Model confuses drowning<->distress on held-out video; that must not reset the timer."""
    p = AlertPolicy(threshold=0.85, sustain={'drowning': 2.0, 'distress': 3.0}, cooldown=30)
    seq = [P(drowning=0.6, distress=0.35), P(distress=0.55, drowning=0.4),
           P(drowning=0.5, distress=0.45), P(drowning=0.7, distress=0.25)]
    alerts = [p.update('P1', pr, t) for pr, t in zip(seq, (0.0, 0.7, 1.4, 2.1))]
    assert alerts[:3] == [None, None, None]
    assert alerts[3] and alerts[3]['label'] == 'drowning' and alerts[3]['confidence'] >= 0.85


def test_alert_policy_distress_key_spelled_right():
    """Regression: SUSTAIN used to say 'distresss', so distress could never alert."""
    from src import config
    assert set(config.SUSTAIN_SECONDS) == {'distress', 'drowning'}
