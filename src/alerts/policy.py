"""
Decides *when* a classifier output becomes an alert.

Kept separate from the video loop so it can be unit-tested without a camera.
A single confident prediction never alerts on its own: danger must hold
continuously for SUSTAIN_SECONDS, mirroring how a lifeguard confirms a scan
before committing to a rescue.

Danger is scored as P(distress) + P(drowning). On held-out video the model
often confuses those two with each other but rarely with normal, so tracking
them separately would let a real victim whose label flips between
"distress" and "drowning" reset the timer and never alert.
"""
from src import config


class AlertPolicy:
    def __init__(self, threshold=config.ALERT_THRESHOLD,
                 sustain=config.SUSTAIN_SECONDS,
                 cooldown=config.ALERT_COOLDOWN_S):
        self.threshold = threshold
        self.sustain = dict(sustain)        # danger label -> seconds required
        self.cooldown = cooldown
        self._since = {}        # key -> t when the current danger streak started
        self._last_alert = {}   # key -> t of last alert

    def danger(self, proba):
        """(combined danger probability, most likely danger label)."""
        p = sum(proba.get(l, 0.0) for l in self.sustain)
        label = max(self.sustain, key=lambda l: proba.get(l, 0.0))
        return p, label

    def update(self, key, proba, t):
        """
        Feed one prediction (class -> probability) for one track at video time t (s).
        Returns an alert dict when an alert should fire, else None.
        """
        p, label = self.danger(proba)
        if p < self.threshold:
            self._since.pop(key, None)
            return None

        start = self._since.setdefault(key, t)
        held = t - start
        if held < self.sustain[label]:
            return None
        if t - self._last_alert.get(key, float('-inf')) < self.cooldown:
            return None

        self._last_alert[key] = t
        return {"person_id": key, "label": label, "confidence": round(p, 3),
                "sustained_s": round(held, 2)}

    def forget(self, key):
        self._since.pop(key, None)
        self._last_alert.pop(key, None)
