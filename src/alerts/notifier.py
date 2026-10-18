"""
SMS alerts via Twilio.

Credentials come from environment variables (put them in .env, never in git):
  TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_FROM, ALERT_TO

If any are missing the notifier runs in dry-run mode and just prints the
message, so the rest of the system works without a Twilio account.
"""
import os


class SMSNotifier:
    def __init__(self):
        self.sid = os.getenv('TWILIO_ACCOUNT_SID')
        self.token = os.getenv('TWILIO_AUTH_TOKEN')
        self.from_ = os.getenv('TWILIO_FROM')
        self.to = [n.strip() for n in os.getenv('ALERT_TO', '').split(',') if n.strip()]
        self.enabled = bool(self.sid and self.token and self.from_ and self.to)
        self._client = None

    @staticmethod
    def format(inc):
        return (f"AquaGuard ALERT: {inc.get('alert_type', inc['label'])}. "
                f"Swimmer {inc['person_id']}, {inc['confidence']:.0%} confidence"
                f"{', held ' + str(inc['sustained_s']) + 's' if inc.get('sustained_s') else ''}. "
                f"{inc.get('timestamp', '')}")

    def send(self, inc):
        """Returns a short status string that gets stored with the incident."""
        body = os.getenv('TWILIO_TRIAL_TEMPLATE') or self.format(inc)
        if not self.enabled:
            print(f"[SMS dry-run] {body}")
            return 'skipped: twilio not configured'
        try:
            if self._client is None:
                from twilio.rest import Client
                self._client = Client(self.sid, self.token)
            for number in self.to:
                self._client.messages.create(body=body, from_=self.from_, to=number)
            return f'sent to {len(self.to)}'
        except Exception as e:      # an SMS outage must never take down the API
            print(f"[SMS failed] {e}")
            return f'failed: {type(e).__name__}'
