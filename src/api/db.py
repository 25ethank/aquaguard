"""SQLite incident log. One file, no server, survives restarts."""
import os
import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS incidents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id   TEXT NOT NULL,
    label       TEXT NOT NULL,
    confidence  REAL NOT NULL,
    alert_type  TEXT,
    sustained_s REAL,
    source      TEXT,
    timestamp   TEXT NOT NULL,          -- when the pipeline raised it
    logged_at   TEXT NOT NULL,          -- when the API stored it
    sms_status  TEXT                    -- sent / skipped / failed: ...
);
CREATE INDEX IF NOT EXISTS idx_incidents_logged ON incidents(logged_at);
"""

COLUMNS = ['id', 'person_id', 'label', 'confidence', 'alert_type', 'sustained_s',
           'source', 'timestamp', 'logged_at', 'sms_status']


class IncidentDB:
    def __init__(self, path):
        if path != ':memory:':
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def add(self, inc):
        cols = [c for c in COLUMNS if c != 'id' and c in inc]
        cur = self.conn.execute(
            f"INSERT INTO incidents ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            [inc[c] for c in cols])
        self.conn.commit()
        return cur.lastrowid

    def set_sms_status(self, incident_id, status):
        self.conn.execute("UPDATE incidents SET sms_status=? WHERE id=?", (status, incident_id))
        self.conn.commit()

    def recent(self, limit=100):
        rows = self.conn.execute(
            "SELECT * FROM incidents ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def count(self):
        return self.conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
