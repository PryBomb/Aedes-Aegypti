"""Database repositories for MosquitoSentinel adhering to Phase 1 SQLite schema."""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional, Union

from sentinel.detection.event_builder import Event
from sentinel.storage.db import DatabaseManager, utc_now_iso


class SessionRepository:
    """Repository managing the 'sessions' table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create(
        self,
        mode: str,
        model_name: str,
        model_backend: str,
        config_json: str | dict,
        deployment_label: Optional[str] = None,
        git_hash: Optional[str] = None,
        started_at: Optional[str] = None,
    ) -> int:
        """Create a new recording session and return its id."""
        if isinstance(config_json, dict):
            config_str = json.dumps(config_json)
        else:
            config_str = str(config_json)

        ts = started_at or utc_now_iso()
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sessions (
                    deployment_label, started_at, mode, model_name,
                    model_backend, config_json, git_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    deployment_label,
                    ts,
                    mode,
                    model_name,
                    model_backend,
                    config_str,
                    git_hash,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get(self, session_id: int) -> Optional[sqlite3.Row]:
        """Fetch session by id."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
            return cursor.fetchone()

    def end_session(self, session_id: int, ended_at: Optional[str] = None) -> None:
        """Set ended_at timestamp for session."""
        ts = ended_at or utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute("UPDATE sessions SET ended_at = ? WHERE id = ?", (ts, session_id))
            conn.commit()


class EventRepository:
    """Repository managing the 'events' table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create_event(self, data: Union[Dict[str, Any], Event]) -> int:
        """Insert a finalized detection event into the database. Returns event id."""
        if isinstance(data, Event):
            row_dict = data.to_db_dict()
        else:
            row_dict = dict(data)

        keys = [
            "session_id",
            "monitoring_point",
            "started_at",
            "ended_at",
            "duration_s",
            "n_frames",
            "max_conf",
            "mean_conf",
            "yolo_class",
            "cnn_label",
            "cnn_conf",
            "track_id",
            "status",
            "snapshot_path",
            "clip_path",
            "t_first_detect",
            "t_confirmed",
            "t_saved",
            "verified_label",
            "verified_by",
            "verified_at",
            "notes",
        ]

        values = [row_dict.get(k) for k in keys]
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)

        with self.db.get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO events ({columns}) VALUES ({placeholders})",
                values,
            )
            conn.commit()
            event_id = int(cursor.lastrowid)

        if isinstance(data, Event):
            data.id = event_id

        return event_id

    def close_event(self, event_id: int, data: Dict[str, Any]) -> None:
        """Update an event row with finalized close values."""
        allowed_keys = [
            "ended_at",
            "duration_s",
            "n_frames",
            "max_conf",
            "mean_conf",
            "yolo_class",
            "cnn_label",
            "cnn_conf",
            "status",
            "snapshot_path",
            "clip_path",
            "t_confirmed",
            "t_saved",
        ]
        updates = []
        values = []
        for k, v in data.items():
            if k in allowed_keys:
                updates.append(f"{k} = ?")
                values.append(v)

        if not updates:
            return

        values.append(event_id)
        sql = f"UPDATE events SET {', '.join(updates)} WHERE id = ?"
        with self.db.get_connection() as conn:
            conn.execute(sql, values)
            conn.commit()

    def get(self, event_id: int) -> Optional[sqlite3.Row]:
        """Fetch event by primary key id."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM events WHERE id = ?", (event_id,))
            return cursor.fetchone()

    def get_recent(self, limit: int = 50) -> List[sqlite3.Row]:
        """Fetch the most recent events ordered by started_at DESC."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM events ORDER BY started_at DESC LIMIT ?", (limit,))
            return cursor.fetchall()

    def get_by_status(self, status: str) -> List[sqlite3.Row]:
        """Fetch events matching status ('confirmed', 'rejected', 'non_target')."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM events WHERE status = ? ORDER BY started_at ASC", (status,))
            return cursor.fetchall()

    def get_unverified_evidence_oldest_first(self) -> List[sqlite3.Row]:
        """Query unverified events that have evidence files.

        Retention order:
        - Prune evidence of 'rejected' and 'non_target' events first.
        - Then prune the oldest unverified confirmed events.
        - Verified events (verified_label IS NOT NULL) are strictly excluded from deletion.
        """
        sql = """
        SELECT id, snapshot_path, clip_path, started_at, status, verified_label
        FROM events
        WHERE (snapshot_path IS NOT NULL OR clip_path IS NOT NULL)
          AND (verified_label IS NULL)
        ORDER BY
          CASE
            WHEN status IN ('rejected', 'non_target') THEN 0
            ELSE 1
          END ASC,
          started_at ASC
        """
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql)
            return cursor.fetchall()

    def update_evidence_paths(
        self,
        event_id: int,
        snapshot_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        t_saved: Optional[str] = None,
    ) -> None:
        """Update snapshot/clip paths for an event (or set to None when pruned)."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE events
                SET snapshot_path = ?, clip_path = ?, t_saved = coalesce(?, t_saved)
                WHERE id = ?
                """,
                (snapshot_path, clip_path, t_saved, event_id),
            )
            conn.commit()

    def verify_event(
        self,
        event_id: int,
        verified_label: str,
        verified_by: str,
        notes: Optional[str] = None,
        verified_at: Optional[str] = None,
    ) -> None:
        """Mark an event as verified ('true' or 'false'). Protects evidence from retention pruning."""
        ts = verified_at or utc_now_iso()
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE events
                SET verified_label = ?, verified_by = ?, verified_at = ?, notes = coalesce(?, notes)
                WHERE id = ?
                """,
                (verified_label, verified_by, ts, notes, event_id),
            )
            conn.commit()

    def count_by_status(self, session_id: Optional[int] = None) -> Dict[str, int]:
        """Return counts of events by status."""
        sql = "SELECT status, COUNT(*) as count FROM events"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " GROUP BY status"

        counts: Dict[str, int] = {}
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                counts[row["status"]] = row["count"]
        return counts

    def count_by_class(self, session_id: Optional[int] = None) -> Dict[str, int]:
        """Return counts of events by yolo_class."""
        sql = "SELECT yolo_class, COUNT(*) as count FROM events"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " GROUP BY yolo_class"

        counts: Dict[str, int] = {}
        with self.db.get_connection() as conn:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                counts[row["yolo_class"]] = row["count"]
        return counts


ALLOWED_HEALTH_COLUMNS = {
    "ts", "session_id", "fps", "infer_ms_avg", "infer_ms_p95",
    "dropped_frames", "queue_depth", "cpu_pct", "ram_pct",
    "cpu_temp_c", "disk_free_gb", "camera_ok", "clock_synced"
}


class HealthRepository:
    """Repository managing system health checks."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create(self, data: Dict[str, Any]) -> int:
        """Insert health metric log row into 'health_logs' with whitelisted columns."""
        filtered = {k: v for k, v in data.items() if k in ALLOWED_HEALTH_COLUMNS}
        if "ts" not in filtered:
            filtered["ts"] = utc_now_iso()
        keys = list(filtered.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO health_logs ({columns}) VALUES ({placeholders})",
                list(filtered.values()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get(self, health_id: int) -> Optional[sqlite3.Row]:
        """Fetch health log row by id."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM health_logs WHERE id = ?", (health_id,))
            return cursor.fetchone()

    def get_recent(self, limit: int = 50) -> List[sqlite3.Row]:
        """Fetch recent health log rows."""
        with self.db.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM health_logs ORDER BY ts DESC LIMIT ?", (limit,))
            return cursor.fetchall()


# ─── Detection Index Repository ─────────────────────────────────────────────

ALLOWED_INDEX_COLUMNS = {
    "monitoring_point", "bin_start", "bin_end", "raw_value", "verified_value",
    "n_events", "n_verified_true", "n_unverified", "valid", "active_hours",
    "bin_coverage", "cusum_value", "ears_value", "level",
}


class DetectionIndexRepository:
    """Repository managing the 'detection_index' table.

    All writes are idempotent: upsert on (monitoring_point, bin_start).
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def upsert(self, data: Dict[str, Any]) -> None:
        """Insert or replace a detection index bin (whitelisted columns).

        Required keys: monitoring_point, bin_start, bin_end, raw_value,
        n_events, active_hours, bin_coverage, valid, level.
        """
        filtered = {k: v for k, v in data.items() if k in ALLOWED_INDEX_COLUMNS}
        keys = list(filtered.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        update_pairs = ", ".join(f"{k} = excluded.{k}" for k in keys)

        sql = (
            f"INSERT INTO detection_index ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(monitoring_point, bin_start) DO UPDATE SET {update_pairs}"
        )
        with self.db.get_connection() as conn:
            conn.execute(sql, list(filtered.values()))
            conn.commit()

    def update_alert_values(
        self,
        monitoring_point: str,
        bin_start: str,
        level: str,
        cusum_value: Optional[float] = None,
        ears_value: Optional[float] = None,
    ) -> None:
        """Update level, cusum_value and ears_value for an existing bin."""
        with self.db.get_connection() as conn:
            conn.execute(
                """
                UPDATE detection_index
                SET level = ?, cusum_value = ?, ears_value = ?
                WHERE monitoring_point = ? AND bin_start = ?
                """,
                (level, cusum_value, ears_value, monitoring_point, bin_start),
            )
            conn.commit()

    def get_by_monitoring_point(
        self,
        monitoring_point: str,
        valid_only: bool = False,
        limit: int = 500,
    ) -> List[sqlite3.Row]:
        """Fetch detection index rows for a monitoring point, ordered by bin_start ASC."""
        sql = "SELECT * FROM detection_index WHERE monitoring_point = ?"
        params: list = [monitoring_point]
        if valid_only:
            sql += " AND valid = 1"
        sql += " ORDER BY bin_start ASC LIMIT ?"
        params.append(limit)
        with self.db.get_connection() as conn:
            return conn.execute(sql, params).fetchall()

    def get_recent(
        self,
        monitoring_point: str,
        limit: int = 50,
        valid_only: bool = True,
    ) -> List[sqlite3.Row]:
        """Fetch the most recent bins for a monitoring point."""
        sql = "SELECT * FROM detection_index WHERE monitoring_point = ?"
        params: list = [monitoring_point]
        if valid_only:
            sql += " AND valid = 1"
        sql += " ORDER BY bin_start DESC LIMIT ?"
        params.append(limit)
        with self.db.get_connection() as conn:
            return conn.execute(sql, params).fetchall()


# ─── Alerts Repository ───────────────────────────────────────────────────────

ALLOWED_ALERT_COLUMNS = {
    "monitoring_point", "level", "method", "t_alert", "t_displayed",
    "cusum_value", "ears_value", "bin_start", "trigger_event_id",
    "acknowledged_at", "verified_outcome", "notes",
}


class AlertsRepository:
    """Repository managing the 'alerts' table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def create(self, data: Dict[str, Any]) -> int:
        """Insert an alert row (whitelisted columns). Returns the new row id."""
        filtered = {k: v for k, v in data.items() if k in ALLOWED_ALERT_COLUMNS}
        if "t_alert" not in filtered:
            filtered["t_alert"] = utc_now_iso()
        keys = list(filtered.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        with self.db.get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO alerts ({columns}) VALUES ({placeholders})",
                list(filtered.values()),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get(self, alert_id: int) -> Optional[sqlite3.Row]:
        """Fetch an alert row by primary key."""
        with self.db.get_connection() as conn:
            return conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()

    def get_recent(self, limit: int = 50) -> List[sqlite3.Row]:
        """Fetch the most recent alert rows."""
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT * FROM alerts ORDER BY t_alert DESC LIMIT ?", (limit,)
            ).fetchall()

    def get_by_monitoring_point(self, monitoring_point: str, limit: int = 50) -> List[sqlite3.Row]:
        """Fetch alert rows for a monitoring point."""
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT * FROM alerts WHERE monitoring_point = ? ORDER BY t_alert ASC LIMIT ?",
                (monitoring_point, limit),
            ).fetchall()


    def acknowledge(self, alert_id: int, notes: Optional[str] = None) -> None:
        """Set acknowledged_at timestamp."""
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged_at = ?, notes = coalesce(?, notes) WHERE id = ?",
                (utc_now_iso(), notes, alert_id),
            )
            conn.commit()

    def verify(self, alert_id: int, outcome: str, notes: Optional[str] = None) -> None:
        """Set verified_outcome ('true_alert' or 'false_alert')."""
        if outcome not in ("true_alert", "false_alert"):
            raise ValueError(f"Invalid outcome: {outcome!r}")
        with self.db.get_connection() as conn:
            conn.execute(
                "UPDATE alerts SET verified_outcome = ?, notes = coalesce(?, notes) WHERE id = ?",
                (outcome, notes, alert_id),
            )
            conn.commit()


# ─── Baselines Repository ────────────────────────────────────────────────────

ALLOWED_BASELINE_COLUMNS = {
    "monitoring_point", "window_group", "mu", "sigma",
    "n_bins", "computed_at", "source",
}


class BaselinesRepository:
    """Repository managing the 'baselines' table (v2+).

    Upsert on (monitoring_point, window_group) so recomputation is idempotent.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db = db_manager

    def upsert(self, data: Dict[str, Any]) -> None:
        """Insert or replace a baseline row.

        Required keys: monitoring_point, window_group, mu, sigma,
                       n_bins, computed_at, source.
        """
        filtered = {k: v for k, v in data.items() if k in ALLOWED_BASELINE_COLUMNS}
        if "computed_at" not in filtered:
            filtered["computed_at"] = utc_now_iso()
        keys = list(filtered.keys())
        placeholders = ", ".join(["?"] * len(keys))
        columns = ", ".join(keys)
        update_pairs = ", ".join(f"{k} = excluded.{k}" for k in keys)

        sql = (
            f"INSERT INTO baselines ({columns}) VALUES ({placeholders}) "
            f"ON CONFLICT(monitoring_point, window_group) DO UPDATE SET {update_pairs}"
        )
        with self.db.get_connection() as conn:
            conn.execute(sql, list(filtered.values()))
            conn.commit()

    def get(self, monitoring_point: str, window_group: str) -> Optional[sqlite3.Row]:
        """Fetch baseline for a (monitoring_point, window_group) pair."""
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT * FROM baselines WHERE monitoring_point = ? AND window_group = ?",
                (monitoring_point, window_group),
            ).fetchone()

    def get_all(self, monitoring_point: str) -> List[sqlite3.Row]:
        """Fetch all baselines for a monitoring point."""
        with self.db.get_connection() as conn:
            return conn.execute(
                "SELECT * FROM baselines WHERE monitoring_point = ? ORDER BY window_group",
                (monitoring_point,),
            ).fetchall()
