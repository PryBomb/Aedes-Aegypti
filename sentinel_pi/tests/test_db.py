"""Tests for sentinel/storage/db.py and schema.sql."""
import sqlite3
import pytest
from datetime import datetime, timezone
from sentinel.storage.db import DatabaseManager, utc_now_iso


# ---------------------------------------------------------------------------
# utc_now_iso helper
# ---------------------------------------------------------------------------

def test_utc_now_iso_format():
    ts = utc_now_iso()
    # Must end with Z and have milliseconds: e.g. 2023-10-10T14:30:00.123Z
    assert ts.endswith("Z"), f"Expected 'Z' suffix, got: {ts}"
    assert "T" in ts
    # Must be parseable
    parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None


def test_utc_now_iso_has_milliseconds():
    ts = utc_now_iso()
    # The fractional seconds part should exist
    time_part = ts.split("T")[1].rstrip("Z")
    assert "." in time_part, f"Expected milliseconds in {ts}"


# ---------------------------------------------------------------------------
# DB initialization
# ---------------------------------------------------------------------------

def test_db_file_created(db_manager):
    assert db_manager.db_path.exists()


def test_wal_mode_enabled(db_manager):
    with db_manager.get_connection() as conn:
        mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert mode.lower() == "wal"


def test_synchronous_normal(db_manager):
    with db_manager.get_connection() as conn:
        # PRAGMA synchronous returns 1 for NORMAL
        val = conn.execute("PRAGMA synchronous;").fetchone()[0]
        assert val == 1


def test_foreign_keys_enabled(db_manager):
    with db_manager.get_connection() as conn:
        val = conn.execute("PRAGMA foreign_keys;").fetchone()[0]
        assert val == 1


def test_busy_timeout_set(db_manager):
    with db_manager.get_connection() as conn:
        val = conn.execute("PRAGMA busy_timeout;").fetchone()[0]
        assert val == 5000


def test_user_version_is_2(db_manager):
    """Schema v2 after Phase 4 migration."""
    with db_manager.get_connection() as conn:
        val = conn.execute("PRAGMA user_version;").fetchone()[0]
        assert val == 2


# ---------------------------------------------------------------------------
# Table existence
# ---------------------------------------------------------------------------

EXPECTED_TABLES = [
    "sessions", "events", "detection_index", "alerts",
    "health_logs", "manual_counts", "survey_responses",
    "baselines",  # v2
]

@pytest.mark.parametrize("table_name", EXPECTED_TABLES)
def test_table_exists(db_manager, table_name):
    with db_manager.get_connection() as conn:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;",
            (table_name,)
        ).fetchone()
        assert row is not None, f"Table '{table_name}' not found"


# ---------------------------------------------------------------------------
# Schema: events CHECK constraints
# ---------------------------------------------------------------------------

def _insert_session(conn):
    conn.execute("""
        INSERT INTO sessions (started_at, mode, model_name, model_backend, config_json)
        VALUES ('2023-10-10T06:00:00.000Z', 'file', 'yolo11n', 'pt', '{}')
    """)
    conn.commit()
    return conn.execute("SELECT last_insert_rowid()").fetchone()[0]


def test_events_valid_status_accepted(db_manager):
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        for status in ("candidate", "confirmed", "rejected", "non_target"):
            conn.execute("""
                INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                    duration_s, n_frames, max_conf, mean_conf, yolo_class, status)
                VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                    1.0, 10, 0.9, 0.8, 'Aedes aegypti', ?)
            """, (sid, status))
        conn.commit()


def test_events_invalid_status_rejected(db_manager):
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                    duration_s, n_frames, max_conf, mean_conf, yolo_class, status)
                VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                    1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'UNKNOWN')
            """, (sid,))


def test_events_verified_label_null_accepted(db_manager):
    """NULL must be allowed for verified_label (NOT NULL IN (..., NULL))."""
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        conn.execute("""
            INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                duration_s, n_frames, max_conf, mean_conf, yolo_class, status, verified_label)
            VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'confirmed', NULL)
        """, (sid,))
        conn.commit()


def test_events_verified_label_valid_values(db_manager):
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        for label in ("true", "false"):
            conn.execute("""
                INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                    duration_s, n_frames, max_conf, mean_conf, yolo_class, status, verified_label)
                VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                    1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'confirmed', ?)
            """, (sid, label))
        conn.commit()


def test_events_verified_label_invalid_rejected(db_manager):
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                    duration_s, n_frames, max_conf, mean_conf, yolo_class, status, verified_label)
                VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                    1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'confirmed', 'yes')
            """, (sid,))


# ---------------------------------------------------------------------------
# Schema: detection_index constraints
# ---------------------------------------------------------------------------

def test_detection_index_valid_level(db_manager):
    with db_manager.get_connection() as conn:
        # Each row needs a distinct bin_start due to UNIQUE(monitoring_point, bin_start)
        bins = {
            "NORMAL": ("2023-10-10T06:00:00.000Z", "2023-10-10T06:30:00.000Z"),
            "WATCH":  ("2023-10-10T06:30:00.000Z", "2023-10-10T07:00:00.000Z"),
            "ALERT":  ("2023-10-10T07:00:00.000Z", "2023-10-10T07:30:00.000Z"),
        }
        for level, (bin_start, bin_end) in bins.items():
            conn.execute("""
                INSERT INTO detection_index
                    (monitoring_point, bin_start, bin_end, raw_value, n_events, active_hours, bin_coverage, level)
                VALUES ('KAS-1', ?, ?, 1.5, 3, 0.5, 0.9, ?)
            """, (bin_start, bin_end, level))
        conn.commit()


def test_detection_index_invalid_level_rejected(db_manager):
    with db_manager.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO detection_index
                    (monitoring_point, bin_start, bin_end, raw_value, n_events, active_hours, bin_coverage, level)
                VALUES ('KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:30:00.000Z',
                    1.5, 3, 0.5, 0.9, 'HIGH')
            """)


def test_detection_index_unique_constraint(db_manager):
    with db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO detection_index
                (monitoring_point, bin_start, bin_end, raw_value, n_events, active_hours, bin_coverage, level)
            VALUES ('KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:30:00.000Z', 1.5, 3, 0.5, 0.9, 'NORMAL')
        """)
        conn.commit()
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO detection_index
                    (monitoring_point, bin_start, bin_end, raw_value, n_events, active_hours, bin_coverage, level)
                VALUES ('KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:30:00.000Z', 2.0, 5, 0.5, 0.9, 'WATCH')
            """)


# ---------------------------------------------------------------------------
# Schema: alerts constraints
# ---------------------------------------------------------------------------

def test_alerts_valid_method(db_manager):
    with db_manager.get_connection() as conn:
        for method in ("cusum", "ears", "both"):
            conn.execute("""
                INSERT INTO alerts (monitoring_point, level, method, t_alert)
                VALUES ('KAS-1', 'ALERT', ?, '2023-10-10T06:00:00.000Z')
            """, (method,))
        conn.commit()


def test_alerts_invalid_method_rejected(db_manager):
    with db_manager.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO alerts (monitoring_point, level, method, t_alert)
                VALUES ('KAS-1', 'ALERT', 'webhook', '2023-10-10T06:00:00.000Z')
            """)


def test_alerts_verified_outcome_null_accepted(db_manager):
    with db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO alerts (monitoring_point, level, method, t_alert, verified_outcome)
            VALUES ('KAS-1', 'WATCH', 'cusum', '2023-10-10T06:00:00.000Z', NULL)
        """)
        conn.commit()


def test_alerts_verified_outcome_invalid_rejected(db_manager):
    with db_manager.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO alerts (monitoring_point, level, method, t_alert, verified_outcome)
                VALUES ('KAS-1', 'ALERT', 'cusum', '2023-10-10T06:00:00.000Z', 'maybe')
            """)


# ---------------------------------------------------------------------------
# Schema: sessions deployment_label; manual_counts new columns
# ---------------------------------------------------------------------------

def test_sessions_has_deployment_label(db_manager):
    with db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO sessions (deployment_label, started_at, mode, model_name, model_backend, config_json)
            VALUES ('lab-run-01', '2023-10-10T06:00:00.000Z', 'file', 'yolo11n', 'pt', '{}')
        """)
        conn.commit()
        row = conn.execute("SELECT deployment_label FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
        assert row["deployment_label"] == "lab-run-01"


def test_manual_counts_new_columns(db_manager):
    with db_manager.get_connection() as conn:
        conn.execute("""
            INSERT INTO manual_counts
                (ts, monitoring_point, method, period_start, period_end, stage, count, inspector)
            VALUES
                ('2023-10-10T06:00:00.000Z', 'KAS-1', 'visual',
                 '2023-10-10T06:00:00.000Z', '2023-10-10T09:00:00.000Z',
                 'adults', 5, 'Inspector A')
        """)
        conn.commit()
        row = conn.execute("SELECT stage, period_start FROM manual_counts LIMIT 1").fetchone()
        assert row["stage"] == "adults"


def test_manual_counts_stage_check(db_manager):
    with db_manager.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO manual_counts
                    (ts, monitoring_point, method, period_start, period_end, stage, count, inspector)
                VALUES
                    ('2023-10-10T06:00:00.000Z', 'KAS-1', 'visual',
                     '2023-10-10T06:00:00.000Z', '2023-10-10T09:00:00.000Z',
                     'pupae', 3, 'Inspector A')
            """)


def test_events_has_track_id(db_manager):
    with db_manager.get_connection() as conn:
        sid = _insert_session(conn)
        conn.execute("""
            INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                duration_s, n_frames, max_conf, mean_conf, yolo_class, status, track_id)
            VALUES (?, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'confirmed', 42)
        """, (sid,))
        conn.commit()
        row = conn.execute("SELECT track_id FROM events ORDER BY id DESC LIMIT 1").fetchone()
        assert row["track_id"] == 42


# ---------------------------------------------------------------------------
# Foreign key enforcement
# ---------------------------------------------------------------------------

def test_event_rejects_invalid_session_fk(db_manager):
    with db_manager.get_connection() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO events (session_id, monitoring_point, started_at, ended_at,
                    duration_s, n_frames, max_conf, mean_conf, yolo_class, status)
                VALUES (999, 'KAS-1', '2023-10-10T06:00:00.000Z', '2023-10-10T06:00:01.000Z',
                    1.0, 10, 0.9, 0.8, 'Aedes aegypti', 'confirmed')
            """)
            conn.commit()
