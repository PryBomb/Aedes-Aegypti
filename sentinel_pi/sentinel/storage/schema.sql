-- SQLite Schema for MosquitoSentinel
-- Uses WAL mode for better concurrency during reads and writes
-- TIMESTAMPS: All DATETIME fields must store UTC ISO-8601 strings with milliseconds (e.g. '2023-10-10T14:30:00.123Z')
-- Schema version history:
--   v1: initial schema
--   v2: detection_index gets valid + n_unverified; new baselines table

PRAGMA user_version = 2;
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deployment_label TEXT,
    started_at DATETIME NOT NULL,
    ended_at DATETIME,
    mode TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_backend TEXT NOT NULL,
    config_json TEXT NOT NULL,
    git_hash TEXT
);

-- Note: events are written when closed (and open events flushed on shutdown)
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL,
    monitoring_point TEXT NOT NULL,
    started_at DATETIME NOT NULL,
    ended_at DATETIME NOT NULL,
    duration_s REAL NOT NULL,
    n_frames INTEGER NOT NULL,
    max_conf REAL NOT NULL,
    mean_conf REAL NOT NULL,
    yolo_class TEXT NOT NULL,
    cnn_label TEXT,
    cnn_conf REAL,
    track_id INTEGER,
    status TEXT NOT NULL CHECK (status IN ('candidate','confirmed','rejected','non_target')),
    snapshot_path TEXT,
    clip_path TEXT,
    t_first_detect DATETIME,
    t_confirmed DATETIME,
    t_saved DATETIME,
    verified_label TEXT CHECK (verified_label IS NULL OR verified_label IN ('true','false')),
    verified_by TEXT,
    verified_at DATETIME,
    notes TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
CREATE INDEX IF NOT EXISTS idx_events_started_at ON events(started_at);
CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id);

CREATE TABLE IF NOT EXISTS detection_index (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitoring_point TEXT NOT NULL,
    bin_start DATETIME NOT NULL,
    bin_end DATETIME NOT NULL,
    raw_value REAL NOT NULL,
    verified_value REAL,
    n_events INTEGER NOT NULL,
    n_verified_true INTEGER,
    -- v2 additions --
    valid INTEGER NOT NULL DEFAULT 1 CHECK (valid IN (0, 1)),  -- 0 = excluded (low coverage)
    n_unverified INTEGER NOT NULL DEFAULT 0,                   -- confirmed events not yet verified
    active_hours REAL NOT NULL,
    bin_coverage REAL NOT NULL,
    cusum_value REAL,
    ears_value REAL,
    level TEXT NOT NULL CHECK (level IN ('NORMAL','WATCH','ALERT')),
    UNIQUE(monitoring_point, bin_start)
);
CREATE INDEX IF NOT EXISTS idx_index_bin_start ON detection_index(bin_start);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitoring_point TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('NORMAL','WATCH','ALERT')),
    method TEXT NOT NULL CHECK (method IN ('cusum','ears','both')),
    t_alert DATETIME NOT NULL,
    t_displayed DATETIME,
    cusum_value REAL,
    ears_value REAL,
    bin_start DATETIME,
    trigger_event_id INTEGER,
    acknowledged_at DATETIME,
    verified_outcome TEXT CHECK (verified_outcome IS NULL OR verified_outcome IN ('true_alert','false_alert')),
    notes TEXT,
    FOREIGN KEY (trigger_event_id) REFERENCES events (id)
);
CREATE INDEX IF NOT EXISTS idx_alerts_t_alert ON alerts(t_alert);

CREATE TABLE IF NOT EXISTS health_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME NOT NULL,
    session_id INTEGER NOT NULL,
    fps REAL NOT NULL,
    infer_ms_avg REAL NOT NULL,
    infer_ms_p95 REAL NOT NULL,
    dropped_frames INTEGER NOT NULL,
    queue_depth INTEGER NOT NULL,
    cpu_pct REAL NOT NULL,
    ram_pct REAL NOT NULL,
    cpu_temp_c REAL,
    disk_free_gb REAL NOT NULL,
    camera_ok BOOLEAN NOT NULL,
    clock_synced BOOLEAN NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);
CREATE INDEX IF NOT EXISTS idx_health_ts ON health_logs(ts);

CREATE TABLE IF NOT EXISTS manual_counts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME NOT NULL,
    monitoring_point TEXT NOT NULL,
    method TEXT NOT NULL,
    period_start DATETIME NOT NULL,
    period_end DATETIME NOT NULL,
    stage TEXT NOT NULL CHECK (stage IN ('adults','larvae','eggs')),
    count INTEGER NOT NULL,
    inspector TEXT NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS idx_manual_counts_ts ON manual_counts(ts);

CREATE TABLE IF NOT EXISTS survey_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts DATETIME NOT NULL,
    respondent_role TEXT NOT NULL,
    answers_json TEXT NOT NULL,
    comments TEXT
);

-- v2: baselines table (window-group statistics used by CUSUM/EARS)
CREATE TABLE IF NOT EXISTS baselines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    monitoring_point TEXT NOT NULL,
    window_group TEXT NOT NULL,    -- 'morning' | 'afternoon'
    mu REAL NOT NULL,
    sigma REAL NOT NULL,
    n_bins INTEGER NOT NULL,
    computed_at DATETIME NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('computed', 'manual')),
    UNIQUE(monitoring_point, window_group)
);
