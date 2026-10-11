"""Database manager for MosquitoSentinel with versioned schema migrations.

Schema versions:
  v1 – initial (detection_index without valid/n_unverified; no baselines).
  v2 – detection_index.valid INTEGER (0/1) + n_unverified INTEGER;
       new table: baselines(monitoring_point, window_group, mu, sigma, n_bins,
                            computed_at, source).

Migration is applied once at startup inside _init_db(); it is idempotent (safe
to run repeatedly) because it uses ADD COLUMN IF NOT EXISTS / CREATE TABLE IF
NOT EXISTS semantics via try/except on column-already-exists errors.
"""
import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)

_CURRENT_VERSION = 2


def utc_now_iso() -> str:
    """Returns the current UTC time as an ISO-8601 string with milliseconds and 'Z' suffix."""
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')


class DatabaseManager:
    def __init__(self, db_path: Path | str, schema_path: Path | str):
        self.db_path = Path(db_path)
        self.schema_path = Path(schema_path)
        self._init_db()

    def _init_db(self):
        """Initialise database with schema if needed, run migrations, and set WAL mode."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self.get_connection() as conn:
            user_version = conn.execute("PRAGMA user_version").fetchone()[0]

            if user_version == 0:
                # Fresh database – apply full schema
                if self.schema_path.exists():
                    with open(self.schema_path, "r", encoding="utf-8") as f:
                        schema_sql = f.read()
                    conn.executescript(schema_sql)
                    logger.info("Initialised fresh database at schema v%d.", _CURRENT_VERSION)
                else:
                    logger.warning(
                        "Schema file not found at %s – unable to init tables.", self.schema_path
                    )
            elif user_version < _CURRENT_VERSION:
                logger.info(
                    "Migrating database from v%d to v%d.", user_version, _CURRENT_VERSION
                )
                self._migrate(conn, from_version=user_version)
            else:
                logger.debug("Database is at current schema version %d.", user_version)

    def _migrate(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Apply incremental migrations starting from *from_version*.

        Each migration step is idempotent: ALTER TABLE ADD COLUMN is wrapped in
        try/except so a re-run does not fail on an already-migrated database.
        """
        if from_version < 2:
            self._migrate_v1_to_v2(conn)

    @staticmethod
    def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
        """v1 → v2:
        - detection_index: add valid INTEGER NOT NULL DEFAULT 1
        - detection_index: add n_unverified INTEGER NOT NULL DEFAULT 0
        - create baselines table (if not exists)
        """
        # Add 'valid' column (idempotent)
        try:
            conn.execute(
                "ALTER TABLE detection_index ADD COLUMN valid INTEGER NOT NULL DEFAULT 1"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

        # Add 'n_unverified' column (idempotent)
        try:
            conn.execute(
                "ALTER TABLE detection_index ADD COLUMN n_unverified INTEGER NOT NULL DEFAULT 0"
            )
        except sqlite3.OperationalError as exc:
            if "duplicate column" not in str(exc).lower():
                raise

        # Create baselines table (idempotent)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                monitoring_point TEXT NOT NULL,
                window_group TEXT NOT NULL,
                mu REAL NOT NULL,
                sigma REAL NOT NULL,
                n_bins INTEGER NOT NULL,
                computed_at DATETIME NOT NULL,
                source TEXT NOT NULL CHECK (source IN ('computed', 'manual')),
                UNIQUE(monitoring_point, window_group)
            )
            """
        )

        conn.execute("PRAGMA user_version = 2")
        conn.commit()
        logger.info("Migration v1 → v2 complete.")

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Provides a context-managed database connection with required pragmas and guaranteed closure."""
        conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        conn.row_factory = sqlite3.Row

        # Apply strict pragmas to every connection
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.execute("PRAGMA busy_timeout=5000;")

        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
