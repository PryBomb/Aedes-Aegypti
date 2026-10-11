import pytest
from pathlib import Path

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_config_yaml(tmp_path):
    """A minimal valid config for baseline tests."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("""
site:
  monitoring_point: "TEST-POINT-1"
mode: "file"
capture:
  fps: 15
camera:
  format: "RGB888"
""")
    return config_file


@pytest.fixture
def full_config_yaml(tmp_path):
    """Full config matching all dataclass defaults."""
    import shutil
    src = Path(__file__).parent.parent / "config.yaml"
    dst = tmp_path / "config.yaml"
    shutil.copy(src, dst)
    return dst


@pytest.fixture
def db_manager(tmp_path):
    """A fresh DatabaseManager backed by the real schema."""
    from sentinel.storage.db import DatabaseManager
    schema_path = Path(__file__).parent.parent / "sentinel" / "storage" / "schema.sql"
    db_file = tmp_path / "test.db"
    return DatabaseManager(db_file, schema_path)
