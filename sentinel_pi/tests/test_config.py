"""Tests for sentinel/config.py: loading, validation, and rejection of bad input."""
import pytest
from pathlib import Path
from sentinel.config import load_config, ConfigValidationError, AppConfig


# ---------------------------------------------------------------------------
# Basic loading
# ---------------------------------------------------------------------------

def test_load_valid_minimal(sample_config_yaml):
    """A sparse config file should load with defaults filling in missing fields."""
    config = load_config(sample_config_yaml)
    assert config.site.monitoring_point == "TEST-POINT-1"
    assert config.mode == "file"
    assert config.capture.fps == 15
    # Defaults still present
    assert config.capture.width == 1280
    assert config.yolo.conf == 0.25


def test_load_full_config(full_config_yaml):
    """The canonical config.yaml must load without errors."""
    config = load_config(full_config_yaml)
    assert config.mode == "pi"
    assert config.yolo.backend == "pt"
    assert config.alert.method == "both"


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="not found"):
        load_config("does_not_exist.yaml")


# ---------------------------------------------------------------------------
# New config sections
# ---------------------------------------------------------------------------

def test_paths_section(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("paths:\n  db: 'mydb/sentinel.db'\n  data: 'mydata'\n  logs: 'mylogs'\ncamera:\n  format: 'RGB888'\n")
    config = load_config(f)
    assert config.paths.db == "mydb/sentinel.db"
    assert config.paths.data == "mydata"
    assert config.paths.logs == "mylogs"


def test_logging_section(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("logging:\n  level: 'DEBUG'\ncamera:\n  format: 'RGB888'\n")
    config = load_config(f)
    assert config.logging.level == "DEBUG"


def test_camera_section(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\n  usb_index: 2\n")
    config = load_config(f)
    assert config.camera.format == "RGB888"
    assert config.camera.usb_index == 2


# ---------------------------------------------------------------------------
# Unknown key rejection
# ---------------------------------------------------------------------------

def test_unknown_top_level_key_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nnonsense_key: 42\n")
    with pytest.raises(ConfigValidationError, match="Unknown key 'nonsense_key'"):
        load_config(f)


def test_unknown_nested_key_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\ncapture:\n  fps: 30\n  turbo_mode: true\n")
    with pytest.raises(ConfigValidationError, match="Unknown key 'turbo_mode' at capture"):
        load_config(f)


# ---------------------------------------------------------------------------
# Type errors
# ---------------------------------------------------------------------------

def test_wrong_type_int_expected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\ncapture:\n  fps: 'fast'\n")
    with pytest.raises(ConfigValidationError, match="Expected int at capture.fps"):
        load_config(f)


def test_wrong_type_dict_expected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\ncapture: 30\n")
    with pytest.raises(ConfigValidationError, match="Expected dict at capture"):
        load_config(f)


# ---------------------------------------------------------------------------
# Range validation
# ---------------------------------------------------------------------------

def test_conf_out_of_range_high(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nyolo:\n  conf: 1.5\n")
    with pytest.raises(ConfigValidationError, match="yolo.conf must be between 0 and 1"):
        load_config(f)


def test_conf_out_of_range_low(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nyolo:\n  conf: -0.1\n")
    with pytest.raises(ConfigValidationError, match="yolo.conf must be between 0 and 1"):
        load_config(f)


def test_iou_out_of_range(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nyolo:\n  iou: 2.0\n")
    with pytest.raises(ConfigValidationError, match="yolo.iou must be between 0 and 1"):
        load_config(f)


def test_bin_minutes_zero_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nindex:\n  bin_minutes: 0\n")
    with pytest.raises(ConfigValidationError, match="index.bin_minutes must be > 0"):
        load_config(f)


def test_bin_minutes_negative_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nindex:\n  bin_minutes: -5\n")
    with pytest.raises(ConfigValidationError, match="index.bin_minutes must be > 0"):
        load_config(f)


# ---------------------------------------------------------------------------
# Allowed value validation
# ---------------------------------------------------------------------------

def test_invalid_mode(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nmode: cloud\n")
    with pytest.raises(ConfigValidationError, match="Invalid mode 'cloud'"):
        load_config(f)


def test_invalid_yolo_backend(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nyolo:\n  backend: tensorrt\n")
    with pytest.raises(ConfigValidationError, match="Invalid yolo.backend 'tensorrt'"):
        load_config(f)


def test_all_valid_backends(tmp_path):
    for backend in ("pt", "ncnn", "onnx", "openvino"):
        f = tmp_path / "config.yaml"
        f.write_text(f"camera:\n  format: 'RGB888'\nyolo:\n  backend: {backend}\n")
        config = load_config(f)
        assert config.yolo.backend == backend


def test_invalid_alert_method(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nalert:\n  method: webhook\n")
    with pytest.raises(ConfigValidationError, match="Invalid alert.method 'webhook'"):
        load_config(f)


def test_all_valid_alert_methods(tmp_path):
    for method in ("cusum", "ears", "both"):
        f = tmp_path / "config.yaml"
        f.write_text(f"camera:\n  format: 'RGB888'\nalert:\n  method: {method}\n")
        config = load_config(f)
        assert config.alert.method == method


def test_invalid_camera_format_rejected(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: BGR888\n")
    with pytest.raises(ConfigValidationError, match="Invalid camera.format"):
        load_config(f)


# ---------------------------------------------------------------------------
# Schedule window format validation
# ---------------------------------------------------------------------------

def test_valid_schedule_windows(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nschedule:\n  windows: [['06:00','09:00'],['16:00','19:00']]\n")
    config = load_config(f)
    assert config.schedule.windows[0] == ["06:00", "09:00"]


def test_invalid_window_time_format(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nschedule:\n  windows: [['6:00','9:00']]\n")
    with pytest.raises(ConfigValidationError, match="Invalid time format"):
        load_config(f)


def test_invalid_window_bad_hour(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\nschedule:\n  windows: [['25:00','26:00']]\n")
    with pytest.raises(ConfigValidationError, match="Invalid time format"):
        load_config(f)


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def test_resolve_path_relative(full_config_yaml, tmp_path):
    config = load_config(full_config_yaml, project_root=tmp_path)
    resolved = config.resolve_path("data/sentinel.db")
    assert resolved == tmp_path / "data" / "sentinel.db"
    assert resolved.is_absolute()


def test_resolve_path_absolute(full_config_yaml, tmp_path):
    config = load_config(full_config_yaml, project_root=tmp_path)
    # Build a real OS-native absolute path so the test is cross-platform
    abs_path = tmp_path / "some_absolute" / "file.db"
    resolved = config.resolve_path(str(abs_path))
    assert resolved == abs_path
    assert resolved.is_absolute()


# ---------------------------------------------------------------------------
# to_dict for sessions.config_json
# ---------------------------------------------------------------------------

def test_to_dict_serializable(full_config_yaml):
    import json
    config = load_config(full_config_yaml)
    d = config.to_dict()
    assert "_project_root" not in d
    # Must be JSON-serializable (for sessions.config_json)
    json_str = json.dumps(d)
    assert '"monitoring_point"' in json_str


def test_to_dict_nested_values(full_config_yaml):
    config = load_config(full_config_yaml)
    d = config.to_dict()
    assert d["yolo"]["backend"] == "pt"
    assert d["alert"]["cusum"]["k"] == 0.5


def test_alert_min_events_default_and_validation(tmp_path):
    f = tmp_path / "config.yaml"
    f.write_text("camera:\n  format: 'RGB888'\n")
    config = load_config(f)
    assert config.alert.min_events == 3
    assert config.alert.watch_min_events == 2
    assert config.alert.clear_bins == 2
    assert config.alert.ears.lag_bins == 3


    # Reject if min_events < 1
    f_bad = tmp_path / "config_bad.yaml"
    f_bad.write_text("camera:\n  format: 'RGB888'\nalert:\n  min_events: 0\n")
    with pytest.raises(ConfigValidationError, match="alert.min_events must be >= 1"):
        load_config(f_bad)

    # Reject if watch_min_events < 1
    f_bad_watch = tmp_path / "config_bad_watch.yaml"
    f_bad_watch.write_text("camera:\n  format: 'RGB888'\nalert:\n  watch_min_events: 0\n")
    with pytest.raises(ConfigValidationError, match="alert.watch_min_events must be >= 1"):
        load_config(f_bad_watch)

    # Reject if clear_bins < 1
    f_bad_clear = tmp_path / "config_bad_clear.yaml"
    f_bad_clear.write_text("camera:\n  format: 'RGB888'\nalert:\n  clear_bins: 0\n")
    with pytest.raises(ConfigValidationError, match="alert.clear_bins must be >= 1"):
        load_config(f_bad_clear)

    # Reject if ears.lag_bins < 1
    f_bad_lag = tmp_path / "config_bad_lag.yaml"
    f_bad_lag.write_text("camera:\n  format: 'RGB888'\nalert:\n  ears:\n    lag_bins: 0\n")
    with pytest.raises(ConfigValidationError, match="alert.ears.lag_bins must be >= 1"):
        load_config(f_bad_lag)

