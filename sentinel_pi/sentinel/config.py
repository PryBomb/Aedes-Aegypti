import yaml
import re
import dataclasses
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Type, TypeVar, Any, Dict

T = TypeVar('T')

class ConfigValidationError(Exception):
    pass

@dataclass
class SiteConfig:
    monitoring_point: str = "KAS-1E-01"
    timezone: str = "Asia/Manila"

@dataclass
class PathsConfig:
    data_dir: str = "C:/sentinel_data"
    db: str = "data/sentinel.db"
    data: str = "data"
    logs: str = "data/logs"

@dataclass
class LoggingConfig:
    level: str = "INFO"

@dataclass
class FileSourceConfig:
    path: str = ""

@dataclass
class CameraConfig:
    format: str = "RGB888"
    usb_index: int = 0

@dataclass
class ScheduleConfig:
    windows: List[List[str]] = field(default_factory=lambda: [["06:00", "09:00"], ["16:00", "19:00"]])
    test_mode: bool = False

@dataclass
class CaptureConfig:
    width: int = 1280
    height: int = 720
    fps: int = 30
    preroll_s: int = 3
    postroll_s: int = 3

@dataclass
class YoloConfig:
    weights: str = "models/best.pt"
    backend: str = "pt"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.5
    target_class_names: List[str] = field(default_factory=lambda: ["Aedes Aegypti"])
    non_target_class_names: List[str] = field(default_factory=lambda: ["Aedes Albopictus"])
    use_tracker: bool = False

@dataclass
class EventsConfig:
    event_gap_s: float = 1.5
    min_frames: int = 3
    min_mean_conf: float = 0.4

@dataclass
class CnnConfig:
    enabled: bool = False
    model: str = "models/mosquito_model.tflite"
    max_crops_per_event: int = 3
    reject_threshold: float = 0.8

@dataclass
class IndexConfig:
    # NOTE: alert delay is bounded by bin_minutes; shorter bins = lower latency but noisier index.
    bin_minutes: int = 30
    method: str = "events_per_hour"
    # Bins whose coverage fraction is below min_coverage are excluded (valid=0) from baseline
    # and alerting. This prevents low-activity bins from distorting statistics.
    min_coverage: float = 0.5

@dataclass
class BaselineConfig:
    days: int = 7
    # Minimum number of valid bins required before baseline is considered ready.
    min_bins: int = 12
    # Minimum sigma to apply to avoid division-by-zero in CUSUM/EARS.
    min_sigma: float = 0.5

@dataclass
class CusumConfig:
    k: float = 0.5
    h: float = 4.0
    # If True, the CUSUM statistic S_t is reset to 0 after an alert fires.
    reset_after_alert: bool = True

@dataclass
class EarsConfig:
    # Baseline window in valid bins inside the current monitoring window (tuned on real data).
    lag_bins: int = 3
    z: float = 3.0
    min_sd: float = 0.5


@dataclass
class AlertConfig:
    method: str = "both"
    watch_fraction: float = 0.6
    cooldown_min: int = 30
    min_events: int = 3
    watch_min_events: int = 2
    clear_bins: int = 2
    cusum: CusumConfig = field(default_factory=CusumConfig)
    ears: EarsConfig = field(default_factory=EarsConfig)

@dataclass
class GpioConfig:
    led_pin: int = 17
    buzzer_pin: int = 27
    buzzer_enabled: bool = True

@dataclass
class EvidenceConfig:
    max_gb: int = 20
    save_clips: bool = True

@dataclass
class ThermalConfig:
    throttle_c: int = 80

@dataclass
class DashboardConfig:
    host: str = "0.0.0.0"
    port: int = 5000

@dataclass
class AppConfig:
    site: SiteConfig = field(default_factory=SiteConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    mode: str = "pi"
    file_source: FileSourceConfig = field(default_factory=FileSourceConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    schedule: ScheduleConfig = field(default_factory=ScheduleConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    yolo: YoloConfig = field(default_factory=YoloConfig)
    events: EventsConfig = field(default_factory=EventsConfig)
    cnn: CnnConfig = field(default_factory=CnnConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    gpio: GpioConfig = field(default_factory=GpioConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Store the absolute path of the project root to resolve paths against
    _project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        d.pop("_project_root", None)
        return d

    def resolve_path(self, p: str) -> Path:
        """Resolve a string path against the project root if it's relative."""
        path = Path(p)
        if path.is_absolute():
            return path
        return self._project_root / path

    def get_data_dir(self) -> Path:
        """Return the base data directory (isolated outside OneDrive)."""
        p = Path(self.paths.data_dir)
        if p.is_absolute():
            return p
        return self._project_root / p

    def resolve_data_path(self, p: str | Path) -> Path:
        """Resolve a relative path against the configured data_dir."""
        path = Path(p)
        if path.is_absolute():
            return path
        return self.get_data_dir() / path

    def to_data_relative(self, p: str | Path) -> str:
        """Convert a path inside data_dir to a POSIX relative path for DB portability."""
        path = Path(p)
        base = self.get_data_dir()
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            return path.as_posix()

def _validate_time_str(val: str, path: str):
    if not re.match(r"^([01]\d|2[0-3]):([0-5]\d)$", val):
        raise ConfigValidationError(f"Invalid time format at {path}: '{val}' (expected HH:MM)")

def _parse_dict(cls: Type[T], data: dict, current_path: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        return data

    field_types = {f.name: f for f in dataclasses.fields(cls) if f.name != "_project_root"}
    kwargs = {}
    
    for key, value in data.items():
        if key not in field_types:
            raise ConfigValidationError(f"Unknown key '{key}' at {current_path or 'root'}")
        
        f = field_types[key]
        field_path = f"{current_path}.{key}" if current_path else key
        
        if dataclasses.is_dataclass(f.type):
            if not isinstance(value, dict):
                raise ConfigValidationError(f"Expected dict at {field_path}, got {type(value).__name__}")
            kwargs[key] = _parse_dict(f.type, value, field_path)
        else:
            # Basic type checking
            origin_type = getattr(f.type, '__origin__', f.type)
            if origin_type is list or origin_type is List:
                if not isinstance(value, list):
                    raise ConfigValidationError(f"Expected list at {field_path}, got {type(value).__name__}")
                # For windows, validate nested lists and times
                if key == "windows":
                    for idx, w in enumerate(value):
                        if not isinstance(w, list) or len(w) != 2:
                            raise ConfigValidationError(f"Expected [start, end] list at {field_path}[{idx}]")
                        _validate_time_str(str(w[0]), f"{field_path}[{idx}][0]")
                        _validate_time_str(str(w[1]), f"{field_path}[{idx}][1]")
            elif f.type in (int, float, str, bool):
                # Allow int to float cast
                if f.type is float and isinstance(value, int):
                    value = float(value)
                elif not isinstance(value, f.type):
                    raise ConfigValidationError(f"Expected {f.type.__name__} at {field_path}, got {type(value).__name__}")
            
            kwargs[key] = value

    obj = cls(**kwargs)
    return obj

def _post_validate(config: AppConfig):
    # Mode
    if config.mode not in ("pi", "usb", "file"):
        raise ConfigValidationError(f"Invalid mode '{config.mode}' (must be pi, usb, or file)")
    
    # YOLO
    if config.yolo.backend not in ("pt", "ncnn", "onnx", "openvino"):
        raise ConfigValidationError(f"Invalid yolo.backend '{config.yolo.backend}' (must be pt, ncnn, onnx, or openvino)")
    if not (0 <= config.yolo.conf <= 1.0):
        raise ConfigValidationError(f"yolo.conf must be between 0 and 1, got {config.yolo.conf}")
    if not (0 <= config.yolo.iou <= 1.0):
        raise ConfigValidationError(f"yolo.iou must be between 0 and 1, got {config.yolo.iou}")
    
    # Index
    if config.index.bin_minutes <= 0:
        raise ConfigValidationError(f"index.bin_minutes must be > 0, got {config.index.bin_minutes}")
    
    # Alert
    if config.alert.method not in ("cusum", "ears", "both"):
        raise ConfigValidationError(f"Invalid alert.method '{config.alert.method}' (must be cusum, ears, or both)")
    if config.alert.min_events < 1:
        raise ConfigValidationError(f"alert.min_events must be >= 1, got {config.alert.min_events}")
    if config.alert.watch_min_events < 1:
        raise ConfigValidationError(f"alert.watch_min_events must be >= 1, got {config.alert.watch_min_events}")
    if config.alert.clear_bins < 1:
        raise ConfigValidationError(f"alert.clear_bins must be >= 1, got {config.alert.clear_bins}")
    if config.alert.ears.lag_bins < 1:
        raise ConfigValidationError(f"alert.ears.lag_bins must be >= 1, got {config.alert.ears.lag_bins}")
        
    # Camera Format
    if config.camera.format != "RGB888":
        raise ConfigValidationError(f"Invalid camera.format '{config.camera.format}' (must be RGB888 as per thesis requirements)")

def load_config(path: str | Path = "config.yaml", project_root: Optional[Path] = None) -> AppConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {path} not found.")
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    config = _parse_dict(AppConfig, data)
    
    if project_root:
        config._project_root = project_root.resolve()
        
    _post_validate(config)
    
    return config
