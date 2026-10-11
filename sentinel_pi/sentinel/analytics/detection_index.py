"""Detection Index calculator for MosquitoSentinel.

The Detection Index quantifies mosquito activity per bin:

    raw_value = (confirmed Aedes Aegypti events in bin) / active_hours_in_bin

Design rules:
- Bins are aligned to Asia/Manila (UTC+8) clock boundaries of bin_minutes width,
  but only within configured monitoring windows (e.g. 06:00–09:00, 16:00–19:00).
- Timestamps are stored in UTC in the database.
- A bin with coverage < min_coverage is marked valid=0 and excluded from
  baseline computation and alerting.
- raw_value counts only 'confirmed' events whose started_at falls in the bin
  AND whose yolo_class is in target_class_names (case-insensitive).
- non_target and rejected events are NEVER counted.
- verified_value counts only confirmed events with verified_label='true'.
- n_unverified counts confirmed events without a verified_label.
- Recompute is idempotent (UPSERT on monitoring_point + bin_start).

Pure functions are defined at module level; the thin repository layer is in
sentinel/storage/repositories.py (DetectionIndexRepository).
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import List, NamedTuple, Optional, Sequence, Tuple

# Asia/Manila = UTC+8 (no DST)
_MANILA_OFFSET = timedelta(hours=8)
_MANILA_TZ = timezone(_MANILA_OFFSET)


class ActiveInterval(NamedTuple):
    """A half-open [start, end) interval during which the system was active."""
    start_utc: datetime  # timezone-aware UTC
    end_utc: datetime    # timezone-aware UTC


class BinSpec(NamedTuple):
    """Description of one detection-index bin."""
    bin_start: datetime   # UTC, tz-aware
    bin_end: datetime     # UTC, tz-aware
    window_group: str     # 'morning' | 'afternoon'


class BinResult(NamedTuple):
    """Computed values for one bin, ready to upsert into detection_index."""
    monitoring_point: str
    bin_start: datetime
    bin_end: datetime
    window_group: str
    raw_value: float
    verified_value: Optional[float]
    n_events: int
    n_verified_true: int
    n_unverified: int
    active_hours: float
    bin_coverage: float
    valid: int            # 1 = included; 0 = excluded (low coverage)


def _align_floor(dt: datetime, bin_minutes: int) -> datetime:
    """Floor *dt* to the nearest bin boundary in Asia/Manila time, return as UTC.

    Boundaries are aligned to midnight Manila time so that bin edges are
    predictable regardless of when monitoring started.
    """
    manila = dt.astimezone(_MANILA_TZ)
    # Minutes since midnight Manila
    mins_since_midnight = manila.hour * 60 + manila.minute
    floored_mins = (mins_since_midnight // bin_minutes) * bin_minutes
    floored_manila = manila.replace(
        hour=floored_mins // 60,
        minute=floored_mins % 60,
        second=0,
        microsecond=0,
    )
    return floored_manila.astimezone(timezone.utc)


def generate_bins(
    day_start_utc: datetime,
    day_end_utc: datetime,
    bin_minutes: int,
    windows: Sequence[Tuple[str, str]],
) -> List[BinSpec]:
    """Generate all bin specs for a given day that fall inside monitoring windows.

    Args:
        day_start_utc:  Inclusive start of the range to scan (UTC, tz-aware).
        day_end_utc:    Exclusive end of the range (UTC, tz-aware).
        bin_minutes:    Bin duration in minutes.
        windows:        List of (start_HH:MM, end_HH:MM) strings in local Manila time.

    Returns:
        List of BinSpec ordered by bin_start.
    """
    bin_delta = timedelta(minutes=bin_minutes)
    # Parse windows into Manila-local (hour, minute) pairs
    parsed_windows: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    for i, (ws, we) in enumerate(windows):
        sh, sm = map(int, ws.split(":"))
        eh, em = map(int, we.split(":"))
        label = "morning" if i == 0 else "afternoon"
        parsed_windows.append(((sh, sm), (eh, em), label))

    bins: List[BinSpec] = []
    current = _align_floor(day_start_utc, bin_minutes)

    while current < day_end_utc:
        next_bin = current + bin_delta
        bin_mid = current + bin_delta / 2

        # Classify via Manila time
        manila_mid = bin_mid.astimezone(_MANILA_TZ)
        mid_mins = manila_mid.hour * 60 + manila_mid.minute

        matched_label: Optional[str] = None
        for (sh, sm), (eh, em), label in parsed_windows:
            win_start_mins = sh * 60 + sm
            win_end_mins = eh * 60 + em
            if win_start_mins <= mid_mins < win_end_mins:
                matched_label = label
                break

        if matched_label is not None:
            bins.append(BinSpec(
                bin_start=current,
                bin_end=next_bin,
                window_group=matched_label,
            ))

        current = next_bin

    return bins


def compute_active_hours(
    bin_start: datetime,
    bin_end: datetime,
    active_intervals: Sequence[ActiveInterval],
) -> float:
    """Compute total active hours within a bin by intersecting with active intervals.

    Args:
        bin_start:        UTC start of the bin (inclusive).
        bin_end:          UTC end of the bin (exclusive).
        active_intervals: List of (start_utc, end_utc) active periods.

    Returns:
        Float hours of active monitoring within this bin.
    """
    total_seconds = 0.0
    for iv in active_intervals:
        overlap_start = max(bin_start, iv.start_utc)
        overlap_end = min(bin_end, iv.end_utc)
        if overlap_end > overlap_start:
            total_seconds += (overlap_end - overlap_start).total_seconds()
    return total_seconds / 3600.0


def compute_bin_result(
    spec: BinSpec,
    monitoring_point: str,
    active_intervals: Sequence[ActiveInterval],
    confirmed_events: Sequence[Tuple[datetime, Optional[str]]],
    target_class_names: Sequence[str],
    min_coverage: float,
) -> BinResult:
    """Compute one BinResult from raw event data.

    Args:
        spec:              The BinSpec (start, end, window_group).
        monitoring_point:  Site identifier.
        active_intervals:  System uptime segments.
        confirmed_events:  Sequence of (started_at_utc, verified_label) for
                           all 'confirmed' events with yolo_class in target names.
        target_class_names: Case-insensitive target class names (already lowercased).
        min_coverage:      Fraction below which bin is marked valid=0.

    Returns:
        BinResult ready for upsert.
    """
    bin_duration_hours = (spec.bin_end - spec.bin_start).total_seconds() / 3600.0
    active_hours = compute_active_hours(spec.bin_start, spec.bin_end, active_intervals)
    bin_coverage = (active_hours / bin_duration_hours) if bin_duration_hours > 0 else 0.0
    valid = 1 if bin_coverage >= min_coverage else 0

    # Count events in bin
    events_in_bin = [
        (ts, vl)
        for ts, vl in confirmed_events
        if spec.bin_start <= ts < spec.bin_end
    ]
    n_events = len(events_in_bin)
    n_verified_true = sum(1 for _, vl in events_in_bin if vl == "true")
    n_unverified = sum(1 for _, vl in events_in_bin if vl is None)

    # raw_value and verified_value use events_per_hour
    # SOP 2.2.2: verified_value must be NULL when valid == 0 (zero or low coverage),
    # or when the bin is not FULLY verified (n_unverified > 0).
    # When valid == 1 and fully verified (n_unverified == 0), it equals n_verified_true / active_hours,
    # including 0.0 when every event was marked false or when there were zero events.
    if active_hours > 0:
        raw_value = n_events / active_hours
    else:
        raw_value = 0.0

    if valid == 0 or n_unverified > 0:
        verified_value = None
    else:
        verified_value = (n_verified_true / active_hours) if active_hours > 0 else 0.0


    return BinResult(
        monitoring_point=monitoring_point,
        bin_start=spec.bin_start,
        bin_end=spec.bin_end,
        window_group=spec.window_group,
        raw_value=raw_value,
        verified_value=verified_value,
        n_events=n_events,
        n_verified_true=n_verified_true,
        n_unverified=n_unverified,
        active_hours=active_hours,
        bin_coverage=bin_coverage,
        valid=valid,
    )
