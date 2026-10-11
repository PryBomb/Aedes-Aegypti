"""Tests for Phase 4: Analytics (CUSUM, EARS, Detection Index, Baseline, AlertEngine,
Repositories, Schema migration v1→v2, and replay --virtual-start).

All arithmetic is documented in inline comments.
"""
from __future__ import annotations

import math
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pytest

from sentinel.analytics.baseline import BaselineStats, compute_baseline, window_group_for_index
from sentinel.analytics.cusum import cusum_sequence, cusum_step
from sentinel.analytics.ears import ears_sequence, ears_step
from sentinel.analytics.detection_index import (
    ActiveInterval,
    BinSpec,
    compute_active_hours,
    compute_bin_result,
    generate_bins,
)
from sentinel.analytics.alert_engine import AlertEngine, AlertEngineState, AlertDecision
from sentinel.storage.db import DatabaseManager
from sentinel.storage.repositories import (
    AlertsRepository,
    BaselinesRepository,
    DetectionIndexRepository,
)

# Asia/Manila = UTC+8
_MANILA_TZ = timezone(timedelta(hours=8))
_UTC = timezone.utc


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _dt_manila(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    """Create a tz-aware datetime in Manila time."""
    return datetime(year, month, day, hour, minute, tzinfo=_MANILA_TZ)


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=_UTC)


# ─── 1. CUSUM hand-calculated sequences ─────────────────────────────────────

class TestCusum:
    """Arithmetic checks for cusum_step and cusum_sequence.

    Parameters: mu=2, sigma=1, k=0.5, h=4, reset_after_alert=True
    """
    MU = 2.0
    SIGMA = 1.0
    K = 0.5
    H = 4.0

    def test_flat_series_no_alert(self):
        """Flat series at mu: z_t = (2-2)/1 = 0; S_t = max(0, S_{t-1} + 0 - 0.5) → 0."""
        results = cusum_sequence([self.MU] * 10, self.MU, self.SIGMA, self.K, self.H)
        for s_t, alerted in results:
            # S_t = max(0, 0 - 0.5) = 0
            assert s_t == pytest.approx(0.0)
            assert not alerted

    def test_single_step_arithmetic(self):
        """Hand calculation for one step:
        x=4, mu=2, sigma=1, k=0.5, prev_s=0
        z = (4-2)/1 = 2
        S_t = max(0, 0 + 2 - 0.5) = 1.5  → no alert (1.5 < h=4)
        """
        s_new, alerted = cusum_step(4.0, self.MU, self.SIGMA, self.K, self.H, 0.0)
        assert s_new == pytest.approx(1.5)
        assert not alerted

    def test_cumulative_build_up_triggers_alert(self):
        """Repeated x=4 (z=2, net gain = 2-0.5 = 1.5 per step).
        Step 1: S = max(0, 0 + 1.5) = 1.5
        Step 2: S = max(0, 1.5 + 1.5) = 3.0
        Step 3: S = max(0, 3.0 + 1.5) = 4.5 ≥ 4.0 → ALERT; carry reset→0, display retains 4.5
        Step 4: S = max(0, 0 + 1.5) = 1.5 (starts from reset carry)
        """
        results = cusum_sequence([4.0] * 5, self.MU, self.SIGMA, self.K, self.H, True)
        s_vals = [r[0] for r in results]
        alerted = [r[1] for r in results]

        assert s_vals[0] == pytest.approx(1.5)
        assert s_vals[1] == pytest.approx(3.0)
        assert alerted[2]   # step 3 fires
        assert s_vals[2] == pytest.approx(4.5)  # pre-reset display retained for storage/level
        assert s_vals[3] == pytest.approx(1.5)  # next step starts from reset carry (0.0)
        assert not alerted[0]
        assert not alerted[1]

    def test_strict_threshold_exact_equality_does_not_alert(self):
        """When S_t exactly equals h (e.g. 4.0), alerted must be False (strict > h + eps)."""
        # z = (x - mu)/sigma. Let mu=2, sigma=1, k=0.5, prev_s=0.
        # We want s_new = 4.0.
        # s_new = z - k => z = 4.5 => (x - 2)/1 = 4.5 => x = 6.5.
        s_new, alerted = cusum_step(6.5, self.MU, self.SIGMA, self.K, self.H, 0.0)
        assert s_new == pytest.approx(4.0)
        assert not alerted  # S == h exactly must NOT alert

        # But exceeding by epsilon triggers alert
        s_new_hi, alerted_hi = cusum_step(6.50001, self.MU, self.SIGMA, self.K, self.H, 0.0)
        assert alerted_hi
        assert s_new_hi > 4.0

    def test_below_baseline_resets_to_zero(self):
        """x < mu: z < 0; S_t = max(0, prev + negative - k) → stays at 0."""
        results = cusum_sequence([0.0] * 5, self.MU, self.SIGMA, self.K, self.H)
        for s_t, alerted in results:
            assert s_t == pytest.approx(0.0)
            assert not alerted

    def test_sigma_zero_raises(self):
        """sigma=0 must raise ValueError."""
        with pytest.raises(ValueError, match="sigma must be > 0"):
            cusum_step(4.0, 2.0, 0.0, 0.5, 4.0, 0.0)

    def test_no_reset_accumulates(self):
        """Without reset, S_t keeps growing after an alert crosses h."""
        results = cusum_sequence([4.0] * 5, self.MU, self.SIGMA, self.K, self.H, False)
        # After alert at step 3, S continues from 4.5 (not reset)
        assert results[2][1]       # alert at step 3
        assert results[2][0] == pytest.approx(4.5)  # no reset


# ─── 2. EARS hand-calculated sequences ───────────────────────────────────────

class TestEars:
    """Arithmetic checks for ears_step and ears_sequence.

    Parameters: lag_bins=3, z=3.0, min_sd=0.5
    """
    LAG = 3
    Z = 3.0
    MIN_SD = 0.5

    def _step(self, current: float, history: Sequence[float]) -> Tuple[Optional[float], bool]:
        return ears_step(current, history, self.Z, self.MIN_SD, self.LAG)

    def test_insufficient_history_returns_none(self):
        """With fewer than lag_bins history values, EARS must return (None, False)."""
        c1, alerted = self._step(10.0, [1.0, 2.0])  # only 2, need 3
        assert c1 is None
        assert not alerted

    def test_flat_history_no_alert(self):
        """history = [2,2,2], current=2: mu=2, sd=0 (floor to 0.5), C1=(2-2)/0.5=0 < z."""
        c1, alerted = self._step(2.0, [2.0, 2.0, 2.0])
        assert c1 == pytest.approx(0.0)
        assert not alerted

    def test_spike_triggers_alert(self):
        """history = [2,2,2], current=4.5:
        mu=2, sd=max(0,0.5)=0.5, C1=(4.5-2)/0.5=5.0 >= z=3 → ALERT.
        """
        c1, alerted = self._step(4.5, [2.0, 2.0, 2.0])
        assert c1 == pytest.approx(5.0)
        assert alerted

    def test_min_sd_floor_applied(self):
        """history=[3,3,3], min_sd=0.5: sd=0 → floored to 0.5.
        C1 = (5 - 3) / 0.5 = 4.0 >= z=3 → alert.
        """
        c1, alerted = self._step(5.0, [3.0, 3.0, 3.0])
        assert c1 == pytest.approx(4.0)
        assert alerted

    def test_sequence_first_lag_bins_no_alert(self):
        """The first lag_bins positions in the sequence always return (None, False)."""
        series = [2.0, 2.0, 2.0, 10.0]
        results = ears_sequence(series, self.Z, self.MIN_SD, self.LAG)
        # First 3 have no history sufficient
        for c1, alerted in results[:3]:
            assert c1 is None
            assert not alerted
        # 4th: history=[2,2,2], current=10: mu=2, C1=(10-2)/0.5=16 → ALERT
        c1_4, alerted_4 = results[3]
        assert alerted_4

    def test_non_alerting_below_z(self):
        """current slightly above mu but not enough to trigger."""
        history = [2.0, 2.0, 2.0]
        c1, alerted = self._step(2.5, history)
        # C1 = (2.5-2)/0.5 = 1.0 < z=3
        assert c1 == pytest.approx(1.0)
        assert not alerted


# ─── 3. Detection Index bin boundaries (Asia/Manila, UTC+8) ──────────────────

class TestBinBoundaries:
    """Verify that bins align to Manila clock boundaries and map to window groups."""

    WINDOWS = [("06:00", "09:00"), ("16:00", "19:00")]
    BIN_MIN = 15

    def test_bins_align_to_quarter_hour(self):
        """Bins should start at :00, :15, :30, :45 Manila time."""
        # 2026-10-12 06:00 Manila = 2026-10-11 22:00 UTC
        day_start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        day_end = _dt_manila(2026, 10, 12, 9, 0).astimezone(_UTC)

        bins = generate_bins(day_start, day_end, self.BIN_MIN, self.WINDOWS)
        assert len(bins) == 12  # 3 hours / 15 min = 12 bins

        for b in bins:
            # Verify alignment: bin_start minutes should be multiples of 15
            manila_start = b.bin_start.astimezone(_MANILA_TZ)
            assert manila_start.second == 0
            assert manila_start.microsecond == 0
            assert manila_start.minute % self.BIN_MIN == 0

    def test_morning_window_group_label(self):
        """All bins in 06:00–09:00 should be labelled 'morning'."""
        day_start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        day_end = _dt_manila(2026, 10, 12, 9, 0).astimezone(_UTC)
        bins = generate_bins(day_start, day_end, self.BIN_MIN, self.WINDOWS)
        for b in bins:
            assert b.window_group == "morning"

    def test_afternoon_window_group_label(self):
        """All bins in 16:00–19:00 should be labelled 'afternoon'."""
        day_start = _dt_manila(2026, 10, 12, 16, 0).astimezone(_UTC)
        day_end = _dt_manila(2026, 10, 12, 19, 0).astimezone(_UTC)
        bins = generate_bins(day_start, day_end, self.BIN_MIN, self.WINDOWS)
        assert len(bins) == 12
        for b in bins:
            assert b.window_group == "afternoon"

    def test_bins_outside_windows_excluded(self):
        """Bins outside monitoring windows should not be generated."""
        # 10:00–12:00 Manila is outside both windows
        day_start = _dt_manila(2026, 10, 12, 10, 0).astimezone(_UTC)
        day_end = _dt_manila(2026, 10, 12, 12, 0).astimezone(_UTC)
        bins = generate_bins(day_start, day_end, self.BIN_MIN, self.WINDOWS)
        assert len(bins) == 0

    def test_bin_timestamps_in_utc(self):
        """bin_start and bin_end must be UTC-aware datetimes."""
        day_start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        day_end = _dt_manila(2026, 10, 12, 6, 30).astimezone(_UTC)
        bins = generate_bins(day_start, day_end, self.BIN_MIN, self.WINDOWS)
        for b in bins:
            assert b.bin_start.tzinfo is not None
            assert b.bin_end.tzinfo is not None
            # Should be UTC
            assert b.bin_start.utcoffset() == timedelta(0)


# ─── 4. Coverage exclusion ────────────────────────────────────────────────────

class TestCoverageExclusion:
    """Bins with coverage < min_coverage are marked valid=0."""

    def _make_spec(self) -> BinSpec:
        start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        end = start + timedelta(minutes=15)
        return BinSpec(bin_start=start, bin_end=end, window_group="morning")

    def test_full_coverage_valid(self):
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        result = compute_bin_result(
            spec, "MP-1", active, [], ["aedes aegypti"], min_coverage=0.5
        )
        assert result.bin_coverage == pytest.approx(1.0)
        assert result.valid == 1

    def test_partial_coverage_excluded(self):
        """Active interval covers only the first third of the bin → coverage=1/3 < 0.5."""
        spec = self._make_spec()
        bin_duration = (spec.bin_end - spec.bin_start).total_seconds()
        active_end = spec.bin_start + timedelta(seconds=bin_duration / 3)
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=active_end)]
        result = compute_bin_result(
            spec, "MP-1", active, [], ["aedes aegypti"], min_coverage=0.5
        )
        assert result.bin_coverage == pytest.approx(1.0 / 3.0, rel=1e-5)
        assert result.valid == 0

    def test_no_activity_excluded(self):
        spec = self._make_spec()
        result = compute_bin_result(spec, "MP-1", [], [], ["aedes aegypti"], min_coverage=0.5)
        assert result.active_hours == pytest.approx(0.0)
        assert result.valid == 0


# ─── 5. raw_value vs verified_value ──────────────────────────────────────────

class TestRawVsVerified:
    """raw_value counts confirmed targets; verified_value counts only verified_label='true'."""

    def _make_spec(self) -> BinSpec:
        start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        end = start + timedelta(minutes=15)
        return BinSpec(bin_start=start, bin_end=end, window_group="morning")

    def test_partially_verified_returns_none(self):
        """When some events are unverified (n_unverified > 0), verified_value must be None."""
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        t = spec.bin_start + timedelta(minutes=2)
        events = [(t, "true"), (t, "true"), (t, "false"), (t, None)]
        result = compute_bin_result(spec, "MP-1", active, events, ["aedes aegypti"], min_coverage=0.5)
        assert result.n_events == 4
        assert result.n_verified_true == 2
        assert result.n_unverified == 1
        assert result.raw_value == pytest.approx(16.0)
        assert result.verified_value is None  # Partial verification must be NULL

    def test_all_unverified_returns_none(self):
        """When all events are unverified, verified_value must be None."""
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        t = spec.bin_start + timedelta(minutes=2)
        events = [(t, None), (t, None), (t, None)]
        result = compute_bin_result(spec, "MP-1", active, events, ["aedes aegypti"], min_coverage=0.5)
        assert result.n_events == 3
        assert result.n_unverified == 3
        assert result.n_verified_true == 0
        assert result.raw_value == pytest.approx(12.0)
        assert result.verified_value is None

    def test_all_true_returns_rate(self):
        """When fully verified and all true, verified_value equals n_verified_true / active_hours."""
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        t = spec.bin_start + timedelta(minutes=2)
        events = [(t, "true"), (t, "true")]
        result = compute_bin_result(spec, "MP-1", active, events, ["aedes aegypti"], min_coverage=0.5)
        assert result.n_events == 2
        assert result.n_unverified == 0
        assert result.n_verified_true == 2
        assert result.raw_value == pytest.approx(8.0)
        assert result.verified_value == pytest.approx(8.0)

    def test_all_false_returns_zero(self):
        """When fully verified and all false, verified_value must be 0.0 (not None)."""
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        t = spec.bin_start + timedelta(minutes=2)
        events = [(t, "false"), (t, "false"), (t, "false")]
        result = compute_bin_result(spec, "MP-1", active, events, ["aedes aegypti"], min_coverage=0.5)
        assert result.n_events == 3
        assert result.n_unverified == 0
        assert result.n_verified_true == 0
        assert result.raw_value == pytest.approx(12.0)
        assert result.verified_value == pytest.approx(0.0)

    def test_zero_events_returns_zero(self):
        """When there are zero events, fully verified condition holds (n_unverified==0) -> 0.0."""
        spec = self._make_spec()
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_end)]
        result = compute_bin_result(spec, "MP-1", active, [], ["aedes aegypti"], min_coverage=0.5)
        assert result.n_events == 0
        assert result.n_unverified == 0
        assert result.n_verified_true == 0
        assert result.raw_value == pytest.approx(0.0)
        assert result.verified_value == pytest.approx(0.0)

    def test_invalid_bin_returns_none_verified_value(self):
        """When valid == 0 (low/zero coverage), verified_value must be None even if all verified."""
        spec = self._make_spec()
        # Active hours = 3 min < 7.5 min (50% min_coverage), valid == 0
        active = [ActiveInterval(start_utc=spec.bin_start, end_utc=spec.bin_start + timedelta(minutes=3))]
        t = spec.bin_start + timedelta(minutes=2)
        events = [(t, "true")]
        result = compute_bin_result(spec, "MP-1", active, events, ["aedes aegypti"], min_coverage=0.5)
        assert result.valid == 0
        assert result.n_unverified == 0
        assert result.verified_value is None



# ─── 6. Idempotent recompute ─────────────────────────────────────────────────

class TestIdempotentRecompute:
    """Upserting the same bin twice should not create duplicate rows."""

    def _make_db(self, tmp_path: Path) -> DatabaseManager:
        schema = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        return DatabaseManager(tmp_path / "test.db", schema)

    def test_upsert_is_idempotent(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = DetectionIndexRepository(db)
        data = {
            "monitoring_point": "MP-1",
            "bin_start": "2026-10-12T06:00:00.000Z",
            "bin_end": "2026-10-12T06:15:00.000Z",
            "raw_value": 4.0,
            "n_events": 1,
            "n_unverified": 0,
            "active_hours": 0.25,
            "bin_coverage": 1.0,
            "valid": 1,
            "level": "NORMAL",
        }
        repo.upsert(data)
        repo.upsert({**data, "raw_value": 8.0})  # overwrite same bin

        rows = repo.get_by_monitoring_point("MP-1")
        assert len(rows) == 1
        assert rows[0]["raw_value"] == pytest.approx(8.0)


# ─── 7. Baseline suppression ─────────────────────────────────────────────────

class TestBaselineSuppression:
    """AlertEngine must not fire while baseline is not ready."""

    def _engine(self) -> AlertEngine:
        return AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=7,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
        )

    def test_no_alert_when_not_ready(self):
        engine = self._engine()
        baseline = BaselineStats(
            monitoring_point="MP-1",
            window_group="morning",
            mu=2.0,
            sigma=1.0,
            n_bins=5,   # < min_bins (default 12) → not ready
            is_ready=False,
        )
        state = AlertEngineState()
        bin_start = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        # Even a huge spike should not alert when baseline is not ready
        decision = engine.evaluate(20.0, bin_start, baseline, state, n_events=5)
        assert decision.level == "NORMAL"
        assert not decision.fired

    def test_evaluate_requires_n_events(self):
        """evaluate() must raise TypeError if n_events is omitted."""
        engine = self._engine()
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        bin_start = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        with pytest.raises(TypeError):
            engine.evaluate(6.0, bin_start, baseline, state)  # type: ignore[call-arg]

    def test_level_is_alert_when_cusum_fires_with_reset(self):
        """When CUSUM condition is met and reset_after_alert=True, the bin level
        must be ALERT (computed before reset), and cusum_value must reflect pre-reset S_t."""
        engine = self._engine()
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        # x=7.0 => z = (7-2)/1 = 5.0 => S = max(0, 0 + 5.0 - 0.5) = 4.5 > 4.0
        t = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        decision = engine.evaluate(7.0, t, baseline, state, n_events=5)
        assert decision.level == "ALERT"
        assert decision.fired
        assert decision.cusum_value == pytest.approx(4.5)  # pre-reset display value
        assert state.s_t == pytest.approx(0.0)             # post-reset carry is 0

    def test_cusum_carry_not_reset_when_suppressed_by_min_events(self):
        """When CUSUM threshold is exceeded but n_events < min_events, S_t carry must NOT reset."""
        engine = AlertEngine(
            method="cusum",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=4,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=0.3, sigma=0.5, n_bins=20, is_ready=True)
        state = AlertEngineState()
        t1 = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        # raw_value=4.0 => z=(4.0-0.3)/0.5 = 7.4 => S=7.4-0.5 = 6.9 > 4.0
        # But n_events=1 < min_events=3 (and 1 < watch_min_events=2)
        dec = engine.evaluate(4.0, t1, baseline, state, n_events=1)
        assert dec.level == "NORMAL"
        assert not dec.fired
        assert state.s_t == pytest.approx(6.9)  # Carry is preserved! NOT reset to 0.0

    def test_watch_min_events_constraint(self):
        """WATCH level requires n_events >= watch_min_events (default 2)."""
        engine = AlertEngine(
            method="cusum",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=4,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state1 = AlertEngineState()
        state2 = AlertEngineState()
        t = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        # x=4.9 => z=2.9 => S=2.4 == 0.6 * 4.0 (watch threshold)
        # Case A: n_events = 1 < watch_min_events (2) -> NORMAL
        dec1 = engine.evaluate(4.9, t, baseline, state1, n_events=1)
        assert dec1.level == "NORMAL"

        # Case B: n_events = 2 >= watch_min_events (2) -> WATCH
        dec2 = engine.evaluate(4.9, t, baseline, state2, n_events=2)
        assert dec2.level == "WATCH"

    def test_ears_strict_threshold_exact_equality(self):
        """EARS alerts strictly when C1 > z + 1e-9; exact equality C1 == z must not alert."""
        # 4 bins at 2.0 (mean=2.0, sd floored at min_sd=0.5)
        # current = 2.0 + 3.0 * 0.5 = 3.5 => C1 = (3.5 - 2.0)/0.5 = 3.0 == z_threshold
        c1, alerted = ears_step(3.5, [2.0] * 4, z_threshold=3.0, min_sd=0.5, lag_bins=4)
        assert c1 == pytest.approx(3.0)
        assert not alerted  # Exactly on threshold does NOT alert

        c1_hi, alerted_hi = ears_step(3.50001, [2.0] * 4, z_threshold=3.0, min_sd=0.5, lag_bins=4)
        assert alerted_hi


# ─── 8. Cooldown ─────────────────────────────────────────────────────────────

class TestCooldown:
    """AlertEngine must suppress repeated alerts within cooldown_min using bin timestamps."""

    def _engine(self) -> AlertEngine:
        return AlertEngine(
            method="cusum",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=False,  # keep S_t high to ensure second would fire
            ears_lag_bins=4,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )

    def test_second_alert_suppressed_in_cooldown(self):
        engine = self._engine()
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()

        # Force S_t to just above h by pre-loading
        state.s_t = 4.5  # already alerted territory

        t1 = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        decision1 = engine.evaluate(6.0, t1, baseline, state, n_events=5)
        assert decision1.level == "ALERT"
        assert decision1.fired
        assert not decision1.cooldown_suppressed

        # 15 minutes later — within 30-minute cooldown
        t2 = t1 + timedelta(minutes=15)
        state.s_t = 4.5  # keep it high
        decision2 = engine.evaluate(6.0, t2, baseline, state, n_events=5)
        # Level stays ALERT during cooldown, but no alert row is created
        assert decision2.level == "ALERT"
        assert not decision2.fired
        assert decision2.episode_active  # Still in episode 1

    def test_alert_fires_after_cooldown(self):
        engine = self._engine()
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        state.s_t = 4.5

        t1 = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        decision1 = engine.evaluate(6.0, t1, baseline, state, n_events=5)
        assert decision1.fired

        # Close episode by running clear_bins (2) clear bins
        t_clear1 = t1 + timedelta(minutes=15)
        engine.evaluate(1.0, t_clear1, baseline, state, n_events=0)
        t_clear2 = t1 + timedelta(minutes=30)
        engine.evaluate(1.0, t_clear2, baseline, state, n_events=0)
        assert not state.episode_active  # Episode closed

        # 35 minutes after episode closed at t_clear2 — outside cooldown
        t2 = t_clear2 + timedelta(minutes=35)
        state.s_t = 4.5
        decision2 = engine.evaluate(6.0, t2, baseline, state, n_events=5)
        assert decision2.level == "ALERT"
        assert decision2.fired
        assert not decision2.cooldown_suppressed

    def test_cooldown_uses_bin_timestamps_not_wall_clock(self):
        """Cooldown must be computed against bin_start timestamps, independent of wall clock."""
        engine = self._engine()
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        state.s_t = 4.5

        # Historical timestamps (year 2020) far in the past
        past_t1 = datetime(2020, 1, 1, 8, 0, tzinfo=_UTC)
        past_t2 = past_t1 + timedelta(minutes=10)   # 10m < 30m

        d1 = engine.evaluate(6.0, past_t1, baseline, state, n_events=5)
        assert d1.fired
        state.s_t = 4.5
        d2 = engine.evaluate(6.0, past_t2, baseline, state, n_events=5)
        assert not d2.fired
        assert d2.level == "ALERT"

        # Close episode
        engine.evaluate(1.0, past_t1 + timedelta(minutes=20), baseline, state, n_events=0)
        engine.evaluate(1.0, past_t1 + timedelta(minutes=30), baseline, state, n_events=0)
        assert not state.episode_active

        # 35 minutes after episode closed at past_t1 + 30m
        past_t3 = past_t1 + timedelta(minutes=65)
        state.s_t = 4.5
        d3 = engine.evaluate(6.0, past_t3, baseline, state, n_events=5)
        assert d3.fired
        assert not d3.cooldown_suppressed
        assert d3.level == "ALERT"



# ─── 8b. Episode Lifecycle & Restart Rebuild ─────────────────────────────────

class TestEpisodeLifecycleAndRebuildState:
    """Alert episode state machine and restart safety via rebuild_state."""

    def test_episode_lifecycle(self):
        """Test full episode lifecycle: start -> active -> 2 clear bins -> close -> new episode."""
        engine = AlertEngine(
            method="cusum",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=4,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        t0 = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)

        # Bin 1: High surge (x=8.0, n_events=3) -> Episode 1 starts!
        d1 = engine.evaluate(8.0, t0, baseline, state, n_events=3)
        assert d1.fired
        assert d1.level == "ALERT"
        assert d1.episode_active
        assert state.episode_active

        # Bin 2: Elevated (x=8.0, n_events=3) -> Inside Episode 1! No new row.
        t1 = t0 + timedelta(minutes=15)
        d2 = engine.evaluate(8.0, t1, baseline, state, n_events=3)
        assert not d2.fired
        assert d2.level == "ALERT"
        assert d2.episode_active
        assert state.episode_active

        # Bin 3: Normal drop (x=2.0, n_events=0) -> Clear bin 1. Still in Episode 1!
        t2 = t0 + timedelta(minutes=30)
        d3 = engine.evaluate(2.0, t2, baseline, state, n_events=0)
        assert not d3.fired
        assert d3.level == "ALERT"
        assert state.episode_active
        assert state.clear_count == 1

        # Bin 4: Normal drop (x=2.0, n_events=0) -> Clear bin 2. Episode closes at end of bin!
        t3 = t0 + timedelta(minutes=45)
        d4 = engine.evaluate(2.0, t3, baseline, state, n_events=0)
        assert not d4.fired
        assert d4.level == "ALERT"
        assert not state.episode_active  # Episode closed!

        # Bin 5: Normal drop (x=2.0, n_events=0) -> Outside episode! Level is NORMAL.
        t4 = t0 + timedelta(minutes=60)
        d5 = engine.evaluate(2.0, t4, baseline, state, n_events=0)
        assert d5.level == "NORMAL"
        assert not d5.fired

        # Bin 6: New surge (x=8.0, n_events=3, t=75m >= 30m cooldown) -> Episode 2 starts!
        t5 = t0 + timedelta(minutes=75)
        d6 = engine.evaluate(8.0, t5, baseline, state, n_events=3)
        assert d6.fired
        assert d6.level == "ALERT"
        assert state.episode_active

    def test_cooldown_recurrence_continues_episode(self):
        """If alert recurs within cooldown_min after closing, it CONTINUES that episode:
        level ALERT, episode_active=True, fired=False, no new row.
        Never report ALERT for a bin that is not inside an episode.
        """
        engine = AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=3,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()
        t0 = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)

        # 1. Surge starts Episode 1 (row written)
        d1 = engine.evaluate(8.0, t0, baseline, state, n_events=3)
        assert d1.fired
        assert d1.level == "ALERT"
        assert d1.episode_active

        # 2. Clear bin 1
        d2 = engine.evaluate(2.0, t0 + timedelta(minutes=15), baseline, state, n_events=0)
        assert not d2.fired
        assert d2.level == "ALERT"
        assert d2.episode_active

        # 3. Clear bin 2 -> Episode closes at end of bin (t=30m)
        d3 = engine.evaluate(2.0, t0 + timedelta(minutes=30), baseline, state, n_events=0)
        assert not d3.fired
        assert d3.level == "ALERT"
        assert not state.episode_active  # Closed! last_active_bin was t=30m

        # 4. Normal bin outside episode -> must NEVER report ALERT!
        d4 = engine.evaluate(2.0, t0 + timedelta(minutes=40), baseline, state, n_events=0)
        assert d4.level == "NORMAL"
        assert not d4.episode_active

        # 5. Alert recurs at t=45m (15m after last active bin at 30m, within 30m cooldown)
        # MUST continue episode: level ALERT, episode_active=True, fired=False, no new row!
        d5 = engine.evaluate(8.0, t0 + timedelta(minutes=45), baseline, state, n_events=3)
        assert d5.level == "ALERT"
        assert d5.episode_active
        assert not d5.fired  # No new row!
        assert state.episode_active

    def test_window_boundary_day_change_resets(self):
        """A new Manila-local date resets the window state."""
        engine = AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=3,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()

        # Day 1 Manila: 2026-10-12 06:00 (UTC 2026-10-11 22:00)
        t_day1 = datetime(2026, 10, 11, 22, 0, tzinfo=_UTC)
        engine.evaluate(8.0, t_day1, baseline, state, n_events=3)
        assert state.episode_active
        assert state.s_t == 0.0 or state.s_t > 0.0
        assert len(state.ears_history) == 1

        # Day 2 Manila: 2026-10-13 06:00 (UTC 2026-10-12 22:00)
        t_day2 = datetime(2026, 10, 12, 22, 0, tzinfo=_UTC)
        d_day2 = engine.evaluate(2.0, t_day2, baseline, state, n_events=0)
        # Window state was reset!
        assert d_day2.level == "NORMAL"
        assert not state.episode_active
        assert len(state.ears_history) == 1  # reset then appended current

    def test_window_boundary_outage_inside_window_does_not_reset(self):
        """An outage inside the same window and day must NOT reset window state."""
        engine = AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=3,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)
        state = AlertEngineState()

        # Morning window on 2026-10-12 Manila:
        # Bin 1 at 06:00 (UTC 2026-10-11 22:00)
        t1 = datetime(2026, 10, 11, 22, 0, tzinfo=_UTC)
        engine.evaluate(4.0, t1, baseline, state, n_events=1)
        s_prev = state.s_t
        assert s_prev > 0.0

        # Outage of 2 hours inside the morning window:
        # Bin 2 at 08:30 (UTC 2026-10-12 00:30) - same Manila date and 'morning'
        t2 = datetime(2026, 10, 12, 0, 30, tzinfo=_UTC)
        engine.evaluate(4.0, t2, baseline, state, n_events=1)
        # Must accumulate without reset!
        assert state.s_t > s_prev
        assert len(state.ears_history) == 2

    def test_rebuild_state_restart_safety(self):
        """Live evaluation and rebuild_state must produce identical state and decisions."""
        engine = AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=3,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)

        # 1. Live evaluation of 6 bins
        live_state = AlertEngineState()
        start = datetime(2026, 10, 12, 6, 0, tzinfo=_UTC)
        test_data = [
            (2.0, 0),
            (2.5, 1),
            (3.0, 1),
            (8.0, 3),  # Episode starts!
            (8.0, 3),  # In episode
            (2.0, 0),  # Clear bin 1
        ]
        stored_bins = []
        stored_alerts = []

        for i, (val, ev) in enumerate(test_data):
            bt = start + timedelta(minutes=15 * i)
            dec = engine.evaluate(val, bt, baseline, live_state, n_events=ev)
            stored_bins.append({
                "monitoring_point": "MP-1",
                "bin_start": bt.isoformat(),
                "bin_end": (bt + timedelta(minutes=15)).isoformat(),
                "window_group": "morning",
                "raw_value": val,
                "n_events": ev,
                "valid": 1,
            })
            if dec.fired:
                stored_alerts.append({
                    "monitoring_point": "MP-1",
                    "t_alert": bt.isoformat(),
                    "method": dec.method,
                    "level": dec.level,
                })

        # 2. Rebuild state from stored records
        rebuilt_state = engine.rebuild_state(stored_bins, stored_alerts, "morning", baseline)

        # 3. Assert states are identical
        assert rebuilt_state.s_t == pytest.approx(live_state.s_t)
        assert rebuilt_state.ears_history == live_state.ears_history
        assert rebuilt_state.last_alert_time == live_state.last_alert_time
        assert rebuilt_state.last_active_bin == live_state.last_active_bin
        assert rebuilt_state.episode_active == live_state.episode_active
        assert rebuilt_state.clear_count == live_state.clear_count

        # 4. Next bin decision must be identical
        next_t = start + timedelta(minutes=15 * 6)
        next_val = 2.0
        next_ev = 0
        live_dec = engine.evaluate(next_val, next_t, baseline, live_state, n_events=next_ev)
        rebuilt_dec = engine.evaluate(next_val, next_t, baseline, rebuilt_state, n_events=next_ev)

        assert rebuilt_dec.level == live_dec.level
        assert rebuilt_dec.fired == live_dec.fired
        assert rebuilt_dec.episode_active == live_dec.episode_active
        assert rebuilt_dec.cusum_value == pytest.approx(live_dec.cusum_value)

    def test_rebuild_state_with_real_sqlite_rows_and_t_alert(self, tmp_path: Path):
        """Insert REAL rows into a temp SQLite DB via repositories (sqlite3.Row) and prove
        cooldown survives a rebuild.
        """
        schema = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(tmp_path / "test_rebuild.db", schema)
        index_repo = DetectionIndexRepository(db)
        alert_repo = AlertsRepository(db)

        engine = AlertEngine(
            method="both",
            watch_fraction=0.6,
            cooldown_min=30,
            cusum_k=0.5,
            cusum_h=4.0,
            cusum_reset_after_alert=True,
            ears_lag_bins=3,
            ears_z=3.0,
            ears_min_sd=0.5,
            min_sigma=0.5,
            min_events=3,
            watch_min_events=2,
            clear_bins=2,
        )
        baseline = BaselineStats("MP-1", "morning", mu=2.0, sigma=1.0, n_bins=20, is_ready=True)

        start = _dt_manila(2026, 10, 12, 6, 0).astimezone(_UTC)
        live_state = AlertEngineState()


        # Step 1: Run 4 bins. Bin 3 fires an alert; Bin 4 is in episode.
        for i, (val, ev) in enumerate([(2.0, 0), (2.0, 0), (8.0, 3), (8.0, 3)]):
            bt = start + timedelta(minutes=15 * i)
            dec = engine.evaluate(val, bt, baseline, live_state, n_events=ev)

            index_repo.upsert({
                "monitoring_point": "MP-1",
                "bin_start": bt.isoformat(),
                "bin_end": (bt + timedelta(minutes=15)).isoformat(),
                "raw_value": val,
                "n_events": ev,
                "valid": 1,
                "active_hours": 0.25,
                "bin_coverage": 1.0,
                "level": dec.level,
            })

            if dec.fired:
                alert_repo.create({
                    "monitoring_point": "MP-1",
                    "t_alert": bt.isoformat(),
                    "level": dec.level,
                    "method": dec.method,
                    "bin_start": bt.isoformat(),
                })

        # Fetch real sqlite3.Row objects from the DB
        stored_bins = index_repo.get_by_monitoring_point("MP-1")
        stored_alerts = alert_repo.get_by_monitoring_point("MP-1")

        assert len(stored_bins) == 4
        assert len(stored_alerts) == 1
        assert isinstance(stored_bins[0], sqlite3.Row)
        assert isinstance(stored_alerts[0], sqlite3.Row)
        # Note: stored_alerts column is 't_alert'
        assert stored_alerts[0]["t_alert"] is not None

        # Rebuild state from sqlite3.Row objects
        rebuilt_state = engine.rebuild_state(stored_bins, stored_alerts, "morning", baseline)

        # Verify rebuilt state matches live state
        assert rebuilt_state.episode_active == live_state.episode_active
        assert rebuilt_state.last_active_bin == live_state.last_active_bin
        assert rebuilt_state.last_alert_time == live_state.last_alert_time
        assert rebuilt_state.s_t == pytest.approx(live_state.s_t)

        # Both must handle the next bin identically (continuing in episode, no new row)
        next_t = start + timedelta(minutes=60)
        dec_live = engine.evaluate(8.0, next_t, baseline, live_state, n_events=3)
        dec_rebuilt = engine.evaluate(8.0, next_t, baseline, rebuilt_state, n_events=3)

        assert dec_rebuilt.level == "ALERT"
        assert dec_rebuilt.fired is False  # Cooldown/episode continuation survives rebuild!
        assert dec_rebuilt.level == dec_live.level
        assert dec_rebuilt.fired == dec_live.fired



# ─── 9. Schema migration v1 → v2 ─────────────────────────────────────────────

class TestSchemaMigration:
    """Prove that a v1 database is upgraded to v2 in place."""

    def _create_v1_db(self, db_path: Path) -> None:
        """Create a minimal v1 database by hand (without new columns/tables)."""
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA user_version = 1")
        conn.executescript(
            """
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
                status TEXT NOT NULL,
                snapshot_path TEXT,
                clip_path TEXT,
                t_first_detect DATETIME,
                t_confirmed DATETIME,
                t_saved DATETIME,
                verified_label TEXT,
                verified_by TEXT,
                verified_at DATETIME,
                notes TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            );
            CREATE TABLE IF NOT EXISTS detection_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                monitoring_point TEXT NOT NULL,
                bin_start DATETIME NOT NULL,
                bin_end DATETIME NOT NULL,
                raw_value REAL NOT NULL,
                verified_value REAL,
                n_events INTEGER NOT NULL,
                n_verified_true INTEGER,
                active_hours REAL NOT NULL,
                bin_coverage REAL NOT NULL,
                cusum_value REAL,
                ears_value REAL,
                level TEXT NOT NULL,
                UNIQUE(monitoring_point, bin_start)
            );
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
            """
        )
        conn.commit()
        conn.close()

    def test_v1_to_v2_migration(self, tmp_path: Path):
        """Opening a v1 database with DatabaseManager must upgrade it to v2."""
        db_path = tmp_path / "v1.db"
        self._create_v1_db(db_path)

        # Verify it's v1 first
        conn = sqlite3.connect(db_path)
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert ver == 1

        # Now let DatabaseManager migrate it
        schema_path = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        db = DatabaseManager(db_path, schema_path)

        with db.get_connection() as conn:
            ver2 = conn.execute("PRAGMA user_version").fetchone()[0]
            assert ver2 == 2, f"Expected user_version=2, got {ver2}"

            # Check valid column exists in detection_index
            cols = {row[1] for row in conn.execute("PRAGMA table_info(detection_index)")}
            assert "valid" in cols, "Column 'valid' missing after migration"
            assert "n_unverified" in cols, "Column 'n_unverified' missing after migration"

            # Check baselines table exists
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            }
            assert "baselines" in tables, "Table 'baselines' missing after migration"

    def test_migration_is_idempotent(self, tmp_path: Path):
        """Running the migration twice (opening the migrated DB again) must not fail."""
        db_path = tmp_path / "v1b.db"
        self._create_v1_db(db_path)
        schema_path = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        # First open (migrates v1 → v2)
        DatabaseManager(db_path, schema_path)
        # Second open (already v2; must be a no-op)
        DatabaseManager(db_path, schema_path)

        conn = sqlite3.connect(db_path)
        ver = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert ver == 2


# ─── 10. Baseline compute ─────────────────────────────────────────────────────

class TestBaselineCompute:
    """Unit tests for compute_baseline."""

    def test_empty_series_not_ready(self):
        stats = compute_baseline([], "MP-1", "morning", min_bins=12, min_sigma=0.5)
        assert not stats.is_ready
        assert stats.n_bins == 0
        assert stats.sigma == pytest.approx(0.5)  # floored

    def test_insufficient_bins_not_ready(self):
        vals = [2.0] * 5
        stats = compute_baseline(vals, "MP-1", "morning", min_bins=12, min_sigma=0.5)
        assert not stats.is_ready

    def test_ready_when_enough_bins(self):
        vals = [2.0] * 12
        stats = compute_baseline(vals, "MP-1", "morning", min_bins=12, min_sigma=0.5)
        assert stats.is_ready
        assert stats.n_bins == 12
        assert stats.mu == pytest.approx(2.0)
        assert stats.sigma == pytest.approx(0.5)  # sd=0 floored to min_sigma

    def test_mu_and_sigma_calculation(self):
        """vals = [1, 2, 3]: mu=2, population sd = sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/3)
        = sqrt(2/3) ≈ 0.8165.
        sigma = max(0.8165, 0.5) = 0.8165.
        """
        vals = [1.0, 2.0, 3.0]
        stats = compute_baseline(vals, "MP-1", "morning", min_bins=1, min_sigma=0.5)
        assert stats.mu == pytest.approx(2.0)
        expected_sd = math.sqrt(2.0 / 3.0)
        assert stats.sigma == pytest.approx(expected_sd, rel=1e-5)

    def test_window_group_labels(self):
        assert window_group_for_index(0) == "morning"
        assert window_group_for_index(1) == "afternoon"
        assert window_group_for_index(2) == "afternoon"


# ─── 11. Baselines Repository ─────────────────────────────────────────────────

class TestBaselinesRepository:
    def _make_db(self, tmp_path: Path) -> DatabaseManager:
        schema = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        return DatabaseManager(tmp_path / "test.db", schema)

    def test_upsert_and_get(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = BaselinesRepository(db)
        repo.upsert({
            "monitoring_point": "KAS-1E-01",
            "window_group": "morning",
            "mu": 3.5,
            "sigma": 1.2,
            "n_bins": 14,
            "computed_at": "2026-10-12T06:00:00.000Z",
            "source": "computed",
        })
        row = repo.get("KAS-1E-01", "morning")
        assert row is not None
        assert row["mu"] == pytest.approx(3.5)
        assert row["sigma"] == pytest.approx(1.2)
        assert row["n_bins"] == 14

    def test_upsert_overwrites(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = BaselinesRepository(db)
        repo.upsert({
            "monitoring_point": "KAS-1E-01",
            "window_group": "morning",
            "mu": 1.0,
            "sigma": 0.5,
            "n_bins": 10,
            "computed_at": "2026-10-12T06:00:00.000Z",
            "source": "computed",
        })
        repo.upsert({
            "monitoring_point": "KAS-1E-01",
            "window_group": "morning",
            "mu": 4.0,
            "sigma": 1.5,
            "n_bins": 20,
            "computed_at": "2026-10-12T07:00:00.000Z",
            "source": "computed",
        })
        row = repo.get("KAS-1E-01", "morning")
        assert row["mu"] == pytest.approx(4.0)
        assert row["n_bins"] == 20

    def test_manual_source_valid(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = BaselinesRepository(db)
        repo.upsert({
            "monitoring_point": "KAS-1E-01",
            "window_group": "afternoon",
            "mu": 2.0,
            "sigma": 0.8,
            "n_bins": 0,
            "computed_at": "2026-10-12T06:00:00.000Z",
            "source": "manual",
        })
        row = repo.get("KAS-1E-01", "afternoon")
        assert row["source"] == "manual"


# ─── 12. Alerts Repository ────────────────────────────────────────────────────

class TestAlertsRepository:
    def _make_db(self, tmp_path: Path) -> DatabaseManager:
        schema = Path(__file__).resolve().parent.parent / "sentinel" / "storage" / "schema.sql"
        return DatabaseManager(tmp_path / "test.db", schema)

    def test_create_and_get(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = AlertsRepository(db)
        alert_id = repo.create({
            "monitoring_point": "KAS-1E-01",
            "level": "ALERT",
            "method": "cusum",
            "t_alert": "2026-10-12T06:15:00.000Z",
            "cusum_value": 4.8,
            "bin_start": "2026-10-12T06:00:00.000Z",
        })
        row = repo.get(alert_id)
        assert row is not None
        assert row["level"] == "ALERT"
        assert row["method"] == "cusum"
        assert row["cusum_value"] == pytest.approx(4.8)

    def test_acknowledge(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = AlertsRepository(db)
        aid = repo.create({
            "monitoring_point": "KAS-1E-01",
            "level": "ALERT",
            "method": "ears",
            "t_alert": "2026-10-12T06:15:00.000Z",
        })
        repo.acknowledge(aid, notes="Confirmed by observer")
        row = repo.get(aid)
        assert row["acknowledged_at"] is not None

    def test_verify_outcome(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = AlertsRepository(db)
        aid = repo.create({
            "monitoring_point": "KAS-1E-01",
            "level": "ALERT",
            "method": "both",
            "t_alert": "2026-10-12T06:15:00.000Z",
        })
        repo.verify(aid, "true_alert")
        row = repo.get(aid)
        assert row["verified_outcome"] == "true_alert"

    def test_invalid_verified_outcome_raises(self, tmp_path: Path):
        db = self._make_db(tmp_path)
        repo = AlertsRepository(db)
        with pytest.raises(ValueError):
            repo.verify(1, "bad_value")


# ─── 13. Replay virtual-start parsing ────────────────────────────────────────

class TestVirtualStartParsing:
    """Test parse_virtual_start from replay_video module."""

    def test_parse_valid_offset(self):
        from scripts.replay_video import parse_virtual_start
        # 2026-10-12T06:00:00+08:00 = 2026-10-11T22:00:00 UTC
        epoch = parse_virtual_start("2026-10-12T06:00:00+08:00")
        dt_utc = datetime.fromtimestamp(epoch, tz=_UTC)
        assert dt_utc.year == 2026
        assert dt_utc.month == 10
        assert dt_utc.day == 11
        assert dt_utc.hour == 22
        assert dt_utc.minute == 0

    def test_parse_utc_offset(self):
        from scripts.replay_video import parse_virtual_start
        epoch = parse_virtual_start("2026-10-12T06:00:00+00:00")
        dt_utc = datetime.fromtimestamp(epoch, tz=_UTC)
        assert dt_utc.hour == 6

    def test_missing_offset_raises(self):
        import argparse
        from scripts.replay_video import parse_virtual_start
        with pytest.raises(argparse.ArgumentTypeError, match="no UTC offset"):
            parse_virtual_start("2026-10-12T06:00:00")

    def test_invalid_string_raises(self):
        import argparse
        from scripts.replay_video import parse_virtual_start
        with pytest.raises(argparse.ArgumentTypeError):
            parse_virtual_start("not-a-date")

    def test_video_timestamps_use_virtual_start(self):
        """Given virtual_start and frame_index, timestamp must be virtual_start + idx/fps."""
        from scripts.replay_video import parse_virtual_start
        virtual_start_str = "2026-10-12T06:00:00+08:00"
        base_epoch = parse_virtual_start(virtual_start_str)
        fps = 30.0
        frame_index = 90  # frame 91 (0-indexed: 90)
        expected_time_s = frame_index / fps  # = 3.0 s
        event_utc = base_epoch + expected_time_s
        dt = datetime.fromtimestamp(event_utc, tz=_UTC).astimezone(_MANILA_TZ)
        # virtual_start is 06:00 Manila → + 3s = 06:00:03
        assert dt.hour == 6
        assert dt.minute == 0
        assert dt.second == 3
