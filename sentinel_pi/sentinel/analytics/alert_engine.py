"""AlertEngine: pure alert decision logic for MosquitoSentinel.

Responsibilities:
- Per closed valid bin: evaluate CUSUM and/or EARS and return NORMAL / WATCH / ALERT.
- Manage CUSUM state (S_t) and EARS history per (monitoring_point, window_group).
  CUSUM carry and EARS history never mix morning and afternoon windows and are
  reset at the start of each monitoring window.
- EARS needs lag_bins (default 4) valid bins inside the current window before it can fire.
- Manage Alert Episodes:
  - An alerts row is written ONLY when an episode STARTS (fired=True).
  - While an episode is open, the level stays ALERT in every bin (no ALERT/WATCH flapping).
  - The episode closes after clear_bins (default 2) consecutive bins in which neither
    method is at/above its WATCH threshold.
  - After closing, a new alert starts a new episode and writes a new row (subject to cooldown_min).
- Enforce min_events and watch_min_events:
  - ALERT requires n_events >= alert.min_events (default 3).
  - WATCH requires n_events >= alert.watch_min_events (default 2).
  - Below watch_min_events, the level is NORMAL even if statistics are elevated.
- Do not reset CUSUM carry unless an ALERT is actually declared (and reset_after_alert is true).
  Bins suppressed by min_events do NOT reset the carry.
- Strict threshold semantics:
  - CUSUM: alert when S_t > h + 1e-9 (strict).
  - EARS: alert when C1 > z + 1e-9 (strict).
- Restart safety: rebuild_state(bins, alerts, window_group_now, baseline) reconstructs
  running state and episode lifecycle from stored rows.

This module does NOT touch GPIO, Flask, or any I/O beyond repository writes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence, Tuple

from sentinel.analytics.baseline import BaselineStats
from sentinel.analytics.cusum import cusum_step
from sentinel.analytics.ears import ears_step

logger = logging.getLogger(__name__)


# Asia/Manila = UTC+8 (no DST)
_MANILA_OFFSET = timedelta(hours=8)
_MANILA_TZ = timezone(_MANILA_OFFSET)


def _parse_dt(dt_val: Any) -> Optional[datetime]:
    if dt_val is None:
        return None
    if isinstance(dt_val, datetime):
        if dt_val.tzinfo is None:
            return dt_val.replace(tzinfo=timezone.utc)
        return dt_val
    if isinstance(dt_val, str):
        # Support ISO formats like '2026-10-12 06:00:00' or '2026-10-12T06:00:00+00:00'
        s = dt_val.replace(" ", "T")
        try:
            d = datetime.fromisoformat(s)
            if d.tzinfo is None:
                d = d.replace(tzinfo=timezone.utc)
            return d
        except ValueError:
            return None
    return None


@dataclass
class AlertDecision:
    """Result of evaluating one bin."""
    level: str                  # 'NORMAL' | 'WATCH' | 'ALERT'
    method: Optional[str]       # 'cusum' | 'ears' | 'both' | None
    cusum_value: Optional[float]
    ears_value: Optional[float]
    fired: bool                 # True = an alerts row should be written (episode start)
    episode_active: bool = False       # True = currently inside an open alert episode
    cooldown_suppressed: bool = False  # True = alert condition met, but suppressed by cooldown


@dataclass
class AlertEngineState:
    """Mutable running state for AlertEngine (one per (monitoring_point, window_group))."""
    s_t: float = 0.0                            # CUSUM statistic
    ears_history: List[float] = field(default_factory=list)  # recent valid raw_values within current window
    last_alert_time: Optional[datetime] = None  # bin timestamp (UTC) of last alert episode start (fired=True)
    last_active_bin: Optional[datetime] = None  # bin timestamp (UTC) of last bin in which episode was active
    episode_active: bool = False                # True if currently inside an alert episode
    clear_count: int = 0                        # consecutive bins below watch threshold during episode
    current_window_key: Optional[Tuple[Any, str]] = None  # (manila_date, window_group)
    last_bin_start: Optional[datetime] = None   # UTC bin_start of previous evaluated bin

    def reset_window(self) -> None:
        """Reset CUSUM carry and EARS history at the start of a monitoring window."""
        self.s_t = 0.0
        self.ears_history.clear()
        self.episode_active = False
        self.clear_count = 0
        self.last_active_bin = None
        self.last_alert_time = None



class AlertEngine:
    """Stateful alert evaluator for MosquitoSentinel."""

    def __init__(
        self,
        method: str,
        watch_fraction: float,
        cooldown_min: int,
        cusum_k: float,
        cusum_h: float,
        cusum_reset_after_alert: bool,
        ears_lag_bins: int,
        ears_z: float,
        ears_min_sd: float,
        min_sigma: float,
        min_events: int = 3,
        watch_min_events: int = 2,
        clear_bins: int = 2,
    ) -> None:
        if method not in ("cusum", "ears", "both"):
            raise ValueError(f"Invalid alert method: {method!r}")
        if min_events < 1:
            raise ValueError(f"min_events must be >= 1, got {min_events}")
        if watch_min_events < 1:
            raise ValueError(f"watch_min_events must be >= 1, got {watch_min_events}")
        if clear_bins < 1:
            raise ValueError(f"clear_bins must be >= 1, got {clear_bins}")
        if ears_lag_bins < 1:
            raise ValueError(f"ears_lag_bins must be >= 1, got {ears_lag_bins}")

        self.method = method
        self.watch_fraction = watch_fraction
        self.cooldown_min = cooldown_min
        self.cusum_k = cusum_k
        self.cusum_h = cusum_h
        self.cusum_reset_after_alert = cusum_reset_after_alert
        self.ears_lag_bins = ears_lag_bins
        self.ears_z = ears_z
        self.ears_min_sd = ears_min_sd
        self.min_sigma = min_sigma
        self.min_events = min_events
        self.watch_min_events = watch_min_events
        self.clear_bins = clear_bins

    @classmethod
    def from_config(cls, config: "object") -> "AlertEngine":
        """Construct from an AppConfig object."""
        return cls(
            method=config.alert.method,  # type: ignore[attr-defined]
            watch_fraction=config.alert.watch_fraction,
            cooldown_min=config.alert.cooldown_min,
            cusum_k=config.alert.cusum.k,
            cusum_h=config.alert.cusum.h,
            cusum_reset_after_alert=config.alert.cusum.reset_after_alert,
            ears_lag_bins=config.alert.ears.lag_bins,
            ears_z=config.alert.ears.z,
            ears_min_sd=config.alert.ears.min_sd,
            min_sigma=config.baseline.min_sigma,
            min_events=getattr(config.alert, "min_events", 3),
            watch_min_events=getattr(config.alert, "watch_min_events", 2),
            clear_bins=getattr(config.alert, "clear_bins", 2),
        )

    def evaluate(
        self,
        raw_value: float,
        bin_start: datetime,
        baseline: BaselineStats,
        state: AlertEngineState,
        n_events: int,
    ) -> AlertDecision:
        """Evaluate one closed valid bin.

        Args:
            raw_value:  The bin's raw detection index value (events/hour).
            bin_start:  UTC datetime of bin start (for cooldown and window boundary bookkeeping).
            baseline:   Baseline stats for this window_group.
            state:      Mutable running state for (monitoring_point, window_group).
            n_events:   REQUIRED: Number of confirmed target events in this bin.

        Returns:
            AlertDecision describing the outcome.
        """
        # ── Window boundary check ─────────────────────────────────────────────
        # A new Manila-local date or window_group resets the window state.
        # Outages inside the same window (same date and window_group) do NOT reset.
        b_utc = bin_start if bin_start.tzinfo is not None else bin_start.replace(tzinfo=timezone.utc)
        manila_date = b_utc.astimezone(_MANILA_TZ).date()
        window_key = (manila_date, baseline.window_group)

        if state.current_window_key is not None and state.current_window_key != window_key:
            logger.debug("New window boundary detected (%s -> %s) — resetting window state.", state.current_window_key, window_key)
            state.reset_window()
        state.current_window_key = window_key
        state.last_bin_start = b_utc

        # ── Baseline readiness check ──────────────────────────────────────────
        if not baseline.is_ready:
            logger.debug("Baseline not ready (%d bins) — suppressing alerting.", baseline.n_bins)
            state.ears_history.append(raw_value)
            return AlertDecision(
                level="NORMAL",
                method=None,
                cusum_value=None,
                ears_value=None,
                fired=False,
                episode_active=False,
                cooldown_suppressed=False,
            )

        sigma = max(baseline.sigma, self.min_sigma)

        # ── CUSUM step (pure calculation without carry reset) ──────────────────
        cusum_val: Optional[float] = None
        cusum_alert = False
        if self.method in ("cusum", "both"):
            s_new, cusum_alert = cusum_step(
                x=raw_value,
                mu=baseline.mu,
                sigma=sigma,
                k=self.cusum_k,
                h=self.cusum_h,
                prev_s=state.s_t,
            )
            cusum_val = s_new
        else:
            s_new = state.s_t

        # ── EARS step (strict > z + 1e-9) ──────────────────────────────────────
        ears_val: Optional[float] = None
        ears_alert = False
        if self.method in ("ears", "both"):
            c1, ears_alert = ears_step(
                current=raw_value,
                history=state.ears_history,
                z_threshold=self.ears_z,
                min_sd=self.ears_min_sd,
                lag_bins=self.ears_lag_bins,
            )
            ears_val = c1

        # Append to window EARS history after evaluating this bin
        state.ears_history.append(raw_value)

        # ── Threshold triggers ────────────────────────────────────────────────
        alert_condition_met = (
            (self.method == "cusum" and cusum_alert)
            or (self.method == "ears" and ears_alert)
            or (self.method == "both" and (cusum_alert or ears_alert))
        )
        events_qualify_alert = (n_events >= self.min_events)
        alert_trigger = (alert_condition_met and events_qualify_alert)

        # Watch thresholds
        watch_cusum = (cusum_val is not None and cusum_val >= self.watch_fraction * self.cusum_h)
        watch_ears = (ears_val is not None and ears_val >= self.watch_fraction * self.ears_z)
        watch_threshold_met = (watch_cusum or watch_ears or alert_condition_met)
        events_qualify_watch = (n_events >= self.watch_min_events)
        is_clear_bin = not (watch_threshold_met and events_qualify_watch)

        # ── Episode state machine ─────────────────────────────────────────────
        if state.episode_active:
            # Episode is currently open: level stays ALERT in every bin (no flapping)
            state.last_active_bin = bin_start
            method_str = self._fired_method(cusum_alert, ears_alert)

            # CUSUM carry reset: only reset when ALERT condition is declared
            if cusum_alert and events_qualify_alert and self.cusum_reset_after_alert:
                state.s_t = 0.0
            else:
                state.s_t = s_new

            # Track clearing bins to close episode
            if is_clear_bin:
                state.clear_count += 1
                if state.clear_count >= self.clear_bins:
                    # Episode closes at the end of this bin
                    state.episode_active = False
                    state.clear_count = 0
            else:
                state.clear_count = 0

            return AlertDecision(
                level="ALERT",
                method=method_str,
                cusum_value=cusum_val,
                ears_value=ears_val,
                fired=False,  # Row written ONLY when episode starts
                episode_active=True,
                cooldown_suppressed=False,
            )

        else:
            # Episode is not active entering this bin
            if alert_trigger:
                # Check cooldown: measured from the last bin in which the episode was active
                in_cooldown = False
                if self.cooldown_min > 0 and state.last_active_bin is not None:
                    elapsed = (bin_start - state.last_active_bin).total_seconds() / 60.0
                    if elapsed < self.cooldown_min:
                        in_cooldown = True

                if in_cooldown:
                    # Alert condition recurs within cooldown_min after an episode closed:
                    # CONTINUES that episode: level ALERT, episode_active=True, fired=False, no new row.
                    state.episode_active = True
                    state.clear_count = 0
                    state.last_active_bin = bin_start
                    method_str = self._fired_method(cusum_alert, ears_alert)

                    if self.cusum_reset_after_alert:
                        state.s_t = 0.0
                    else:
                        state.s_t = s_new

                    return AlertDecision(
                        level="ALERT",
                        method=method_str,
                        cusum_value=cusum_val,
                        ears_value=ears_val,
                        fired=False,
                        episode_active=True,
                        cooldown_suppressed=True,
                    )
                else:
                    # START NEW EPISODE! Write alert row
                    state.episode_active = True
                    state.clear_count = 0
                    state.last_alert_time = bin_start
                    state.last_active_bin = bin_start
                    method_str = self._fired_method(cusum_alert, ears_alert)

                    if self.cusum_reset_after_alert:
                        state.s_t = 0.0
                    else:
                        state.s_t = s_new

                    return AlertDecision(
                        level="ALERT",
                        method=method_str,
                        cusum_value=cusum_val,
                        ears_value=ears_val,
                        fired=True,
                        episode_active=True,
                        cooldown_suppressed=False,
                    )

            else:
                # No alert condition met: episode remains inactive.
                # DO NOT reset CUSUM carry!
                # Even if CUSUM crossed h, if suppressed by min_events, carry is preserved.
                state.s_t = s_new

                if watch_threshold_met and events_qualify_watch:
                    level = "WATCH"
                else:
                    level = "NORMAL"

                return AlertDecision(
                    level=level,
                    method=None,
                    cusum_value=cusum_val,
                    ears_value=ears_val,
                    fired=False,
                    episode_active=False,
                    cooldown_suppressed=False,
                )

    def rebuild_state(
        self,
        bins: Sequence[Any],
        alerts: Sequence[Any],
        window_group_now: str,
        baseline: Optional[BaselineStats] = None,
    ) -> AlertEngineState:
        """Reconstruct AlertEngineState from stored detection_index and alerts rows.

        Ensures restart safety: live evaluation and rebuild_state produce identical
        internal state and an identical next decision.

        Args:
            bins:              Stored detection_index rows (dicts, Rows, or objects).
            alerts:            Stored alerts rows for this monitoring point.
            window_group_now:  Window group for the state being rebuilt ('morning' | 'afternoon').
            baseline:          Baseline stats for this window group (optional).

        Returns:
            Reconstructed AlertEngineState.
        """
        state = AlertEngineState()

        # Helper to extract dict / attribute / mapping items (safe for sqlite3.Row)
        def _get(obj: Any, key: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(key, default)
            try:
                return obj[key]
            except (KeyError, TypeError, IndexError):
                pass
            if hasattr(obj, key):
                return getattr(obj, key)
            return default

        # 1. Parse valid alert times from alerts table (column is t_alert; fallback created_at)
        alert_times = [_parse_dt(_get(a, "t_alert") or _get(a, "created_at")) for a in alerts]
        valid_ats = [at for at in alert_times if at is not None]

        # 2. Filter bins for the matching window_group and sort by bin_start
        filtered: List[Any] = []
        for b in bins:
            bs = _parse_dt(_get(b, "bin_start"))
            if bs is None:
                continue
            wg = _get(b, "window_group")
            if wg is None:
                # detection_index table does not store window_group; derive from Manila local time
                m_hour = bs.astimezone(_MANILA_TZ).hour
                wg = "morning" if m_hour < 12 else "afternoon"
            if wg == window_group_now:
                filtered.append((bs, b))

        filtered.sort(key=lambda item: item[0])


        if not filtered:
            if valid_ats:
                state.last_alert_time = max(valid_ats)
                state.last_active_bin = max(valid_ats)
            return state

        # Pre-seed last_alert_time and last_active_bin with alerts that occurred
        # strictly BEFORE the first bin being replayed
        first_bin_start = filtered[0][0]
        prior_alerts = [at for at in valid_ats if at < first_bin_start]
        if prior_alerts:
            state.last_alert_time = max(prior_alerts)
            state.last_active_bin = max(prior_alerts)

        # 3. Construct a fallback baseline if none was passed
        if baseline is None:
            valid_raws = [
                float(_get(b, "raw_value", 0.0))
                for _, b in filtered
                if int(_get(b, "valid", 1)) == 1
            ]
            mu = sum(valid_raws) / len(valid_raws) if valid_raws else 0.0
            sigma = self.min_sigma
            baseline = BaselineStats(
                monitoring_point="REBUILD",
                window_group=window_group_now,
                mu=mu,
                sigma=sigma,
                n_bins=len(valid_raws),
                is_ready=True,
            )

        # 4. Replay the bins through evaluate to reconstruct the state
        for bs, b in filtered:
            valid = int(_get(b, "valid", 1))
            raw_val = float(_get(b, "raw_value", 0.0))
            n_ev = int(_get(b, "n_events", 0))

            if valid == 0:
                # Still check window boundary during downtime
                b_utc = bs if bs.tzinfo is not None else bs.replace(tzinfo=timezone.utc)
                m_date = b_utc.astimezone(_MANILA_TZ).date()
                w_key = (m_date, baseline.window_group)
                if state.current_window_key is not None and state.current_window_key != w_key:
                    state.reset_window()
                state.current_window_key = w_key
                state.last_bin_start = b_utc
                continue

            self.evaluate(
                raw_value=raw_val,
                bin_start=bs,
                baseline=baseline,
                state=state,
                n_events=n_ev,
            )

        if valid_ats:
            max_at = max(valid_ats)
            if state.last_alert_time is None or max_at > state.last_alert_time:
                state.last_alert_time = max_at
            if state.last_active_bin is None or max_at > state.last_active_bin:
                state.last_active_bin = max_at

        return state


    def _fired_method(self, cusum_alert: bool, ears_alert: bool) -> str:
        if cusum_alert and ears_alert:
            return "both"
        elif cusum_alert:
            return "cusum"
        else:
            return "ears"
