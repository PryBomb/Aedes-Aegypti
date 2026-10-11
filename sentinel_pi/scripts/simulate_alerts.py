"""simulate_alerts.py – Synthetic bin series to demonstrate CUSUM + EARS-C1 alerting.

Simulates alerting under Phase 4 specifications:
  (a) flat    – baseline Poisson counts; values remain within baseline mean, no alerts.
  (b) gradual – slowly rising whole-number counts; CUSUM detects trend, enters ONE episode.
  (c) spike   – sudden spike in one bin; EARS detects, starts episode, closes after clear_bins.
  (d) downtime gap – flat series with a gap of low-coverage (invalid=0) bins, then rise.
  (e) sparse  – low baseline (mu=0.3), isolated 1-event bins (suppressed by watch_min_events=2),
                then 4-event bin (alerts) and episode lifecycle.
  (f) low-rate surge comparison – baseline mu=1.0, sigma=0.5 with realistic surge across
                bin_minutes / min_events combinations (15/3, 30/3, 60/3, 60/2).

In all series, whole-number counts are generated per bin and value = n_events / bin_hours.
Round()-derived counts have been completely removed.
Per series, the output prints whether an alert fired and, if not, why.

If matplotlib is installed, saves a PNG chart of series (a)-(e).

Usage:
    python scripts/simulate_alerts.py [--data-dir C:/sentinel_data]
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure project root on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel.analytics.baseline import BaselineStats
from sentinel.analytics.alert_engine import AlertEngine, AlertEngineState

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# ─── Simulation parameters (match ARCHITECTURE.md defaults) ──────────────────

MU = 2.0          # Baseline mean (events per hour)
SIGMA = 0.8       # Baseline std-dev
CUSUM_K = 0.5
CUSUM_H = 4.0
RESET_AFTER_ALERT = True
EARS_LAG = 3      # provisional default 3 per Phase 4 review 3
EARS_Z = 3.0
EARS_MIN_SD = 0.5
MIN_SIGMA = 0.5
WATCH_FRACTION = 0.6
COOLDOWN_MIN = 30
MIN_EVENTS = 3
WATCH_MIN_EVENTS = 2
CLEAR_BINS = 2
BIN_MINUTES = 30  # provisional default 30 per Phase 4 review 3
BIN_HOURS = BIN_MINUTES / 60.0  # 0.5 hr


def poisson_counts(rates: List[float], bin_hours: float, seed: int = 42) -> List[int]:
    """Generate whole-number counts per bin using seeded Poisson sampling."""
    rng = random.Random(seed)
    counts: List[int] = []
    for r in rates:
        lam = r * bin_hours
        if lam <= 0:
            counts.append(0)
            continue
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= rng.random()
        counts.append(k - 1)
    return counts


def run_simulation(
    name: str,
    events: List[int],
    valid_flags: Optional[List[int]] = None,
    mu: float = MU,
    sigma: float = SIGMA,
    bin_minutes: int = BIN_MINUTES,
    min_events: int = MIN_EVENTS,
    watch_min_events: int = WATCH_MIN_EVENTS,
    clear_bins: int = CLEAR_BINS,
    ears_lag_bins: int = EARS_LAG,
) -> Tuple[bool, Optional[int], Optional[str], str]:
    """Run AlertEngine over a whole-number count series and print formatted per-bin table.

    Returns:
        (alert_fired, first_alert_bin, first_alert_method, summary_reason)
    """
    bin_hours = bin_minutes / 60.0
    if valid_flags is None:
        valid_flags = [1] * len(events)
    raw_values = [ev / bin_hours for ev in events]

    print(f"\n{'=' * 80}")
    print(f" Series ({name})")
    print(f"  Baseline:   mu={mu:.2f}, sigma={sigma:.2f}")
    print(f"  CUSUM:      k={CUSUM_K}, h={CUSUM_H}, reset_after_alert={RESET_AFTER_ALERT} (sigma units)")
    print(f"  EARS:       lag={ears_lag_bins}, z={EARS_Z}, min_sd={EARS_MIN_SD} (strict > z + 1e-9)")
    print(f"  Alerting:   cooldown={COOLDOWN_MIN}min, min_events={min_events}, watch_min_events={watch_min_events}, clear_bins={clear_bins}")
    print(f"{'=' * 80}")
    header = f"{'Bin':>4} | {'Value':>7} | {'Events':>6} | {'Valid':>5} | {'S_t':>8} | {'EARS C1':>8} | {'Level':<8} | Note"
    print(header)
    print("-" * 80)

    baseline = BaselineStats(
        monitoring_point="SIM-1",
        window_group="morning",
        mu=mu,
        sigma=sigma,
        n_bins=100,
        is_ready=True,
    )

    engine = AlertEngine(
        method="both",
        watch_fraction=WATCH_FRACTION,
        cooldown_min=COOLDOWN_MIN,
        cusum_k=CUSUM_K,
        cusum_h=CUSUM_H,
        cusum_reset_after_alert=RESET_AFTER_ALERT,
        ears_lag_bins=ears_lag_bins,
        ears_z=EARS_Z,
        ears_min_sd=EARS_MIN_SD,
        min_sigma=MIN_SIGMA,
        min_events=min_events,
        watch_min_events=watch_min_events,
        clear_bins=clear_bins,
    )

    state = AlertEngineState()
    # 06:00 Manila (UTC+8) = 2026-10-11 22:00 UTC
    start_time = datetime(2026, 10, 11, 22, 0, tzinfo=timezone.utc)
    first_alert_method: Optional[str] = None
    first_alert_bin: Optional[int] = None
    alert_fired = False
    episodes_started = 0
    suppression_reasons: List[str] = []

    for i, (x, valid, ev) in enumerate(zip(raw_values, valid_flags, events)):
        bin_time = start_time + timedelta(minutes=bin_minutes * i)

        if not valid:
            s_display = state.s_t
            c1_str = "   N/A  "
            level = "NORMAL"
            note = "(excluded)"
        else:
            decision = engine.evaluate(
                raw_value=x,
                bin_start=bin_time,
                baseline=baseline,
                state=state,
                n_events=ev,
            )
            level = decision.level
            s_display = decision.cusum_value if decision.cusum_value is not None else 0.0
            c1_str = f"{decision.ears_value:>8.3f}" if decision.ears_value is not None else "   N/A  "

            note = ""
            if decision.fired:
                alert_fired = True
                episodes_started += 1
                fired_by = decision.method or "alert"
                note = f"<< {fired_by.upper()} ALERT (Episode {episodes_started} started, row written)"
                if first_alert_method is None:
                    first_alert_method = fired_by
                    first_alert_bin = i + 1
            elif decision.episode_active:
                note = f"(in Episode {episodes_started}: level ALERT)"
            elif decision.cooldown_suppressed:
                fired_by = decision.method or "alert"
                note = f"<< {fired_by.upper()} ALERT (suppressed by cooldown)"
                suppression_reasons.append("cooldown")
            elif ev < min_events and (
                s_display > CUSUM_H + 1e-9
                or (decision.ears_value is not None and decision.ears_value > EARS_Z + 1e-9)
            ):
                note = f"(suppressed: n_events={ev} < min_events={min_events})"
                suppression_reasons.append("below min_events")
            elif ev < watch_min_events and (
                s_display >= WATCH_FRACTION * CUSUM_H
                or (decision.ears_value is not None and decision.ears_value >= WATCH_FRACTION * EARS_Z)
            ):
                note = f"(below watch_min_events={watch_min_events})"

        print(f"{i+1:>4} | {x:>7.2f} | {ev:>6} | {valid:>5} | {s_display:>8.3f} | {c1_str} | {level:<8} | {note}")

    print("-" * 80)
    if alert_fired:
        summary = f"ALERT fired at bin {first_alert_bin} via {first_alert_method.upper()} ({episodes_started} episode started)"
        print(f"  >> {summary}")
    else:
        if suppression_reasons:
            reason = suppression_reasons[0]
            summary = f"No alert fired: {reason}"
        else:
            summary = "No alert fired: values remain within baseline mean"
        print(f"  >> {summary}")
    print()
    return alert_fired, first_alert_bin, first_alert_method, summary


def simulate_series_f() -> None:
    """Run Series (f): low-rate baseline mu=1.0, sigma=0.5 deterministic counts.

    Surge scenario over 3 hours (180 min):
      Hour 1: 1 count
      Hour 2: 5 counts
      Hour 3: 6 counts
    Evaluates 4 configurations: (15/3, 30/3, 60/3, 60/2).
    """
    print(f"\n{'=' * 80}")
    print(" Series (f – Low-Rate Baseline Surge Comparison: mu=1.0, sigma=0.5)")
    print(" Deterministic 3-hour surge: 1 event in hr 1, 5 events in hr 2, 6 events in hr 3")
    print(f"{'=' * 80}")

    configs = [
        ("15 min / min_events 3", 15, 3, [0, 1, 0, 0, 1, 2, 1, 1, 2, 1, 2, 1]),
        ("30 min / min_events 3", 30, 3, [1, 0, 2, 3, 3, 3]),
        ("60 min / min_events 3", 60, 3, [1, 5, 6]),
        ("60 min / min_events 2", 60, 2, [1, 5, 6]),
    ]

    results_table = []
    for label, bm, me, counts in configs:
        bh = bm / 60.0
        raw_values = [cnt / bh for cnt in counts]

        baseline = BaselineStats("SIM-F", "morning", mu=1.0, sigma=0.5, n_bins=50, is_ready=True)
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
            min_events=me,
            watch_min_events=min(2, me),
            clear_bins=2,
        )
        state = AlertEngineState()
        start = datetime(2026, 10, 11, 22, 0, tzinfo=timezone.utc)

        fired_bin = None
        fired_method = None
        for i, (cnt, val) in enumerate(zip(counts, raw_values)):
            t = start + timedelta(minutes=bm * i)
            dec = engine.evaluate(val, t, baseline, state, cnt)
            if dec.fired and fired_bin is None:
                fired_bin = i + 1
                fired_method = dec.method

        if fired_bin is not None:
            status = f"YES (Bin {fired_bin}, {fired_bin * bm}m)"
            note = f"Fired via {fired_method.upper()}"
        else:
            status = "NO"
            note = f"Suppressed (max count/bin = {max(counts)} < {me})"

        results_table.append((label, len(counts), me, status, note))

    tbl_header = f"{'Configuration':<24} | {'Bins':>4} | {'Min Ev':>6} | {'Fired?':<18} | Outcome"
    print(tbl_header)
    print("-" * 80)
    for cfg_lbl, n_b, me, stat, note in results_table:
        print(f"{cfg_lbl:<24} | {n_b:>4} | {me:>6} | {stat:<18} | {note}")
    print("-" * 80)
    print("  >> Note: 15min/3 suppresses low-rate surges because count/bin is too sparse;")
    print("           30min and 60min bins aggregate sufficient counts to detect the surge.\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Simulate CUSUM+EARS alerting with synthetic series.")
    p.add_argument(
        "--data-dir",
        default="C:/sentinel_data",
        help="Directory to save optional matplotlib PNG chart.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ── (a) Flat series – no alert expected ────────────────────────────────────
    # Baseline mu=2.0. Whole-number seeded Poisson counts from rate MU.
    flat_counts = poisson_counts([MU] * 20, BIN_HOURS, seed=42)
    run_simulation("a – Flat (no alert)", flat_counts)

    # ── (b) Gradual rise – CUSUM detects, enters ONE episode ───────────────────
    # Rising whole-number counts from baseline (1) up to 5 events per 30m bin.
    # At bin 12..14: CUSUM accumulates, n_events >= 3 -> Episode 1 starts.
    # Subsequent bins remain in Episode 1 with level=ALERT; no new rows written.
    gradual_counts = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]
    run_simulation("b – Gradual rise (CUSUM episode lifecycle)", gradual_counts)

    # ── (c) Sudden spike – EARS detects, starts episode, clears after 2 bins ───
    # 10 bins at baseline (1 event). Bin 11 spikes to 6 events.
    # EARS fires on bin 11 -> Episode 1 starts. Bins 12..20 return to 1 event; closes after 2 bins.
    spike_counts = [1] * 10 + [6] + [1] * 9
    run_simulation("c – Sudden spike (EARS detects & clear_bins closes episode)", spike_counts)

    # ── (d) Downtime gap then rise – invalid bins excluded ─────────────────────
    # 8 valid baseline bins, 4 downtime bins (valid=0, excluded), then 8 rising bins.
    gap_counts = [1] * 8 + [0] * 4 + [1, 1, 2, 2, 2, 2, 2, 2]
    valid_flags_d = [1] * 8 + [0] * 4 + [1] * 8
    run_simulation("d – Downtime gap then rise (invalid bins excluded)", gap_counts, valid_flags_d)

    # ── (e) Sparse series – watch_min_events & min_events ──────────────────────
    # mu=0.3, sigma=0.5. Isolated 1-event bins (suppressed: 1 < watch_min_events=2, level NORMAL).
    # Bin 12 has 4 events >= 3 -> Episode 1 starts!
    # Bin 13 has 4 events -> inside Episode 1 (level ALERT, no new row).
    # Bins 14-15 drop to 0 -> episode closes.
    events_e = [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 4, 4, 0, 0]
    run_simulation(
        "e – Sparse (watch_min_events & min_events suppression)",
        events_e,
        mu=0.3,
        sigma=0.5,
    )

    # ── (f) Low-rate baseline surge comparison ─────────────────────────────────
    simulate_series_f()

    # ── Optional matplotlib chart for (a)-(e) ──────────────────────────────────
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore

        fig, axes = plt.subplots(3, 2, figsize=(14, 11))
        fig.suptitle("Simulate Alerts – Synthetic Series (Phase 4 Review Round 3)", fontsize=13)

        series_data = [
            ("a – Flat", [c / BIN_HOURS for c in flat_counts], [1]*20, MU, SIGMA),
            ("b – Gradual rise", [c / BIN_HOURS for c in gradual_counts], [1]*20, MU, SIGMA),
            ("c – Sudden spike", [c / BIN_HOURS for c in spike_counts], [1]*20, MU, SIGMA),
            ("d – Downtime gap", [c / BIN_HOURS for c in gap_counts], valid_flags_d, MU, SIGMA),
            ("e – Sparse", [c / BIN_HOURS for c in events_e], [1]*len(events_e), 0.3, 0.5),
        ]

        axes_flat = axes.flatten()
        for idx, (title, s_vals, vflags, m, s) in enumerate(series_data):
            ax = axes_flat[idx]
            xs = list(range(1, len(s_vals) + 1))
            valid_xs = [x for x, v in zip(xs, vflags) if v]
            valid_ys = [y for y, v in zip(s_vals, vflags) if v]
            invalid_xs = [x for x, v in zip(xs, vflags) if not v]
            invalid_ys = [y for y, v in zip(s_vals, vflags) if not v]

            ax.plot(valid_xs, valid_ys, "b-o", ms=4, label="raw_value (valid)")
            if invalid_xs:
                ax.plot(invalid_xs, invalid_ys, "rx", ms=6, label="excluded bin")
            ax.axhline(m, color="green", linestyle="--", linewidth=0.8, label=f"mu={m:.1f}")
            ax.axhline(m + EARS_Z * s, color="red", linestyle=":", linewidth=0.8,
                       label=f"alert ({m + EARS_Z*s:.1f})")
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("Bin")
            ax.set_ylabel("Events / hour")
            ax.legend(fontsize=7)

        fig.delaxes(axes_flat[5])
        plt.tight_layout()
        data_dir.mkdir(parents=True, exist_ok=True)
        out_path = data_dir / "simulate_alerts.png"
        plt.savefig(out_path, dpi=120)
        print(f"[chart] Saved to {out_path}")
    except ImportError:
        print("[chart] matplotlib not installed – skipping PNG.")


if __name__ == "__main__":
    main()
