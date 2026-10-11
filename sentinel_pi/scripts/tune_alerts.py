"""tune_alerts.py – Monte Carlo simulation to tune bin_minutes and min_events.

Simulates alerting under Phase 4 specifications across:
- Baselines: 1.0 and 2.0 events/hour
- Bin durations: 15, 30, and 60 minutes
- min_events: 2 and 3

Evaluates:
(a) No surge: false-alert episodes per 3-hour monitoring window.
(b) Surges of 2x, 3x, and 5x baseline starting at a random time inside the window:
    detection probability and median minutes from surge start to the alert row.

Usage:
    python scripts/tune_alerts.py [--days 500] [--seed 1] [--data-dir C:/sentinel_data]
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sentinel.analytics.baseline import BaselineStats
from sentinel.analytics.alert_engine import AlertEngine, AlertEngineState

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def sample_poisson(lam: float, rng: random.Random) -> int:
    """Draw a single Poisson sample using Knuth's algorithm."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while p > L:
        k += 1
        p *= rng.random()
    return k - 1


def simulate_window_no_surge(
    engine: AlertEngine,
    baseline: BaselineStats,
    bin_minutes: int,
    mu: float,
    window_mins: int,
    rng: random.Random,
    start_dt: datetime,
) -> int:
    """Simulate one 3-hour window with baseline Poisson events. Returns number of episodes started."""
    state = AlertEngineState()
    n_bins = window_mins // bin_minutes
    bin_hours = bin_minutes / 60.0
    lam = mu * bin_hours
    episodes = 0

    for i in range(n_bins):
        bt = start_dt + timedelta(minutes=i * bin_minutes)
        cnt = sample_poisson(lam, rng)
        raw_val = cnt / bin_hours
        dec = engine.evaluate(raw_val, bt, baseline, state, cnt)
        if dec.fired:
            episodes += 1

    return episodes


def simulate_window_surge(
    engine: AlertEngine,
    baseline: BaselineStats,
    bin_minutes: int,
    mu: float,
    multiplier: float,
    window_mins: int,
    rng: random.Random,
    start_dt: datetime,
) -> Optional[float]:
    """Simulate one 3-hour window with a surge starting at a random time.
    
    Returns:
        Detection latency in minutes from surge start to alert row write, or None if not detected.
    """
    state = AlertEngineState()
    n_bins = window_mins // bin_minutes
    bin_hours = bin_minutes / 60.0

    # Random surge start time between 0 and 120 minutes (allowing >= 60 min surge)
    t_surge_min = rng.uniform(0.0, 120.0)
    detected_delay: Optional[float] = None

    for i in range(n_bins):
        b_start_min = i * bin_minutes
        b_end_min = (i + 1) * bin_minutes
        bt = start_dt + timedelta(minutes=b_start_min)

        # Compute pre-surge and post-surge duration in hours for this bin
        pre_min = max(0.0, min(b_end_min, t_surge_min) - b_start_min)
        post_min = max(0.0, b_end_min - max(b_start_min, t_surge_min))

        lam = (pre_min / 60.0) * mu + (post_min / 60.0) * (mu * multiplier)
        cnt = sample_poisson(lam, rng)
        raw_val = cnt / bin_hours

        dec = engine.evaluate(raw_val, bt, baseline, state, cnt)

        # Did alert fire on or after surge start?
        if dec.fired and b_end_min > t_surge_min and detected_delay is None:
            # Alert row written at the close of this bin (b_end_min)
            detected_delay = b_end_min - t_surge_min

    return detected_delay


def run_tuning(days: int = 500, seed: int = 1, data_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    window_mins = 180  # 3 hours
    total_windows = days * 2  # 2 windows per day (morning & afternoon)

    baselines_mu = [1.0, 2.0]
    bin_options = [15, 30, 60]
    min_event_options = [2, 3]
    multipliers = [2.0, 3.0, 5.0]

    results: List[Dict[str, Any]] = []
    base_start = datetime(2026, 10, 12, 6, 0, tzinfo=timezone.utc)

    for mu in baselines_mu:
        # Standard deviation floored at 0.5
        sigma = max(math.sqrt(mu), 0.5)

        for bm in bin_options:
            for me in min_event_options:
                watch_me = min(2, me)
                baseline_stats = BaselineStats(
                    monitoring_point="TUNE",
                    window_group="morning",
                    mu=mu,
                    sigma=sigma,
                    n_bins=100,
                    is_ready=True,
                )

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
                    watch_min_events=watch_me,
                    clear_bins=2,
                )

                # 1. No surge simulation (false alert rate per window)
                total_false_episodes = 0
                for w in range(total_windows):
                    total_false_episodes += simulate_window_no_surge(
                        engine=engine,
                        baseline=baseline_stats,
                        bin_minutes=bm,
                        mu=mu,
                        window_mins=window_mins,
                        rng=rng,
                        start_dt=base_start + timedelta(days=w),
                    )
                false_rate_per_win = total_false_episodes / total_windows

                # 2. Surge simulations (2x, 3x, 5x)
                row_res: Dict[str, Any] = {
                    "mu": mu,
                    "bin_minutes": bm,
                    "min_events": me,
                    "false_rate_per_win": false_rate_per_win,
                }

                for m in multipliers:
                    delays: List[float] = []
                    for w in range(total_windows):
                        d = simulate_window_surge(
                            engine=engine,
                            baseline=baseline_stats,
                            bin_minutes=bm,
                            mu=mu,
                            multiplier=m,
                            window_mins=window_mins,
                            rng=rng,
                            start_dt=base_start + timedelta(days=w),
                        )
                        if d is not None:
                            delays.append(d)

                    det_prob = len(delays) / total_windows
                    med_delay = statistics.median(delays) if delays else None
                    m_label = f"{int(m)}x"
                    row_res[f"{m_label}_prob"] = det_prob
                    row_res[f"{m_label}_delay"] = med_delay

                results.append(row_res)

    # Print Table
    header = (
        f"{'Mu (ev/h)':<10} | {'Bin (m)':<7} | {'MinEv':<5} | {'False Ep/Win':<12} | "
        f"{'2x Det':<7} | {'2x Delay':<9} | {'3x Det':<7} | {'3x Delay':<9} | {'5x Det':<7} | {'5x Delay':<9}"
    )
    print("\n" + "=" * 105)
    print(" MosquitoSentinel Alert Tuning Simulation (500 Days / 1000 Windows per Config)")
    print("=" * 105)
    print(header)
    print("-" * 105)

    for r in results:
        mu_str = f"{r['mu']:.1f}"
        bm_str = f"{r['bin_minutes']}m"
        me_str = f"{r['min_events']}"
        fp_str = f"{r['false_rate_per_win']:.4f}"
        
        d2_str = f"{r['2x_prob']*100:.1f}%"
        del2_str = f"{r['2x_delay']:.1f}m" if r['2x_delay'] is not None else "N/A"
        
        d3_str = f"{r['3x_prob']*100:.1f}%"
        del3_str = f"{r['3x_delay']:.1f}m" if r['3x_delay'] is not None else "N/A"
        
        d5_str = f"{r['5x_prob']*100:.1f}%"
        del5_str = f"{r['5x_delay']:.1f}m" if r['5x_delay'] is not None else "N/A"

        print(
            f"{mu_str:<10} | {bm_str:<7} | {me_str:<5} | {fp_str:<12} | "
            f"{d2_str:<7} | {del2_str:<9} | {d3_str:<7} | {del3_str:<9} | {d5_str:<7} | {del5_str:<9}"
        )
    print("=" * 105 + "\n")

    # Save CSV
    if data_dir is not None:
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / "tune_alerts.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "baseline_mu", "bin_minutes", "min_events", "false_episodes_per_window",
                "2x_det_prob", "2x_median_delay_min",
                "3x_det_prob", "3x_median_delay_min",
                "5x_det_prob", "5x_median_delay_min",
            ])
            for r in results:
                writer.writerow([
                    r["mu"], r["bin_minutes"], r["min_events"], f"{r['false_rate_per_win']:.4f}",
                    f"{r['2x_prob']:.4f}", f"{r['2x_delay']:.1f}" if r['2x_delay'] is not None else "",
                    f"{r['3x_prob']:.4f}", f"{r['3x_delay']:.1f}" if r['3x_delay'] is not None else "",
                    f"{r['5x_prob']:.4f}", f"{r['5x_delay']:.1f}" if r['5x_delay'] is not None else "",
                ])
        print(f"[tune] Saved tuning results CSV to {csv_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Monte Carlo alert tuning simulation.")
    parser.add_argument("--days", type=int, default=500, help="Number of simulated days per config.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for reproducibility.")
    parser.add_argument("--data-dir", type=str, default="C:/sentinel_data", help="Directory to save CSV.")
    args = parser.parse_args()

    run_tuning(days=args.days, seed=args.seed, data_dir=Path(args.data_dir))


if __name__ == "__main__":
    main()
