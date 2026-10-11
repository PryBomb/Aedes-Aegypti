"""CUSUM (Cumulative Sum) control chart for mosquito detection alerting.

Implements the one-sided upper CUSUM (Page's test) used to detect a shift
upward from the baseline mean.

Algorithm (computed in sigma units):
    z_t   = (x_t - mu) / sigma          -- standardised observation
    k     = k_param                      -- allowance / reference value (sigma units)
    S_t   = max(0, S_{t-1} + z_t - k)  -- one-sided cumulative sum (sigma units)
    alert = S_t > h + epsilon            -- strict threshold (sigma units)

Working in sigma units is equivalent to the natural-unit formulation
    S_t = max(0, S_{t-1} + x_t - mu - k*sigma)   vs   h*sigma
but avoids repeated division and makes the threshold h directly interpretable
as "number of baseline standard deviations".

Reference:
    Page (1954) "Continuous inspection schemes". Biometrika 41(1/2):100-115.

Design notes:
- sigma is floored at min_sigma to prevent division-by-zero on flat baselines.
- Alerting uses STRICT inequality with a small epsilon (1e-9) to prevent
  floating-point noise at the boundary from causing spurious alerts.
  A value landing exactly on h is NOT an alert.
- cusum_step is pure computation and does NOT reset carry state.
  The calling AlertEngine resets carry S_t to 0 only when an ALERT is actually
  declared (satisfying min_events and episode rules, if reset_after_alert is true).
- cusum_step returns (s_new, alerted):
    s_new   – computed cumulative sum for this step.
    alerted – True when s_new > h + 1e-9 (strict).
- Pure Python; no numpy required (compatible with Pi's constrained environment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

# Tolerance band to prevent floating-point noise at the boundary.
# S_t must exceed h by MORE than _EPSILON to alert.
_EPSILON = 1e-9


@dataclass
class CusumState:
    """Mutable running state for the CUSUM statistic."""
    s_t: float = 0.0            # current cumulative sum
    alerted: bool = False       # True if last step crossed threshold h


def cusum_step(
    x: float,
    mu: float,
    sigma: float,
    k: float,
    h: float,
    prev_s: float,
) -> tuple[float, bool]:
    """Compute one CUSUM step (pure statistic without carry reset).

    Args:
        x:       Observed bin value (raw_value or events_per_hour).
        mu:      Baseline mean (from baselines table).
        sigma:   Baseline std-dev (floored externally to min_sigma).
        k:       Allowance parameter (reference value, in sigma units).
        h:       Decision threshold (in sigma units).
        prev_s:  S_{t-1} carry value (pass 0.0 for the first step).

    Returns:
        (s_new, alerted):
            s_new   – computed cumulative sum for this step.
            alerted – True when s_new > h + 1e-9 (strict).
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0 (got {sigma}); floor with min_sigma before calling.")
    z = (x - mu) / sigma
    s_new = max(0.0, prev_s + z - k)
    alerted = s_new > h + _EPSILON
    return s_new, alerted


def cusum_sequence(
    series: Sequence[float],
    mu: float,
    sigma: float,
    k: float,
    h: float,
    reset_after_alert: bool = True,
) -> List[tuple[float, bool]]:
    """Run CUSUM over an entire series.

    Args:
        series:            Ordered sequence of bin values.
        mu:                Baseline mean.
        sigma:             Baseline std-dev (must be > 0).
        k, h:              CUSUM parameters (in sigma units).
        reset_after_alert: Reset carry S_t after each alert when advancing.

    Returns:
        List of (s_new, alerted) tuples, one per element in *series*.
    """
    results: List[tuple[float, bool]] = []
    s = 0.0
    for x in series:
        s_new, alerted = cusum_step(x, mu, sigma, k, h, s)
        results.append((s_new, alerted))
        s = 0.0 if (alerted and reset_after_alert) else s_new
    return results
