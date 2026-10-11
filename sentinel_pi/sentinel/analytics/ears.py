"""EARS-C1 (Early Aberration Reporting System, method C1) for mosquito alerting.

EARS-C1 is a simple time-series aberration detection method that compares a
current observation to the recent historical mean + z * sd.

Algorithm:
    baseline window: the last lag_bins valid observations (excluding today).
    mu_lag  = mean(y_{t-lag} ... y_{t-1})
    sd_lag  = std(y_{t-lag} ... y_{t-1})  (population std, ddof=0; floored at min_sd)
    C1_t    = (y_t - mu_lag) / max(sd_lag, min_sd)
    alert   = C1_t > z_threshold + 1e-9  (strict threshold with epsilon, same as CUSUM)

Reference:
    Hutwagner et al. (2003) "The bioterrorism preparedness and response early
    aberration reporting system (EARS)". J Urban Health 80(2 Suppl 1):i89-96.

Design notes:
- Pure Python, no numpy required.
- Uses population standard deviation (ddof=0) with min_sd floor to prevent division-by-zero.
- Alerting uses STRICT inequality with epsilon (1e-9), identical to CUSUM semantics:
  a C1 value landing exactly on z_threshold does NOT alert.
- If fewer than lag_bins historical observations are available, EARS is not
  computed (returns None, False) — same behaviour as "baseline not ready".
- The function operates on raw numeric values; the caller filters for valid bins.
"""
from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

# Tolerance band to prevent floating-point noise at the boundary.
_EPSILON = 1e-9


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: Sequence[float]) -> float:
    """Population standard deviation (ddof=0)."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def ears_step(
    current: float,
    history: Sequence[float],
    z_threshold: float,
    min_sd: float,
    lag_bins: int,
) -> Tuple[Optional[float], bool]:
    """Compute one EARS-C1 step.

    Args:
        current:      The observed value for the current bin.
        history:      Ordered historical valid values (most recent last).
                      Only the last *lag_bins* values are used.
        z_threshold:  Alerting threshold (z-score; default 3.0).
        min_sd:       Floor for standard deviation (prevents division-by-zero).
        lag_bins:     Number of historical bins to use as baseline window.

    Returns:
        (c1_value, alerted):
            c1_value – The computed C1 statistic (None if not enough history).
            alerted  – True if c1_value > z_threshold + 1e-9 (strict).
    """
    window = list(history[-lag_bins:])
    if len(window) < lag_bins:
        # Not enough history; suppress alerting.
        return None, False

    mu = _mean(window)
    sd = max(_std(window), min_sd)
    c1 = (current - mu) / sd
    alerted = c1 > z_threshold + _EPSILON
    return c1, alerted


def ears_sequence(
    series: Sequence[float],
    z_threshold: float,
    min_sd: float,
    lag_bins: int,
) -> List[Tuple[Optional[float], bool]]:
    """Run EARS-C1 over an entire series.

    Each position uses the *lag_bins* values immediately before it as history.
    The first *lag_bins* positions cannot alert (insufficient history).

    Args:
        series:       Ordered sequence of valid bin values.
        z_threshold:  Alert threshold (z-score).
        min_sd:       Floor for standard deviation.
        lag_bins:     Baseline window width.

    Returns:
        List of (c1_value, alerted) tuples, one per element in *series*.
    """
    results: List[Tuple[Optional[float], bool]] = []
    for i, current in enumerate(series):
        history = list(series[:i])
        c1, alerted = ears_step(current, history, z_threshold, min_sd, lag_bins)
        results.append((c1, alerted))
    return results
