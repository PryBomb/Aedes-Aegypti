"""Baseline statistics for MosquitoSentinel detection index.

The baseline characterises the "normal" level of mosquito activity for each
monitoring window (morning / afternoon) using historical valid bins.

window_group assignment:
    The first configured window  → 'morning'
    The second configured window → 'afternoon'
    (Additional windows are mapped to 'afternoon' with a warning.)

Baseline is considered "ready" only when n_bins >= min_bins.
mu is the mean of valid raw_values; sigma is the std-dev floored at min_sigma.

Pure functions are defined here; persistence is done via BaselinesRepository
in sentinel/storage/repositories.py.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class BaselineStats:
    """Computed baseline statistics for one monitoring_point × window_group."""
    monitoring_point: str
    window_group: str        # 'morning' | 'afternoon'
    mu: float
    sigma: float             # floored at min_sigma
    n_bins: int
    is_ready: bool           # True when n_bins >= min_bins


def compute_baseline(
    values: Sequence[float],
    monitoring_point: str,
    window_group: str,
    min_bins: int,
    min_sigma: float,
) -> BaselineStats:
    """Compute baseline statistics from a sequence of valid raw_values.

    Args:
        values:           Valid (non-excluded) raw_values for this window group.
        monitoring_point: Site identifier.
        window_group:     'morning' or 'afternoon'.
        min_bins:         Minimum number of bins required for baseline to be ready.
        min_sigma:        Floor for sigma (prevents division-by-zero in CUSUM/EARS).

    Returns:
        BaselineStats with mu, sigma, n_bins and is_ready flag.
    """
    n = len(values)
    if n == 0:
        return BaselineStats(
            monitoring_point=monitoring_point,
            window_group=window_group,
            mu=0.0,
            sigma=min_sigma,
            n_bins=0,
            is_ready=False,
        )

    mu = sum(values) / n

    # Population standard deviation
    if n >= 2:
        variance = sum((v - mu) ** 2 for v in values) / n
        sigma_raw = math.sqrt(variance)
    else:
        sigma_raw = 0.0

    sigma = max(sigma_raw, min_sigma)

    return BaselineStats(
        monitoring_point=monitoring_point,
        window_group=window_group,
        mu=mu,
        sigma=sigma,
        n_bins=n,
        is_ready=n >= min_bins,
    )


def window_group_for_index(window_index: int) -> str:
    """Map a window index (0-based) to a window_group label.

    Args:
        window_index: Position of the window in the schedule.windows list.

    Returns:
        'morning' for index 0, 'afternoon' for any other index.
    """
    if window_index == 0:
        return "morning"
    return "afternoon"
