"""Adaptive thresholds (project description, Section 10):  T_t = f(H_{t-k:t}, D_t).

Range rule for each numerical column, recomputed every batch:

    History H (last k batch profiles, median across the window so one unusual batch
    does not dominate):  mu_H, sigma_H, median_H, IQR_H
    Current drift D_t, measured robustly from the current profile:
        location shift  = median_t - median_H
        scale ratio     = IQR_t / IQR_H
    Threshold T_t: the historical mu_H +/- k*sigma_H, moved by the location shift and
    stretched by the scale ratio:
        sigma_t = sigma_H * scale ratio
        mu_t    = median_t + (mu_H - median_H) * scale ratio
        range   = mu_t +/- k * sigma_t

Medians and IQRs barely move when a few percent of records are defective, but they do
move when the whole distribution drifts. So the threshold follows legitimate drift
without also following degradation. With no drift (shift ~ 0, ratio ~ 1) it reduces to
the historical mu_H +/- k*sigma_H.

Categorical columns: the valid set is every category seen in at least
``category_min_batches`` of the last k batches. New legitimate categories are learned
once they recur; an invalid value that appears in a single batch is not learned.

Design history (kept for the methodology section):
  v1  mu_t = (1 - D)*mu_H + D*mu_current with D the normalised drift score. The threshold
      lagged behind gradual drift (BTS false-positive rate 5.9% under drift vs 4.5% stable).
  v2  robust location/scale adjustment (above), with the history averaged by mean.
      Retail recall 0.96: one batch with an extreme legitimate quantity inflated sigma_H.
  v3  (current) same as v2 with the median across the history window.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def adaptive_range_rules(history: List[Dict[str, Any]], current: Dict[str, Any], drift: Dict[str, float],
                         numeric: List[str], k: float) -> List[Dict[str, Any]]:
    from profiling.profiler import numeric_with_stats
    rules = []
    for c in numeric_with_stats(list(history) + [current], numeric):
        hist = [h["columns"][c] for h in history]
        mu_h = float(np.median([h["mean"] for h in hist]))
        sd_h = float(np.median([h["std"] for h in hist]))
        med_h = float(np.median([h["median"] for h in hist]))
        iqr_h = float(np.median([h["quantiles"]["p75"] - h["quantiles"]["p25"] for h in hist]))
        cur = current["columns"][c]
        iqr_c = cur["quantiles"]["p75"] - cur["quantiles"]["p25"]
        shift = cur["median"] - med_h                       # location drift
        scale = iqr_c / iqr_h if iqr_h > 0 else 1.0         # scale drift
        mu, sd = mu_h * scale + (cur["median"] - med_h * scale), sd_h * scale
        rules.append({"type": "range", "column": c, "low": mu - k * sd, "high": mu + k * sd,
                      "location_shift": shift, "scale_ratio": scale, "drift": float(drift.get(c, 0.0))})
    return rules


def adaptive_category_rules(history: List[Dict[str, Any]], categorical: List[str],
                            min_batches: int, warm: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Valid set = categories seen in >= min_batches of the history window, plus (when ``warm`` is given) every
    category of the warm-up batches. Without the warm-up set a RARE legitimate category that was accepted in the
    warm-up is dropped again whenever it is missing from two of the last five batches, and every later occurrence is
    a false alarm (dev run 3: most of the extra false alarms of the rolling methods on Retail StockCode / Country and
    Chicago Facility Type). The warm-up is trusted the same way by every other method (fixed_pooled, frozen)."""
    need = min(min_batches, len(history))
    rules = []
    for c in categorical:
        seen: Dict[str, int] = {}
        for h in history:
            for cat in h["columns"][c]["frequencies"]:
                seen[cat] = seen.get(cat, 0) + 1
        allowed = {k for k, n in seen.items() if n >= need}
        for w in warm or []:
            allowed |= set(w["columns"][c]["frequencies"])
        rules.append({"type": "category", "column": c, "allowed": sorted(allowed)})
    return rules


def wide_gate_rules(history: List[Dict[str, Any]], numeric: List[str], width: float = 6.0) -> List[Dict[str, Any]]:
    """Coarse mu_H +/- 6 sigma_H gate used only to keep gross outliers out of the drift estimate."""
    from profiling.profiler import numeric_with_stats
    rules = []
    for c in numeric_with_stats(history, numeric):
        mu = float(np.mean([h["columns"][c]["mean"] for h in history]))
        sd = float(np.mean([h["columns"][c]["std"] for h in history]))
        rules.append({"type": "range", "column": c, "low": mu - width * sd, "high": mu + width * sd})
    return rules
