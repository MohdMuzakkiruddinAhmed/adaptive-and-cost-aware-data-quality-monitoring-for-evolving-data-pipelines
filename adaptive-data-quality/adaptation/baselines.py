"""Threshold policies for RQ1 / H1a (supervisor experiment 1; manuscript Sections 2.2 and 6.4).

Every policy produces range rules for the numerical columns and category rules for the categorical
columns of batch t from information available before (or, for the proposed method, at) batch t:

  warm      profiles of the warm-up batches T1..T5
  history   profiles of the previous ``window`` batches (window = experiment.yaml adaptive.history_window)
  past      profiles of ALL previous batches (only EWMA uses more than ``window``; with lambda = 0.3
            its effective memory, (2 - lambda) / lambda = 5.7 batches, matches the window)
  current   profile of batch t itself (only the proposed "adaptive" policy uses it)

Numerical policies                                         bounds
  fixed_t1        PoC baseline: T1 profile                   mean_T1 +/- k sd_T1
  fixed_pooled    all five warm-up batches pooled            pooled mean +/- k pooled sd
  rolling_meansd  previous window batches pooled             pooled mean +/- k pooled sd
  rolling_mad     previous window, robust                    median +/- mad_k * 1.4826 * MAD
  rolling_iqr     previous window, robust (Tukey far out)    Q1 - iqr_k IQR, Q3 + iqr_k IQR
  ewma            exponentially weighted batch moments       m_t +/- k sqrt(v_t)
  adaptive        proposed: history moved to the CURRENT batch's median / IQR (adaptive_thresholds.py)
  adaptive_prev   the same formula, location/scale taken from the most recent PREVIOUS batch only
  rolling_mad_cal / rolling_iqr_cal
                  the robust rules with k CALIBRATED on the warm-up instead of the textbook k = 3 (which flagged
                  27-29% of valid BTS and Retail records in dev run 2: skewed columns). k is chosen per column so
                  that, leave-one-out over the warm-up batches, the robust rule flags the same share of records as
                  the primary comparator's pooled mean +/- k sd rule (calibrate_robust_k): both baselines then
                  start from the same false-positive level, and the comparison isolates how each handles drift.

Category policies
  fixed_t1        categories observed in T1
  fixed_pooled    categories observed in any warm-up batch
  recurrence      categories observed in >= category_min_batches of the previous window batches (proposed)

Robust medians / MADs / quartiles of the window are the medians of the per-batch statistics (the same
aggregation the proposed method uses), so no raw history is re-read.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adaptation.adaptive_thresholds import adaptive_category_rules, adaptive_range_rules
from profiling.profiler import numeric_with_stats

NUMERIC_POLICIES = ["fixed_t1", "fixed_pooled", "rolling_meansd", "rolling_mad", "rolling_iqr", "ewma",
                    "adaptive", "adaptive_prev", "rolling_mad_cal", "rolling_iqr_cal"]
CALIBRATED = {"rolling_mad_cal": "rolling_mad", "rolling_iqr_cal": "rolling_iqr"}
CATEGORY_POLICIES = ["fixed_t1", "fixed_pooled", "recurrence"]
MAD_TO_SD = 1.4826


def pooled_moments(profiles: List[Dict[str, Any]], col: str):
    """Count-weighted pooled mean and sd of several batch profiles (exact for the union of the batches)."""
    n = np.array([p["columns"][col]["count"] for p in profiles], dtype=float)
    m = np.array([p["columns"][col]["mean"] for p in profiles], dtype=float)
    s = np.array([p["columns"][col]["std"] for p in profiles], dtype=float)
    N = n.sum()
    mean = float((n * m).sum() / N)
    var = float(((n - 1) * s ** 2).sum() + (n * (m - mean) ** 2).sum()) / max(N - 1, 1)
    return mean, float(np.sqrt(max(var, 0.0)))


def _range(col: str, low: float, high: float, policy: str, **extra) -> Dict[str, Any]:
    return {"type": "range", "column": col, "low": float(low), "high": float(high), "policy": policy, **extra}


def _robust_spread(p: Dict[str, Any]) -> float:
    """MAD in sd units, falling back to IQR / 1.349 and then sd when the MAD is 0 (discrete data)."""
    mad = p.get("mad")
    if mad:
        return MAD_TO_SD * mad
    iqr = p["quantiles"]["p75"] - p["quantiles"]["p25"]
    return iqr / 1.349 if iqr > 0 else p["std"]


def _robust_center_spread(history: List[Dict[str, Any]], col: str, kind: str):
    """(median, MAD-based sd) or (Q1, Q3) of a window: medians of the per-batch statistics."""
    hs = [h["columns"][col] for h in history]
    if kind == "mad":
        return float(np.median([h["median"] for h in hs])), float(np.median([_robust_spread(h) for h in hs]))
    return (float(np.median([h["quantiles"]["p25"] for h in hs])),
            float(np.median([h["quantiles"]["p75"] for h in hs])))


def calibrate_robust_k(policy: str, warm_dfs, warm_profiles: List[Dict[str, Any]], numeric: List[str],
                       k_fixed: float) -> Dict[str, float]:
    """Per-column k of a calibrated robust policy (rolling_mad_cal / rolling_iqr_cal), from the warm-up only.

    Leave-one-out over the warm-up batches j: the robust bounds are built from the OTHER batches' profiles,
    exactly as the policy builds them later from its history window; z_j = standardised distance of batch j's
    values beyond the robust centre (MAD) or the quartiles (IQR). The target share is what the primary
    comparator's rule (pooled mean +/- k_fixed sd of the other batches) flags in batch j, averaged over j; k is
    the (1 - target) quantile of the pooled z values. No labels are used."""
    kind = "mad" if CALIBRATED[policy] == "rolling_mad" else "iqr"
    out: Dict[str, float] = {}
    for c in numeric_with_stats(warm_profiles, numeric):
        zs, rates = [], []
        for j, df in enumerate(warm_dfs):
            others = [p for i, p in enumerate(warm_profiles) if i != j]
            x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
            if not len(x):
                continue
            mu, sd = pooled_moments(others, c)
            rates.append(float(np.mean((x < mu - k_fixed * sd) | (x > mu + k_fixed * sd))))
            a, b = _robust_center_spread(others, c, kind)
            if kind == "mad":
                z = np.abs(x - a) / b if b > 0 else np.full(len(x), np.inf)
            else:
                w = b - a
                z = np.maximum(a - x, x - b) / w if w > 0 else np.where((x < a) | (x > b), np.inf, 0.0)
            zs.append(z)
        if not zs:
            continue
        z = np.concatenate(zs)
        target = float(np.mean(rates))
        z_finite = z[np.isfinite(z)]
        if target <= 0 or not len(z_finite):
            out[c] = float(np.nanmax(z_finite)) if len(z_finite) else k_fixed
            continue
        out[c] = float(np.quantile(z, 1.0 - target)) if np.isfinite(np.quantile(z, 1.0 - target)) else float(z_finite.max())
    return out


def ewma_state(past: List[Dict[str, Any]], col: str, lam: float, init: Optional[List[Dict[str, Any]]] = None):
    """EWMA of batch mean and variance. ``past`` = all previous profiles T1..T(t-1), oldest first.
    Initialised with the pooled ``init`` profiles (the warm-up T1..T5, which are the first entries of
    ``past``), then updated with every later past batch: m <- lam * mean_s + (1 - lam) * m (same for var)."""
    init = list(init) if init else past[:1]
    m, s = pooled_moments(init, col)
    v = s ** 2
    for p in past[len(init):]:
        c = p["columns"][col]
        m = lam * c["mean"] + (1 - lam) * m
        v = lam * c["std"] ** 2 + (1 - lam) * v
    return m, float(np.sqrt(v))


def numeric_rules(policy: str, numeric: List[str], k: float, warm: List[Dict[str, Any]],
                  history: List[Dict[str, Any]], past: List[Dict[str, Any]], current: Optional[Dict[str, Any]],
                  params: Dict[str, Any]) -> List[Dict[str, Any]]:
    if policy not in NUMERIC_POLICIES:
        raise ValueError(f"unknown numeric policy {policy!r}")
    # columns without numeric values in some reference profile get no range rule (type checks still apply)
    numeric = numeric_with_stats(list(warm) + list(history) + list(past) + ([current] if current else []), numeric)
    if policy == "adaptive":
        return [dict(r, policy=policy) for r in adaptive_range_rules(history, current, {}, numeric, k)]
    if policy == "adaptive_prev":
        return [dict(r, policy=policy) for r in adaptive_range_rules(history, history[-1], {}, numeric, k)]
    rules = []
    k_cal = params.get("k_cal", {}) if policy in CALIBRATED else {}
    base = CALIBRATED.get(policy, policy)
    for c in numeric:
        if base == "rolling_mad":
            spread_c = _robust_center_spread(history, c, "mad")
            kk = k_cal.get(c, params.get("mad_k", 3.0))
            rules.append(_range(c, spread_c[0] - kk * spread_c[1], spread_c[0] + kk * spread_c[1], policy))
            continue
        if base == "rolling_iqr":
            q1, q3 = _robust_center_spread(history, c, "iqr")
            kk = k_cal.get(c, params.get("iqr_k", 3.0))
            rules.append(_range(c, q1 - kk * (q3 - q1), q3 + kk * (q3 - q1), policy))
            continue
        if policy == "fixed_t1":
            p = warm[0]["columns"][c]
            rules.append(_range(c, p["mean"] - k * p["std"], p["mean"] + k * p["std"], policy))
        elif policy in ("fixed_pooled", "rolling_meansd"):
            mu, sd = pooled_moments(warm if policy == "fixed_pooled" else history, c)
            rules.append(_range(c, mu - k * sd, mu + k * sd, policy))
        elif policy == "ewma":
            m, sd = ewma_state(past, c, params.get("ewma_lambda", 0.3), init=warm)
            rules.append(_range(c, m - k * sd, m + k * sd, policy))
    return rules


def category_rules(policy: str, categorical: List[str], warm: List[Dict[str, Any]], history: List[Dict[str, Any]],
                   min_batches: int, keep_warmup: bool = False) -> List[Dict[str, Any]]:
    if policy not in CATEGORY_POLICIES:
        raise ValueError(f"unknown category policy {policy!r}")
    if policy == "recurrence":
        return [dict(r, policy=policy) for r in
                adaptive_category_rules(history, categorical, min_batches, warm if keep_warmup else None)]
    src = warm[:1] if policy == "fixed_t1" else warm
    return [{"type": "category", "column": c, "policy": policy,
             "allowed": sorted(set().union(*[set(p["columns"][c]["frequencies"]) for p in src]))}
            for c in categorical]


def method_rules(method: Dict[str, str], numeric: List[str], categorical: List[str], k: float,
                 warm, history, past, current, params: Dict[str, Any], min_batches: int,
                 keep_warmup: bool = False) -> List[Dict[str, Any]]:
    """Range + category rules of a baseline method {numeric: policy, category: policy}."""
    return (numeric_rules(method["numeric"], numeric, k, warm, history, past, current, params)
            + category_rules(method["category"], categorical, warm, history, min_batches, keep_warmup))
