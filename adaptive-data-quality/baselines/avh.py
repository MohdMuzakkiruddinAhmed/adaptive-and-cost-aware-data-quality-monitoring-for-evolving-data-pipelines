"""AVH-style batch-level comparator (Tu et al., "Auto-Validate by-History", KDD 2023).

WHAT IS REPRODUCED. AVH monitors recurring pipeline partitions through many per-column statistics
("metrics") and learns, from the history of earlier partitions, constraints of the form
|metric_t - mean_hist| <= k * sd_hist whose false-alarm probability is bounded with distribution-free
(Chebyshev-type) inequalities; the false-alarm budget delta is shared across the selected constraints.

SIMPLIFICATION (state it in the paper). The original selects a constraint subset by solving an
optimisation problem that maximises expected recall under the precision constraint, and it uses a larger
metric library. Here every metric of a fixed library is kept and the budget delta is split equally (union
bound): delta_c = delta / K per constraint. ``bound``:
  student     (default) prediction bound for a new observation from n history points:
              k_c = t quantile(1 - delta_c / 2, n - 1) * sqrt(1 + 1/n). Accounts for the uncertainty of the mean and
              sd estimated from only 4-15 past batches (t quantile by a Cornish-Fisher expansion, within ~1% of the
              exact value for n - 1 >= 4 and ~3% for n - 1 = 3).
  gaussian    k_c = normal quantile 1 - delta_c / 2: too many false alarms with short histories (sd underestimated).
  chebyshev   k_c = 1 / sqrt(delta_c) (distribution-free; about 24 sd for K = 30): with ~30 metrics and only
              5-15 history points it alarmed on 11% of evaluation batches although two thirds contained drift
              or degradation, i.e. it was a strawman comparator (dev run, 5 Oct 2026).
It is a stand-in for AVH, not a reimplementation. Its native output is a BATCH alarm (with the metrics
that fired); it provides no record-level localisation, and is evaluated only at batch level.

Four-class mapping (defined in advance, Section 6.4): metrics are grouped into "distribution" metrics
(location, spread, quantile distance, category divergence) and "quality" metrics (missing share,
unseen-category share, duplicate share, row count). A distribution alarm -> drift; a quality alarm ->
degradation. Outliers can move both groups; this ambiguity is a property of the comparator.

History (``history``), both reported (protocol.yaml classification.comparators):
  all       (default, as published) every previous batch, alarmed or not (expanding window, at least
            ``min_history``). A persistent change is absorbed after one or two batches (dev run 2: drift recall
            0.12-0.20, degradation recall 0.02-0.10 against labels that describe change since the warm-up), while
            onsets are found without delay. AVH targets anomalous partitions, not persistent states.
  accepted  "quarantine": only batches that raised no alarm enter the history; ``prev`` (previous-batch metrics)
            is the last accepted batch. Persistent changes keep alarming, but censoring the alarmed batches
            shrinks the history sd, so false alarms rise above the budget (synthetic noise, 6 x 40 batches:
            18% of ready batches instead of 5% with ``all``).
No oracle is used in either mode: the decision to keep a batch uses only AVH's own alarm.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from drift.drift_detector import categorical_drift

DISTRIBUTION = "distribution"
QUALITY = "quality"


def batch_metrics(profile: Dict[str, Any], prev: Dict[str, Any], history: List[Dict[str, Any]],
                  numeric: List[str], categorical: List[str]) -> Dict[str, tuple]:
    """Metric name -> (value, group). ``prev`` = previous batch profile, ``history`` = all earlier profiles."""
    out: Dict[str, tuple] = {}
    from profiling.profiler import has_numeric_stats
    for c in numeric:
        p, q = profile["columns"][c], prev["columns"][c]
        out[f"{c}:missing"] = (p.get("missing_pct", 0.0) or 0.0, QUALITY)
        if not (has_numeric_stats(p) and has_numeric_stats(q)):
            continue                     # no numeric values: only the missing/non-numeric share is monitored
        qs = p.get("quantiles", {})
        iqr_prev = (q["quantiles"]["p75"] - q["quantiles"]["p25"]) or q["std"] or 1.0
        out[f"{c}:mean"] = (p.get("mean", np.nan), DISTRIBUTION)
        out[f"{c}:median"] = (p.get("median", np.nan), DISTRIBUTION)
        out[f"{c}:std"] = (p.get("std", np.nan), DISTRIBUTION)
        out[f"{c}:iqr"] = (qs.get("p75", np.nan) - qs.get("p25", np.nan), DISTRIBUTION)
        out[f"{c}:p01"] = (qs.get("p01", np.nan), DISTRIBUTION)
        out[f"{c}:p99"] = (qs.get("p99", np.nan), DISTRIBUTION)
        qk = [k for k in qs if k in q["quantiles"]]
        out[f"{c}:qdist_prev"] = (float(np.mean([abs(qs[k] - q["quantiles"][k]) for k in qk])) / iqr_prev, DISTRIBUTION)
    for c in categorical:
        p, q = profile["columns"][c], prev["columns"][c]
        seen = set().union(*[set(h["columns"][c]["frequencies"]) for h in history])
        freq = p["frequencies"]
        out[f"{c}:js_prev"] = (categorical_drift(q["frequencies"], freq), DISTRIBUTION)
        out[f"{c}:top1_share"] = (max(freq.values()) if freq else 0.0, DISTRIBUTION)
        out[f"{c}:cardinality"] = (float(p["cardinality"]), DISTRIBUTION)
        out[f"{c}:unseen_share"] = (float(sum(v for k, v in freq.items() if k not in seen)), QUALITY)
        out[f"{c}:missing"] = (p.get("missing_pct", 0.0) or 0.0, QUALITY)
    out["_records:duplicate_pct"] = (profile["dataset"].get("duplicate_pct", 0.0) or 0.0, QUALITY)
    out["_records:row_count"] = (float(profile["dataset"]["record_count"]), QUALITY)
    return out


def t_quantile(p: float, nu: float) -> float:
    """Student-t quantile by a Cornish-Fisher expansion around the normal quantile (no SciPy dependency)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    g = [(z ** 3 + z) / 4, (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96,
         (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384,
         (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160]
    return z + sum(gi / nu ** (i + 1) for i, gi in enumerate(g))


def constraint_k(bound: str, delta_c: float, n: int) -> float:
    """Multiplier of the history sd for one constraint with false-alarm budget ``delta_c`` and n history points."""
    if bound == "chebyshev":
        return 1.0 / np.sqrt(delta_c)
    if bound == "gaussian":
        from statistics import NormalDist
        return NormalDist().inv_cdf(1 - delta_c / 2)
    return t_quantile(1 - delta_c / 2, max(n - 1, 1)) * np.sqrt(1 + 1 / n)


class AVHMonitor:
    """Sequential AVH-style monitor. Call ``observe(profile)`` for T1, T2, ... in order."""

    def __init__(self, numeric: List[str], categorical: List[str], fpr_budget: float = 0.05,
                 min_history: int = 4, bound: str = "student", history: str = "all") -> None:
        if bound not in ("student", "gaussian", "chebyshev"):
            raise ValueError(f"unknown bound {bound!r}")
        if history not in ("accepted", "all"):
            raise ValueError(f"unknown history policy {history!r}")
        self.history = history
        self.numeric, self.categorical = numeric, categorical
        self.delta, self.min_history, self.bound = fpr_budget, min_history, bound
        self.profiles: List[Dict[str, Any]] = []
        self.values: List[Dict[str, tuple]] = []

    def observe(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        res: Dict[str, Any] = {"alarm": False, "drift": False, "degradation": False, "fired": [], "ready": False}
        if self.profiles:
            m = batch_metrics(profile, self.profiles[-1], self.profiles, self.numeric, self.categorical)
            hist = self.values[1:] if len(self.values) > 1 else []     # first entry has no "previous"
            if len(hist) >= self.min_history:
                res["ready"] = True
                K = len(m)
                for name, (v, group) in m.items():
                    h = np.array([x[name][0] for x in hist if name in x], dtype=float)
                    h = h[np.isfinite(h)]
                    if len(h) < self.min_history or not np.isfinite(v):
                        continue
                    kc = constraint_k(self.bound, self.delta / K, len(h))
                    mu, sd = float(h.mean()), float(h.std(ddof=1))
                    sd = max(sd, 1e-9, 1e-6 * abs(mu))
                    if abs(v - mu) > kc * sd:
                        res["fired"].append(name)
                        res["drift" if group == DISTRIBUTION else "degradation"] = True
                res["alarm"] = bool(res["fired"])
            if res["alarm"] and self.history == "accepted":
                return res                       # alarmed batch: not learned from
            self.values.append(m)
        else:
            self.values.append({})
        self.profiles.append(profile)
        return res
