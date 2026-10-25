"""Paired uncertainty analysis (manuscript Section 6.5; supervisor: "report paired confidence intervals",
"thousands of rows or batches are not thousands of independent replications").

Unit of replication = the simulation RUN (seed). Batches of a run share warm-up history and records, and
the same seed is used by every method, so every comparison is paired by seed and the bootstrap resamples
whole seeds (cluster bootstrap). Record- and batch-level counts are pooled inside each resampled set of
seeds before the ratio (recall, FPR, ...) is computed, matching the micro-averaged point estimates.

Provided:
  paired_ratio_diff   difference of two pooled ratios (e.g. recall of A minus recall of B), cluster
                      bootstrap percentile CI, bootstrap p-value, and a t interval on per-seed differences
  paired_mean_diff    the same for per-seed scalar metrics (e.g. macro-F1, runtime)
  noninferior         CI lower bound of (new - comparator) > -margin  (H1a)
  holm                Holm step-down adjustment for secondary comparisons
Results are reported per dataset; pooling across datasets is a separate, clearly labelled summary.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 19: 2.093, 24: 2.064, 29: 2.045, 49: 2.010}


def _t_quantile(df: int, level: float) -> float:
    if abs(level - 0.95) > 1e-9:
        from statistics import NormalDist
        return NormalDist().inv_cdf(0.5 + level / 2)       # normal approximation for other levels
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    for k in keys:
        if k > df:
            return _T975[k]
    return 1.96


def _ratio(num: np.ndarray, den: np.ndarray) -> float:
    d = den.sum()
    return float(num.sum() / d) if d else float("nan")


def _summary(est: float, boot: np.ndarray, per_cluster: np.ndarray, level: float) -> Dict[str, Any]:
    boot = boot[np.isfinite(boot)]
    a = (1 - level) / 2
    lo, hi = (np.quantile(boot, [a, 1 - a]) if len(boot) else (np.nan, np.nan))
    p = float(min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))) if len(boot) else float("nan")
    pc = per_cluster[np.isfinite(per_cluster)]
    if len(pc) > 1:
        se = pc.std(ddof=1) / math.sqrt(len(pc))
        q = _t_quantile(len(pc) - 1, level)
        t_lo, t_hi = pc.mean() - q * se, pc.mean() + q * se
    else:
        t_lo = t_hi = float("nan")
    return {"estimate": est, "ci_low": float(lo), "ci_high": float(hi), "p_value": p,
            "t_ci_low": float(t_lo), "t_ci_high": float(t_hi), "n_clusters": int(len(per_cluster)),
            "level": level}


def paired_ratio_diff(num_a: Sequence[float], den_a: Sequence[float], num_b: Sequence[float],
                      den_b: Sequence[float], B: int = 5000, level: float = 0.95, seed: int = 0) -> Dict[str, Any]:
    """Pooled ratio(A) - ratio(B), one entry per cluster (seed) in each array, same order for A and B."""
    num_a, den_a, num_b, den_b = (np.asarray(x, dtype=float) for x in (num_a, den_a, num_b, den_b))
    k = len(num_a)
    est = _ratio(num_a, den_a) - _ratio(num_b, den_b)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(B, k))
    da, db = den_a[idx].sum(axis=1), den_b[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        boot = num_a[idx].sum(axis=1) / da - num_b[idx].sum(axis=1) / db
    with np.errstate(invalid="ignore", divide="ignore"):
        per = num_a / den_a - num_b / den_b
    out = _summary(est, boot, per, level)
    out.update(ratio_a=_ratio(num_a, den_a), ratio_b=_ratio(num_b, den_b))
    return out


def paired_mean_diff(a: Sequence[float], b: Sequence[float], B: int = 5000, level: float = 0.95,
                     seed: int = 0) -> Dict[str, Any]:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    out = _summary(float(np.nanmean(d)), np.nanmean(d[idx], axis=1), d, level)
    out.update(mean_a=float(np.nanmean(a)), mean_b=float(np.nanmean(b)))
    return out


def noninferior(result: Dict[str, Any], margin: float) -> bool:
    """New method non-inferior: the CI of (new - comparator) excludes losses larger than ``margin``."""
    return bool(np.isfinite(result["ci_low"]) and result["ci_low"] > -margin)


def holm(pvalues: Sequence[float]) -> List[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def compare_methods(df: pd.DataFrame, by: List[str], method_col: str, comparator: str, cluster_col: str,
                    ratios: Dict[str, tuple], B: int = 5000, level: float = 0.95, margin: Optional[float] = None,
                    seed: int = 0) -> pd.DataFrame:
    """Paired comparisons of every method against ``comparator`` within groups ``by``.

    ``df`` holds one row per (group, method, cluster) with count columns; ``ratios`` maps a metric name to
    (numerator column, denominator column), e.g. {"recall": ("TP", "P"), "fpr": ("FP", "N")}."""
    rows = []
    for key, g in df.groupby(by) if by else [((), df)]:
        key = key if isinstance(key, tuple) else (key,)
        ref = g[g[method_col] == comparator].set_index(cluster_col)
        for m, gm in g.groupby(method_col):
            if m == comparator:
                continue
            gm = gm.set_index(cluster_col)
            common = sorted(set(gm.index) & set(ref.index))
            if not common:
                continue
            for metric, (nc, dc) in ratios.items():
                r = paired_ratio_diff(gm.loc[common, nc], gm.loc[common, dc], ref.loc[common, nc],
                                      ref.loc[common, dc], B, level, seed)
                row = dict(zip(by, key))
                row.update(method=m, comparator=comparator, metric=metric, **r)
                if margin is not None and metric == "recall":
                    row["noninferior"] = noninferior(r, margin)
                rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = np.nan
        for _, idx in out.groupby(by + ["metric"] if by else ["metric"]).groups.items():
            out.loc[idx, "p_holm"] = holm(out.loc[idx, "p_value"].fillna(1.0).to_numpy())
    return out
