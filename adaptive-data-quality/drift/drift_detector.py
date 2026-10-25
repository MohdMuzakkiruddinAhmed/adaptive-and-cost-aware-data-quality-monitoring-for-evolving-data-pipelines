"""Drift detection (project description, Section 10).

Numerical columns: Wasserstein distance, computed from quantiles p05..p95 (the average
absolute difference between current and reference quantiles), divided by the reference
inter-quartile range. Quantiles are unaffected by tied values (e.g. delays in whole minutes).
Categorical columns: Jensen-Shannon divergence (base 2) over the 20 most common reference
categories plus one "other" bucket. Bucketing keeps sampling noise in high-cardinality
columns (e.g. ~4,900 stock codes) from dominating the score.

Both give a drift score D in [0, 1] per column (numerical scores are clipped at 1).
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

from profiling.profiler import QUANTILES, _qkey, has_numeric_stats, numeric_with_stats

# p05..p95: the extreme 1% tails are noisy in heavy-tailed columns and are where
# borderline outliers land, so they are left out of the drift score.
DRIFT_Q = [q for q in QUANTILES if 0.05 <= q <= 0.95]
QKEYS = [_qkey(q) for q in DRIFT_Q]
TOP_N = 20
_EPS = 1e-12


def reference_quantiles(history: List[Dict[str, Any]], column: str) -> np.ndarray:
    return np.mean([[h["columns"][column]["quantiles"][k] for k in QKEYS] for h in history], axis=0)


def reference_frequencies(history: List[Dict[str, Any]], column: str) -> Dict[str, float]:
    total: Dict[str, float] = {}
    for h in history:
        for k, v in h["columns"][column]["frequencies"].items():
            total[k] = total.get(k, 0.0) + v / len(history)
    return total


def numeric_drift(ref_q: np.ndarray, values: np.ndarray):
    if values.size == 0:
        return None                     # no numeric value: insufficient evidence, not "no drift"
    cur_q = np.quantile(values, DRIFT_Q)
    w = float(np.mean(np.abs(cur_q - ref_q)))
    iqr = ref_q[QKEYS.index("p75")] - ref_q[QKEYS.index("p25")]
    scale = iqr if iqr > 0 else (np.std(ref_q) or 1.0)
    return min(1.0, w / scale)


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = p / p.sum() if p.sum() else p
    q = q / q.sum() if q.sum() else q
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.maximum(b[mask], _EPS))))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def categorical_drift(ref: Dict[str, float], cur: Dict[str, float], top_n: int = TOP_N) -> float:
    top = [k for k, _ in sorted(ref.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    p = np.array([ref.get(k, 0.0) for k in top] + [max(0.0, 1 - sum(ref.get(k, 0.0) for k in top))])
    q = np.array([cur.get(k, 0.0) for k in top] + [max(0.0, 1 - sum(cur.get(k, 0.0) for k in top))])
    return jsd(p, q)


def drift_scores(df: pd.DataFrame, reference: List[Dict[str, Any]], numeric: List[str],
                 categorical: List[str]) -> Dict[str, float]:
    """Drift score D in [0, 1] per column: current batch ``df`` vs. the reference profiles."""
    scores: Dict[str, float] = {}
    ok = set(numeric_with_stats(reference, numeric))
    for c in numeric:
        if c not in ok:
            scores[c] = None
            continue
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy(dtype=float)
        scores[c] = numeric_drift(reference_quantiles(reference, c), x)
    for c in categorical:
        cur = df[c].dropna().astype(str).value_counts(normalize=True).to_dict()
        scores[c] = categorical_drift(reference_frequencies(reference, c), cur)
    return scores


def reference_summary(reference: List[Dict[str, Any]], numeric: List[str], categorical: List[str]) -> Dict[str, Any]:
    """Pre-computed reference quantiles / frequencies (compute once for a fixed reference)."""
    return {"q": {c: reference_quantiles(reference, c) for c in numeric_with_stats(reference, numeric)},
            "f": {c: reference_frequencies(reference, c) for c in categorical}}


def profile_drift_scores(current: Dict[str, Any], reference, numeric: List[str],
                         categorical: List[str]) -> Dict[str, float]:
    """Drift score per column computed from stored profiles only (no pass over the raw batch).
    Used by the risk score, which must be computed before deciding which checks to run.
    ``reference`` is a list of profiles or a pre-computed ``reference_summary``."""
    ref = reference if isinstance(reference, dict) else reference_summary(reference, numeric, categorical)
    scores: Dict[str, float] = {}
    for c in numeric:
        if c not in ref["q"] or not has_numeric_stats(current["columns"][c]):
            scores[c] = None            # no numeric statistics: insufficient evidence
            continue
        cur = np.array([current["columns"][c]["quantiles"][k] for k in QKEYS])
        rq = ref["q"][c]
        iqr = rq[QKEYS.index("p75")] - rq[QKEYS.index("p25")]
        scale = iqr if iqr > 0 else (np.std(rq) or 1.0)
        scores[c] = min(1.0, float(np.mean(np.abs(cur - rq))) / scale)
    for c in categorical:
        scores[c] = categorical_drift(ref["f"][c], current["columns"][c]["frequencies"])
    return scores


# ---------------------------------------------------------------------------------- resampling null
def resampled_null_scores(pool: pd.DataFrame, gate_pass: np.ndarray, numeric: List[str], categorical: List[str],
                          folds: int, draws: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    """Null distribution of the drift score of a STABLE batch, by resampling the warm-up records.

    The warm-up records (all ``folds`` batches pooled) are randomly split into ``folds`` pseudo-batches of equal
    size, ``draws`` times. Every pseudo-batch is scored once against the other ``folds - 1`` pseudo-batches, exactly
    as the leave-one-out calibration scores a real warm-up batch: the current batch contributes only its records
    that pass the error-filtering gate (``gate_pass``), the reference uses all records (the reference profiles of
    the monitor are built from complete batches). This gives folds * draws scores per column instead of the five
    leave-one-out scores, so the upper quantile of the score of a stable batch is estimated directly rather than
    through a normal approximation of five values (dev run 3: discrete, heavy-tailed columns such as Retail
    Quantity gave five nearly equal LOO scores, a tiny sd and 15-17% false drift alarms).
    Exchangeable warm-up batches are assumed; between-batch differences that are not sampling noise are captured
    by the leave-one-out bound, which the classifier keeps (the threshold is the larger of the two)."""
    N = len(pool)
    m = N // folds
    if m < 2:
        return {}
    gate_pass = np.asarray(gate_pass, dtype=bool)
    num = {c: pd.to_numeric(pool[c], errors="coerce").to_numpy(dtype=float) for c in numeric}
    num = {c: v for c, v in num.items() if np.isfinite(v).sum() >= folds * 2}
    cats = {}
    for c in categorical:
        s = pool[c]
        codes, uniques = pd.factorize(s.map(lambda v: None if pd.isna(v) else str(v)), use_na_sentinel=True)
        if len(uniques):
            cats[c] = (codes, np.array([str(u) for u in uniques]))
    acc: Dict[str, List[np.ndarray]] = {c: [] for c in list(num) + list(cats)}
    block = max(1, min(draws, 4_000_000 // max(N, 1)))           # bound memory: block x N values at a time
    done = 0
    import warnings
    while done < draws:
        nb = min(block, draws - done)
        done += nb
        idx = np.stack([rng.permutation(N)[: m * folds].reshape(folds, m) for _ in range(nb)])   # nb x folds x m
        gp = gate_pass[idx]
        for c, v in num.items():
            x = v[idx]
            cur = np.where(gp, x, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                q_all = np.nanquantile(x, DRIFT_Q, axis=2)            # nq x nb x folds
                q_cur = np.nanquantile(cur, DRIFT_Q, axis=2)
                ref = (q_all.sum(axis=2, keepdims=True) - q_all) / (folds - 1)   # mean of the other folds
                iqr = ref[QKEYS.index("p75")] - ref[QKEYS.index("p25")]
                scale = np.where(iqr > 0, iqr, np.std(ref, axis=0))
                scale = np.where(scale > 0, scale, 1.0)
                w = np.nanmean(np.abs(q_cur - ref), axis=0) / scale
            acc[c].append(np.clip(w[np.isfinite(w)], 0.0, 1.0).ravel())
        for c, (codes, names) in cats.items():
            K = len(names)
            cd = codes[idx]
            valid = cd >= 0
            flat = np.arange(nb * folds).reshape(nb, folds, 1) * K + np.where(valid, cd, 0)
            cnt_all = np.bincount(flat[valid], minlength=nb * folds * K).reshape(nb, folds, K).astype(float)
            vg = valid & gp
            cnt_cur = np.bincount(flat[vg], minlength=nb * folds * K).reshape(nb, folds, K).astype(float)
            f_all = cnt_all / np.maximum(cnt_all.sum(axis=2, keepdims=True), 1)
            f_cur = cnt_cur / np.maximum(cnt_cur.sum(axis=2, keepdims=True), 1)
            ref = (f_all.sum(axis=1, keepdims=True) - f_all) / (folds - 1)
            sc = []
            for d in range(nb):
                for k in range(folds):
                    r, q = ref[d, k], f_cur[d, k]
                    order = np.lexsort((names, -r))[:TOP_N]           # top-N reference categories, ties by name
                    p_vec = np.append(r[order], max(0.0, 1.0 - r[order].sum()))
                    q_vec = np.append(q[order], max(0.0, 1.0 - q[order].sum()))
                    sc.append(jsd(p_vec, q_vec))
            acc[c].append(np.asarray(sc))
    out = {c: np.concatenate(v) for c, v in acc.items() if v}
    return out
