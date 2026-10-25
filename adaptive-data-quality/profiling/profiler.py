"""Data Profiler (project description, Section 5).

Extracts column- and dataset-level statistics for a batch:

* Numerical:   count, min, max, mean, median, std, MAD, quantiles, missing %, unique count
* Categorical: number of categories, most common categories, category frequencies,
               missing %, cardinality
* Dataset:     record count, column count, duplicate %, schema, data types

Profiles are stored per batch (see profiling/store.py) so later batches can be
compared with history. The statistical rules (Section 8.2), drift detection and
adaptive thresholds (Section 10) all read from these stored profiles.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from pandas.api import types as ptypes

QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95, 0.99]
TOP_N = 10


def _qkey(q: float) -> str:
    return f"p{int(round(q * 100)):02d}"


def profile_numerical(s: pd.Series) -> Dict[str, Any]:
    n = len(s)
    x = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "datatype": "numerical",
        "count": int(x.size),
        "missing_pct": 100.0 * (n - x.size) / n if n else None,
        "unique_count": int(np.unique(x).size),
    }
    if x.size:
        out.update({
            "min": float(x.min()),
            "max": float(x.max()),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
            "mad": float(np.median(np.abs(x - np.median(x)))),     # robust spread for the MAD baseline
            "quantiles": {_qkey(q): float(v) for q, v in zip(QUANTILES, np.quantile(x, QUANTILES))},
        })
    return out


def profile_categorical(s: pd.Series) -> Dict[str, Any]:
    n = len(s)
    vals = s.dropna().astype(str)
    freq = vals.value_counts(normalize=True)
    return {
        "datatype": "categorical",
        "count": int(len(vals)),
        "missing_pct": 100.0 * (n - len(vals)) / n if n else None,
        "cardinality": int(freq.size),
        "most_common": [str(v) for v in freq.index[:TOP_N]],
        "frequencies": {str(k): float(v) for k, v in freq.items()},
    }


def has_numeric_stats(col_profile: Dict[str, Any]) -> bool:
    """False for a numerical column with no numeric value in the batch (e.g. every value carries a unit:
    "12.0 oz"): such a column has completeness/type checks but no range or drift statistics."""
    return bool(col_profile.get("count")) and "mean" in col_profile and "quantiles" in col_profile


def numeric_with_stats(profiles: Iterable[Dict[str, Any]], numeric: Iterable[str]) -> List[str]:
    """The numerical columns that have statistics in EVERY given profile."""
    profiles = [p for p in profiles if p is not None]
    return [c for c in numeric if all(has_numeric_stats(p["columns"][c]) for p in profiles)]


def profile_batch(
    df: pd.DataFrame,
    numeric: Iterable[str],
    categorical: Iterable[str],
    batch: Optional[str] = None,
    exclude_from_duplicates: Iterable[str] = ("record_id",),
    return_artifacts: bool = False,
):
    """Profile one batch. ``numeric`` and ``categorical`` name the columns to profile.

    ``return_artifacts=True`` returns (profile, artifacts) where artifacts holds in-memory results that
    later checks can reuse instead of recomputing (the whole-record duplicate mask and its columns)."""
    numeric, categorical = list(numeric), list(categorical)
    dup_cols = [c for c in df.columns if c not in set(exclude_from_duplicates)]
    n = len(df)
    dup_mask = df.duplicated(subset=dup_cols).to_numpy() if n else np.zeros(0, dtype=bool)
    dup = int(dup_mask.sum())
    columns: Dict[str, Any] = {}
    for c in numeric:
        columns[c] = profile_numerical(df[c])
    for c in categorical:
        columns[c] = profile_categorical(df[c])
    profile = {
        "batch": batch,
        "dataset": {
            "record_count": n,
            "column_count": int(df.shape[1]),
            "duplicate_pct": 100.0 * dup / n if n else None,
            "schema": {c: str(t) for c, t in df.dtypes.items()},
        },
        "columns": columns,
    }
    if return_artifacts:
        return profile, {"dup_mask": dup_mask, "dup_columns": dup_cols}
    return profile
