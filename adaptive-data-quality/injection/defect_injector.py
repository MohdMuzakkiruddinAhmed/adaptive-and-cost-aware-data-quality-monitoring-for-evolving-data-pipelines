"""Controlled defect injection and ground truth (project description, Section 6).

Record-level defects (each written to the ground truth with record ID, column,
defect type, severity and batch):
    inject_missing_values()      a value in an experiment column is set to null
    inject_duplicates()          a copy of an existing record is appended (new record ID)
    inject_invalid_categories()  a categorical value is replaced by a value that is not a valid category
    inject_outliers()            a numerical value is set to mean +/- U(4, 12) * std

Experiment 3b (semantic-defect scenarios):
    inject_semantic_defects()    realistic domain violations (e.g. negative taxi-out time, a
                                 well-formed but non-existent code, a misspelled category)

Batch-level change:
    inject_distribution_shift()  legitimate distribution drift (Section 7). It is recorded
                                 in the scenario labels, not as record-level defects.
    inject_schema_change()       optional in the project description; not implemented.

Each defective record receives exactly one defect, so every ground-truth row maps to
one record.
"""
from __future__ import annotations

import string
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

GT_COLUMNS = ["record_id", "column", "defect_type", "severity", "batch"]


def _gt(record_ids, columns, defect_type, severity, batch) -> pd.DataFrame:
    return pd.DataFrame({"record_id": list(record_ids), "column": list(columns), "defect_type": defect_type,
                         "severity": severity, "batch": batch})


def inject_missing_values(df, rows, columns, rng, batch, severity="Medium"):
    cols = rng.choice(columns, size=len(rows))
    for c in sorted(set(cols)):
        df.loc[rows[cols == c], c] = np.nan
    return _gt(df.loc[rows, "record_id"], cols, "missing", severity, batch)


def inject_invalid_categories(df, rows, columns, rng, batch, severity="Medium"):
    cols = rng.choice(columns, size=len(rows))
    letters = np.array(list(string.ascii_uppercase))
    for c in sorted(set(cols)):
        target = rows[cols == c]
        tokens = ["INVALID_" + "".join(rng.choice(letters, 5)) for _ in range(len(target))]
        df[c] = df[c].astype(object)
        df.loc[target, c] = tokens
    return _gt(df.loc[rows, "record_id"], cols, "invalid_category", severity, batch)


def inject_outliers(df, rows, columns, rng, batch, sigma_range=(4, 12), high_sigma=8, mild_sigma=None):
    """Severity is High when the value is >= high_sigma std from the mean, Mild when it is below
    ``mild_sigma`` (if given), otherwise Medium."""
    cols = rng.choice(columns, size=len(rows))
    severity = np.empty(len(rows), dtype=object)
    for c in sorted(set(cols)):
        sel = cols == c
        target = rows[sel]
        mean, std = float(df[c].mean()), float(df[c].std())
        size = rng.uniform(sigma_range[0], sigma_range[1], len(target))
        df[c] = df[c].astype(float)
        df.loc[target, c] = mean + size * rng.choice([-1, 1], len(target)) * std
        sev = np.where(size >= high_sigma, "High", "Medium")
        if mild_sigma is not None:
            sev = np.where(size < mild_sigma, "Mild", sev)
        severity[sel] = sev
    g = _gt(df.loc[rows, "record_id"], cols, "outlier", "", batch)
    g["severity"] = severity
    return g


def inject_duplicates(df, rows, rng, batch, severity="Low"):
    """Append a copy of each row in ``rows`` (new record ID). The copies reflect the rows as they are
    NOW, so this must be the last mutation of a batch (see append_duplicates / defer_duplicates)."""
    copies = df.loc[rows].copy()
    copies["record_id"] = [f"{batch}-D{i:06d}" for i in range(len(copies))]
    out = pd.concat([df, copies], ignore_index=True)
    return out, _gt(copies["record_id"], ["<row>"] * len(copies), "duplicate", severity, batch)


def append_duplicates(df: pd.DataFrame, source_ids: Sequence[str], batch: str,
                      severity: str = "Low") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Benchmark-integrity fix (manuscript Section 6.2): duplicates are appended only after every
    other mutation of the batch, as exact copies of the FINAL state of their source records. The
    source records must not receive any other mutation (callers reserve them), so each copy is a
    true whole-record duplicate of an unmodified original and its label matches the final batch."""
    if len(source_ids) == 0:
        return df, pd.DataFrame(columns=GT_COLUMNS)
    pos = pd.Index(df["record_id"]).get_indexer(list(source_ids))
    if (pos < 0).any():
        raise ValueError("duplicate source record not found in batch")
    return inject_duplicates(df, pos, None, batch, severity)


def inject_distribution_shift(df, level, numeric, categorical, top_categories, rng):
    """Legitimate drift of size ``level``: numeric columns x (1 + level); a fraction ``level``
    of each categorical column re-drawn uniformly from its 10 most common categories."""
    if level <= 0:
        return df
    df = df.copy()
    for c in numeric:
        df[c] = df[c].astype(float) * (1.0 + level)
    for c in categorical:
        mask = rng.random(len(df)) < level
        df.loc[mask, c] = rng.choice(top_categories[c], size=int(mask.sum()))
    return df


def inject_defects(df: pd.DataFrame, rate: float, numeric: Sequence[str], categorical: Sequence[str],
                   rng: np.random.Generator, batch: str, cfg: Dict[str, Any], defer_duplicates: bool = False,
                   exclude_ids=None):
    """Inject ``rate`` * n defects, split equally over the configured defect types.

    ``defer_duplicates=False`` (legacy): duplicates are appended immediately and (df, gt) is returned.
    ``defer_duplicates=True``: duplicates are NOT appended; (df, gt, dup_source_ids) is returned and
    the caller must call append_duplicates(df, dup_source_ids, ...) after every later mutation (e.g.
    semantic defects), never touching the source records in between. The random draws are identical
    in both modes, so batches without later mutations are bit-identical to the legacy ones.
    ``exclude_ids``: record IDs that must not receive a defect (already used by another injection);
    with no exclusions the random draws are those of the legacy code."""
    types = list(cfg["types"])
    sev = cfg["severity"]
    n_total = int(round(rate * len(df)))
    df = df.copy()
    pool = rng.permutation(len(df))           # disjoint rows, one defect per record
    if exclude_ids:
        pool = pool[~df["record_id"].iloc[pool].isin(set(exclude_ids)).to_numpy()]
        n_total = min(n_total, len(pool))
    per_type = {t: n_total // len(types) for t in types}
    for t in types[: n_total % len(types)]:
        per_type[t] += 1

    pos = 0
    take = lambda k: pool[pos:pos + k]
    gts: List[pd.DataFrame] = []

    rows = take(per_type.get("missing", 0)); pos += len(rows)
    if len(rows):
        gts.append(inject_missing_values(df, rows, list(numeric) + list(categorical), rng, batch, sev["missing"]))
    rows = take(per_type.get("invalid_category", 0)); pos += len(rows)
    if len(rows):
        gts.append(inject_invalid_categories(df, rows, list(categorical), rng, batch, sev["invalid_category"]))
    rows = take(per_type.get("outlier", 0)); pos += len(rows)
    if len(rows):
        gts.append(inject_outliers(df, rows, list(numeric), rng, batch, tuple(cfg["outlier_sigma"]),
                                   cfg.get("outlier_high_sigma", 8), cfg.get("outlier_mild_sigma")))
    rows = take(per_type.get("duplicate", 0)); pos += len(rows)
    dup_sources = list(df.loc[rows, "record_id"]) if len(rows) else []
    if len(rows) and not defer_duplicates:
        df, g = inject_duplicates(df, rows, rng, batch, sev["duplicate"])
        gts.append(g)

    gt = pd.concat([g for g in gts if len(g)], ignore_index=True) if any(len(g) for g in gts) \
        else pd.DataFrame(columns=GT_COLUMNS)
    # duplicates stay at the end of the batch: a re-sent record arrives after the original
    if defer_duplicates:
        return df, gt[GT_COLUMNS], dup_sources
    return df, gt[GT_COLUMNS]


def inject_semantic_defects(df: pd.DataFrame, rate: float, specs: Sequence[Dict[str, Any]],
                            rng: np.random.Generator, batch: str, exclude_ids: set,
                            severity: str = "Medium") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Realistic domain violations (Experiment 3b): each spec sets a column either to a value drawn
    uniformly from ``uniform: [low, high]`` or to one of ``values``. Records that already carry a
    standard defect (``exclude_ids``) are not touched, so each record has at most one defect.
    Ground truth: defect_type "semantic_numeric" or "semantic_category"."""
    n = int(round(rate * len(df)))
    if n == 0:
        return df, pd.DataFrame(columns=GT_COLUMNS)
    df = df.copy()
    eligible = np.flatnonzero(~df["record_id"].isin(exclude_ids).to_numpy())
    rows = rng.choice(eligible, size=min(n, len(eligible)), replace=False)
    pick = rng.integers(0, len(specs), size=len(rows))
    cols, types = [], []
    for i in sorted(set(pick)):
        target = rows[pick == i]
        spec = specs[i]
        c = spec["column"]
        if "uniform" in spec:
            lo, hi = spec["uniform"]
            df[c] = df[c].astype(float)
            df.loc[target, c] = rng.uniform(lo, hi, len(target))
            kind = "semantic_numeric"
        else:
            vals = list(spec["values"])
            new = [vals[j] for j in rng.integers(0, len(vals), len(target))]
            if all(isinstance(v, (int, float)) for v in vals):
                df[c] = df[c].astype(float)
            else:
                df[c] = df[c].astype(object)
            df.loc[target, c] = new
            kind = "semantic_numeric" if all(isinstance(v, (int, float)) for v in vals) else "semantic_category"
        cols += [(r, c, kind) for r in target]
    order = sorted(cols)
    gt = pd.DataFrame({"record_id": df.loc[[r for r, _, _ in order], "record_id"].to_numpy(),
                       "column": [c for _, c, _ in order], "defect_type": [k for _, _, k in order],
                       "severity": severity, "batch": batch})
    return df, gt[GT_COLUMNS]
