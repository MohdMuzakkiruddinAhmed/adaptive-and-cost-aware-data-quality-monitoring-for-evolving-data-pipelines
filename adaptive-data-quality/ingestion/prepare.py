"""Clean source data and sequential batches (Sections 6 and 7).

Clean data: rows that pass the dataset's clean filters, with no missing values in the
experiment columns and no duplicate records. Defects are injected into this data, so
every defect in an experimental batch is known.

Batches: the clean data is shuffled with a fixed seed and split into T1..T20 of equal
size. Shuffling means the only drift in the batches is the drift we inject, so the
drift labels in the ground truth are exact.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def clean(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    report = {"raw_rows": len(df)}
    for expr in cfg.get("clean_filters") or []:
        df = df[df.eval(expr, engine="python")]
        report[f"after: {expr}"] = len(df)
    keep = list(dict.fromkeys(cfg["key_columns"] + cfg["numeric"] + cfg["categorical"]))
    df = df[keep].dropna(subset=cfg["numeric"] + cfg["categorical"])
    report["after: no missing values"] = len(df)
    df = df.drop_duplicates()
    report["after: no duplicate records"] = len(df)
    return df.reset_index(drop=True), report


def make_batches(clean_df: pd.DataFrame, n_batches: int, batch_size: int, seed: int) -> List[pd.DataFrame]:
    need = n_batches * batch_size
    if len(clean_df) < need:
        raise ValueError(f"clean data has {len(clean_df)} rows, {need} needed")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(clean_df))[:need]
    batches = []
    for t in range(n_batches):
        b = clean_df.iloc[idx[t * batch_size:(t + 1) * batch_size]].reset_index(drop=True)
        b.insert(0, "record_id", [f"T{t + 1:02d}-{i:06d}" for i in range(len(b))])
        batches.append(b)
    return batches
