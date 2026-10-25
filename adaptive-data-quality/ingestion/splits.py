"""Disjoint development and test record pools (manuscript Section 6.3).

The cleaned source rows are split once, by a fixed ``split_seed`` that is independent of the run
seeds, into a development pool (debugging, exploratory runs, LLM prompt examples) and a test pool
(confirmatory runs only). Runs of either split draw their T1..T20 batches from their own pool only,
so no record used during development or rule generation can appear in a confirmatory batch.

Values held out for the category-novelty scenarios (H2b) are removed from both pools for those
scenarios (``exclude_values``), so a "new" valid category is genuinely absent from the warm-up.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Tuple

import numpy as np
import pandas as pd

SPLITS = ("dev", "test")


def split_pools(clean_df: pd.DataFrame, dev_fraction: float, split_seed: int) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(len(clean_df))
    n_dev = int(round(dev_fraction * len(clean_df)))
    return {"dev": clean_df.iloc[np.sort(perm[:n_dev])].reset_index(drop=True),
            "test": clean_df.iloc[np.sort(perm[n_dev:])].reset_index(drop=True)}


def exclude_values(df: pd.DataFrame, values: Mapping[str, Iterable[str]]) -> pd.DataFrame:
    """Rows whose value in any listed column is one of the listed values are removed."""
    keep = np.ones(len(df), dtype=bool)
    for col, vals in values.items():
        keep &= ~df[col].astype(str).isin({str(v) for v in vals}).to_numpy()
    return df[keep].reset_index(drop=True)


def batch_size_for(pool_rows: int, batch_size: int, n_batches: int) -> int:
    return int(min(batch_size, pool_rows // n_batches))


def check_split_allowed(split: str, protocol: Dict[str, Any]) -> None:
    """Confirmatory (test) runs are only allowed once the protocol is frozen (Section 6.5)."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; use one of {SPLITS}")
    if split == "test" and not protocol["protocol"].get("frozen", False):
        raise PermissionError("split 'test' is reserved for confirmatory runs: set protocol.frozen: true in "
                              "configs/protocol.yaml after all tuning choices are final, then rerun")


def pool_overlap(a: pd.DataFrame, b: pd.DataFrame, key_columns) -> int:
    """Number of records (by key columns) present in both frames: 0 for valid pools."""
    ka = pd.MultiIndex.from_frame(a[list(key_columns)].astype(str))
    kb = pd.MultiIndex.from_frame(b[list(key_columns)].astype(str))
    return int(ka.isin(kb).sum())
