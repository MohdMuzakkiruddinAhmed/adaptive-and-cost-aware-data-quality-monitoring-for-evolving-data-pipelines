"""External dirty/clean benchmarks as batch replays (configs/benchmarks.yaml; manuscript Section 6.3).

``BenchmarkReplay`` offers the same interface as injection.scenario.ScenarioReplay (batch(t), n_batches,
cfg, dataset, seed, scenario, split), so the same monitors and evaluation code run on it. Ground truth: one
row per erroneous cell (dirty != clean under the configured normalisation), used only for evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from common.utils import batch_label, data_dir
from injection.scenario import GT_V2_COLUMNS, Batch


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.strip(), errors="coerce")


def error_cells(dirty: pd.DataFrame, clean: pd.DataFrame, normalization: str = "numeric_equal") -> pd.DataFrame:
    """Boolean frame: True where a dirty cell differs from its clean counterpart."""
    if dirty.shape != clean.shape:
        raise ValueError(f"dirty {dirty.shape} and clean {clean.shape} tables are not aligned")
    clean = clean.copy()
    clean.columns = dirty.columns
    out = pd.DataFrame(False, index=dirty.index, columns=dirty.columns)
    for c in dirty.columns:
        a, b = dirty[c].astype(str), clean[c].astype(str)
        if normalization == "exact":
            out[c] = (a != b).to_numpy()
            continue
        a, b = a.str.strip(), b.str.strip()
        diff = a != b
        na, nb = _num(a), _num(b)
        both = na.notna() & nb.notna()
        diff &= ~(both & np.isclose(na.fillna(0), nb.fillna(0), rtol=1e-12, atol=1e-12))
        out[c] = diff.to_numpy()
    return out


def benchmark_cfg(name: str, bench: Dict[str, Any]) -> Dict[str, Any]:
    b = bench["benchmarks"][name]
    return {"name": name, "numeric": list(b["numeric"]), "categorical": list(b["categorical"]),
            "nullable": list(b.get("nullable", [])), "drift_numeric": [], "drift_categorical": [],
            "key_columns": [], "importance": {}, "batch_size": None, "description": b.get("description", ""),
            "column_descriptions": dict(b.get("column_descriptions", {}))}


def prompt_pool_index(n_rows: int, bench: Dict[str, Any], split_seed: int) -> np.ndarray:
    """Rows reserved for the LLM prompts (v2.1 leakage fix): a fixed random share of the table, chosen by the split
    seed, that never occurs in any replay batch. Before v2.1 the prompt sample was a reference replay's warm-up, whose
    records also appeared in evaluation batches of the other seeds."""
    pp = bench.get("prompt_pool") or {}
    if not pp:
        return np.array([], dtype=int)
    k = max(int(round(float(pp.get("fraction", 0.1)) * n_rows)), int(pp.get("min_rows", 0)))
    k = min(k, n_rows // 2)
    rng = np.random.default_rng([int(split_seed), 991])
    return np.sort(rng.permutation(n_rows)[:k])


def load_tables(name: str, bench: Dict[str, Any]):
    b = bench["benchmarks"][name]
    dirty = pd.read_csv(data_dir("external", b["files"]["dirty"]), dtype=str, keep_default_na=False)
    clean = pd.read_csv(data_dir("external", b["files"]["clean"]), dtype=str, keep_default_na=False)
    return dirty, clean


class BenchmarkReplay:
    def __init__(self, name: str, bench: Dict[str, Any], exp: Dict[str, Any], seed: int,
                 dirty: Optional[pd.DataFrame] = None, clean: Optional[pd.DataFrame] = None,
                 split_seed: Optional[int] = None) -> None:
        b = bench["benchmarks"][name]
        self.dataset, self.seed, self.scenario, self.split = name, seed, "external", "external"
        self.cfg = benchmark_cfg(name, bench)
        self.context = [c for c in b.get("context_columns", [])]
        if dirty is None:
            dirty, clean = load_tables(name, bench)
        clean = clean.copy()
        clean.columns = dirty.columns
        drop = [c for c in b.get("drop_columns", []) if c in dirty.columns]
        dirty, clean = dirty.drop(columns=drop), clean.drop(columns=drop)
        err = error_cells(dirty, clean, bench.get("label_normalization", "numeric_equal"))
        if bench.get("empty_as_missing", True):
            dirty = dirty.replace({"": np.nan})
        if split_seed is not None:                       # rows reserved for the LLM prompts never enter a batch
            reserved = prompt_pool_index(len(dirty), bench, split_seed)
            self.prompt_rows = dirty.iloc[reserved].reset_index(drop=True)
            keep = np.setdiff1d(np.arange(len(dirty)), reserved)
            dirty, err = dirty.iloc[keep].reset_index(drop=True), err.iloc[keep].reset_index(drop=True)
        else:
            self.prompt_rows = dirty.iloc[:0]
        self.n_batches = int(bench.get("n_batches", 20))
        self.warm = exp["warmup_batches"]
        rng = np.random.default_rng([seed, 4242])
        order = rng.permutation(len(dirty))
        size = len(dirty) // self.n_batches
        self.cfg["batch_size"] = size
        self.batches = []
        for t in range(1, self.n_batches + 1):
            idx = order[(t - 1) * size: t * size]
            df = dirty.iloc[idx].reset_index(drop=True)
            e = err.iloc[idx].reset_index(drop=True)
            lab = batch_label(t)
            df.insert(0, "record_id", [f"{lab}-{i:06d}" for i in range(len(df))])
            rows = [(df.at[i, "record_id"], c) for c in e.columns for i in np.flatnonzero(e[c].to_numpy())]
            gt = pd.DataFrame({"record_id": [r for r, _ in rows], "column": [c for _, c in rows],
                               "defect_type": "benchmark_error", "severity": "", "batch": lab, "incident": ""},
                              columns=GT_V2_COLUMNS)
            meta = {"batch": lab, "t": t, "drift_level": 0.0, "defect_rate": np.nan, "drift_label": 0,
                    "degradation_label": 0, "evaluation": int(t > self.warm), "records": len(df),
                    "defects": int(gt.record_id.nunique()),
                    "monitored_error_cells": int(e[[c for c in e.columns if c in self.cfg["numeric"] + self.cfg["categorical"]]].to_numpy().sum()),
                    "all_error_cells": int(e.to_numpy().sum())}
            self.batches.append(Batch(t=t, label=lab, df=df, gt=gt, novelty=pd.DataFrame(columns=["record_id", "column", "value", "kind", "batch"]), meta=meta))
            self.batches[-1].error_mask = e

    def batch(self, t: int) -> Batch:
        return self.batches[t - 1]
