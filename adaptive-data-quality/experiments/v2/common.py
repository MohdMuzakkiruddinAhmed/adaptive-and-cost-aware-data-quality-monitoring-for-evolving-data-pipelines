"""Shared machinery of the v2 experiment runners.

Every runner:
  * takes --split dev|test (test only when protocol.yaml protocol.frozen is true), optional --seeds,
    --datasets, --scenarios subsets, --jobs N (parallel units), --force (ignore cached unit results)
  * processes independent UNITS (dataset, seed) in parallel; each unit regenerates its batches from the
    clean data (identical across experiments and methods, so all comparisons are paired)
  * writes raw per-batch results per unit to results/v2/<experiment>/<split>/raw/ (resumable), then
    summary tables, paired confidence intervals and a provenance manifest to results/v2/<experiment>/<split>/
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.provenance import config_hash, file_hash, write_manifest  # noqa: E402
from common.utils import (data_dir, datasets_config, experiment_config, protocol_config, results_dir,  # noqa: E402
                          v2_results_dir, write_json)
from ingestion.splits import check_split_allowed  # noqa: E402
from injection.scenario import DataSource, ScenarioReplay  # noqa: E402
from profiling.profiler import profile_batch  # noqa: E402


# ----------------------------------------------------------------------------- arguments
def parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="subset of the split's seeds")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--jobs", type=int, default=1, help="parallel worker processes")
    ap.add_argument("--force", action="store_true", help="recompute units whose raw results exist")
    ap.add_argument("--summary-only", action="store_true", help="only rebuild tables from existing raw results")
    return ap


class Context:
    def __init__(self, experiment: str, args: argparse.Namespace) -> None:
        self.experiment, self.args = experiment, args
        self.datasets_all, self.exp, self.protocol = datasets_config(), experiment_config(), protocol_config()
        check_split_allowed(args.split, self.protocol)
        self.split = args.split
        self.seeds = list(args.seeds) if args.seeds else list(self.protocol["splits"]["seeds"][args.split])
        self.datasets = {k: v for k, v in self.datasets_all.items() if not args.datasets or k in args.datasets}
        default_scen = self.protocol["experiments"].get(experiment, [])
        self.scenarios = list(args.scenarios) if args.scenarios else list(default_scen)
        self.out = v2_results_dir(experiment, self.split)
        self.raw = self.out / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()

    @property
    def criteria(self) -> Dict[str, Any]:
        return self.protocol["criteria"]

    def units(self) -> List[Tuple[str, int]]:
        return [(ds, s) for ds in self.datasets for s in self.seeds]

    def unit_path(self, ds: str, seed: int, suffix: str = "") -> Path:
        return self.raw / f"{ds}__seed{seed}{suffix}.parquet"

    def manifest(self, outputs: Dict[str, str], extra: Optional[Dict[str, Any]] = None) -> Path:
        cfg = {"experiment": self.exp, "protocol": self.protocol, "datasets": self.datasets}
        data = {ds: {"clean_parquet_sha256": file_hash(data_dir("clean", f"{ds}.parquet"))} for ds in self.datasets}
        return write_manifest(self.out / "manifest.json", self.experiment, self.split, self.seeds, cfg, data, outputs,
                              {"scenarios": self.scenarios, "runtime_s": time.perf_counter() - self.t0, **(extra or {})})


# ----------------------------------------------------------------------------- unit execution
def cache_key(ctx: "Context", payload: Dict[str, Any]) -> str:
    """Fingerprint of everything a unit result depends on (Section 6.6): code, all configuration files and the
    runner's payload. Cached unit results are reused only while this key is unchanged."""
    from common.provenance import code_fingerprint, config_files_hashes
    return config_hash({"code": code_fingerprint(), "configs": config_files_hashes(), "payload": payload})


def run_units(ctx: Context, worker: Callable[[str, int, Dict[str, Any]], pd.DataFrame], payload: Dict[str, Any],
              suffix: str = "") -> pd.DataFrame:
    """Run ``worker(dataset, seed, payload) -> DataFrame`` for every unit (cached per unit), concatenate.
    Cached units are discarded when the code or any configuration file changed since they were computed."""
    payload = dict(payload, split=ctx.split, scenarios=ctx.scenarios)
    key = cache_key(ctx, payload)
    key_file = ctx.raw / f"_cache_key{suffix}.json"
    have_units = any(ctx.unit_path(ds, s, suffix).exists() for ds, s in ctx.units())
    if key_file.exists():
        stale = json.loads(key_file.read_text()).get("key") != key
    else:
        stale = have_units          # results without a fingerprint cannot be trusted (predate fingerprinting)
    if stale and not ctx.args.summary_only:
        print(f"[{ctx.experiment}] code or configuration changed since the cached results: recomputing all units", flush=True)
    force = ctx.args.force or stale
    todo = [(ds, s) for ds, s in ctx.units() if force or not ctx.unit_path(ds, s, suffix).exists()]
    if ctx.args.summary_only:
        todo = []
    elif not key_file.exists() or force:
        for ds, s in ctx.units():                       # drop results of the old key before writing new ones
            if force and ctx.unit_path(ds, s, suffix).exists():
                ctx.unit_path(ds, s, suffix).unlink()
        key_file.write_text(json.dumps({"key": key}))
    if todo:
        print(f"[{ctx.experiment}] {len(todo)} unit(s) to run ({ctx.split}, jobs={ctx.args.jobs})", flush=True)
    if ctx.args.jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=ctx.args.jobs) as pool:
            futs = {pool.submit(_safe_worker, worker, ds, s, payload): (ds, s) for ds, s in todo}
            for f in as_completed(futs):
                ds, s = futs[f]
                df, err = f.result()
                if err:
                    raise RuntimeError(f"unit {ds} seed {s} failed:\n{err}")
                _save(df, ctx.unit_path(ds, s, suffix))
                print(f"[{ctx.experiment}] done {ds} seed {s}", flush=True)
    else:
        for ds, s in todo:
            t0 = time.perf_counter()
            _save(worker(ds, s, payload), ctx.unit_path(ds, s, suffix))
            print(f"[{ctx.experiment}] done {ds} seed {s} ({time.perf_counter() - t0:.0f}s)", flush=True)
    frames = [pd.read_parquet(ctx.unit_path(ds, s, suffix)) for ds, s in ctx.units() if ctx.unit_path(ds, s, suffix).exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _safe_worker(worker, ds, s, payload):
    try:
        return worker(ds, s, payload), None
    except Exception:  # noqa: BLE001
        return None, traceback.format_exc()


def _save(df: pd.DataFrame, path: Path) -> None:
    """Parquet needs one type per column. Columns mixing values and gaps (rows of different kinds) keep
    their type: booleans -> nullable boolean, numbers -> float, other values -> strings with gaps as null.
    (Never str(): "False" would become a truthy string.)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    for c in df.columns:
        if df[c].dtype != object and str(df[c].dtype) not in ("str", "string"):
            continue
        s = df[c].map(lambda v: json.dumps(v, default=str) if isinstance(v, (dict, list, set, tuple)) else v)
        vals = s[s.notna()]
        kinds = {type(v) for v in vals}
        if not len(vals):
            df[c] = s.astype("string")
        elif kinds <= {bool, np.bool_}:
            df[c] = s.astype("boolean")
        elif all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))
                 for v in vals):
            df[c] = pd.to_numeric(s, errors="coerce")
        else:
            df[c] = s.map(lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)).astype("string")
    df.to_parquet(path, index=False)


# ----------------------------------------------------------------------------- worker helpers
_SOURCES: Dict[int, DataSource] = {}


def data_source() -> DataSource:
    """One DataSource per process (clean data and split pools are loaded once per worker)."""
    pid = os.getpid()
    if pid not in _SOURCES:
        _SOURCES[pid] = DataSource(datasets_config(), experiment_config(), protocol_config())
    return _SOURCES[pid]


class ProfiledReplay:
    """A replay plus its batch profiles (computed once, shared by every method of the unit)."""

    def __init__(self, replay: ScenarioReplay) -> None:
        self.replay = replay
        self.cfg = replay.cfg
        self._p: Dict[int, tuple] = {}

    def batch(self, t: int):
        return self.replay.batch(t)

    def profile(self, t: int):
        if t not in self._p:
            b = self.replay.batch(t)
            self._p[t] = profile_batch(b.df, self.cfg["numeric"], self.cfg["categorical"], batch=b.label,
                                       return_artifacts=True)
        return self._p[t]

    def __iter__(self):
        for t in range(1, self.replay.n_batches + 1):
            yield t, self.batch(t), self.profile(t)


def run_monitor_over(pr: ProfiledReplay, monitor, upstream: bool = True, reference: bool = False,
                     extra: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Run a Monitor over every batch; return one evaluation row per batch (ground truth used here only)."""
    from evaluation.metrics import record_counts
    from execution.local_executor import execute
    rows = []
    for t, b, (prof, art) in pr:
        step = monitor.step(b.df, b.label, prof if upstream else None, art if upstream else None)
        valid_novel = set(b.novelty.loc[b.novelty.kind == "valid", "record_id"]) if len(b.novelty) else set()
        row = {**b.meta, **(extra or {}), "scenario": pr.replay.scenario, "seed": pr.replay.seed,
               "dataset": pr.replay.dataset, "split": pr.replay.split,
               **record_counts(step["failed_ids"], b.gt, b.df["record_id"], valid_novel if valid_novel else None),
               "rules_available": step["rules_available"], "rules_executed": step["rules_executed"],
               "bytes_scanned": step["bytes_scanned"], "ms_total": step["ms_total"],
               **{f"ms_{k}": v for k, v in step["ms"].items()}}
        lab = step["label"] or {}
        for k in ("label", "drift", "degradation", "drift_insufficient", "drifted_columns", "degraded_checks",
                  "drift_signal", "failure_rate", "gate_excluded_fraction", "gate_coverage", "check_coverage",
                  "max_staleness", "degradation_evidence", "drift_threshold_k"):
            if k in lab:
                row[f"pred_{k}" if k in ("label", "drift", "degradation") else k] = lab[k]
        sch = step["scheduler"] or {}
        for k in ("coverage", "max_staleness", "standard_total", "standard_executed", "max_risk", "complete_suite"):
            if k in sch:
                row[f"sched_{k}"] = sch[k]
        if reference and not step["warmup"]:
            ref = execute(monitor.last_suite, b.df)          # evaluation-only full-suite pass
            row["ref_TP"] = len(ref["failed_ids"] & b.defective)
            row["ref_FP"] = len(ref["failed_ids"] - b.defective)
            row["missed_by_selection"] = len((ref["failed_ids"] & b.defective) - step["failed_ids"])
        rows.append(row)
    return rows


def true_class(row) -> str:
    from monitor.classifier import to_class
    return to_class(bool(row["drift_label"]), bool(row["degradation_label"]))


def summarize_counts(df: pd.DataFrame, by: List[str]) -> pd.DataFrame:
    from evaluation.metrics import ratios
    cols = ["TP", "FP", "FN", "TN", "P", "N", "records"]
    extra = [c for c in ("rules_executed", "bytes_scanned", "ms_total", "fp_valid_novel", "n_valid_novel")
             if c in df.columns]
    g = df.groupby(by)[cols + extra].sum().reset_index()
    g["seeds"] = df.groupby(by)["seed"].nunique().to_numpy() if "seed" in df.columns else 1
    return ratios(g)
