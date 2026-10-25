"""Integrated monitoring loop (manuscript Section 5.1, Figure 1; supervisor experiments 5 and 6).

One loop does everything for each batch, in the order of Figure 1, and times every phase of the cost
model C_t = C_read + C_profile + C_drift + C_adapt + C_select + C_execute + C_persist (Equation 8):

  read      load the batch (only when a path is given; in-memory replays charge 0)
  profile   profile the batch (or accept an upstream profile: "profiles supplied upstream")
  drift     profile drift vs the warm-up (input to the risk score); later also the gated drift of the
            batch label, which needs the executed checks
  adapt     build the candidate suite Q_t (threshold method, category policy, LLM/manual rules)
  select    scheduler chooses A_t (risk score, periodic, random, ...)
  execute   run A_t
  classify  batch label from drift evidence and the EXECUTED checks (no oracle on skipped checks)
  persist   write the profile and the monitor state (last runs, last failure rates) to disk

Classification and selection share the loop, so a skipped check really is missing evidence for the label.
The loop never sees ground truth. A full-suite "reference pass" for evaluation is run by the experiment
code on a copy of the suite (``last_suite``) after ``step`` returns; nothing from it flows back here.

Warm-up T1..T5: every check runs; per-check failure rates are recorded; at the end of T5 the classifier
(leave-one-out) and the scheduler (failure baselines, drift thresholds) are calibrated. That one-off
calibration cost is reported separately ("calibrate").
"""
from __future__ import annotations

import json
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from adaptation.baselines import CALIBRATED, calibrate_robust_k
from drift.drift_detector import profile_drift_scores, reference_summary
from execution.local_executor import column_bytes, execute
from monitor.classifier import BatchClassifier
from monitor.schedulers import Scheduler
from monitor.suite import SuiteBuilder
from profiling.profiler import profile_batch
from risk.risk_score import RECORDS

PHASES = ["read", "profile", "drift", "adapt", "select", "execute", "classify", "persist"]


@dataclass
class MonitorConfig:
    name: str
    method: str = "adaptive"                      # protocol.yaml baselines.methods key
    category_policy: Optional[str] = None         # guard policy (H2b) or None = the method's own
    guard_constraints: Optional[Dict[str, List]] = None
    extra_rules: List[Dict[str, Any]] = field(default_factory=list)
    include_statistical: bool = True
    include_base: bool = True
    dedupe_extra: bool = True
    scheduler: str = "all"
    scheduler_kw: Dict[str, Any] = field(default_factory=dict)
    drift_source: str = "gated"
    degradation_rule: str = "union_rate"
    reuse_profile_stats: bool = True              # checks reuse profile statistics (identical results, less work;
                                                  # rq3 --runtime measures both settings)
    persist: bool = False
    classify: bool = True
    drift_reference: str = "warmup"               # warmup: change since T1-T5 | rolling: vs the previous window
    drift_threshold: Optional[Dict[str, Any]] = None   # overrides protocol.yaml classification.drift_threshold


def _schema_changed(cur: Dict[str, Any], prev: Optional[Dict[str, Any]], units: List[str]) -> Dict[str, bool]:
    schema = cur["dataset"]["schema"]
    return {u: (u not in schema) or (prev is not None and prev["dataset"]["schema"].get(u) != schema.get(u))
            for u in units if u != RECORDS}


class Monitor:
    def __init__(self, cfg: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any], mc: MonitorConfig,
                 seed: int = 0, persist_dir: Optional[Path] = None) -> None:
        self.cfg, self.exp, self.protocol, self.mc = cfg, exp, protocol, mc
        self.warm_n, self.window = exp["warmup_batches"], exp["adaptive"]["history_window"]
        self.num, self.cat = cfg["numeric"], cfg["categorical"]
        self.units = self.num + self.cat + [RECORDS]
        self.suite = SuiteBuilder(cfg, exp, protocol, mc.method, mc.category_policy, mc.guard_constraints,
                                  mc.extra_rules, dedupe_extra=mc.dedupe_extra,
                                  include_statistical=mc.include_statistical, include_base=mc.include_base)
        self.scheduler = Scheduler(mc.scheduler, exp, self.units, cfg.get("importance", {}), seed=seed,
                                   **mc.scheduler_kw)
        ccfg = protocol.get("classification", {})
        self.classifier = BatchClassifier(cfg, exp, mc.drift_source, mc.degradation_rule,
                                          min_gate_records=ccfg.get("min_gate_records", 50),
                                          drift_threshold=mc.drift_threshold or ccfg.get("drift_threshold"),
                                          seed=seed) \
            if mc.classify else None
        self.profiles: List[Dict[str, Any]] = []
        self.warm_dfs: List[pd.DataFrame] = []
        self.warm_reuse: List[Optional[Dict[str, Any]]] = []
        self.warm_rates: List[Dict[str, float]] = []
        self.baseline_ref = None
        self.drift_tau: Dict[str, float] = {}
        self.calibrate_ms = 0.0
        self.last_suite: List[Dict[str, Any]] = []
        self.t = 0
        self._tmp = None
        if mc.persist:
            if persist_dir is None:
                self._tmp = tempfile.TemporaryDirectory(prefix="adq_monitor_")
                persist_dir = Path(self._tmp.name)
            self.persist_dir = Path(persist_dir)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.persist_dir = None

    # ------------------------------------------------------------------ helpers
    def _drift_thresholds(self, warm: List[Dict[str, Any]], z: float = 3.0) -> Dict[str, float]:
        rows = []
        for j, p in enumerate(warm):
            others = [q for i, q in enumerate(warm) if i != j]
            rows.append(profile_drift_scores(p, others, self.num, self.cat))
        df = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce")
        out = {}
        for c in df.columns:
            x = df[c].dropna()
            out[c] = max(float(x.mean() + z * x.std(ddof=1)), 1e-6) if len(x) > 1 else float("nan")
        return out

    def _calibrate(self) -> None:
        t0 = time.perf_counter()
        warm = self.profiles[: self.warm_n]
        self.baseline_ref = reference_summary(warm, self.num, self.cat)
        self.drift_tau = self._drift_thresholds(warm)
        n = len(self.warm_dfs[-1])
        num_policy = self.suite.method.get("numeric") if self.suite.method else None
        if num_policy in CALIBRATED:                 # robust baselines with k calibrated on the warm-up
            self.suite.params["k_cal"] = calibrate_robust_k(num_policy, self.warm_dfs, warm, self.num,
                                                            self.exp["range_k"])
        self.scheduler.calibrate(self.warm_rates, n, self.drift_tau)
        if self.classifier is not None:
            self.classifier.calibrate(self.warm_dfs, warm, self.suite,
                                      self.warm_reuse if self.mc.reuse_profile_stats else None)
        self.warm_dfs, self.warm_reuse = [], []          # raw warm-up batches are no longer needed
        self.calibrate_ms = (time.perf_counter() - t0) * 1000

    # ------------------------------------------------------------------ one batch
    def step(self, batch: Any, label: str, profile: Optional[Dict[str, Any]] = None,
             artifacts: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process one batch. ``batch`` is a DataFrame or a path to a parquet file. ``profile`` (and its
        ``artifacts``) may be supplied upstream; then the profile phase costs 0 for the monitor."""
        self.t += 1
        t = self.t
        ms: Dict[str, float] = {p: 0.0 for p in PHASES}
        clock = time.perf_counter

        t0 = clock()
        df = pd.read_parquet(batch) if isinstance(batch, (str, Path)) else batch
        ms["read"] = (clock() - t0) * 1000 if isinstance(batch, (str, Path)) else 0.0

        t0 = clock()
        upstream = profile is not None
        if not upstream:
            profile, artifacts = profile_batch(df, self.num, self.cat, batch=label, return_artifacts=True)
            ms["profile"] = (clock() - t0) * 1000
        reuse = {"profile": profile, **(artifacts or {})} if self.mc.reuse_profile_stats else None

        warm = self.profiles[: self.warm_n]
        history = self.profiles[max(0, len(self.profiles) - self.window):]
        past = list(self.profiles)
        in_warmup = t <= self.warm_n

        # drift from the profile (input to risk; cheap, no pass over the records)
        t0 = clock()
        d_raw = profile_drift_scores(profile, self.baseline_ref, self.num, self.cat) if not in_warmup else {}
        ms["drift"] += (clock() - t0) * 1000

        # adapt: candidate suite
        t0 = clock()
        if t == 1:   # no history yet: rules from the batch itself (T1 rules), as in the proof of concept
            rules, admissions = self.suite.build(list(df.columns), [profile], [profile], [profile], profile)
        else:
            rules, admissions = self.suite.build(list(df.columns), warm, history, past, profile)
        ms["adapt"] = (clock() - t0) * 1000

        # select
        t0 = clock()
        if in_warmup:
            chosen, sinfo = rules, {"policy": "warmup", "coverage": 1.0, "max_staleness": 0,
                                    "standard_total": None, "standard_executed": None}
        else:
            changed = _schema_changed(profile, self.profiles[-1] if self.profiles else None, self.units)
            chosen, sinfo = self.scheduler.select(rules, t, d_raw, changed)
        ms["select"] = (clock() - t0) * 1000

        # execute
        t0 = clock()
        cb = column_bytes(df)
        need_masks = self.classifier is not None and self.mc.drift_source in ("gated", "gated_executed")
        res = execute(chosen, df, cb, reuse=reuse, return_masks=need_masks)
        ms["execute"] = (clock() - t0) * 1000
        n = len(df)
        observed = {p["key"]: p["failures"] / max(n, 1) for p in res["per_rule"]}

        # classify (drift evidence of the label is charged to the drift phase)
        label_info: Dict[str, Any] = {}
        if self.classifier is not None and not in_warmup:
            t0 = clock()
            executed = [(r, res["masks"][r["key"]]) for r in chosen] if need_masks else None
            rolling = self.mc.drift_reference == "rolling"
            ev = self.classifier.drift_evidence(df, profile, history if rolling else warm, history, executed, reuse,
                                                cached_reference=not rolling)
            ms["drift"] += (clock() - t0) * 1000
            t0 = clock()
            label_info = self.classifier.classify(ev, res, n, sinfo.get("coverage", 1.0), sinfo.get("max_staleness", 0))
            label_info["drift_scores"] = ev["scores"]
            ms["classify"] = (clock() - t0) * 1000

        # update state
        if in_warmup:
            self.scheduler.mark_warmup(t, observed)
            if t >= 2:
                self.warm_rates.append(observed)
            self.warm_dfs.append(df)
            self.warm_reuse.append(reuse)
        else:
            self.scheduler.update(t, observed)
        self.profiles.append(profile)
        if t == self.warm_n:
            self._calibrate()

        # persist
        if self.persist_dir is not None:
            t0 = clock()
            (self.persist_dir / f"{label}.json").write_text(json.dumps(profile, default=float), encoding="utf-8")
            state = {"t": t, "last_run": self.scheduler.last_run,
                     "last_f": self.scheduler.tracker.last_f if self.scheduler.tracker else {}}
            (self.persist_dir / "state.json").write_text(json.dumps(state, default=float), encoding="utf-8")
            ms["persist"] = (clock() - t0) * 1000

        self.last_suite = [dict(r) for r in rules]
        total = sum(ms.values())
        return {"t": t, "batch": label, "warmup": in_warmup, "records": n,
                "failed_ids": res["failed_ids"], "failure_rate": res["failure_rate"],
                "rules_available": len(rules), "rules_executed": res["rules_executed"],
                "bytes_scanned": res["bytes_scanned"], "per_rule": res["per_rule"],
                "scheduler": sinfo, "label": label_info, "admissions": admissions,
                "dropped_extra": list(self.suite.dropped),
                "ms": ms, "ms_total": total, "ms_total_upstream_profile": total - ms["profile"],
                "profile_supplied": upstream, "calibrate_ms": self.calibrate_ms if t == self.warm_n else 0.0}

    def close(self) -> None:
        if self._tmp is not None:
            self._tmp.cleanup()
