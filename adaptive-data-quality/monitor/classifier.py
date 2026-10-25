"""Joint drift / degradation batch labelling inside the monitoring loop (manuscript Section 5.5; H1b).

Drift evidence d_t (``drift_source``):
  gated           proof-of-concept gate: records failing completeness, type, uniqueness, the recurrence
                  category rule or a coarse 6-sigma range are excluded; drift is computed on the rest
                  against the warm-up profiles. Gate checks that the monitor did not execute this batch are
                  computed here and charged to the drift phase.
  gated_executed  the same gate built ONLY from the checks executed this batch (+ the 6-sigma gate); with
                  selective execution the gate is weaker, and its coverage is reported
  ungated         drift from all records (current profile vs warm-up): the gate ablation
  none            no drift evidence (the proof-of-concept static comparator)
A gate that leaves fewer than ``min_gate_records`` records gives "insufficient" drift evidence instead of the
proof-of-concept zero score (Section 5.3: an empty gated sample is not evidence of no drift).

Degradation evidence g_t (``degradation_rule``):
  union_rate  share of records failing any executed check > warm-up threshold (proof-of-concept rule;
              biased low when checks are skipped, because the threshold was set with the complete suite)
  per_check   some EXECUTED check fails clearly more often than in the warm-up:
              rate_c > mu_c + z * sd_c (sd floored at the binomial sampling sd). Skipped checks give no
              evidence either way; coverage and staleness are reported with the label.

Thresholds: leave-one-out over the warm-up batches T1..T5 (each scored with the other four as history and
reference). Drift thresholds per column (protocol.yaml classification.drift_threshold.method):
  zscore            mean + z * sd (z = experiment.yaml classification.calibration_z; the proof-of-concept rule).
                    With only n = 5 calibration scores the sd is very uncertain: for one column the chance that a
                    new stable batch exceeds mean + 3 sd is P(T_4 > 3 / sqrt(1 + 1/5)) = 2.6%, and the label
                    takes the maximum over all monitored columns (5-7), so about 15-20% of stable batches were
                    labelled drift in dev run 2 (observed: 18-22% on all three datasets).
  prediction_bound  mean + t(1 - alpha / C, n - 1) * sqrt(1 + 1/n) * sd: one-sided prediction bound for a new
                    score with a Bonferroni split of the batch-level false-drift budget alpha over the C columns
                    (the same construction as the AVH-style comparator). Assumes roughly normal LOO scores; the
                    scores are non-negative and right-skewed, so alpha is approximate.
  min_effect        optional floor per score type ({numeric: w, categorical: js}); a change must also exceed a
                    minimum size to count as drift (practical significance). 0 = no floor.
Degradation thresholds (union_rate and per_check) keep mean + z * sd (sd floored at the binomial sampling sd).
Both are empirical rules, not significance tests (Section 5.5).

Labels: stable (0,0), drift (1,0), degradation (0,1), drift+degradation (1,1); "insufficient" when drift
evidence is insufficient and no degradation is detected.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from adaptation.adaptive_thresholds import adaptive_category_rules, wide_gate_rules
from drift.drift_detector import drift_scores, profile_drift_scores, reference_summary
from execution.local_executor import check_with_reuse, execute
from rules.static_rules import base_rules, check

CLASSES = ["stable", "drift", "degradation", "drift+degradation"]
INSUFFICIENT = "insufficient"
DRIFT_SOURCES = ["gated", "gated_executed", "ungated", "none"]
DEGRADATION_RULES = ["union_rate", "per_check"]


def to_class(drift: bool, degr: bool) -> str:
    return CLASSES[int(drift) + 2 * int(degr)]


def _sig(rule: Dict[str, Any]) -> str:
    keep = {k: v for k, v in rule.items() if k in ("type", "column", "columns", "low", "high", "allowed", "pattern")}
    return json.dumps(keep, sort_keys=True, default=str)


_NULL_CACHE: Dict[str, Dict[str, np.ndarray]] = {}


def _cached_null(pool: pd.DataFrame, gate_pass: np.ndarray, num: List[str], cat: List[str], folds: int, draws: int,
                 seed: int) -> Dict[str, np.ndarray]:
    """Resampled null scores, cached per process: every gated comparator of a run sees the same warm-up records and
    the same gate, so the (deterministic) resampling is done once per run instead of once per comparator."""
    import hashlib
    from drift.drift_detector import resampled_null_scores
    h = hashlib.sha256(pd.util.hash_pandas_object(pool[num + cat], index=False).to_numpy().tobytes())
    h.update(np.packbits(gate_pass).tobytes())
    h.update(repr((num, cat, folds, draws, seed)).encode())
    key = h.hexdigest()
    if key not in _NULL_CACHE:
        if len(_NULL_CACHE) > 64:
            _NULL_CACHE.clear()
        _NULL_CACHE[key] = resampled_null_scores(pool, gate_pass, num, cat, folds, draws,
                                                 np.random.default_rng([seed, 7919]))
    return _NULL_CACHE[key]


class BatchClassifier:
    def __init__(self, cfg: Dict[str, Any], exp: Dict[str, Any], drift_source: str = "gated",
                 degradation_rule: str = "union_rate", z: Optional[float] = None,
                 per_check_z: Optional[float] = None, min_gate_records: int = 50,
                 drift_threshold: Optional[Dict[str, Any]] = None, seed: int = 0) -> None:
        if drift_source not in DRIFT_SOURCES:
            raise ValueError(f"unknown drift source {drift_source!r}")
        if degradation_rule not in DEGRADATION_RULES:
            raise ValueError(f"unknown degradation rule {degradation_rule!r}")
        self.cfg, self.exp = cfg, exp
        self.num, self.cat = cfg["numeric"], cfg["categorical"]
        self.drift_source, self.degradation_rule = drift_source, degradation_rule
        self.z = exp["classification"]["calibration_z"] if z is None else z
        self.per_check_z = self.z if per_check_z is None else per_check_z
        self.min_gate = min_gate_records
        dt = dict(drift_threshold or {})
        self.drift_method = dt.get("method", "zscore")
        if self.drift_method not in ("zscore", "prediction_bound"):
            raise ValueError(f"unknown drift threshold method {self.drift_method!r}")
        self.drift_alpha = float(dt.get("alpha", 0.05))
        self.min_effect = {"numeric": float((dt.get("min_effect") or {}).get("numeric", 0.0)),
                           "categorical": float((dt.get("min_effect") or {}).get("categorical", 0.0))}
        rs = dt.get("resampling") or {}
        self.resample_draws = int(rs.get("draws", 0)) if rs.get("enabled", bool(rs)) else 0
        self.seed = seed
        self.tau_parts: Dict[str, Dict[str, float]] = {}
        self.drift_k: Optional[float] = None
        self.tau_drift: Dict[str, float] = {}
        self.tau_f: float = np.inf
        self.check_base: Dict[str, Dict[str, float]] = {}
        self.calibration: List[Dict[str, Any]] = []
        self.baseline_ref = None

    # ------------------------------------------------------------------ gate + drift
    def gate_rules(self, columns: List[str], history: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # the gate keeps the recurrence-only category rule even when the monitored category checks also keep the
        # warm-up categories (experiment.yaml adaptive.category_keep_warmup): the drift thresholds are calibrated
        # leave-one-out, where a value seen in only one warm-up batch is never "known", so a gate that admitted all
        # warm-up values at evaluation time would be looser than the one the thresholds were calibrated with
        # (dev harness: Retail false drift on Price 4.7% -> 8.7% with the looser gate)
        rules = base_rules(columns, self.num, self.cat, self.cfg.get("nullable", []))
        rules += adaptive_category_rules(history, self.cat, self.exp["adaptive"]["category_min_batches"])
        rules += wide_gate_rules(history, self.num)
        return rules

    def drift_evidence(self, df: pd.DataFrame, profile: Dict[str, Any], baseline: List[Dict[str, Any]],
                       history: List[Dict[str, Any]], executed: Optional[List[Tuple[Dict, np.ndarray]]] = None,
                       reuse: Optional[Dict[str, Any]] = None, cached_reference: bool = True) -> Dict[str, Any]:
        """Drift scores vs the reference profiles ``baseline`` (None per column = insufficient evidence).
        ``cached_reference=False`` when the reference changes from batch to batch (rolling reference)."""
        if self.drift_source == "none":
            return {"scores": {}, "excluded_fraction": 0.0, "gate_coverage": None, "insufficient": False}
        if self.drift_source == "ungated":
            ref = (self.baseline_ref if cached_reference else None) or reference_summary(baseline, self.num, self.cat)
            return {"scores": profile_drift_scores(profile, ref, self.num, self.cat), "excluded_fraction": 0.0,
                    "gate_coverage": None, "insufficient": False}
        done = {_sig(r): m for r, m in (executed or [])}
        # (type, column) of every executed check: a gate rule whose check was executed this batch in a DIFFERENT
        # form still counts as executed and is computed in the gate's own form. Needed since v2.1: the monitored
        # category check also keeps the warm-up categories (category_keep_warmup), the gate keeps the
        # recurrence-only rule (see gate_rules), so the two never match exactly and gated_executed silently lost
        # its whole category gate even with every check executed (dev run 4: gate coverage 0.77-0.84 with the
        # full suite; invalid categories entered the drift scores and degradation batches were labelled drift)
        done_units = {(r.get("type"), r.get("column")) for r, _ in (executed or [])}
        failed = np.zeros(len(df), dtype=bool)
        gate = self.gate_rules(list(df.columns), history)
        used = 0
        for r in gate:
            s = _sig(r)
            if s in done:
                failed |= done[s]; used += 1
            elif self.drift_source == "gated" or r.get("type") == "range" or \
                    (r.get("type"), r.get("column")) in done_units:      # the 6-sigma gate is always computed
                m = check_with_reuse(r, df, reuse)[0] if reuse else check(r, df).fillna(False).to_numpy(dtype=bool)
                failed |= m; used += 1
        passing = df[~failed]
        out = {"excluded_fraction": float(failed.mean()) if len(df) else 0.0,
               "gate_coverage": used / len(gate) if gate else 1.0, "gate_failed": failed}
        if len(passing) < self.min_gate:
            out.update(scores={c: None for c in self.num + self.cat}, insufficient=True)
            return out
        out.update(scores=drift_scores(passing, baseline, self.num, self.cat), insufficient=False)
        return out

    # ------------------------------------------------------------------ calibration
    def drift_multiplier(self, n: int, n_columns: int) -> float:
        """Multiplier of the LOO sd for a drift threshold from n calibration scores (see module docstring)."""
        if self.drift_method == "zscore":
            k = float(self.z)
        else:
            from baselines.avh import t_quantile
            k = float(t_quantile(1.0 - self.drift_alpha / max(n_columns, 1), max(n - 1, 1)) * np.sqrt(1.0 + 1.0 / n))
        self.drift_k = k
        return k

    def calibrate(self, warm_dfs: List[pd.DataFrame], warm_profiles: List[Dict[str, Any]], suite_builder,
                  reuse_list: Optional[List[Dict[str, Any]]] = None) -> None:
        """Leave-one-out over the warm-up batches with the configuration's own suite."""
        rows, check_rates, gate_pass = [], [], []
        n_last = len(warm_dfs[-1])
        for j, (df, prof) in enumerate(zip(warm_dfs, warm_profiles)):
            others = [p for i, p in enumerate(warm_profiles) if i != j]
            rules, _ = suite_builder.build(list(df.columns), others, others, others, prof)
            reuse = reuse_list[j] if reuse_list else None
            res = execute(rules, df, reuse=reuse, return_masks=True)
            executed = [(r, res["masks"][r["key"]]) for r in rules]
            ev = self.drift_evidence(df, prof, others, others, executed, reuse)
            gf = ev.get("gate_failed")
            gate_pass.append(np.ones(len(df), dtype=bool) if gf is None else ~gf)
            rows.append({"failure_rate": res["failure_rate"],
                         **{f"D_{c}": v for c, v in ev["scores"].items()}})
            check_rates.append({p["key"]: p["failures"] / len(df) for p in res["per_rule"]})
        cal = pd.DataFrame(rows)
        self.calibration = rows
        scores = {c[2:]: pd.to_numeric(cal[c], errors="coerce").dropna() for c in cal.columns if c.startswith("D_")}
        usable = {c: x for c, x in scores.items() if len(x) > 1}
        null = {}
        if self.resample_draws and self.drift_source != "none" and usable:
            pool = pd.concat(warm_dfs, ignore_index=True)
            null = _cached_null(pool, np.concatenate(gate_pass), [c for c in self.num if c in usable],
                                [c for c in self.cat if c in usable], len(warm_dfs), self.resample_draws,
                                int(self.seed))
        for c, x in scores.items():
            if c not in usable:
                self.tau_drift[c] = np.inf
                continue
            k = self.drift_multiplier(len(x), len(usable))
            floor = self.min_effect["categorical" if c in self.cat else "numeric"]
            loo = float(x.mean() + k * x.std(ddof=1))
            res = float(np.quantile(null[c], 1.0 - self.drift_alpha / len(usable), method="higher")) \
                if c in null and len(null[c]) else -np.inf
            self.tau_parts[c] = {"loo": loo, "resampled": res}
            self.tau_drift[c] = max(loo, res, floor)
        self.tau_f = float(cal.failure_rate.mean() + self.z * cal.failure_rate.std(ddof=1))
        keys = sorted(set().union(*check_rates))
        for k in keys:
            x = np.array([r.get(k, 0.0) for r in check_rates])
            mu = float(x.mean())
            p = min(max(mu, 1.0 / n_last), 1.0 - 1.0 / n_last)
            sd = max(float(x.std(ddof=1)), float(np.sqrt(p * (1 - p) / n_last)))
            self.check_base[k] = {"mu": mu, "sd": sd}
        self.baseline_ref = reference_summary(warm_profiles, self.num, self.cat)

    # ------------------------------------------------------------------ decision
    def classify(self, drift_ev: Dict[str, Any], exec_result: Dict[str, Any], n_records: int,
                 coverage: float = 1.0, max_staleness: int = 0) -> Dict[str, Any]:
        scores = drift_ev["scores"]
        exceed = {c: (v is not None and c in self.tau_drift and v > self.tau_drift[c]) for c, v in scores.items()}
        d = any(exceed.values())
        insufficient = drift_ev.get("insufficient", False) and not d
        if self.degradation_rule == "union_rate":
            g = exec_result["failure_rate"] > self.tau_f
            fired = []
        else:
            fired = []
            for p in exec_result["per_rule"]:
                b = self.check_base.get(p["key"])
                if b is not None and p["failures"] / max(n_records, 1) > b["mu"] + self.per_check_z * b["sd"]:
                    fired.append(p["key"])
            g = bool(fired)
        label = INSUFFICIENT if (insufficient and not g) else to_class(d, g)
        signal = [v / self.tau_drift[c] for c, v in scores.items()
                  if v is not None and c in self.tau_drift and self.tau_drift[c] > 0]
        return {"label": label, "drift": bool(d), "degradation": bool(g),
                "drift_insufficient": bool(drift_ev.get("insufficient", False)),
                "drifted_columns": ",".join(c for c, v in exceed.items() if v),
                "degraded_checks": ",".join(fired),
                "drift_signal": max(signal) if signal else None,
                "failure_rate": exec_result["failure_rate"], "failure_threshold": self.tau_f,
                "gate_excluded_fraction": drift_ev.get("excluded_fraction"),
                "gate_coverage": drift_ev.get("gate_coverage"),
                "check_coverage": coverage, "max_staleness": max_staleness, "drift_threshold_k": self.drift_k,
                "degradation_evidence": "complete" if coverage >= 1.0 else "partial"}
