"""Scenario engine v2: deterministic in-memory replay of T1..T20 for one dataset, split, seed, scenario.

Every v2 experiment builds its batches here, so paired comparisons always see identical records,
drift and incidents (same dataset, split, seed and scenario -> bit-identical batches). Batches are
generated on demand instead of being stored (15+ scenarios x 15 seeds would not fit on disk); the
ground truth, labels, novelty log and label audit of every batch are kept and can be written out.

Order of operations for batch t (each record receives at most one change, so labels stay exact):

  1. base batch      T1..T20 drawn from the split's record pool (holdout values removed for novelty)
  2. drift           legitimate distribution change (experiment.yaml drift schedule)
  3. valid novelty   real but previously unseen categories (H2b), legitimate, NOT defects
     -> reference snapshot (the batch as it would be without any defect)
  4. invalid novelty well-formed values outside the domain, recurring like the valid ones (defects)
  5. standard defects missing, invalid category, outlier; duplicate SOURCES reserved, not yet copied
  6. incidents        extra defects of selected families (RQ3), sources of duplicates reserved
  7. semantic defects realistic domain violations (Experiment 3b)
  8. contamination   sustained systematic error on a share of records (RQ1)
  9. duplicates      appended LAST, as copies of the final source records (integrity fix, Section 6.2)
 10. label audit     labels verified against the final batch; a mismatch raises

Random streams: np.random.default_rng([seed, dataset index, t, stream]) with one stream per step, so
adding a step to one scenario never changes the draws of another step.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.utils import batch_label
from ingestion.prepare import make_batches
from ingestion.splits import batch_size_for, exclude_values, split_pools
from injection.defect_injector import (GT_COLUMNS, append_duplicates, inject_defects, inject_distribution_shift,
                                       inject_invalid_categories, inject_missing_values, inject_outliers,
                                       inject_semantic_defects)
from injection.label_audit import audit_batch

GT_V2_COLUMNS = GT_COLUMNS + ["incident"]
STREAM = {"drift": 1, "defects": 2, "semantic": 3, "novelty": 4, "contamination": 5, "incident": 10}


@dataclass
class Batch:
    t: int
    label: str
    df: pd.DataFrame
    gt: pd.DataFrame                      # GT_V2_COLUMNS
    novelty: pd.DataFrame                 # record_id, column, value, kind (valid|invalid), batch
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def defective(self) -> set:
        return set(self.gt["record_id"])


def _empty_gt() -> pd.DataFrame:
    return pd.DataFrame(columns=GT_V2_COLUMNS)


def _concat(frames: List[pd.DataFrame], columns: List[str]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return pd.DataFrame(columns=columns)
    out = pd.concat(frames, ignore_index=True)
    for c in columns:
        if c not in out.columns:
            out[c] = ""
    return out[columns]


def resolve_scenario(name: str, protocol: Dict[str, Any], exp: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(protocol["scenarios"][name])
    spec["name"] = name
    spec["drift_schedule"] = exp["schedules"][spec["drift"]]
    spec["defect_schedule"] = exp["schedules"][spec["defects"]]
    spec["semantic_schedule"] = exp["semantic_schedules"][spec["semantic"]] if spec.get("semantic") else None
    prof = copy.deepcopy(exp["defects"])
    prof.update(protocol.get("defect_profiles", {}).get("default", {}))
    if spec.get("defect_profile"):
        prof.update(protocol["defect_profiles"][spec["defect_profile"]])
    spec["defect_cfg"] = prof
    return spec


class ScenarioReplay:
    """Batches of one (dataset, split, seed, scenario). ``batch(t)`` is deterministic and cached."""

    def __init__(self, dataset: str, cfg: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any],
                 pool: pd.DataFrame, ds_index: int, seed: int, scenario: str, split: str = "dev",
                 audit: bool = True, cache: bool = True) -> None:
        self.dataset, self.cfg, self.exp, self.protocol = dataset, cfg, exp, protocol
        self.ds_index, self.seed, self.split = ds_index, seed, split
        self.spec = resolve_scenario(scenario, protocol, exp)
        self.scenario = scenario
        self.n_batches, self.warm = exp["n_batches"], exp["warmup_batches"]
        self.dscen = protocol.get("dataset_scenarios", {}).get(dataset, {})
        self.novelty_specs = self.dscen.get("novelty", []) if self.spec.get("novelty") else []
        holdout: Dict[str, List[str]] = {}
        for nv in self.novelty_specs:
            holdout.setdefault(nv["column"], []).extend(nv["valid"])
        if holdout:
            pool = exclude_values(pool, holdout)
        self.holdout = holdout
        self.batch_size = batch_size_for(len(pool), cfg["batch_size"], self.n_batches)
        self.base = make_batches(pool, self.n_batches, self.batch_size, seed=seed + ds_index)
        # legitimate categorical drift redraws values from the pool's COMMON categories (manuscript Section 6.2):
        # the top_n most frequent with at least category_min_share of the pool. Without the floor, a value seen in
        # 0.01% of records (e.g. Chicago Risk "All") would become about 7% of records at drift level 0.30.
        dcfg = protocol.get("drift", {})
        top_n, min_share = int(dcfg.get("category_top_n", 10)), float(dcfg.get("category_min_share", 0.0))
        # drifting columns: datasets.yaml, overridable per dataset in protocol.yaml dataset_scenarios (a scaled
        # column must stay physically plausible, e.g. route distances cannot grow by 30%)
        self.drift_numeric = list(self.dscen.get("drift_numeric", cfg["drift_numeric"]))
        self.drift_categorical = list(self.dscen.get("drift_categorical", cfg["drift_categorical"]))
        self.top = {}
        for c in self.drift_categorical:
            vc = pool[c].astype(str).value_counts(normalize=True)
            self.top[c] = [v for v in vc.index[:top_n] if vc[v] >= min_share] or vc.index[:1].tolist()
        self.audit = audit
        self._cache: Dict[int, Batch] = {} if cache else None
        self.audits: Dict[int, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ helpers
    def _rng(self, t: int, stream: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, self.ds_index, t, stream])

    def labels(self, t: int) -> Dict[str, Any]:
        s = self.spec
        drift = s["drift_schedule"][t - 1]
        rate = 0.0 if s.get("zero_defects") else s["defect_schedule"][t - 1]
        base_rate = 0.0 if s.get("zero_defects") else s["defect_schedule"][0]
        incident = [i["name"] for i in s.get("incidents", []) if i["start"] <= t <= i["end"]]
        contam = bool(s.get("contamination")) and t >= s["contamination"]["start"]
        sem = s["semantic_schedule"][t - 1] if s["semantic_schedule"] else 0.0
        nov_invalid = any(t >= nv["start"] and nv.get("invalid") for nv in self.novelty_specs)
        # degradation = more erroneous records than the warm-up baseline, whatever their kind: a higher standard
        # defect rate, an incident, contamination, semantic violations or recurring invalid categories
        degr = rate > base_rate or bool(incident) or contam or sem > 0 or nov_invalid
        return {"batch": batch_label(t), "t": t, "drift_level": drift, "defect_rate": rate,
                "semantic_rate": sem, "drift_label": int(drift > 0), "degradation_label": int(degr),
                "incident": ",".join(incident), "contamination": int(contam),
                "semantic_active": int(sem > 0), "novelty_invalid_active": int(nov_invalid),
                "evaluation": int(t > self.warm)}

    @staticmethod
    def _pick(df: pd.DataFrame, n: int, excluded: set, rng: np.random.Generator) -> np.ndarray:
        eligible = np.flatnonzero(~df["record_id"].isin(excluded).to_numpy())
        n = min(n, len(eligible))
        return np.sort(rng.choice(eligible, size=n, replace=False)) if n else np.array([], dtype=int)

    # ------------------------------------------------------------------ build
    def batch(self, t: int) -> Batch:
        if self._cache is not None and t in self._cache:
            return self._cache[t]
        cfg, s = self.cfg, self.spec
        b = batch_label(t)
        lab = self.labels(t)
        df = inject_distribution_shift(self.base[t - 1], lab["drift_level"], self.drift_numeric,
                                       self.drift_categorical, self.top, self._rng(t, STREAM["drift"]))
        df = df.copy()
        used: set = set()
        novelty_rows: List[Dict[str, Any]] = []
        gts: List[pd.DataFrame] = []

        # 3./4. category novelty (H2b): valid values (legitimate) and invalid values (defects), same share
        active = [nv for nv in self.novelty_specs if t >= nv["start"]]
        rng_n = self._rng(t, STREAM["novelty"])
        planned: List[Tuple[str, str, str, np.ndarray]] = []
        for nv in active:
            n_each = int(round(nv["share"] * len(df)))
            for kind in ("valid", "invalid"):
                for value in nv[kind]:
                    rows = self._pick(df, n_each, used, rng_n)
                    used |= set(df.loc[rows, "record_id"])
                    planned.append((nv["column"], value, kind, rows))
        for col, value, kind, rows in planned:
            if kind == "valid":
                df[col] = df[col].astype(object)
                df.loc[rows, col] = value
        reference = df.copy()                               # the batch without any defect
        for col, value, kind, rows in planned:
            ids = df.loc[rows, "record_id"].tolist()
            novelty_rows += [{"record_id": r, "column": col, "value": value, "kind": kind, "batch": b} for r in ids]
            if kind == "invalid":
                df[col] = df[col].astype(object)
                df.loc[rows, col] = value
                gts.append(pd.DataFrame({"record_id": ids, "column": col, "defect_type": "invalid_novel",
                                         "severity": "Medium", "batch": b, "incident": ""}))

        # 5. standard background defects (duplicates deferred)
        dup_sources: List[Tuple[str, str]] = []              # (source record_id, incident name)
        if lab["defect_rate"] > 0:
            df, g, src = inject_defects(df, lab["defect_rate"], cfg["numeric"], cfg["categorical"],
                                        self._rng(t, STREAM["defects"]), b, s["defect_cfg"],
                                        defer_duplicates=True, exclude_ids=used)
            gts.append(g.assign(incident=""))
            used |= set(g["record_id"]) | set(src)
            dup_sources += [(r, "") for r in src]

        # 6. incidents: extra defects of selected families
        for k, inc in enumerate(s.get("incidents", [])):
            if not inc["start"] <= t <= inc["end"]:
                continue
            rng_i = self._rng(t, STREAM["incident"] + k)
            fams = list(inc["families"])
            n_total = int(round(inc["rate"] * len(df)))
            for j, fam in enumerate(fams):
                n = n_total // len(fams) + (1 if j < n_total % len(fams) else 0)
                rows = self._pick(df, n, used, rng_i)
                if not len(rows):
                    continue
                ids = df.loc[rows, "record_id"].tolist()
                used |= set(ids)
                if fam == "duplicate":
                    dup_sources += [(r, inc["name"]) for r in ids]
                    continue
                cols = self._incident_columns(inc, fam)
                if fam == "invalid_category":
                    g = inject_invalid_categories(df, rows, cols, rng_i, b, "Medium")
                elif fam == "missing":
                    g = inject_missing_values(df, rows, cols, rng_i, b, "Medium")
                elif fam == "outlier":
                    g = inject_outliers(df, rows, cols, rng_i, b, tuple(s["defect_cfg"]["outlier_sigma"]),
                                        s["defect_cfg"].get("outlier_high_sigma", 8),
                                        s["defect_cfg"].get("outlier_mild_sigma"))
                else:
                    raise ValueError(f"unknown incident family {fam!r}")
                gts.append(g.assign(incident=inc["name"]))

        # 7. semantic domain violations
        if s["semantic_schedule"] is not None and lab["semantic_rate"] > 0:
            df, g = inject_semantic_defects(df, lab["semantic_rate"], cfg["semantic_defects"],
                                            self._rng(t, STREAM["semantic"]), b, used)
            gts.append(g.assign(incident=""))
            used |= set(g["record_id"])

        # 8. sustained contamination: a systematic error (e.g. a unit error) on a share of records
        if lab["contamination"]:
            c = s["contamination"]
            rng_c = self._rng(t, STREAM["contamination"])
            rows = self._pick(df, int(round(c["fraction"] * len(df))), used, rng_c)
            cols = self.dscen.get("contamination_columns") or cfg["numeric"][:1]
            which = rng_c.choice(cols, size=len(rows))
            for col in sorted(set(which)):
                sel = rows[which == col]
                df[col] = df[col].astype(float)
                df.loc[sel, col] = df.loc[sel, col] * float(self.dscen.get("contamination_factor", c["factor"]))
            ids = df.loc[rows, "record_id"].tolist()
            used |= set(ids)
            gts.append(pd.DataFrame({"record_id": ids, "column": which, "defect_type": "contamination",
                                     "severity": "Systematic", "batch": b, "incident": "contamination"}))

        # 9. duplicates last: copies of the final source records
        if dup_sources:
            df, gd = append_duplicates(df, [r for r, _ in dup_sources], b,
                                       s["defect_cfg"]["severity"]["duplicate"])
            gd["incident"] = [inc for _, inc in dup_sources]
            gts.append(gd)

        gt = _concat(gts, GT_V2_COLUMNS)
        novelty = pd.DataFrame(novelty_rows, columns=["record_id", "column", "value", "kind", "batch"])
        if self.audit:
            a = audit_batch(df, gt, reference)
            self.audits[t] = a
            if not a["ok"]:
                raise RuntimeError(f"{self.dataset}/{self.scenario}/seed {self.seed}/{b}: labels do not match "
                                   f"the final batch: {a['issues']} {a['examples'][:3]}")
        lab.update(records=len(df), defects=len(gt), realized_defect_fraction=len(gt) / len(df) if len(df) else 0.0)
        out = Batch(t=t, label=b, df=df, gt=gt, novelty=novelty, meta=lab)
        if self._cache is not None:
            self._cache[t] = out
        return out

    def _incident_columns(self, inc: Dict[str, Any], fam: str) -> List[str]:
        if inc.get("column") == "sparse":
            return [self.dscen.get("sparse_category_column") or self.cfg["categorical"][0]]
        if inc.get("column"):
            return [inc["column"]]
        if fam == "invalid_category":
            return list(self.cfg["categorical"])
        if fam == "outlier":
            return list(self.cfg["numeric"])
        return list(self.cfg["numeric"]) + list(self.cfg["categorical"])

    def batches(self):
        for t in range(1, self.n_batches + 1):
            yield self.batch(t)

    def ground_truth(self) -> pd.DataFrame:
        return _concat([self.batch(t).gt for t in range(1, self.n_batches + 1)], GT_V2_COLUMNS)

    def label_table(self) -> pd.DataFrame:
        return pd.DataFrame([self.batch(t).meta for t in range(1, self.n_batches + 1)])


class DataSource:
    """Clean data, split pools and replays for the v2 experiments (one object per run)."""

    def __init__(self, datasets: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any],
                 clean_loader=None) -> None:
        from common.utils import data_dir
        self.datasets, self.exp, self.protocol = datasets, exp, protocol
        self._loader = clean_loader or (lambda name: pd.read_parquet(data_dir("clean", f"{name}.parquet")))
        self._pools: Dict[str, Dict[str, pd.DataFrame]] = {}

    def pools(self, dataset: str) -> Dict[str, pd.DataFrame]:
        if dataset not in self._pools:
            sp = self.protocol["splits"]
            self._pools[dataset] = split_pools(self._loader(dataset), sp["dev_fraction"], sp["split_seed"])
        return self._pools[dataset]

    def replay(self, dataset: str, split: str, seed: int, scenario: str, **kw) -> ScenarioReplay:
        idx = list(self.datasets).index(dataset)
        return ScenarioReplay(dataset, self.datasets[dataset], self.exp, self.protocol,
                              self.pools(dataset)[split], idx, seed, scenario, split, **kw)
