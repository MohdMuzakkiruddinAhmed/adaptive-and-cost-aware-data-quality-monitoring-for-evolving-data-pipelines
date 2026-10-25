"""Chronological Chicago replay (supervisor experiment 7, part 2; manuscript Section 6.3).

Replays the RAW City of Chicago food-inspection records (data/original/Food_Inspections.csv, not the cleaned
pool: real missing values, duplicates and odd categories stay in) in time order, one batch per calendar month,
through the frozen monitor. There are no complete error labels, so the outputs are:

  temporal_monthly.csv       alert burden per month (flagged records, share, failing checks), batch label,
                             drift trajectory per column, gate excluded fraction; once per drift reference:
                             "rolling" (previous 5 months: is the data changing NOW?) and "warmup" (Jan-May 2010:
                             how far from the start? after 16 years every month differs, as the dev run showed);
                             each also with the practical-significance floor on drift (column min_effect;
                             protocol.yaml classification.practical_min_effect), a sensitivity analysis
  temporal_label_summary.csv share of drift / degradation labels per year and reference
  temporal_adjudication.csv  a random sample of the records failing the DEGRADED checks of months labelled
                             degradation (rolling reference;
                             at most adjudication_max_rows rows in total), with the checks
                             they failed and empty columns for INDEPENDENT adjudication (error / not an error /
                             unclear). Only adjudicated cases may be called errors in the paper.
  temporal_objective_alarms.csv  EVERY alarm of the frozen monitor (rolling reference), split by check and by an
                             OBJECTIVE verdict where one exists (objective_verdicts below): coordinates inside / outside
                             the city limits (City of Chicago boundary polygon), missing value, exact duplicate, value
                             outside / inside the documented domain of Risk and Results. Facility Type has no closed
                             domain: its category alarms stay "unresolved". The adjudication sample carries the same
                             verdict, so independent human adjudication is needed only for the unresolved rows.
  temporal_change_points.csv documented external events (see below)
  temporal_change_point_check.csv  labels of every monitor variant in the months around each event

An adverse inspection outcome (e.g. Results = "Fail") is a property of the restaurant, NOT a data-quality
error, and must never be counted as one.

Documented event: the City's dataset description notes a "7/1/2018 change to food inspection procedures that
affects the data in this dataset" (catalog.data.gov/dataset/food-inspections). Verified in the raw data (6 Oct
2026): from July 2018 "Pass w/ Conditions" rises from 12% to 54% of inspections and "Pass" falls from 56% to 12%
(Fail unchanged), and violation codes are renumbered (not monitored). It is a KNOWN LEGITIMATE CHANGE: the
expected label is drift in Results, not degradation. temporal_change_point_check.csv reports what each monitor
variant said in the months around it (a real-data check of the drift / degradation separation).

    python -m experiments.v2.temporal_chicago
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.provenance import file_hash, write_manifest  # noqa: E402
from common.utils import data_dir, datasets_config, experiment_config, protocol_config, v2_results_dir  # noqa: E402
from injection.scenario import Batch, GT_V2_COLUMNS  # noqa: E402
from ingestion.loaders import find_files, load_chicago  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402
from rules.manual_rules import manual_rules  # noqa: E402

CHANGE_POINTS = [{"month": "2018-07", "event": "7/1/2018 change to food inspection procedures (City of Chicago note; "
                  "new food code requirements effective 1 July 2018)",
                  "source": "https://catalog.data.gov/dataset/food-inspections; verified in the raw data 6 Oct 2026",
                  "verified_fields": "Results: Pass 56% -> 12%, Pass w/ Conditions 12% -> 54% between June and July 2018 "
                                     "(Fail unchanged at about 18-21%); Violations renumbered from codes 1-44 and 70 to "
                                     "1-63 (not monitored)",
                  "expected_label": "legitimate drift in Results (procedural change), not degradation",
                  "column": "Results"}]


def objective_verdicts(rule, df: pd.DataFrame, mask: np.ndarray, boundary, domains) -> np.ndarray:
    """Objective verdict per record failing ``rule`` (``mask``): "error", "not_error" or "unresolved".
      completeness            the value is missing: error
      uniqueness              exact copy of an earlier record of the month: error
      range on a coordinate   the position is outside the city limits (or not a number): error; inside: not_error
      category / pattern      value outside the documented domain (datasets.yaml documented_domains): error;
                              inside it: not_error; column without a documented domain: unresolved
      any other range         unresolved"""
    rows = df[mask]
    out = np.full(len(rows), "unresolved", dtype=object)
    t, col = rule["type"], rule.get("column")
    if t == "completeness":
        out[:] = "error"
    elif t == "uniqueness":
        out[:] = "error"
    elif t == "range" and col in ("Latitude", "Longitude") and boundary is not None:
        lon = pd.to_numeric(rows["Longitude"], errors="coerce").to_numpy(dtype=float)
        lat = pd.to_numeric(rows["Latitude"], errors="coerce").to_numpy(dtype=float)
        inside = boundary.contains(lon, lat)
        out[:] = np.where(inside, "not_error", "error")
    elif t in ("category", "pattern") and col in (domains or {}):
        dom = set(map(str, domains[col]))
        out[:] = np.where(rows[col].astype(str).isin(dom), "not_error", "error")
    return out


class MonthlyReplay:
    """Raw records in chronological order, one batch per month (same interface as ScenarioReplay)."""

    def __init__(self, raw: pd.DataFrame, cfg, warm: int, date_col: str = "Inspection Date",
                 max_gap_days: int = 4) -> None:
        raw = raw.dropna(subset=[date_col]).sort_values([date_col, "Inspection ID"]).reset_index(drop=True)
        raw["_month"] = raw[date_col].dt.to_period("M").astype(str)
        self.months = sorted(raw["_month"].unique())
        # the extract ends inside its last month (dev run 2: 62 records for 2026-10); a partial month is not
        # comparable with full months, so it is dropped when the last record is more than max_gap_days before the
        # month's end (inspections are on weekdays, so a full month can end up to 3 days early)
        self.dropped_months: list = []
        last = raw[date_col].max()
        month_end = last.to_period("M").end_time.normalize()
        if (month_end - last.normalize()).days > max_gap_days:
            self.dropped_months.append(self.months.pop())
            raw = raw[raw["_month"] != self.dropped_months[-1]]
        self.cfg, self.dataset, self.seed, self.scenario, self.split = cfg, "chicago_food", 0, "temporal", "temporal"
        self.n_batches, self.warm = len(self.months), warm
        self._groups = {m: g.drop(columns="_month").reset_index(drop=True) for m, g in raw.groupby("_month")}

    def batch(self, t: int) -> Batch:
        m = self.months[t - 1]
        df = self._groups[m].copy()
        df.insert(0, "record_id", [f"{m}-{i:05d}" for i in range(len(df))])
        meta = {"batch": m, "t": t, "month": m, "evaluation": int(t > self.warm), "records": len(df),
                "drift_label": np.nan, "degradation_label": np.nan}
        return Batch(t=t, label=m, df=df, gt=pd.DataFrame(columns=GT_V2_COLUMNS),
                     novelty=pd.DataFrame(columns=["record_id", "column", "value", "kind", "batch"]), meta=meta)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2010-01", help="first month")
    ap.add_argument("--with-manual", action="store_true", help="add the manual domain rules to the frozen suite")
    ap.add_argument("--scheduler", default="all")
    args = ap.parse_args(argv)
    datasets, exp, pr = datasets_config(), experiment_config(), protocol_config()
    cfg = datasets["chicago_food"]
    files = find_files(data_dir("original"), cfg["files"])
    if not files:
        sys.exit("Food_Inspections.csv not found under data/original (python scripts/download_data.py)")
    raw = load_chicago(files, cfg)
    raw = raw[raw["Inspection Date"] >= pd.Period(args.start, "M").start_time]
    tcfg = pr["temporal"]
    rep = MonthlyReplay(raw, cfg, tcfg["warmup_batches"], max_gap_days=int(tcfg.get("incomplete_month_gap_days", 4)))
    if rep.dropped_months:
        print(f"[temporal] incomplete last month dropped: {rep.dropped_months}")
    extra = manual_rules("chicago_food", cfg, ["record_id"] + list(raw.columns)) if args.with_manual else []
    refs = list(tcfg.get("drift_references", ["warmup"]))
    floor = dict(pr["classification"].get("drift_threshold", {}), min_effect=pr["classification"]["practical_min_effect"])
    variants = [(ref, False) for ref in refs] + [(ref, True) for ref in refs] if tcfg.get("min_effect_sensitivity", True) \
        else [(ref, False) for ref in refs]
    monitors = {(ref, me): Monitor(cfg, exp, pr, MonitorConfig(
        name=f"frozen_{ref}{'_min_effect' if me else ''}", method="adaptive", extra_rules=extra, scheduler=args.scheduler,
        drift_source="gated", degradation_rule=pr["schedulers"]["degradation_rule"], drift_reference=ref,
        drift_threshold=floor if me else None)) for ref, me in variants}
    out = v2_results_dir("temporal_chicago")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(pr["splits"]["split_seed"])
    from rules.static_rules import check
    from ingestion.geo import CityBoundary
    try:
        boundary = CityBoundary.load()
    except FileNotFoundError as exc:
        print(f"[temporal] WARNING: {exc}; coordinate alarms stay unresolved", flush=True)
        boundary = None
    domains = cfg.get("documented_domains", {})
    monthly, adjud, objective = [], [], []
    adj_ref = refs[0]                       # adjudication sample from the first (rolling) reference
    for t in range(1, rep.n_batches + 1):
        b = rep.batch(t)
        for (ref, me), m in monitors.items():
            st = m.step(b.df, b.label)
            lab = st["label"] or {}
            fails = {p["key"]: p["failures"] for p in st["per_rule"] if p["failures"]}
            monthly.append({"drift_reference": ref, "min_effect": me, "month": b.label, "t": t, "warmup": st["warmup"],
                            "records": len(b.df), "flagged": len(st["failed_ids"]),
                            "flagged_share": len(st["failed_ids"]) / max(len(b.df), 1),
                            "label": lab.get("label"), "drifted_columns": lab.get("drifted_columns"),
                            "degraded_checks": lab.get("degraded_checks"),
                            "gate_excluded_fraction": lab.get("gate_excluded_fraction"),
                            "change_point": any(cp["month"] == b.label for cp in CHANGE_POINTS),
                            **{f"fail:{k}": v for k, v in fails.items()},
                            **{f"D_{c}": v for c, v in (lab.get("drift_scores") or {}).items()}})
            if ref == adj_ref and not me and not st["warmup"]:
                for x in m.last_suite:
                    mask = check(x, b.df).fillna(False).to_numpy(dtype=bool)
                    if not mask.any():
                        continue
                    v = pd.Series(objective_verdicts(x, b.df, mask, boundary, domains)).value_counts()
                    objective.append({"month": b.label, "label": lab.get("label"), "check": x["key"],
                                      "check_type": x["type"], "source": x.get("source"),
                                      "degraded_check": x["key"] in str(lab.get("degraded_checks") or "").split(","),
                                      "alarms": int(mask.sum()), **{f"verdict_{k}": int(n) for k, n in v.items()}})
            if ref != adj_ref or me or st["warmup"] or "degradation" not in str(lab.get("label")) or not st["failed_ids"]:
                continue
            # sample among records failing the checks that made the batch "degradation" (union_rate labels have
            # no per-check evidence: then any flagged record)
            keys = [k for k in str(lab.get("degraded_checks") or "").split(",") if k]
            rules = [x for x in m.last_suite if x.get("key") in keys]
            if rules:
                mask = np.zeros(len(b.df), dtype=bool)
                for x in rules:
                    mask |= check(x, b.df).fillna(False).to_numpy(dtype=bool)
                ids = sorted(b.df.loc[mask, "record_id"])
            else:
                ids = sorted(st["failed_ids"])
            if not ids:
                continue
            pick = set(rng.choice(ids, size=min(tcfg["adjudication_sample"], len(ids)), replace=False))
            sub = b.df[b.df.record_id.isin(pick)]
            for _, r in sub.iterrows():
                one = sub[sub.record_id == r.record_id]
                failed = [x for x in m.last_suite if x["type"] != "uniqueness" and bool(check(x, one).fillna(False).iloc[0])]
                verdicts = [objective_verdicts(x, one, np.ones(1, dtype=bool), boundary, domains)[0] for x in failed]
                verdict = "error" if "error" in verdicts else ("unresolved" if "unresolved" in verdicts or not verdicts
                                                                else "not_error")
                if not failed:                       # flagged only as a duplicate of an earlier record
                    verdict = "error"
                adjud.append({"month": b.label, "batch_label": lab.get("label"), "degraded_checks": lab.get("degraded_checks"),
                              **r.to_dict(), "failed_checks": ";".join(x["key"] for x in failed) or
                              "uniqueness (duplicate of an earlier record)", "objective_verdict": verdict,
                              "adjudicated": "", "adjudicator": "", "notes": ""})
    mo = pd.DataFrame(monthly)
    mo.to_csv(out / "temporal_monthly.csv", index=False)
    ev = mo[~mo.warmup.astype(bool)].assign(year=lambda d: d.month.str[:4])
    ev.groupby(["drift_reference", "min_effect", "year"]).agg(
        months=("month", "count"), mean_flagged_share=("flagged_share", "mean"),
        drift_share=("label", lambda x: x.astype(str).str.contains("drift").mean()),
        degradation_share=("label", lambda x: x.astype(str).str.contains("degradation").mean())) \
        .reset_index().to_csv(out / "temporal_label_summary.csv", index=False)
    adj = pd.DataFrame(adjud)
    cap = int(tcfg.get("adjudication_max_rows", 300))
    if len(adj) > cap:                       # keep adjudication feasible: random subsample, sorted by month
        adj = adj.iloc[np.sort(rng.choice(len(adj), size=cap, replace=False))]
    adj.to_csv(out / "temporal_adjudication.csv", index=False)
    obj = pd.DataFrame(objective).fillna(0)
    obj.to_csv(out / "temporal_objective_alarms.csv", index=False)
    if len(obj):
        vcols = [c for c in obj.columns if c.startswith("verdict_")]
        summ = obj.groupby(["check_type", "check", "degraded_check"])[["alarms"] + vcols].sum().reset_index()
        summ.to_csv(out / "temporal_objective_summary.csv", index=False)
        print(summ.to_string(index=False))
    pd.DataFrame(CHANGE_POINTS).to_csv(out / "temporal_change_points.csv", index=False)
    checks = []
    for cp in CHANGE_POINTS:
        months = pd.period_range(pd.Period(cp["month"]) - 2, pd.Period(cp["month"]) + 3, freq="M").astype(str)
        for (ref, me), g in mo[mo.month.isin(months)].groupby(["drift_reference", "min_effect"]):
            for _, r in g.iterrows():
                drifted = str(r.drifted_columns or "")
                checks.append({"change_point": cp["month"], "drift_reference": ref, "min_effect": me, "month": r.month,
                               "label": r.label, "drift_in_changed_column": cp["column"] in drifted.split(","),
                               "drifted_columns": r.drifted_columns, "degraded_checks": r.degraded_checks,
                               f"D_{cp['column']}": r.get(f"D_{cp['column']}")})
    pd.DataFrame(checks).to_csv(out / "temporal_change_point_check.csv", index=False)
    write_manifest(out / "manifest.json", "temporal_chicago", "temporal", [], {"experiment": exp, "protocol": pr},
                   {"raw_file": str(files[0]), "raw_sha256": file_hash(files[0])}, {"tables": str(out)},
                   {"months": rep.n_batches, "dropped_months": rep.dropped_months, "with_manual": args.with_manual,
                    "drift_references": refs})
    print(mo[(mo.drift_reference == adj_ref) & ~mo.min_effect][["month", "records", "flagged_share", "label"]]
          .tail(24).to_string(index=False))
    print(ev.groupby(["drift_reference", "min_effect"]).label.apply(lambda x: x.astype(str).str.contains("drift").mean())
          .rename("drift_share").to_string())


if __name__ == "__main__":
    main()
