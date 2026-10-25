"""Validator stress test (v2.1): does rule admission keep deliberately broken rules out, and what would they cost?

The LLM experiments show which proposals the validator admits, but not how it behaves on rules that are KNOWN to be
wrong. Here a corpus of candidates with a known defect is built per run from the run's own warm-up (so every
candidate is plausible-looking for that data), passed through the same admission code as the LLM proposals
(rules/validator.py validate, both redundancy modes are cheap, "logical" is reported), and every candidate is then
executed on the evaluation batches T6..T20 twice: as written (no validation) and as admitted (if admitted, possibly
relaxed). Nothing here uses labels except the final scoring.

Classes of candidates (expected outcome in brackets):
  malformed   unknown column, unknown rule type, min > max, non-numeric bounds, empty category list, regular
              expression that does not compile, range on a text column                         [invalid]
  too_strict  range = warm-up interquartile range, range = 10th-90th percentile, category = three most common values,
              pattern matching only the most common value                                       [rejected / modified]
  wrong       range entirely above the warm-up maximum, category list of fabricated values      [rejected / modified]
  subtle      range = 1st-99th percentile, category without the values under 1% share           [modified]
  sound       wide range around the warm-up support, category = all warm-up values, completeness, type   [accepted]
  vacuous     range (-1e12, 1e12), pattern ".*"                                                 [accepted, harmless]
  redundant   exact copy of a sound rule, a range wider than an admitted one                    [redundant]

Measures per candidate: status; false-positive rate on valid evaluation records and recall on erroneous ones, for the
raw candidate and for the admitted rule. Summary: per class, share of each status and the false alarms the
validator prevented (raw minus admitted, per 10,000 valid records).

    python -m experiments.v2.validator_stress --split dev
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.v2.common import Context, data_source, parser, run_units  # noqa: E402
from rules.static_rules import check  # noqa: E402
from rules.validator import normalise, validate  # noqa: E402

EXPERIMENT = "validator_stress"
SCENARIOS = ["A_stable", "B_drift"]


def corpus(cfg: Dict[str, Any], warm: pd.DataFrame, rng: np.random.Generator) -> List[Dict[str, Any]]:
    """Candidates in the LLM proposal format, each with "stress_class" and "stress_kind"."""
    out: List[Dict[str, Any]] = []

    def add(cls, kind, **rule):
        out.append(dict(rule, stress_class=cls, stress_kind=kind, confidence=round(float(rng.uniform(0.3, 0.95)), 3)))

    num, cat = cfg["numeric"], cfg["categorical"]
    for c in num:
        x = pd.to_numeric(warm[c], errors="coerce").dropna()
        if len(x) < 20:
            continue
        q = x.quantile([0.01, 0.10, 0.25, 0.75, 0.90, 0.99]).to_dict()
        lo, hi = float(x.min()), float(x.max())
        span = (hi - lo) or 1.0
        add("too_strict", "range_iqr", column=c, rule_type="range", min=q[0.25], max=q[0.75])
        add("too_strict", "range_p10_p90", column=c, rule_type="range", min=q[0.10], max=q[0.90])
        add("wrong", "range_above_max", column=c, rule_type="range", min=hi + span, max=hi + 2 * span)
        add("subtle", "range_p1_p99", column=c, rule_type="range", min=q[0.01], max=q[0.99])
        add("sound", "range_wide", column=c, rule_type="range", min=lo - 0.5 * span, max=hi + 0.5 * span)
        add("sound", "type", column=c, rule_type="type")
        add("vacuous", "range_huge", column=c, rule_type="range", min=-1e12, max=1e12)
        add("redundant", "range_wider_than_sound", column=c, rule_type="range", min=lo - span, max=hi + span)
        add("malformed", "range_min_gt_max", column=c, rule_type="range", min=hi, max=lo - 1)
        add("malformed", "range_non_numeric", column=c, rule_type="range", min="low", max="high")
    for c in cat:
        vc = warm[c].dropna().astype(str).value_counts(normalize=True)
        if not len(vc):
            continue
        add("too_strict", "category_top3", column=c, rule_type="category", allowed_values=list(vc.index[:3]))
        add("too_strict", "pattern_most_common_only", column=c, rule_type="pattern",
            pattern=__import__("re").escape(str(vc.index[0])))
        add("wrong", "category_fabricated", column=c, rule_type="category",
            allowed_values=[f"ZZ_{i}_{c[:3].upper()}" for i in range(5)])
        add("subtle", "category_drop_under_1pct", column=c, rule_type="category",
            allowed_values=list(vc[vc >= 0.01].index) or list(vc.index[:1]))
        add("sound", "category_all_warm", column=c, rule_type="category", allowed_values=list(vc.index))
        add("sound", "completeness", column=c, rule_type="completeness")
        add("vacuous", "pattern_anything", column=c, rule_type="pattern", pattern=".*")
        add("redundant", "copy_of_sound_category", column=c, rule_type="category", allowed_values=list(vc.index))
        add("malformed", "category_empty", column=c, rule_type="category", allowed_values=[])
        add("malformed", "pattern_bad_regex", column=c, rule_type="pattern", pattern="([A-Z")
        add("malformed", "range_on_text", column=c, rule_type="range", min=0, max=1)
    add("malformed", "unknown_column", column="no_such_column", rule_type="range", min=0, max=1)
    add("malformed", "unknown_rule_type", column=(num + cat)[0], rule_type="sql", condition="SELECT 1")
    # sound rules first in the admission order so that the redundant copies are judged against them
    for r in out:
        if r["stress_class"] == "sound":
            r["confidence"] = 0.99
        if r["stress_class"] == "redundant":
            r["confidence"] = 0.2
    return out


def _score(rule: Dict[str, Any], batches) -> Dict[str, float]:
    fp = n = tp = p = 0
    for b in batches:
        m = check(rule, b.df).fillna(False).to_numpy(dtype=bool)
        bad = b.df["record_id"].isin(b.defective).to_numpy()
        fp += int((m & ~bad).sum()); n += int((~bad).sum())
        tp += int((m & bad).sum()); p += int(bad.sum())
    return {"fpr": fp / n if n else np.nan, "recall": tp / p if p else np.nan, "fp": fp, "valid": n, "tp": tp}


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp = src.datasets[ds], src.exp
    max_rate = exp["validation"]["max_warmup_failure_rate"]
    rows = []
    for scen in payload["scenarios"]:
        rep = src.replay(ds, payload["split"], seed, scen)
        warm_b = [rep.batch(t).df for t in range(1, rep.warm + 1)]
        evalb = [rep.batch(t) for t in range(rep.warm + 1, rep.n_batches + 1)]
        warm = pd.concat(warm_b, ignore_index=True)
        cands = corpus(cfg, warm, np.random.default_rng([seed, 31337]))
        cols = list(warm_b[0].columns)
        val = validate([{k: v for k, v in c.items() if not k.startswith("stress_")} for c in cands], cfg, warm_b,
                       max_rate, "logical")
        for c, v in zip(cands, val):
            raw, _ = normalise({k: x for k, x in c.items() if not k.startswith("stress_")}, cfg, cols)
            row = {"dataset": ds, "seed": seed, "scenario": scen, "class": c["stress_class"], "kind": c["stress_kind"],
                   "column": c.get("column"), "status": v["status"], "stage": v["stage"], "detail": v["detail"],
                   "warmup_failure_rate": v["warmup_failure_rate"]}
            if raw is not None:
                s = _score(raw, evalb)
                row.update({f"raw_{k}": x for k, x in s.items()})
            if v["status"] in ("accepted", "modified"):
                s = _score(v["rule"], evalb)
                row.update({f"admitted_{k}": x for k, x in s.items()})
            else:
                row.update(admitted_fpr=0.0, admitted_recall=0.0, admitted_fp=0, admitted_tp=0,
                           admitted_valid=row.get("raw_valid"))
            rows.append(row)
    return pd.DataFrame(rows)


def main(argv=None) -> None:
    args = parser(__doc__).parse_args(argv)
    if not args.scenarios:
        args.scenarios = SCENARIOS
    ctx = Context(EXPERIMENT, args)
    raw = run_units(ctx, worker, {})
    raw.to_csv(ctx.out / "validator_stress_detail.csv", index=False)
    st = pd.crosstab([raw.dataset, raw["class"]], raw.status, normalize="index").reset_index()
    st.to_csv(ctx.out / "validator_stress_status.csv", index=False)
    g = raw.groupby(["dataset", "class"]).agg(candidates=("status", "size"), raw_fp=("raw_fp", "sum"),
                                              admitted_fp=("admitted_fp", "sum"), valid=("raw_valid", "sum"),
                                              raw_tp=("raw_tp", "sum"), admitted_tp=("admitted_tp", "sum")).reset_index()
    g["raw_fp_per_10k"] = 1e4 * g.raw_fp / g.valid
    g["admitted_fp_per_10k"] = 1e4 * g.admitted_fp / g.valid
    g["prevented_fp_per_10k"] = g.raw_fp_per_10k - g.admitted_fp_per_10k
    g.to_csv(ctx.out / "validator_stress_summary.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)})
    print(st.round(3).to_string(index=False))
    print(g[["dataset", "class", "candidates", "raw_fp_per_10k", "admitted_fp_per_10k"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
