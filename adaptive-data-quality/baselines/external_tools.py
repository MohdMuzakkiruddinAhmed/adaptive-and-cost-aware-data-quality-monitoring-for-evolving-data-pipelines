"""Comparators from established data-validation tools: Amazon Deequ and TensorFlow Data Validation (TFDV).

The tools run as sidecars (tools/external_validators/) on the exchanged replay batches
(experiments/v2/tool_exchange.py). This module turns their RAW outputs into the two things the experiments
compare:

1. Deequ at record level (H1a, rq1_thresholds methods "deequ_*", protocol.yaml external_tools.rq1_methods).
   Deequ's ConstraintSuggestionRunner proposes constraints from the warm-up; Deequ itself then VERIFIES them on
   every evaluation batch and reports, per constraint, the batch-level status and the violating records (its
   row-level results; tools/external_validators/run_deequ.py). Nothing is re-implemented: a record is flagged
   exactly when Deequ says it violates a constraint (deequ_flags), under one of two readings:
     failed  the record violates a constraint whose batch-level assertion FAILED (operational reading)
     any     the record violates a constraint's row-level predicate, whatever the assertion (e.g. every missing
             value under hasCompleteness(c, _ >= 0.99))
   Two suggestion variants (external_tools.deequ.suggestion_variants):
     observed_default  warm-up as observed (with its ~1% erroneous records), DEFAULT rule set: what a practitioner
                       gets. Dev run 3: the erroneous records suppress most suggestions (a column with one missing
                       value gets hasCompleteness instead of isComplete, a category list is not suggested once an
                       invalid value is present).
     clean_extended    warm-up WITHOUT its erroneous records, Deequ's EXTENDED rules without NUMERICAL
                       (DEFAULT+STRING+COMMON): the tool's best case, an optimistic upper bound (no other method is
                       given clean data). NUMERICAL is left out because its hasMin/hasMax/hasMean/hasStandardDeviation
                       assert EQUALITY with the warm-up value, which no other batch meets, and its HasMean rule uses
                       the column maximum as the mean (Deequ 2.0.x, suggestions/rules/HasMean.scala).
   "+base" adds this project's base checks (completeness, type, uniqueness, domain ranges) to Deequ's flags.
   With these rule sets Deequ suggests no numeric bounds beyond non-negativity (string lengths are bounded in the
   best case), so it finds few numeric outliers: that is the tool's behaviour.
   translate_constraint / deequ_rules below map suggestions to this project's rules; they are kept only for the
   suggestion report (rq1a_deequ_constraints.csv), not for scoring.

2. Batch labels (H1b, rq1_classification comparators), mapped to the four classes in advance:
   deequ_*         one Deequ anomaly check per metric under an anomaly configuration
                   (external_tools.deequ.anomaly_configs): Deequ's documented set-up (OnlineNormalStrategy, defaults,
                   cumulative history) or the warm-up reference with the deviation factor calibrated like the
                   proposed method (BatchNormalStrategy, Student-t prediction factor, Bonferroni over the metrics).
                   Distribution metrics (mean, sd, min, max, median, distinct count, entropy) -> drift; quality
                   metrics (size, completeness, uniqueness) -> degradation; the same split as the AVH-style
                   comparator. Optionally a failed suggested constraint (Deequ verification) is a second
                   degradation signal.
   tfdv_*          drift = some feature's TFDV drift distance (L-infinity categorical, approximate JS divergence
                   numeric) above its threshold; degradation = some TFDV schema anomaly (presence, type, shape,
                   unexpected string values) against the schema TFDV inferred from the warm-up.
                   reference "previous" (TFDV's documented set-up: consecutive spans) or "warmup";
                   thresholds "documented" (protocol.yaml external_tools.tfdv.documented_thresholds, TFDV's
                   inferred schema as is) or "calibrated" (the proposed method's prediction bound applied to
                   TFDV's own warm-up distances, and to the presence and domain-mass tolerances of the schema:
                   TFDV's defaults flag ANY missing value in an always-present feature and ANY unseen string, so
                   with a normal base error rate every batch would be "degradation").
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_STR = r'"((?:[^"\\]|\\.)*)"'


def _unescape(s: str) -> str:
    return re.sub(r"\\(.)", r"\1", s)


def _strings(text: str) -> List[str]:
    return [_unescape(s) for s in re.findall(_STR, text)]


def _array_values(code: str) -> Optional[List[str]]:
    """String literals of the first Array(...) in Deequ's Scala code; quotes and parentheses inside values are
    handled by scanning literal by literal."""
    i = code.find("Array(")
    if i < 0:
        return None
    i += len("Array(")
    vals, lit = [], re.compile(_STR)
    while i < len(code):
        while i < len(code) and code[i] in " ,\n\t":
            i += 1
        if i >= len(code) or code[i] == ")":
            return vals
        m = lit.match(code, i)
        if not m:
            return None
        vals.append(_unescape(m.group(1)))
        i = m.end()
    return None


def translate_constraint(code: str, columns: Dict[str, str], numeric: List[str]) -> Optional[Dict[str, Any]]:
    """One Deequ suggestion (Scala code) -> a rule of this project, or None when it has no record-level form."""
    code = code.strip()
    m = re.match(r"\.(\w+)\(", code)
    if not m:
        return None
    op = m.group(1)
    lits = _strings(code)
    if not lits:
        return None
    col = columns.get(lits[0], lits[0])
    if op == "isComplete":
        return {"type": "completeness", "column": col}
    if op == "isNonNegative":
        return {"type": "range", "column": col, "low": 0.0, "high": None}
    if op == "isUnique":
        return {"type": "uniqueness", "columns": [col]}
    if op == "isContainedIn":
        if re.search(r"\)\s*,\s*_\s*[<>=]", code):          # fractional variant: a batch-level tolerance
            return None
        values = _array_values(code)
        return {"type": "category", "column": col, "allowed": sorted(set(values))} if values else None
    if op == "hasDataType":
        if re.search(r"ConstrainableDataTypes\.(Integral|Fractional|Numeric)", code):
            return {"type": "type", "column": col, "dtype": "numeric"}
        return None
    return None


def deequ_rules(suggestions: Dict[str, Any], numeric: List[str]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    cols = suggestions.get("columns", {})
    rules, report = [], {"translated": [], "batch_level_only": [], "untranslated": []}
    seen = set()
    for s in suggestions.get("suggestions", []):
        code = s.get("code_for_constraint", "")
        r = translate_constraint(code, cols, numeric)
        if r is None:
            key = "batch_level_only" if (".hasCompleteness(" in code or ".hasDataType(" in code or
                                         (".isContainedIn(" in code and re.search(r"_\s*[<>=]", code))) else "untranslated"
            report[key].append(code)
            continue
        sig = json.dumps(r, sort_keys=True)
        if sig not in seen:
            seen.add(sig)
            rules.append(dict(r, source="deequ", deequ_code=code))
            report["translated"].append(code)
    return rules, report


# ----------------------------------------------------------------------------------------------- Deequ: record level
def deequ_flags(verify: Dict[str, Any], variant: str, reading: str) -> Dict[int, Dict[str, Any]]:
    """Deequ's own verdicts per evaluation batch for one suggestion variant:
      positions    row numbers of the records Deequ reports as violating, under ``reading``
                   "failed": a record counts when it violates a constraint that FAILED on the batch (the operational
                             reading: a failed check sends the engineer to its violating rows)
                   "any":    a record counts when it violates any suggested constraint's row-level predicate, whether
                             or not the batch-level assertion failed (e.g. every missing value under hasCompleteness)
      failed       number of suggested constraints whose batch-level status is not Success
      constraints  number of suggested constraints
      rows         rows Deequ saw (checked against the replay batch)"""
    if reading not in ("failed", "any"):
        raise ValueError(f"unknown Deequ reading {reading!r}")
    out = {}
    for b in verify["batches"]:
        status = b["status"].get(variant, [])
        fails = b["fails"].get(variant, [])
        pos = set()
        for st, rows in zip(status, fails):
            if reading == "any" or st != "Success":
                pos.update(int(r) for r in rows)
        out[int(b["t"])] = {"positions": pos, "failed": sum(st != "Success" for st in status),
                            "constraints": len(status), "rows": int(b["rows"]),
                            "failed_constraints": " | ".join(c for c, st in zip(verify["variants"][variant]["constraints"],
                                                                                 status) if st != "Success")}
    return out


# ----------------------------------------------------------------------------------------------- batch labels
_NOT_READY = ("previous results", "There have to be", "requirement failed", "Can't execute")


def deequ_labels(series: Dict[str, Any], config: str = "default",
                 constraint_failures: Optional[Dict[int, int]] = None) -> Dict[int, Dict[str, Any]]:
    """Four-class labels from Deequ's anomaly checks under anomaly configuration ``config``: a distribution metric
    flagged -> drift, a quality metric flagged -> degradation. ``constraint_failures`` (t -> number of failed
    suggested constraints) adds Deequ's verification result as a second degradation signal."""
    out = {}
    for b in series["batches"]:
        checks = b["checks"][config] if isinstance(b["checks"], dict) else b["checks"]   # old single-config format
        fired, errors = [], []
        for c in checks:
            if c.get("constraint_status") == "Success":
                continue
            msg = str(c.get("message") or "")
            if any(s in msg for s in _NOT_READY):
                errors.append(c["metric"])
                continue
            fired.append((c["metric"], c["group"]))
        t = int(b["t"])
        cf = (constraint_failures or {}).get(t, 0)
        out[t] = {"drift": any(g == "distribution" for _, g in fired),
                  "degradation": any(g == "quality" for _, g in fired) or cf > 0,
                  "fired": ",".join(k for k, _ in fired) + (f",constraints:{cf}" if cf else ""),
                  "not_ready": ",".join(errors)}
    return out


def _calibrated(calibration: List[Dict[str, float]], kinds: Dict[str, str], alpha: float) -> Dict[str, float]:
    from baselines.avh import t_quantile
    feats = list(kinds)
    C = len(feats)
    tau = {}
    for f in feats:
        x = np.array([row.get(f, 0.0) for row in calibration], dtype=float)
        n = len(x)
        if n < 2:
            tau[f] = np.inf
            continue
        k = t_quantile(1.0 - alpha / max(C, 1), n - 1) * np.sqrt(1.0 + 1.0 / n)
        tau[f] = float(x.mean() + k * x.std(ddof=1))
    return tau


def tfdv_thresholds(series: Dict[str, Any], reference: str, thresholds: str, documented: Dict[str, float],
                    alpha: float) -> Dict[str, float]:
    kinds = series["feature_kind"]
    if thresholds == "documented":
        return {f: documented["categorical_linf" if k == "categorical" else "numeric_jsd"] for f, k in kinds.items()}
    if thresholds == "calibrated":
        cal = series["loo_warmup"] if reference == "warmup" else series["consecutive_warmup"]
        return _calibrated(cal, kinds, alpha)
    raise ValueError(f"unknown TFDV threshold mode {thresholds!r}")


# TFDV anomaly types whose tolerance the calibrated variant sets from the warm-up (the rest keep TFDV's schema)
_TOLERANCE_TYPES = {"FEATURE_TYPE_LOW_FRACTION_PRESENT", "INVALID_FEATURE_SHAPE", "ENUM_TYPE_UNEXPECTED_STRING_VALUES"}


def _quality_bounds(cal: List[Dict[str, Any]], alpha: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Lower bounds of the present fraction and upper bounds of the out-of-domain fraction per feature, from the
    leave-one-out warm-up values: prediction bound with the budget split over all bounds; the sd is floored at
    the binomial sampling sd of a batch, so a column without any missing value in the warm-up still tolerates
    sampling noise."""
    from baselines.avh import t_quantile
    pres = sorted({f for r in cal for f in r["present_fraction"]})
    unexp = sorted({f for r in cal for f in r["unexpected_fraction"]})
    C = max(len(pres) + len(unexp), 1)
    n = len(cal)
    k = t_quantile(1.0 - alpha / C, max(n - 1, 1)) * np.sqrt(1.0 + 1.0 / n)
    m = float(np.mean([r["records"] for r in cal]))

    def bound(x, sign):
        x = np.asarray(x, dtype=float)
        p = min(max(float(x.mean()), 1.0 / m), 1.0 - 1.0 / m)
        sd = max(float(x.std(ddof=1)) if n > 1 else 0.0, float(np.sqrt(p * (1 - p) / m)))
        return float(x.mean() + sign * k * sd)

    lo = {f: bound([r["present_fraction"].get(f, 1.0) for r in cal], -1) for f in pres}
    hi = {f: bound([r["unexpected_fraction"].get(f, 0.0) for r in cal], +1) for f in unexp}
    return lo, hi


def tfdv_labels(series: Dict[str, Any], reference: str, thresholds: str, documented: Dict[str, float],
                alpha: float = 0.05) -> Dict[int, Dict[str, Any]]:
    """documented: TFDV's inferred schema as is (any missing value in an always-present feature, any value
    outside an inferred string domain is an anomaly) and the documented drift threshold.
    calibrated: drift thresholds AND the presence / domain-mass tolerances calibrated on the warm-up."""
    if reference not in ("previous", "warmup"):
        raise ValueError(f"unknown TFDV reference {reference!r}")
    tau = tfdv_thresholds(series, reference, thresholds, documented, alpha)
    names = series.get("columns", {})
    if thresholds == "calibrated":
        lo, hi = _quality_bounds(series["loo_warmup_quality"], alpha)
    out = {}
    for b in series["batches"]:
        dist = b["distance_previous" if reference == "previous" else "distance_warmup"] or {}
        drifted = [names.get(f, f) for f, d in dist.items() if d > tau.get(f, np.inf)]
        if thresholds == "calibrated":
            anomalies = sorted({f"{names.get(a['feature'], a['feature'])}:{a['type']}" for a in b["schema_anomalies"]
                                if a["type"] not in _TOLERANCE_TYPES})
            anomalies += [f"{names.get(f, f)}:present<{lo[f]:.4f}" for f, v in b["present_fraction"].items()
                          if f in lo and v < lo[f]]
            anomalies += [f"{names.get(f, f)}:unexpected>{hi[f]:.4f}" for f, v in b["unexpected_fraction"].items()
                          if f in hi and v > hi[f]]
        else:
            anomalies = sorted({f"{names.get(a['feature'], a['feature'])}:{a['type']}" for a in b["schema_anomalies"]})
        out[int(b["t"])] = {"drift": bool(drifted), "degradation": bool(anomalies),
                            "fired": ",".join(drifted + anomalies), "not_ready": ""}
    return out


# ----------------------------------------------------------------------------------------------- loading
def load_series(exchange: Path, tool: str, dataset: str, seed: int, scenario: str) -> Optional[Dict[str, Any]]:
    p = exchange / tool / "series" / dataset / f"seed{seed}" / f"{scenario}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_suggestions(exchange: Path, warm_hash: str, variant: str) -> Optional[Dict[str, Any]]:
    p = exchange / "deequ" / "suggestions" / variant / f"{warm_hash}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def load_verify(exchange: Path, dataset: str, seed: int, scenario: str) -> Optional[Dict[str, Any]]:
    p = exchange / "deequ" / "verify" / dataset / f"seed{seed}" / f"{scenario}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
