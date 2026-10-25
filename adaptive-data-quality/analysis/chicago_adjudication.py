"""Adjudication of the Chicago chronological replay (supervisor experiment 7, part 2).

Problem     The frozen replay (experiments/v2/temporal_chicago.py) gives every alarm an OBJECTIVE verdict where one
            exists (missing value, duplicate, coordinates outside the city, documented domain of Risk and Results).
            Facility Type is free text with no documented domain, so its category alarms (3,276 in the test run)
            and the 10 Facility Type rows of the adjudication sample stayed "unresolved", and the sample columns for
            independent adjudication were never filled.
Fix         Adjudicate VALUES, not records: every alarm on Facility Type is the same question ("is this value an
            error?") asked about one of a few hundred distinct strings. One codebook row per distinct flagged value
            resolves every alarm and every sample row that carries it.

    python -m analysis.chicago_adjudication prepare     # replay the frozen monitor once, write the codebook
    (fill adjudicator_1_verdict in the codebook: error / not_error / unclear; optionally a second adjudicator)
    python -m analysis.chicago_adjudication apply       # resolved tables (pending values stay "pending")
    python -m analysis.chicago_adjudication apply --provisional   # use the pre-labels where no human verdict yet

Codebook    results/v2/temporal_chicago/adjudication/facility_type_codebook.csv, one row per distinct flagged value:
            alarms (evaluation months, rolling reference), degraded-check alarms, months flagged, occurrences in
            the raw file, the nearest frequent value, a deterministic RULE pre-label with its reason, a separate
            model SUGGESTION for the values no rule decides (column suggested_verdict, source in suggestion_by),
            and empty columns for human adjudicators. Rerunning prepare keeps every filled human column.

Pre-label rules (deterministic; the paper may report them, but only human verdicts may be called adjudicated)
    R1 placeholder    empty after normalisation, at most 2 characters, or a placeholder word (na, n/a, none,
                      null, unknown, tbd, test): error ("not a facility type"). Digits only (e.g. "1023", a CDPH
                      license code): left to a human
    R2 format variant same text as a frequent value after normalisation (case, punctuation, & = and):
                      not_error, format_variant = True (a representation inconsistency, reported separately)
    R3 spacing/plural same text as a frequent value once spaces and plural "s" endings are ignored
                      ("Day Care Combo (1586)", "MOBILE FROZEN DESSERT VENDOR"): not_error, format_variant = True
    R4 misspelling    optimal-string-alignment distance on that key: 1 (key shorter than 10 characters) or 1-2
                      (longer), key at least 6 characters: error ("probable misspelling of X")
    R5 combination    every part (split at / , & + and) is a frequent value or passes R2-R4 as not_error:
                      not_error ("combination of known types")
    otherwise         unclear: needs a human (a model suggestion is provided separately)
    Frequent value    occurs at least FREQUENT_MIN (100) times in the raw file.

Outputs of apply (never overwrite the frozen replay outputs)
    adjudication/facility_type_alarm_verdicts.csv     Facility Type alarms by final verdict and verdict source
    adjudication/temporal_objective_summary_resolved.csv   objective summary with Facility Type resolved
    adjudication/temporal_adjudication_resolved.csv   the 96-row sample with final_verdict and verdict_source
    adjudication/agreement.csv                        two-adjudicator agreement and Cohen's kappa (when filled)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.provenance import file_hash  # noqa: E402
from common.utils import data_dir, datasets_config, experiment_config, now_iso, protocol_config, v2_results_dir, write_json  # noqa: E402

COLUMN = "Facility Type"
FREQUENT_MIN = 100
VERDICTS = ("error", "not_error", "unclear")
PLACEHOLDERS = {"na", "n a", "none", "null", "unknown", "tbd", "test", "nan", "not available"}
HUMAN_COLS = ["adjudicator_1_verdict", "adjudicator_1", "adjudicator_2_verdict", "adjudicator_2", "resolved_verdict",
              "notes"]
OUT = v2_results_dir("temporal_chicago") / "adjudication"
CODEBOOK = OUT / "facility_type_codebook.csv"
SUGGESTIONS = Path(__file__).resolve().parent / "facility_type_suggestions.csv"


# ----------------------------------------------------------------------------- rules
def norm(s: str) -> str:
    s = str(s).lower().replace("&", " and ").replace("'", "")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def osa(a: str, b: str, cap: int = 3) -> int:
    """Optimal string alignment distance (Levenshtein + adjacent transposition), early exit above ``cap``."""
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            c = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + c)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def compact(n: str) -> str:
    """Normalised text without spaces and without a plural "s" on any word (format-insensitive key)."""
    return "".join(w[:-1] if len(w) > 3 and w.endswith("s") else w for w in n.split())


def _single(v: str, freq: Dict[str, str]) -> Tuple[Optional[str], str, bool, str, Optional[int]]:
    """(verdict or None, rule, format_variant, nearest frequent value, distance) for one value or part."""
    n = norm(v)
    if not n or len(n) <= 2 or n in PLACEHOLDERS:
        return "error", "R1 placeholder or not a facility type", False, "", None
    if re.fullmatch(r"[0-9 ]+", n):
        return None, "numeric code only (possibly a CDPH license code): needs a human", False, "", None
    if n in freq:
        same = freq[n] == v
        return ("not_error", "frequent value" if same else f"R2 format variant of '{freq[n]}'", not same, freq[n], 0)
    comp = {compact(fn): fv for fn, fv in freq.items()}
    if compact(n) in comp:
        return "not_error", f"R3 spacing/plural variant of '{comp[compact(n)]}'", True, comp[compact(n)], 0
    c = compact(n)
    best, dist = "", 99
    for fc, fv in comp.items():
        x = osa(c, fc, 2)
        if x < dist:
            best, dist = fv, x
    allowed = 1 if len(c) < 10 else 2                # short words: one edit only (THEATER is not "Shelter")
    if len(c) >= 6 and 0 < dist <= allowed:
        return "error", f"R4 probable misspelling of '{best}' (distance {dist})", False, best, dist
    return None, "", False, best, dist if dist <= 3 else None


def pre_label(v: str, freq: Dict[str, str]) -> Dict[str, object]:
    verdict, rule, fmt, near, dist = _single(v, freq)
    if verdict is None:
        parts = [p for p in re.split(r"\s*(?:/|,|&|\+|\band\b)\s*", str(v), flags=re.I) if p.strip()]
        if len(parts) > 1:
            sub = [_single(p, freq) for p in parts]
            if all(s[0] == "not_error" for s in sub):
                verdict, rule, fmt = "not_error", "R5 combination of known types: " + " + ".join(s[3] for s in sub), \
                    any(s[2] for s in sub)
    return {"pre_label": verdict or "unclear", "pre_label_rule": rule or _single(v, freq)[1] or "no rule applies (needs a human)",
            "format_variant": bool(fmt), "nearest_frequent_value": near, "distance_to_nearest": dist}


# ----------------------------------------------------------------------------- prepare
def replay_alarm_values() -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Replay the frozen monitor exactly as experiments/v2/temporal_chicago.py does for its adjudication reference
    (first drift reference, no practical floor, every check) and record the VALUE of every Facility Type alarm."""
    from experiments.v2.temporal_chicago import MonthlyReplay
    from ingestion.loaders import find_files, load_chicago
    from monitor.loop import Monitor, MonitorConfig
    from rules.static_rules import check
    datasets, exp, pr = datasets_config(), experiment_config(), protocol_config()
    cfg = datasets["chicago_food"]
    files = find_files(data_dir("original"), cfg["files"])
    if not files:
        sys.exit("Food_Inspections.csv not found under data/original")
    raw = load_chicago(files, cfg)
    raw = raw[raw["Inspection Date"] >= pd.Period("2010-01", "M").start_time]
    tcfg = pr["temporal"]
    rep = MonthlyReplay(raw, cfg, tcfg["warmup_batches"], max_gap_days=int(tcfg.get("incomplete_month_gap_days", 4)))
    ref = list(tcfg.get("drift_references", ["warmup"]))[0]
    m = Monitor(cfg, exp, pr, MonitorConfig(name=f"frozen_{ref}", method="adaptive", extra_rules=[], scheduler="all",
                                            drift_source="gated", degradation_rule=pr["schedulers"]["degradation_rule"],
                                            drift_reference=ref))
    rows = []
    for t in range(1, rep.n_batches + 1):
        b = rep.batch(t)
        st = m.step(b.df, b.label)
        if st["warmup"]:
            continue
        lab = st["label"] or {}
        degraded = set(str(lab.get("degraded_checks") or "").split(","))
        for x in m.last_suite:
            if x.get("column") != COLUMN or x["type"] not in ("category", "pattern"):
                continue
            mask = check(x, b.df).fillna(False).to_numpy(dtype=bool)
            if not mask.any():
                continue
            vc = b.df.loc[mask, COLUMN].astype(str).value_counts()
            for v, n in vc.items():
                rows.append({"month": b.label, "label": lab.get("label"), "check": x["key"],
                             "degraded_check": x["key"] in degraded, "value": v, "alarms": int(n)})
    info = {"raw_file": str(files[0]), "raw_sha256": file_hash(files[0]), "months": rep.n_batches,
            "dropped_months": rep.dropped_months, "drift_reference": ref}
    return pd.DataFrame(rows), {"raw": raw, **info}


def prepare(args) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    alarms, info = replay_alarm_values()
    raw = info.pop("raw")
    alarms.to_csv(OUT / "facility_type_alarm_values.csv", index=False)
    # consistency with the frozen run's objective summary (same monitor, same data)
    summ_path = v2_results_dir("temporal_chicago") / "temporal_objective_summary.csv"
    if summ_path.exists():
        s = pd.read_csv(summ_path)
        s = s[s.check == f"category:{COLUMN}"]
        for deg, g in alarms.groupby("degraded_check"):
            frozen = int(s.loc[s.degraded_check == deg, "alarms"].sum())
            print(f"[adjudication] Facility Type alarms, degraded_check={deg}: replay {int(g.alarms.sum())}, "
                  f"frozen run {frozen}{'' if frozen == int(g.alarms.sum()) else '  <-- MISMATCH'}")
    counts = raw[COLUMN].dropna().astype(str).value_counts()
    freq = {}
    for v, n in counts.items():                      # most frequent spelling represents its normalised form
        if n >= FREQUENT_MIN and norm(v) not in freq:
            freq[norm(v)] = v
    g = alarms.groupby("value").agg(alarms=("alarms", "sum"),
                                    degraded_check_alarms=("alarms", lambda s: int(s[alarms.loc[s.index, "degraded_check"]].sum())),
                                    months_flagged=("month", "nunique"), first_month=("month", "min"),
                                    last_month=("month", "max")).reset_index()
    g["raw_occurrences"] = g.value.map(counts).fillna(0).astype(int)
    lab = pd.DataFrame([pre_label(v, freq) for v in g.value])
    book = pd.concat([g, lab], axis=1).sort_values("alarms", ascending=False)
    if SUGGESTIONS.exists():                         # model suggestions for values no rule decides
        sug = pd.read_csv(SUGGESTIONS, dtype=str, keep_default_na=False)
        book = book.merge(sug[["value", "suggested_verdict", "suggestion_reason", "suggestion_by"]], on="value", how="left")
    for c in ("suggested_verdict", "suggestion_reason", "suggestion_by"):
        if c not in book:
            book[c] = ""
    for c in HUMAN_COLS:
        book[c] = ""
    if CODEBOOK.exists() and not args.reset:         # keep every human entry already made
        old = pd.read_csv(CODEBOOK, dtype=str, keep_default_na=False).set_index("value")
        for c in HUMAN_COLS:
            if c in old:
                book[c] = book.value.map(old[c]).fillna("")
    book = book.fillna({"suggested_verdict": "", "suggestion_reason": "", "suggestion_by": ""})
    book.to_csv(CODEBOOK, index=False)
    write_json({"created_at": now_iso(), "driver_sha256": file_hash(Path(__file__)), "frequent_min": FREQUENT_MIN,
                "frequent_values": len(freq), "distinct_flagged_values": len(book), "alarms": int(book.alarms.sum()),
                **info}, OUT / "codebook_manifest.json")
    pl = book.groupby("pre_label").agg(values=("value", "size"), alarms=("alarms", "sum"))
    print(f"[adjudication] {len(book)} distinct flagged values, {int(book.alarms.sum())} alarms -> {CODEBOOK}")
    print(pl.to_string())


# ----------------------------------------------------------------------------- apply
def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    cats = sorted(set(a) | set(b))
    po = float((a == b).mean())
    pe = sum(float((a == c).mean()) * float((b == c).mean()) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else np.nan


def final_verdicts(book: pd.DataFrame, provisional: bool) -> pd.DataFrame:
    b = book.copy()
    for c in ("adjudicator_1_verdict", "adjudicator_2_verdict", "resolved_verdict"):
        b[c] = b[c].str.strip().str.lower().replace({"not an error": "not_error", "not error": "not_error"})
        bad = b[c][(b[c] != "") & ~b[c].isin(VERDICTS)]
        if len(bad):
            sys.exit(f"{c}: unknown verdict(s) {sorted(set(bad))}; use one of {VERDICTS}")
    v1, v2, rv = b.adjudicator_1_verdict, b.adjudicator_2_verdict, b.resolved_verdict
    two = (v1 != "") & (v2 != "")
    final = np.where(rv != "", rv, np.where(two & (v1 == v2), v1, np.where(two, "unclear",
                     np.where(v1 != "", v1, "pending"))))
    source = np.where(rv != "", "human (resolved disagreement)", np.where(two & (v1 == v2), "human (two agree)",
                      np.where(two, "human (disagreement, unresolved)", np.where(v1 != "", "human (one adjudicator)", ""))))
    b["final_verdict"], b["verdict_source"] = final, source
    if provisional:
        pend = b.final_verdict == "pending"
        decided = pend & (b.pre_label != "unclear")
        b.loc[decided, "final_verdict"] = b.loc[decided, "pre_label"]
        b.loc[decided, "verdict_source"] = "rule pre-label (provisional)"
        sug = pend & ~decided & b.suggested_verdict.isin(VERDICTS)
        b.loc[sug, "final_verdict"] = b.loc[sug, "suggested_verdict"]
        b.loc[sug, "verdict_source"] = "model suggestion (provisional): " + b.loc[sug, "suggestion_by"]
    b.loc[b.verdict_source == "", "verdict_source"] = "pending"
    return b


def apply(args) -> None:
    if not CODEBOOK.exists():
        sys.exit(f"{CODEBOOK} not found: run 'python -m analysis.chicago_adjudication prepare' first")
    book = pd.read_csv(CODEBOOK, dtype=str, keep_default_na=False)
    for c in ("alarms", "degraded_check_alarms"):
        book[c] = pd.to_numeric(book[c])
    b = final_verdicts(book, args.provisional)
    sfx = "_provisional" if args.provisional else ""
    tot = b.groupby(["final_verdict", "verdict_source"]).agg(values=("value", "size"), alarms=("alarms", "sum"),
                                                             degraded_check_alarms=("degraded_check_alarms", "sum"))
    tot = tot.reset_index()
    tot["alarm_share"] = tot.alarms / tot.alarms.sum()
    tot.to_csv(OUT / f"facility_type_alarm_verdicts{sfx}.csv", index=False)
    vmap = dict(zip(b.value, b.final_verdict))
    smap = dict(zip(b.value, b.verdict_source))

    # objective summary with the Facility Type category rows split by the adjudicated verdicts
    tc = v2_results_dir("temporal_chicago")
    summ = pd.read_csv(tc / "temporal_objective_summary.csv")
    av = pd.read_csv(OUT / "facility_type_alarm_values.csv")
    av["final_verdict"] = av.value.astype(str).map(vmap).fillna("pending")
    rows = []
    for _, r in summ.iterrows():
        if r.check == f"category:{COLUMN}":
            g = av[av.degraded_check == bool(r.degraded_check)]
            cnt = g.groupby("final_verdict").alarms.sum()
            rows.append({**r.to_dict(), "verdict_error": int(cnt.get("error", 0)),
                         "verdict_not_error": int(cnt.get("not_error", 0)),
                         "verdict_unresolved": int(cnt.get("unclear", 0) + cnt.get("pending", 0)),
                         "verdict_pending": int(cnt.get("pending", 0)), "resolved_by": "value codebook"})
        else:
            rows.append({**r.to_dict(), "verdict_pending": 0, "resolved_by": "objective (frozen replay)"})
    pd.DataFrame(rows).to_csv(OUT / f"temporal_objective_summary_resolved{sfx}.csv", index=False)

    # adjudication sample: objective verdicts stay; unresolved rows take the codebook verdict of their value
    adj = pd.read_csv(tc / "temporal_adjudication.csv", dtype=str, keep_default_na=False)
    fin, src = [], []
    for _, r in adj.iterrows():
        checks = r.failed_checks.split(";")
        if r.objective_verdict != "unresolved":
            fin.append(r.objective_verdict)
            src.append("objective")
            continue
        v = vmap.get(r[COLUMN], "pending") if f"category:{COLUMN}" in checks else "pending"
        fin.append(v)
        src.append(smap.get(r[COLUMN], "pending") if v != "pending" else "pending")
    adj["final_verdict"], adj["verdict_source"] = fin, src
    adj.to_csv(OUT / f"temporal_adjudication_resolved{sfx}.csv", index=False)

    two = b[(b.adjudicator_1_verdict != "") & (b.adjudicator_2_verdict != "")]
    if len(two):
        pd.DataFrame([{"values_rated_by_both": len(two), "agreement": float((two.adjudicator_1_verdict ==
                                                                           two.adjudicator_2_verdict).mean()),
                       "cohen_kappa": cohen_kappa(two.adjudicator_1_verdict, two.adjudicator_2_verdict),
                       "alarms_covered": int(two.alarms.sum())}]).to_csv(OUT / "agreement.csv", index=False)
    pd.set_option("display.width", 200)
    print(tot.to_string(index=False))
    print(adj.groupby(["objective_verdict", "final_verdict", "verdict_source"]).size().rename("rows").to_string())
    pend = b[b.final_verdict == "pending"]
    if len(pend):
        print(f"\n{len(pend)} value(s) ({int(pend.alarms.sum())} alarms) still pending: fill adjudicator_1_verdict in {CODEBOOK}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="replay the frozen monitor and write the value codebook")
    p.add_argument("--reset", action="store_true", help="discard human entries of an existing codebook")
    a = sub.add_parser("apply", help="resolved tables from the codebook")
    a.add_argument("--provisional", action="store_true",
                   help="where no human verdict exists yet, use the rule pre-label, then the model suggestion")
    args = ap.parse_args(argv)
    prepare(args) if args.cmd == "prepare" else apply(args)


if __name__ == "__main__":
    main()
