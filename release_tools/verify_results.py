"""Verify frozen saved metrics without experiments, data downloads or third-party Python packages."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "adaptive-data-quality"
RESULTS = PROJECT / "results" / "v2"
OUTPUT = ROOT / ".artifacts" / "saved-metrics.json"
CODE_DIRS = ("adaptation", "baselines", "common", "drift", "execution", "experiments", "evaluation",
             "ingestion", "injection", "monitor", "optimizer", "profiling", "risk", "rules")
CLASSES = ("stable", "drift", "degradation", "drift+degradation")
checks = []
details = {}


def rows(relative):
    with (RESULTS / relative).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def check(name, passed, **evidence):
    checks.append({"name": name, "passed": bool(passed), **evidence})


def close(a, b, tolerance=1e-9):
    return math.isclose(float(a), float(b), rel_tol=tolerance, abs_tol=tolerance)


def truth(value):
    return str(value).lower() == "true"


def one(values, **keys):
    matches = [r for r in values if all(r[k] == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {keys}, found {len(matches)}")
    return matches[0]


def provenance():
    files = sorted(p for directory in CODE_DIRS for p in (PROJECT / directory).rglob("*.py")
                   if "__pycache__" not in p.parts)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    fingerprint = digest.hexdigest()[:16]
    config_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                     for p in sorted((PROJECT / "configs").glob("*.yaml"))}
    manifests = []
    for path in sorted(RESULTS.glob("*/test/manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if "code_fingerprint" not in manifest:
            manifests.append({"path": path.relative_to(RESULTS).as_posix(),
                              "note": "Custom tool-exchange manifest; no pipeline fingerprint field."})
            continue
        config_digest = hashlib.sha256(json.dumps(manifest["config"], sort_keys=True).encode()).hexdigest()[:16]
        mismatches = {k: {"current": v, "manifest": manifest.get("config_files", {}).get(k)}
                      for k, v in config_hashes.items() if manifest.get("config_files", {}).get(k) != v}
        check(f"provenance:{path.parent.parent.name}",
              manifest["code_fingerprint"] == fingerprint and not mismatches
              and config_digest == manifest["config_hash"],
              code_fingerprint=fingerprint, config_hash=config_digest, config_file_mismatches=mismatches)
        manifests.append({"path": path.relative_to(RESULTS).as_posix(), "created_at": manifest.get("created_at"),
                          "code_fingerprint": manifest["code_fingerprint"], "config_hash": manifest["config_hash"]})
    check("original_pipeline_fingerprint", fingerprint == "fd749de031ca4f54", actual=fingerprint,
          source_files=len(files))
    details["provenance"] = {"code_fingerprint": fingerprint, "source_files": len(files),
                              "config_file_hashes": config_hashes, "manifests": manifests,
                              "scope": "Pipeline source and YAML configs only; no dataset rehash or public preregistration proof."}


def thresholds():
    decisions = rows("rq1_thresholds/test/rq1a_h1a_decisions.csv")
    counts = defaultdict(lambda: {"scenarios": 0, "supported": 0})
    for r in decisions:
        counts[r["dataset"]]["scenarios"] += 1
        counts[r["dataset"]]["supported"] += truth(r["H1a_supported"])
    expected = {"bts_airline": 6, "chicago_food": 9, "online_retail": 0}
    check("H1a_scenario_verdict_counts", all(counts[d] == {"scenarios": 9, "supported": n}
                                           for d, n in expected.items()), actual=dict(counts))
    record = rows("rq1_thresholds/test/rq1a_record_level.csv")
    anchors = {}
    for dataset in expected:
        a = one(record, dataset=dataset, scenario="B_drift", method="adaptive")
        b = one(record, dataset=dataset, scenario="B_drift", method="fixed_pooled")
        verdict = one(decisions, dataset=dataset, scenario="B_drift")
        diff_fpr, diff_recall = float(a["fpr"]) - float(b["fpr"]), float(a["recall"]) - float(b["recall"])
        check(f"H1a_B_counts_and_differences:{dataset}",
              close(a["fpr"], float(a["FP"]) / float(a["N"]))
              and close(a["recall"], float(a["TP"]) / float(a["P"]))
              and close(diff_fpr, verdict["fpr_diff"]) and close(diff_recall, verdict["recall_diff"]))
        anchors[dataset] = {"adaptive_fpr": float(a["fpr"]), "fixed_fpr": float(b["fpr"]),
                            "adaptive_recall": float(a["recall"]), "fixed_recall": float(b["recall"]),
                            "fpr_difference_pp": 100 * diff_fpr, "recall_difference_pp": 100 * diff_recall}
    contamination = one(decisions, dataset="bts_airline", scenario="F2_contam_drift")
    check("BTS_contamination_loss_29_06pp", close(-100 * float(contamination["recall_diff"]), 29.05925),
          recall_loss_pp=-100 * float(contamination["recall_diff"]))
    details["thresholds"] = {"H1a_counts": dict(counts), "scenario_B": anchors}


def classification():
    summaries = rows("rq1_classification/test/rq1b_classification_summary.csv")
    grouped = defaultdict(list)
    for r in rows("rq1_classification/test/rq1b_confusion_matrices.csv"):
        grouped[(r["dataset"], r["comparator"])].append(r)
    largest_discrepancy = 0.0
    for r in summaries:
        group = grouped[(r["dataset"], r["comparator"])]
        by_true = {x["row_0"]: x for x in group}
        scores = []
        for label in CLASSES:
            tp = int(by_true[label][label])
            predicted = sum(int(x[label]) for x in group)
            actual = sum(int(by_true[label][k]) for k in (*CLASSES, "insufficient"))
            scores.append(2 * tp / (predicted + actual) if predicted + actual else 0.0)
        recomputed = sum(scores) / 4
        largest_discrepancy = max(largest_discrepancy, abs(recomputed - float(r["macro_f1"])))
    check("all_classification_F1_from_confusion_counts", largest_discrepancy < 1e-12,
          method_dataset_rows=len(summaries), maximum_absolute_difference=largest_discrepancy)
    headline = {r["dataset"]: float(r["macro_f1"]) for r in summaries if r["comparator"] == "proposed"}
    expected = {"bts_airline": 0.9657706892775965, "chicago_food": 0.9549718574108819,
                "online_retail": 0.9466120437221626}
    check("classification_headline_F1", all(close(headline[k], v) for k, v in expected.items()), actual=headline)
    details["classification"] = {"pooled_macro_F1": headline,
                                  "note": "Pooled confusion-matrix F1; paired CI tables use mean per-seed differences."}


def semantics_and_categories():
    selected = [r for r in rows("rq2_coverage/test/rq2a_summary.csv")
                if r["scenario"] == "S2_semantic_drift" and r["config"] == "fixed+llm_validated"
                and r["redundancy"] == "logical"]
    recalls = [float(r["semantic_recall_mean"]) for r in selected]
    check("fixed_plus_admitted_LLM_semantic_recall", len(recalls) == 6
          and close(min(recalls), 0.9964746666666666) and close(max(recalls), 1),
          minimum=min(recalls), maximum=max(recalls), configurations=len(recalls))
    guard = [r for r in rows("rq2_category_guard/test/rq2b_summary.csv")
             if r["policy"] == "recurrence_llm_value"]
    # Summaries repeat each distinct value across 10 seeds and 2 scenarios.
    for model in ("primary", "secondary"):
        subset = [r for r in guard if r["model"] == model]
        valid = sum(int(r["values"]) / 20 for r in subset if r["kind"] == "valid")
        invalid = sum(int(r["values"]) / 20 for r in subset if r["kind"] == "invalid")
        accepted_valid = sum(int(r["values"]) / 20 * float(r["accepted_share"]) for r in subset if r["kind"] == "valid")
        accepted_invalid = sum(int(r["values"]) / 20 * float(r["accepted_share"]) for r in subset if r["kind"] == "invalid")
        check(f"per_value_guard_distinct_cases:{model}", valid == 12 and invalid == 12
              and accepted_valid == 12 and accepted_invalid == 1,
              distinct_valid=valid, distinct_invalid=invalid, accepted_valid=accepted_valid,
              rejected_invalid=invalid - accepted_invalid)
    details["semantic_coverage"] = {"S2_fixed_plus_admitted_rules": selected,
                                     "guard_note": "24 distinct cases, repeated over seeds/scenarios; not independent semantic replications."}


def scheduling():
    decision = rows("rq3_scheduling/test/rq3_h3_decisions.csv")
    counts = {}
    for policy in ("risk", "periodic", "random", "failure_only", "drift_only"):
        subset = [r for r in decision if r["policy"] == policy and r["post_onset_recall"]]
        counts[policy] = {"cases": len(subset), "useful": sum(truth(r["useful"]) for r in subset)}
    check("H3_useful_counts", counts["risk"] == {"cases": 18, "useful": 11}
          and counts["periodic"]["useful"] == 4 and counts["random"]["useful"] == 5, actual=counts)
    risk = [r for r in decision if r["policy"] == "risk"]
    details["scheduling"] = {"useful_counts": counts,
                              "checks_saved_percent_range": [min(float(r["saving_checks_pct"]) for r in risk),
                                                               max(float(r["saving_checks_pct"]) for r in risk)],
                              "interpretation": "Usefulness thresholds mean recall, not a lower confidence bound."}


def runtime():
    raw = rows("rq3_scheduling/test/rq3_runtime_trials_raw.csv")
    summary = rows("rq3_scheduling/test/rq3_runtime.csv")
    fields = [k for k in summary[0] if k.endswith("_median")]
    sums, counts = defaultdict(lambda: defaultdict(float)), defaultdict(int)
    for r in raw:
        if int(r["repeat"]) < 0 or truth(r.get("warmup", "")):
            continue
        key = (r["dataset"], r["scenario"], r["policy"], r["reuse"], r["repeat"])
        counts[key] += 1
        for field in fields:
            sums[key][field] += float(r[field.removesuffix("_median")])
    errors = []
    for r in summary:
        key = (r["dataset"], r["scenario"], r["policy"], r["reuse"])
        repeats = [v for k, v in sums.items() if k[:4] == key]
        for field in fields:
            actual = median(v[field] for v in repeats)
            if not close(actual, r[field], tolerance=1e-8):
                errors.append({"configuration": key, "field": field, "raw_median_sum": actual,
                               "stored": float(r[field])})
    check("runtime_summary_matches_raw_15_batch_sums", not errors and set(counts.values()) == {15},
          summary_rows=len(summary), timed_repetitions=len(counts), evaluation_batches_per_repetition=sorted(set(counts.values())),
          mismatches=errors)
    corrected = []
    for r in summary:
        if r["scenario"] != "A_stable" or r["policy"] not in ("all", "risk"):
            continue
        item = {k: r[k] for k in ("dataset", "scenario", "policy", "reuse")}
        item.update({k: float(r[k]) / 15 for k in fields})
        corrected.append(item)
    check("BTS_corrected_runtime_60_295ms", close(one(corrected, dataset="bts_airline", policy="all", reuse="False")
          ["ms_total_median"], 60.295092136948366))
    details["runtime"] = {"stored_estimand": "Median over repetitions of summed times for 15 evaluation batches.",
                          "corrected_estimand": "Median over repetitions of mean milliseconds per evaluation batch.",
                          "divisor": 15, "scenario_A_corrected_ms_per_batch": corrected,
                          "note": "Division preserves relative savings; it does not yield median individual-batch latency."}


def external():
    tax = [r for r in rows("external/test/external_detection.csv") if r["benchmark"] == "tax"]
    valid = all(close(float(r["cell_TP"]) + float(r["cell_FN"]), r["checkable_error_cells"])
                and close(r["cell_recall"], float(r["cell_TP"]) / float(r["checkable_error_cells"]))
                and close(r["checkable_error_coverage"], float(r["checkable_error_cells"]) / float(r["all_error_cells"]))
                for r in tax)
    check("Tax_cell_recall_uses_checkable_error_denominator", valid, methods=len(tax))
    anchors = []
    for method in ("adaptive", "adaptive+deps", "adaptive+xcol_primary", "adaptive+xcol_secondary"):
        r = one(tax, method=method)
        anchors.append({"method": method, "cell_TP": int(r["cell_TP"]),
                        "checkable_error_cells": int(r["checkable_error_cells"]),
                        "all_error_cells": int(r["all_error_cells"]),
                        "cell_recall_checkable": float(r["cell_recall"]),
                        "cell_recall_all_error_cells": float(r["cell_TP"]) / float(r["all_error_cells"]),
                        "record_recall": float(r["recall"]), "record_fpr": float(r["fpr"])})
    deps = one(anchors, method="adaptive+deps")
    check("Tax_dependency_recall_90_81pct_is_conditional", close(deps["cell_recall_checkable"], 0.9081277320886123)
          and close(deps["cell_recall_all_error_cells"], 12257 / 19944),
          checkable_recall=deps["cell_recall_checkable"], all_error_cell_recall=deps["cell_recall_all_error_cells"])
    details["external_Tax"] = {"anchors": anchors,
                                "note": "LLM methods pool five generations, multiplying counts by five; ratios remain comparable. Denominator excludes errors outside checkable columns."}


def main():
    provenance()
    thresholds()
    classification()
    semantics_and_categories()
    scheduling()
    runtime()
    external()
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Saved-artifact consistency audit; no experiment rerun, API call or original-source modification.",
              "project_root": str(PROJECT), "all_checks_passed": all(c["passed"] for c in checks),
              "checks_passed": sum(c["passed"] for c in checks), "checks_total": len(checks),
              "checks": checks, "details": details,
              "known_corrections": ["Frozen runtime summaries are 15-batch totals; the current paper correctly reports totals divided by fifteen.",
                                    "Original secondary Holm values are preserved; paper/latex/reporting_corrections contains the separate comparator-family repair.",
                                    "Tax cell recall is conditional on checkable columns; report all-error coverage alongside it."]}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks_passed": report["checks_passed"],
                      "checks_total": len(checks), "output": str(OUTPUT)}, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
