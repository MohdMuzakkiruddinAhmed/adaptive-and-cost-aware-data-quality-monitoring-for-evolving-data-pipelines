"""Reporting-only RQ1a Holm correction from saved paired comparison statistics.

Standard-library only. Does not change original files or primary CI decisions.
Families are dataset x scenario x metric, including all 16 distinct comparator
configurations. Run from the repository root:
    python release_tools/recompute_reporting_corrections.py

Outputs go to .artifacts/reporting-corrections; archived results are read only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROJECT = ROOT / "adaptive-data-quality"
SOURCE = PROJECT / "results" / "v2" / "rq1_thresholds" / "test" / "rq1a_paired_ci.csv"
DECISIONS = SOURCE.with_name("rq1a_h1a_decisions.csv")
OUTPUT = ROOT / ".artifacts" / "reporting-corrections"
REFERENCE = ROOT / "paper" / "latex" / "reporting_corrections" / "rq1a_holm_corrected.csv"
DESTINATION = OUTPUT / "rq1a_holm_corrected.csv"
VERIFICATION = OUTPUT / "rq1a_holm_verification.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_p(value):
    if value == "" or not math.isfinite(float(value)):
        return 1.0
    p = float(value)
    if not 0 <= p <= 1:
        raise ValueError(f"Invalid p-value: {value}")
    return p


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    before = {SOURCE.relative_to(ROOT).as_posix(): digest(SOURCE), DECISIONS.relative_to(ROOT).as_posix(): digest(DECISIONS)}
    with SOURCE.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields, original = list(reader.fieldnames), list(reader)
    families = defaultdict(list)
    for index, row in enumerate(original):
        families[(row["dataset"], row["scenario"], row["metric"])].append(index)
    corrected = {}
    family_details = []
    all_unique, all_independently_verified = True, True
    for key, indices in sorted(families.items()):
        identities = [(original[i]["method"], original[i]["comparator"]) for i in indices]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate method/comparator hypotheses in {key}")
        ordered = sorted(indices, key=lambda i: (parse_p(original[i]["p_value"]), original[i]["comparator"]))
        size, running = len(ordered), 0.0
        for rank, index in enumerate(ordered):
            running = max(running, min(1.0, (size - rank) * parse_p(original[index]["p_value"])))
            corrected[index] = running
            # Independent direct formula, rather than the sequential running maximum.
            direct = min(1.0, max((size - j) * parse_p(original[ordered[j]]["p_value"])
                                  for j in range(rank + 1)))
            all_independently_verified &= math.isclose(running, direct, rel_tol=1e-12, abs_tol=1e-12)
        family_details.append({"dataset": key[0], "scenario": key[1], "metric": key[2], "family_size": size,
                               "undefined_p_values": sum(original[i]["p_value"] == "" for i in indices),
                               "adjusted_values_changed": sum(not math.isclose(corrected[i], parse_p(original[i]["p_holm"]),
                                                                                rel_tol=1e-12, abs_tol=1e-12)
                                                              for i in indices)})
    output_fields = ["p_holm_original" if field == "p_holm" else field for field in fields]
    output_fields += ["p_holm_corrected", "holm_family_size"]
    with DESTINATION.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for index, row in enumerate(original):
            result = {("p_holm_original" if field == "p_holm" else field): value for field, value in row.items()}
            result["p_holm_corrected"] = repr(corrected[index])
            result["holm_family_size"] = len(families[(row["dataset"], row["scenario"], row["metric"])])
            writer.writerow(result)
    with DESTINATION.open(newline="", encoding="utf-8") as stream:
        reread = list(csv.DictReader(stream))
    with REFERENCE.open(newline="", encoding="utf-8-sig") as stream:
        reference_rows = list(csv.DictReader(stream))
    reference_matches = reread == reference_rows
    unchanged = len(reread) == len(original) and all(
        all(output["p_holm_original" if field == "p_holm" else field] == value for field, value in source.items())
        for output, source in zip(reread, original))
    finite = [i for i, r in enumerate(original) if r["p_value"] != ""]
    changed = [i for i in range(len(original)) if not math.isclose(corrected[i], parse_p(original[i]["p_holm"]),
                                                                rel_tol=1e-12, abs_tol=1e-12)]
    transitions = []
    for alpha in (0.05, 0.01):
        previously = sum(parse_p(original[i]["p_holm"]) < alpha for i in finite)
        now = sum(corrected[i] < alpha for i in finite)
        lost = [i for i in finite if parse_p(original[i]["p_holm"]) < alpha <= corrected[i]]
        transitions.append({"alpha": alpha, "comparison": "strict p < alpha", "original_significant": previously,
                            "corrected_significant": now, "lost_significance": len(lost),
                            "lost_rows": [{k: original[i][k] for k in ("dataset", "scenario", "metric", "comparator", "p_value")}
                                          | {"p_holm_original": parse_p(original[i]["p_holm"]),
                                             "p_holm_corrected": corrected[i]} for i in lost]})
    after = {SOURCE.relative_to(ROOT).as_posix(): digest(SOURCE), DECISIONS.relative_to(ROOT).as_posix(): digest(DECISIONS)}
    checks = {"all_original_columns_preserved": unchanged, "source_and_primary_decisions_unchanged": before == after,
              "all_hypotheses_unique_within_family": all_unique,
              "independent_direct_formula_agrees": all_independently_verified,
              "adjustment_never_smaller_than_unadjusted_p": all(corrected[i] + 1e-12 >= parse_p(r["p_value"])
                                                                for i, r in enumerate(original)),
              "all_corrected_p_in_unit_interval": all(0 <= p <= 1 for p in corrected.values()),
              "matches_supplied_reporting_table": reference_matches,
              "outputs_separate_from_archived_inputs": DESTINATION.parent == OUTPUT and VERIFICATION.parent == OUTPUT
                  and OUTPUT != SOURCE.parent and OUTPUT != REFERENCE.parent}
    report = {"scope": "Reporting correction of saved approximate bootstrap p-values, not a new confirmatory experiment.",
              "family": "dataset x scenario x metric, across all distinct method/comparator comparisons",
              "rows": len(original), "families": len(families), "family_sizes": sorted({len(v) for v in families.values()}),
              "undefined_p_values": len(original) - len(finite), "adjusted_values_changed": len(changed),
              "families_with_changes": sum(f["adjusted_values_changed"] > 0 for f in family_details),
              "zero_p_values_preserved": sum(float(original[i]["p_value"]) == 0 for i in finite),
              "significance_transitions": transitions, "checks": checks, "all_checks_passed": all(checks.values()),
              "original_input_sha256": before, "reference_sha256": digest(REFERENCE), "output_sha256": digest(DESTINATION), "families_detail": family_details,
              "limitations": ["Raw bootstrap sign-tail p-values are retained and approximate; zero is not an exact probability.",
                              "Undefined p-values are assigned 1 solely for conservative family construction; no meaningful recall inference exists in the zero-error control.",
                              "Adjustment is within each dataset/scenario/metric, not across scenarios, datasets or metrics.",
                              "The fixed-pooled comparison participates in this reporting family; frozen primary CI decisions remain unchanged."]}
    VERIFICATION.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "rows", "families", "family_sizes", "adjusted_values_changed",
                                            "families_with_changes", "undefined_p_values")}, indent=2))
    for item in transitions:
        print(f"p < {item['alpha']}: {item['original_significant']} -> {item['corrected_significant']} "
              f"({item['lost_significance']} no longer below threshold)")
    if not report["all_checks_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
