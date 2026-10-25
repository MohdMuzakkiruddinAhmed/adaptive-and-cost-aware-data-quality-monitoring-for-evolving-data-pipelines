"""Benchmark integrity (manuscript Section 6.2): labels must match the final batch."""
import numpy as np
import pandas as pd

from ingestion.prepare import make_batches
from injection.defect_injector import (append_duplicates, inject_defects, inject_semantic_defects)
from injection.label_audit import audit_batch

CFG = {"types": ["missing", "duplicate", "invalid_category", "outlier"],
       "severity": {"missing": "Medium", "duplicate": "Low", "invalid_category": "Medium"},
       "outlier_sigma": [4, 12], "outlier_high_sigma": 8}
NUM, CAT = ["x", "y"], ["c"]
SPECS = [{"column": "x", "uniform": [-10, -1]}, {"column": "c", "values": ["ZZ", "QQ"]}]


def make(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"key": np.arange(n), "x": rng.normal(100, 10, n), "y": rng.gamma(2, 5, n),
                       "c": rng.choice(list("ABCDE"), n)})
    return make_batches(df, 1, n, seed)[0]


def legacy_semantic_batch(ref, seed):
    """The proof-of-concept order: duplicates appended BEFORE the semantic mutation."""
    df, gt = inject_defects(ref, 0.08, NUM, CAT, np.random.default_rng(seed), "T06", CFG)
    df, gs = inject_semantic_defects(df, 0.20, SPECS, np.random.default_rng(seed + 1), "T06", set(gt.record_id))
    return df, pd.concat([gt, gs], ignore_index=True)


def fixed_semantic_batch(ref, seed):
    df, gt, src = inject_defects(ref, 0.08, NUM, CAT, np.random.default_rng(seed), "T06", CFG, defer_duplicates=True)
    df, gs = inject_semantic_defects(df, 0.20, SPECS, np.random.default_rng(seed + 1), "T06", set(gt.record_id) | set(src))
    df, gd = append_duplicates(df, src, "T06")
    return df, pd.concat([gt, gs, gd], ignore_index=True)


def test_legacy_order_produces_inconsistent_duplicate_labels():
    ref = make()
    found = 0
    for seed in range(5):
        df, gt = legacy_semantic_batch(ref, seed)
        found += audit_batch(df, gt, ref)["issues"].get("duplicate_label_not_a_duplicate", 0)
    assert found > 0          # the audit detects the proof-of-concept problem


def test_fixed_order_labels_match_final_batch():
    ref = make()
    for seed in range(5):
        df, gt = fixed_semantic_batch(ref, seed)
        a = audit_batch(df, gt, ref)
        assert a["ok"], a
        assert gt.record_id.is_unique


def test_deferred_duplicates_identical_without_later_mutations():
    ref = make()
    legacy, gl = inject_defects(ref, 0.05, NUM, CAT, np.random.default_rng(3), "T01", CFG)
    df, gt, src = inject_defects(ref, 0.05, NUM, CAT, np.random.default_rng(3), "T01", CFG, defer_duplicates=True)
    df, gd = append_duplicates(df, src, "T01")
    assert legacy.equals(df)
    assert sorted(gl.record_id) == sorted(pd.concat([gt, gd]).record_id)


def test_audit_detects_unlabelled_change_and_unlabelled_duplicate():
    ref = make(500)
    df = ref.copy()
    df.loc[3, "x"] = -999.0                                   # changed, not labelled
    df = pd.concat([df, df.iloc[[7]].assign(record_id="extra")], ignore_index=True)  # copy, not labelled
    gt = pd.DataFrame(columns=["record_id", "column", "defect_type", "severity", "batch"])
    issues = audit_batch(df, gt, ref)["issues"]
    assert issues["unlabelled_record_changed"] == 1
    assert issues["unknown_record"] == 1 and issues["unlabelled_duplicate"] == 1


def test_mild_outlier_severity():
    ref = make()
    cfg = dict(CFG, outlier_sigma=[2, 6], outlier_mild_sigma=4)
    _, gt = inject_defects(ref, 0.2, NUM, CAT, np.random.default_rng(0), "T01", cfg)
    assert set(gt.loc[gt.defect_type == "outlier", "severity"]) == {"Mild", "Medium"}
