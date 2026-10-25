"""Tests for the v2 evaluation module (statistics, metrics, rule analysis)."""
import numpy as np
import pandas as pd
import pytest

from evaluation import metrics as M
from evaluation import stats as S


def test_paired_ratio_diff_detects_clear_difference_and_covers_zero_when_equal():
    rng = np.random.default_rng(0)
    den = rng.integers(900, 1100, 10)
    a = (den * 0.95).round(); b = (den * 0.90).round()
    r = S.paired_ratio_diff(a, den, b, den, B=2000)
    assert r["estimate"] == pytest.approx(0.05, abs=0.002) and r["ci_low"] > 0
    same = S.paired_ratio_diff(a, den, a, den, B=500)
    assert same["estimate"] == 0 and same["ci_low"] <= 0 <= same["ci_high"]
    assert S.noninferior(r, 0.02) and not S.noninferior({"ci_low": -0.03}, 0.02)


def test_holm():
    adj = S.holm([0.01, 0.04, 0.03])
    assert adj == pytest.approx([0.03, 0.06, 0.06])


def test_compare_methods_long_form():
    rows = []
    for seed in range(5):
        rows.append({"dataset": "d", "method": "base", "seed": seed, "TP": 90, "P": 100, "FP": 50, "N": 1000})
        rows.append({"dataset": "d", "method": "new", "seed": seed, "TP": 89 + seed % 2, "P": 100, "FP": 20, "N": 1000})
    out = S.compare_methods(pd.DataFrame(rows), ["dataset"], "method", "base", "seed",
                            {"recall": ("TP", "P"), "fpr": ("FP", "N")}, B=500, margin=0.02)
    rec = out[out.metric == "recall"].iloc[0]
    assert rec["noninferior"] and out[out.metric == "fpr"].iloc[0]["estimate"] == pytest.approx(-0.03)


def test_record_counts_and_breakdown():
    gt = pd.DataFrame({"record_id": ["a", "b", "c"], "column": ["x", "y", "<row>"],
                       "defect_type": ["outlier", "semantic_numeric", "duplicate"], "severity": ["High", "Medium", "Low"],
                       "batch": "T06", "incident": ["", "", "dup"]})
    c = M.record_counts({"a", "c", "z"}, gt, list("abcxyz"), valid_novel_ids={"z", "y"})
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (2, 1, 1, 2)
    assert c["tp_type:outlier"] == 1 and c["n_type:semantic_numeric"] == 1 and c["tp_inc:dup"] == 1
    assert c["fp_valid_novel"] == 1
    long = M.breakdown_long(pd.DataFrame([dict(c, dataset="d")]), ["dataset"])
    assert set(long.breakdown) == {"type", "sev", "col", "inc"}


def test_label_metrics_and_delay():
    true = ["stable", "drift", "drift", "degradation", "drift+degradation"]
    pred = ["stable", "drift", "drift+degradation", "degradation", "drift+degradation"]
    m = M.label_metrics(true, pred)
    assert m["accuracy"] == 0.8 and m["false_degradation_during_drift"] == 0.5
    assert 0 < m["macro_f1"] < 1 and m["balanced_accuracy"] == pytest.approx((1 + .5 + 1 + 1) / 4)
    assert M.onset_delay([6, 7, 8], [False, False, True], 6) == 2
    assert np.isnan(M.onset_delay([6, 7], [False, False], 6))


def test_incident_summary():
    pb = pd.DataFrame({"t": range(6, 21), "n_inc:x": [0, 0, 0, 10, 10, 10, 10, 10, 10, 0, 0, 0, 0, 0, 0],
                       "tp_inc:x": [0, 0, 0, 2, 8, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0],
                       "label": ["stable"] * 4 + ["degradation"] * 6 + ["degradation", "stable"] + ["stable"] * 3})
    s = M.incident_summary(pb, "x", 9, 14, 0.5, 20)
    assert s["detection_delay"] == 1 and s["label_delay"] == 1 and s["recovery_time"] == 3   # stable again at T17
    assert s["post_onset_recall"] == pytest.approx(46 / 60)


def test_predicate_stability_and_admission():
    from evaluation.rule_analysis import admission_outcomes, predicate_stability
    cfg = {"numeric": ["x"], "categorical": ["s"]}
    probe = pd.DataFrame({"x": np.linspace(1, 100, 500), "s": np.random.default_rng(0).choice(["AA", "BB", "CC"], 500)})
    g1 = [{"type": "range", "column": "x", "low": 0, "high": None}, {"type": "category", "column": "s", "allowed": ["AA", "BB", "CC"]}]
    g2 = [{"type": "range", "column": "x", "low": 0, "high": None}, {"type": "pattern", "column": "s", "pattern": "[A-Z]{2}"}]
    st = predicate_stability({1: g1, 2: g2}, probe, cfg, n_probe=100)
    assert st[st.column == "x"].probe_agreement.iloc[0] == 1.0
    assert st[st.column == "s"].probe_agreement.iloc[0] < 1.0       # list and pattern disagree on new codes
    df = pd.DataFrame({"record_id": ["r1", "r2", "r3", "r4"], "x": [5.0, -3.0, 7.0, 8.0], "s": ["AA", "AA", "BB", "AA"]})
    gt = pd.DataFrame({"record_id": ["r2"], "column": ["x"], "defect_type": ["semantic_numeric"], "severity": ["M"], "batch": ["T06"]})
    val = [{"index": 0, "status": "rejected", "column": "x", "rule_type": "range"},
           {"index": 1, "status": "accepted", "column": "x", "rule_type": "range"}]
    cands = [{"column": "x", "rule_type": "range", "min": 0, "max": None}, {"column": "x", "rule_type": "range", "min": 6, "max": None}]
    out = admission_outcomes(val, cands, cfg, [(df, gt, set())], 0.1)
    assert out.loc[0, "incorrectly_rejected"] and out.loc[1, "incorrectly_admitted"]
