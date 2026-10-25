"""Tests for defect injection, rules, drift detection and adaptive thresholds."""
import numpy as np
import pandas as pd
import pytest

from adaptation.adaptive_thresholds import adaptive_category_rules, adaptive_range_rules
from common.metrics import confusion, scores
from drift.drift_detector import categorical_drift, drift_scores, numeric_drift
from execution.local_executor import execute
from ingestion.prepare import make_batches
from injection.defect_injector import inject_defects, inject_distribution_shift
from profiling.profiler import QUANTILES, profile_batch
from rules.static_rules import base_rules, check
from rules.statistical_rules import category_rules, range_rules

CFG = {"types": ["missing", "duplicate", "invalid_category", "outlier"],
       "severity": {"missing": "Medium", "duplicate": "Low", "invalid_category": "Medium"},
       "outlier_sigma": [4, 12], "outlier_high_sigma": 8}
NUM, CAT = ["x", "y"], ["c"]


def make(n=5000, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"key": np.arange(n), "x": rng.normal(100, 10, n), "y": rng.gamma(2, 5, n),
                       "c": rng.choice(list("ABCDE"), n, p=[.4, .3, .15, .1, .05])})
    return make_batches(df, 1, n, seed)[0]


def test_injection_ground_truth():
    df = make()
    out, gt = inject_defects(df, 0.04, NUM, CAT, np.random.default_rng(1), "T01", CFG)
    assert len(gt) == 200 and gt.record_id.is_unique
    assert list(gt.columns) == ["record_id", "column", "defect_type", "severity", "batch"]
    assert (gt.defect_type.value_counts() == 50).all()
    assert len(out) == len(df) + 50                           # duplicates are appended
    miss = gt[gt.defect_type == "missing"]
    for r in miss.itertuples():
        assert pd.isna(out.loc[out.record_id == r.record_id, r.column]).all()
    # every injected defect is caught by rules built from the clean batch
    prof = profile_batch(df, NUM, CAT)
    rules = base_rules(list(out.columns), NUM, CAT) + range_rules(prof, NUM, 3) + category_rules(prof, CAT)
    res = execute(rules, out)
    assert set(gt.record_id) <= res["failed_ids"]


def test_rules():
    df = pd.DataFrame({"record_id": list("abcde"), "v": [1, 2, None, 50, 2], "s": ["A", "B", "Z", "A", "B"],
                       "k": [1, 2, 3, 4, 2]})
    assert check({"type": "completeness", "column": "v"}, df).tolist() == [False, False, True, False, False]
    assert check({"type": "range", "column": "v", "low": 0, "high": 10}, df).tolist() == [False, False, False, True, False]
    assert check({"type": "category", "column": "s", "allowed": ["A", "B"]}, df).tolist() == [False, False, True, False, False]
    assert check({"type": "uniqueness", "columns": ["v", "s", "k"]}, df).tolist() == [False, False, False, False, True]


def test_metrics():
    c = confusion({"a", "b", "x"}, {"a", "b", "c"}, list("abcxyz"))
    assert c == {"TP": 2, "FP": 1, "FN": 1, "TN": 2}
    s = scores(c)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)


def test_drift_scores_detect_shift_not_noise():
    a, b = make(seed=1), make(seed=2)
    ref = [profile_batch(a, NUM, CAT)]
    same = drift_scores(b, ref, NUM, CAT)
    shifted = inject_distribution_shift(b, 0.3, NUM, CAT, {"c": list("ABCDE")}, np.random.default_rng(0))
    moved = drift_scores(shifted, ref, NUM, CAT)
    for c in NUM + CAT:
        assert moved[c] > 3 * same[c]
    assert 0 <= categorical_drift({"A": .5, "B": .5}, {"A": .5, "B": .5}) < 1e-9


def test_adaptive_threshold_follows_drift_but_not_defects():
    base = make(seed=3)
    hist = [profile_batch(make(seed=s), NUM, CAT) for s in range(4, 9)]
    drifted = inject_distribution_shift(base, 0.3, NUM, [], {}, np.random.default_rng(0))
    cur = profile_batch(drifted, NUM, CAT)
    static = range_rules(hist[-1], NUM, 3)
    adaptive = adaptive_range_rules(hist, cur, {}, NUM, 3)
    fp_static = execute(static, drifted)["failure_rate"]
    fp_adapt = execute(adaptive, drifted)["failure_rate"]
    assert fp_adapt < fp_static / 3
    # degraded (not drifted) batch: thresholds stay close to history
    bad, _ = inject_defects(base, 0.10, NUM, CAT, np.random.default_rng(1), "T01", CFG)
    r = adaptive_range_rules(hist, profile_batch(bad, NUM, CAT), {}, NUM, 3)[0]
    h = range_rules(hist[-1], NUM, 3)[0]
    assert abs(r["high"] - h["high"]) / h["high"] < 0.05


def test_adaptive_categories_ignore_one_off_values():
    h = [{"columns": {"c": {"frequencies": {"A": .9, "B": .1}}}},
         {"columns": {"c": {"frequencies": {"A": .9, "B": .05, "INVALID_X": .05}}}},
         {"columns": {"c": {"frequencies": {"A": .8, "B": .1, "NEW": .1}}}},
         {"columns": {"c": {"frequencies": {"A": .8, "B": .1, "NEW": .1}}}}]
    allowed = adaptive_category_rules(h, ["c"], 2)[0]["allowed"]
    assert allowed == ["A", "B", "NEW"]


def test_risk_score_components_and_weights():
    from risk.risk_score import RECORDS, RiskTracker
    w = {"F": 0.25, "D": 0.25, "S": 0.15, "H": 0.20, "I": 0.15}
    base = {"range:x": {"mu": 0.01, "sd": 0.001}, "uniqueness:_records": {"mu": 0.002, "sd": 0.0005}}
    tr = RiskTracker(["x", RECORDS], {"x": 1.0, RECORDS: 0.5}, w, base, {"x": 0.1}, window=5)
    calm = tr.score({"x": 0.0}, {"x": False})
    assert calm["x"]["R"] == pytest.approx(0.15 * 1.0)            # only importance contributes
    assert calm[RECORDS]["R"] == pytest.approx(0.15 * 0.5)
    tr.update({"range:x": 0.02}, calm)                            # failing 10 sd above normal -> F = 1
    hot = tr.score({"x": 0.3}, {"x": True})                        # drift above threshold, schema changed
    assert hot["x"]["F"] == 1 and hot["x"]["D"] == 1 and hot["x"]["S"] == 1
    assert hot["x"]["R"] == pytest.approx(0.25 + 0.25 + 0.15 + 0.20 * 1.0 + 0.15)   # H=1: anomalous last batch
    assert hot[RECORDS]["S"] == 1                                  # any schema change affects record unit


def test_selector_levels_and_refresh():
    from optimizer.rule_selector import select
    tiers = {"completeness": "basic", "type": "basic", "range": "basic", "category": "standard", "uniqueness": "standard"}
    thr = {"moderate": 0.2, "high": 0.5, "very_high": 0.75}
    rules = [{"type": "completeness", "column": "c"}, {"type": "category", "column": "c", "allowed": ["A"]},
             {"type": "uniqueness", "columns": ["c"]}]
    low = {"c": {"R": 0.1}, "_records": {"R": 0.1}}
    chosen, info = select(rules, low, tiers, thr, {"category:c": 9, "uniqueness:_records": 9}, 10, 5)
    assert [r["type"] for r in chosen] == ["completeness"] and info["skipped"] == 2
    chosen, info = select(rules, low, tiers, thr, {"category:c": 5, "uniqueness:_records": 9}, 10, 5)
    assert [r["type"] for r in chosen] == ["completeness", "category"] and info["refresh"] == 1
    chosen, _ = select(rules, {"c": {"R": 0.3}, "_records": {"R": 0.1}}, tiers, thr, {"category:c": 9, "uniqueness:_records": 9}, 10, 5)
    assert [r["type"] for r in chosen] == ["completeness", "category"]
    chosen, info = select(rules, {"c": {"R": 0.8}, "_records": {"R": 0.1}}, tiers, thr, {"category:c": 9, "uniqueness:_records": 9}, 10, 5)
    assert len(chosen) == 3 and info["complete_suite"]


def test_llm_response_parsing():
    from rules.llm_generator import parse_rules
    rules, err = parse_rules('```json\n{"rules": [{"column": "x", "rule_type": "completeness"}]}\n```')
    assert err is None and rules[0]["column"] == "x"
    assert parse_rules("sorry, no JSON")[1] is not None
    assert parse_rules('{"rules": [ {"column": "x"')[1] is not None          # truncated answer


def test_validator_statuses():
    from rules.validator import accepted_rules, validate
    cfg = {"numeric": ["x", "y"], "categorical": ["c"]}
    warm = [make(seed=s) for s in range(20, 25)]
    cands = [
        {"column": "x", "rule_type": "range", "min": 0, "max": 1000, "confidence": 0.9},     # plausible
        {"column": "x", "rule_type": "range", "min": 0, "max": 2000, "confidence": 0.5},     # redundant (looser)
        {"column": "x", "rule_type": "range", "min": 95, "max": 105, "confidence": 0.8},     # too strict -> modified
        {"column": "c", "rule_type": "category", "allowed_values": ["A", "B", "C", "D", "E"], "confidence": 0.9},
        {"column": "c", "rule_type": "pattern", "pattern": "[AB]", "confidence": 0.4},       # rejected
        {"column": "nope", "rule_type": "completeness", "confidence": 0.9},                  # invalid column
        {"column": "x", "rule_type": "sql", "condition": "x > 0"},                           # invalid type
        {"column": "c", "rule_type": "pattern", "pattern": "([", "confidence": 0.9},         # invalid regex
        {"column": "_records", "rule_type": "uniqueness", "confidence": 0.9},
    ]
    st = [v["status"] for v in validate(cands, cfg, warm, 0.005)]
    assert st == ["accepted", "redundant", "modified", "accepted", "rejected", "invalid", "invalid", "invalid", "accepted"]
    keys = [r["key"] for r in accepted_rules(validate(cands, cfg, warm, 0.005))]
    assert len(keys) == len(set(keys)) == 4


def test_budget_guard(tmp_path, monkeypatch):
    from rules import llm_generator as g
    monkeypatch.setattr(g, "results_dir", lambda *p: tmp_path.joinpath(*p))
    cfg = {"model_id": "m", "max_tokens": 8000, "temperature": None, "price_input_per_mtok": 4.4,
           "price_output_per_mtok": 22.0, "max_cost_usd": 0.10}
    client = g.LLMClient(cfg, mock=False)
    with pytest.raises(g.BudgetExceeded):          # worst case 8000 output tokens = $0.176 > $0.10
        client.generate("ds", 1, "prompt", 1000)


def test_semantic_defects():
    from injection.defect_injector import inject_semantic_defects
    df = make()
    specs = [{"column": "x", "uniform": [-10, -1]}, {"column": "c", "values": ["ZZ", "QQ"]}]
    excl = set(df["record_id"].iloc[:1000])
    out, gt = inject_semantic_defects(df, 0.02, specs, np.random.default_rng(3), "T06", excl)
    assert len(gt) == 100 and gt.record_id.is_unique and not (set(gt.record_id) & excl)
    assert set(gt.defect_type) == {"semantic_numeric", "semantic_category"}
    m = out.set_index("record_id")
    num = gt[gt.defect_type == "semantic_numeric"].record_id
    assert m.loc[num, "x"].between(-10, -1).all()
    cat = gt[gt.defect_type == "semantic_category"].record_id
    assert m.loc[cat, "c"].isin(["ZZ", "QQ"]).all()
    assert len(out) == len(df)


def test_sensitivity_helpers(tmp_path):
    from common.utils import data_dir, results_dir, workspace
    from experiments.sensitivity import weight_variants
    main_batches = data_dir("batches", "x")
    with workspace(tmp_path):                        # sensitivity data never touches the main data
        assert data_dir("batches", "x") == tmp_path / "batches" / "x"
        assert data_dir("ground_truth", "x") == tmp_path / "ground_truth" / "x"
        assert results_dir("profiles", "x") == tmp_path / "profiles" / "x"
        assert data_dir("clean", "x.parquet") != tmp_path / "clean" / "x.parquet"   # clean data is shared
    assert data_dir("batches", "x") == main_batches
    w = {"F": 0.25, "D": 0.25, "S": 0.15, "H": 0.20, "I": 0.15}
    v = weight_variants(w, [0, 0.5, 2])
    assert len(v) == 2 + 5 * 3
    assert all(abs(sum(x.values()) - 1) < 1e-9 for x in v.values())
    assert v["D removed"]["D"] == 0 and v["F x2"]["F"] == pytest.approx(0.5 / 1.25)
