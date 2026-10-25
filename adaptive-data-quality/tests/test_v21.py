"""Tests for the v2.1 changes: resampled drift thresholds, warm-up categories kept, domain ranges, objective
verdicts of the Chicago replay, cross-column rules, benchmark prompt pool, validator stress test, H1a verdict,
independent manual-rule author."""
import copy
import json

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import SYNTH_CFG, configs, make_clean


@pytest.fixture(scope="module")
def env():
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    src = DataSource(datasets, exp, pr, clean_loader=lambda name: make_clean())
    return datasets, exp, pr, src


# ----------------------------------------------------------------------------- resampled drift thresholds
def test_resampled_null_is_stable_batch_distribution():
    from drift.drift_detector import resampled_null_scores
    rng = np.random.default_rng(0)
    pool = pd.DataFrame({"x": rng.normal(10, 2, 5000), "q": rng.integers(1, 4, 5000).astype(float),
                         "c": rng.choice(list("ABCD"), 5000, p=[.4, .3, .2, .1])})
    gate = np.ones(len(pool), dtype=bool)
    null = resampled_null_scores(pool, gate, ["x", "q"], ["c"], 5, 50, np.random.default_rng(1))
    assert set(null) == {"x", "q", "c"} and all(len(v) == 250 for v in null.values())
    assert all(np.isfinite(v).all() and (v >= 0).all() for v in null.values())
    # a genuinely shifted batch scores far above the null's upper quantile
    from drift.drift_detector import drift_scores
    from profiling.profiler import profile_batch
    ref = [profile_batch(pool.iloc[i * 1000:(i + 1) * 1000], ["x", "q"], ["c"]) for i in range(5)]
    shifted = pool.iloc[:1000].assign(x=lambda d: d.x * 1.2)
    assert drift_scores(shifted, ref, ["x", "q"], ["c"])["x"] > np.quantile(null["x"], 0.99) * 3
    # memory-bounded blocks give the same number of draws
    big = resampled_null_scores(pd.concat([pool] * 2, ignore_index=True), np.ones(10000, bool), ["x"], [], 5, 7,
                                np.random.default_rng(2))
    assert len(big["x"]) == 35


def test_classifier_threshold_is_max_of_loo_and_resampled(env):
    from monitor.loop import Monitor, MonitorConfig
    datasets, exp, pr, src = env
    dt = dict(pr["classification"]["drift_threshold"], resampling={"draws": 40})
    m = Monitor(SYNTH_CFG, exp, pr, MonitorConfig(name="p", drift_threshold=dt), seed=42)
    for b in src.replay("synth", "dev", 42, "A_stable").batches():
        m.step(b.df, b.label)
    parts = m.classifier.tau_parts
    assert parts and all(np.isfinite(p["resampled"]) for p in parts.values())
    for c, p in parts.items():
        assert m.classifier.tau_drift[c] == pytest.approx(max(p["loo"], p["resampled"]))
    off = Monitor(SYNTH_CFG, exp, pr, MonitorConfig(name="p", drift_threshold=dict(dt, resampling=None)), seed=42)
    for b in src.replay("synth", "dev", 42, "A_stable").batches():
        off.step(b.df, b.label)
    assert not off.classifier.tau_parts or all(p["resampled"] == -np.inf for p in off.classifier.tau_parts.values())


# ----------------------------------------------------------------------------- categories, domain ranges
def test_recurrence_keeps_warmup_categories():
    from adaptation.adaptive_thresholds import adaptive_category_rules
    warm = [{"columns": {"s": {"frequencies": {"AA": .9, "RARE": .1}}}}] + [{"columns": {"s": {"frequencies": {"AA": 1.}}}}] * 4
    hist = [{"columns": {"s": {"frequencies": {"AA": .9, "NEW": .1}}}}] * 2 + [{"columns": {"s": {"frequencies": {"AA": 1.}}}}] * 3
    plain = set(adaptive_category_rules(hist, ["s"], 2)[0]["allowed"])
    kept = set(adaptive_category_rules(hist, ["s"], 2, warm)[0]["allowed"])
    assert plain == {"AA", "NEW"} and kept == {"AA", "NEW", "RARE"}


def test_suite_uses_config_flag_and_domain_ranges(env):
    from monitor.suite import SuiteBuilder
    from profiling.profiler import profile_batch
    datasets, exp, pr, src = env
    cfg = dict(SYNTH_CFG, domain_ranges={"y": [0, 1000]})
    b = src.replay("synth", "dev", 42, "A_stable").batch(1)
    p = profile_batch(b.df, cfg["numeric"], cfg["categorical"])
    rules, _ = SuiteBuilder(cfg, exp, pr, "adaptive").build(list(b.df.columns), [p] * 5, [p] * 5, [p] * 5, p)
    ranges = {r["column"]: r for r in rules if r["type"] == "range"}
    assert ranges["y"]["policy"] == "domain" and ranges["y"]["source"] == "base" and ranges["y"]["high"] == 1000
    assert ranges["x"]["source"] == "stat"
    assert sum(r["type"] == "range" and r["column"] == "y" for r in rules) == 1
    assert exp["adaptive"]["category_keep_warmup"] is True


def test_real_chicago_domain_inside_city_bbox():
    from common.utils import datasets_config
    dr = datasets_config()["chicago_food"]["domain_ranges"]
    assert dr["Latitude"][0] < 41.65 and dr["Latitude"][1] > 42.02
    assert dr["Longitude"][0] < -87.94 and dr["Longitude"][1] > -87.53


# ----------------------------------------------------------------------------- objective verdicts (Chicago replay)
def test_point_in_polygon_and_objective_verdicts():
    from experiments.v2.temporal_chicago import objective_verdicts
    from ingestion.geo import CityBoundary, parse_multipolygon
    square = "MULTIPOLYGON (((0 0, 10 0, 10 10, 0 10, 0 0), (4 4, 6 4, 6 6, 4 6, 4 4)), ((20 20, 21 20, 21 21, 20 20)))"
    b = CityBoundary(parse_multipolygon(square))
    inside = b.contains([1, 5, 11, 20.8, np.nan], [1, 5, 1, 20.2, 1])
    assert inside.tolist() == [True, False, False, True, False]          # hole excluded, second polygon included
    df = pd.DataFrame({"Longitude": [1.0, 11.0, None], "Latitude": [1.0, 1.0, 2.0], "Results": ["Pass", "Pased", None]})
    rng = {"type": "range", "column": "Longitude", "low": 0.5, "high": 0.9}
    v = objective_verdicts(rng, df, np.array([True, True, True]), b, {})
    assert v.tolist() == ["not_error", "error", "error"]                # missing coordinate is not inside the city
    cat = {"type": "category", "column": "Results", "allowed": ["Pass"]}
    assert objective_verdicts(cat, df, np.array([False, True, False]), b, {"Results": ["Pass", "Fail"]}).tolist() == ["error"]
    assert objective_verdicts(cat, df, np.array([True, False, False]), b, {}).tolist() == ["unresolved"]
    assert objective_verdicts({"type": "completeness", "column": "Results"}, df, np.array([False, False, True]),
                              b, {}).tolist() == ["error"]


# ----------------------------------------------------------------------------- cross-column rules
def _tax_like(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    zips = [f"{z:05d}" for z in rng.integers(10000, 99999, 600)]
    zip_city = {z: f"CITY{i % 400}" for i, z in enumerate(zips)}
    z = rng.choice(zips, n)
    kids = rng.choice(["Y", "N"], n)
    df = pd.DataFrame({"zip": z, "city": [zip_city[x] for x in z], "has_child": kids,
                       "child_exemp": np.where(kids == "Y", rng.choice([0, 500, 1000], n, p=[.3, .4, .3]), 0).astype(float),
                       "salary": rng.integers(1, 100, n).astype(float) * 1000})
    return df


def test_dependency_and_conditional_checks():
    from rules.static_rules import check, involved_columns
    df = pd.DataFrame({"zip": ["1", "1", "2", None, "9"], "city": ["A", "B", "C", "A", "Z"],
                       "kid": ["N", "N", "Y", "N", None], "ex": [0, 5, 5, None, 3]})
    dep = {"type": "dependency", "columns": ["zip"], "column": "city", "mapping": {"1": "A", "2": "C"}}
    assert check(dep, df).tolist() == [False, True, False, False, False]   # unknown / missing determinant never fails
    con = {"type": "conditional", "if_column": "kid", "if_values": ["N"], "column": "ex", "low": 0, "high": 0}
    assert check(con, df).tolist() == [False, True, False, False, False]
    assert involved_columns(dep) == ["zip", "city"] and involved_columns(con) == ["kid", "ex"]


def test_discovery_finds_planted_dependencies_and_rejects_noise():
    from rules.crosscolumn import discover_rules
    df = _tax_like()
    keys = {r["key"] for r in discover_rules(df, list(df.columns), 0.005)}
    assert "dependency:zip>city" in keys and "conditional:has_child=N>child_exemp" in keys
    assert not any(k.startswith("dependency:salary") or k.endswith(">salary") for k in keys)


def test_llm_crosscolumn_candidates_validated_on_warmup():
    from rules.crosscolumn import validate_candidates
    df = _tax_like()
    cols = list(df.columns)
    cands = [{"rule_type": "dependency", "determinant": ["zip"], "dependent": "city", "confidence": 0.9},
             {"rule_type": "conditional", "if": {"column": "has_child", "values": ["N"]},
              "then": {"column": "child_exemp", "min": 0, "max": 0}, "confidence": 0.8},
             {"rule_type": "conditional", "if": {"column": "has_child", "values": ["Y"]},
              "then": {"column": "child_exemp", "min": 1, "max": None}, "confidence": 0.7},     # false: 30% have 0
             {"rule_type": "dependency", "determinant": ["zip"], "dependent": "city", "confidence": 0.5},
             {"rule_type": "dependency", "determinant": ["nope"], "dependent": "city"},
             {"rule_type": "sql"}]
    st = [r["status"] for r in validate_candidates(cands, [df.iloc[:3000], df.iloc[3000:]], cols, 0.005)]
    assert st == ["accepted", "accepted", "rejected", "redundant", "invalid", "invalid"]


def test_benchmark_prompt_pool_is_disjoint_from_replays(tmp_path):
    from ingestion.benchmarks import BenchmarkReplay, prompt_pool_index
    from common.utils import experiment_config
    n = 4000
    rng = np.random.default_rng(0)
    clean = pd.DataFrame({"a": rng.integers(0, 50, n).astype(str), "b": rng.choice(list("XYZ"), n),
                          "id": [f"r{i}" for i in range(n)]})
    dirty = clean.copy()
    dirty.loc[::50, "b"] = "W"
    bench = {"n_batches": 20, "prompt_pool": {"fraction": 0.1, "min_rows": 500},
             "benchmarks": {"toy": {"numeric": ["a"], "categorical": ["b"], "files": {}}}}
    idx = prompt_pool_index(n, bench, 7)
    assert len(idx) == 500
    rep = BenchmarkReplay("toy", bench, experiment_config(), 3, dirty=dirty, clean=clean, split_seed=7)
    seen = set(pd.concat([rep.batch(t).df for t in range(1, 21)]).id)
    assert not seen & set(rep.prompt_rows.id) and len(rep.prompt_rows) == 500
    assert len(seen) == (n - 500) // 20 * 20
    # labels stay aligned after the reserved rows are removed
    b = rep.batch(4)
    wrong = set(b.df.loc[b.df.b == "W", "record_id"])
    assert wrong == set(b.gt.record_id)


# ----------------------------------------------------------------------------- validator stress, H1a, author
def test_validator_stress_classes_behave(env, monkeypatch):
    import experiments.v2.validator_stress as vs
    datasets, exp, pr, src = env
    monkeypatch.setattr(vs, "data_source", lambda: src)
    df = vs.worker("synth", 42, {"split": "dev", "scenarios": ["A_stable"]})
    st = df.groupby("class").status.agg(lambda s: set(s))
    assert st["malformed"] == {"invalid"}
    assert st["sound"] == {"accepted"}
    assert "accepted" not in st["too_strict"] and "accepted" not in st["wrong"]
    raw = df.groupby("class")[["raw_fp", "admitted_fp"]].sum()
    assert raw.loc["too_strict", "raw_fp"] > 10 * max(raw.loc["too_strict", "admitted_fp"], 1)


def test_h1a_decision_handles_zero_defect_control():
    import experiments.v2.rq1_thresholds as rq1
    import inspect
    src = inspect.getsource(rq1.main)
    assert "recall_applicable" in src and "drift_scenario" in src


def test_author_prompt_has_no_data():
    from common.utils import datasets_config
    from rules.llm_author import AUTHOR_SYSTEM_PROMPT, author_prompt
    for ds, cfg in datasets_config().items():
        p = author_prompt(ds, cfg)
        assert "statistics" not in p.lower() and "example" not in p.lower()
        for d in cfg.get("semantic_defects", []):                 # the injected error values are never shown
            for v in d.get("values", []):
                if len(str(v)) >= 4:
                    assert str(v) not in p
    assert "have not seen any records" in AUTHOR_SYSTEM_PROMPT
