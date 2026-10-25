"""End-to-end smoke test of every v2 runner on the small synthetic dataset, in a temporary workspace.

Nothing is written to the project's data/ or results/ folders and no real LLM is called (mock answers).
"""
import copy
import os

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import MANUAL, SYNTH_CFG, configs, make_clean


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    from common import utils
    import experiments.v2.common as C
    import experiments.v2.llm_generate as G
    import experiments.v2.rq2_coverage as R2
    import experiments.v2.rq2_category_guard as R2b
    from injection.scenario import DataSource
    tmp = tmp_path_factory.mktemp("ws")
    datasets, exp, pr = configs()
    pr = copy.deepcopy(pr)
    pr["splits"]["seeds"]["dev"] = [42, 43]
    pr["rq2"]["generations"] = 2
    pr["cost"]["runtime_trials"]["repeats"] = 1
    src = DataSource(datasets, exp, pr, clean_loader=lambda n: make_clean())
    mp = pytest.MonkeyPatch()
    for mod in (C, G):
        mp.setattr(mod, "datasets_config", lambda: datasets)
        mp.setattr(mod, "experiment_config", lambda: exp)
        mp.setattr(mod, "protocol_config", lambda: pr)
    mp.setattr(G, "DataSource", lambda *a, **k: src)
    mp.setattr(R2, "manual_config", lambda: MANUAL)
    mp.setattr(R2b, "manual_config", lambda: MANUAL)
    C._SOURCES[os.getpid()] = src
    utils.set_workspace(tmp)
    G.main(["--mock"])                                   # offline LLM stand-in answers
    yield tmp, pr
    utils.set_workspace(None)
    C._SOURCES.pop(os.getpid(), None)
    mp.undo()


def test_rq1_thresholds(synth):
    from experiments.v2 import rq1_thresholds
    tmp, pr = synth
    rq1_thresholds.main(["--scenarios", "A_stable", "B_drift", "F1_contam_stable"])
    out = tmp / "results_v2" / "rq1_thresholds" / "dev"
    t = pd.read_csv(out / "rq1a_record_level.csv")
    assert set(t.method) == set(pr["baselines"]["methods"])
    ci = pd.read_csv(out / "rq1a_paired_ci.csv")
    assert {"recall", "fpr"} <= set(ci.metric) and "noninferior" in ci
    assert (out / "rq1a_h1a_decisions.csv").exists() and (out / "manifest.json").exists()


def test_rq1_classification(synth):
    from experiments.v2 import rq1_classification
    tmp, pr = synth
    rq1_classification.main(["--scenarios", "A_stable", "B_drift", "C_degradation", "D_drift_degradation"])
    out = tmp / "results_v2" / "rq1_classification" / "dev"
    s = pd.read_csv(out / "rq1b_classification_summary.csv")
    native = {k for k, v in pr["classification"]["comparators"].items() if v["drift"] not in ("deequ", "tfdv")}
    assert set(s.comparator) == native              # tool comparators are skipped (with a warning) without outputs
    assert s.set_index("comparator").loc["static_no_drift", "accuracy"] <= 0.5 + 1e-9   # structural ceiling
    assert (out / "rq1b_paired_ci.csv").exists() and (out / "rq1b_delays.csv").exists()


def test_rq2_coverage_mock(synth):
    from experiments.v2 import rq2_coverage
    tmp, pr = synth
    rq2_coverage.main(["--mock", "--scenarios", "S1_semantic_stable"])
    out = tmp / "results_v2" / "rq2_coverage" / "dev"
    s = pd.read_csv(out / "rq2a_summary_mock.csv")
    assert {"stat_fixed", "stat_adaptive", "llm_validated", "llm_raw", "adaptive+llm_validated", "manual",
            "adaptive+manual"} <= set(s.config)
    assert set(s.redundancy.dropna()) >= {"empirical", "logical"}
    man = s[(s.config == "adaptive+manual")].iloc[0]
    assert man.semantic_recall_mean > s[s.config == "stat_adaptive"].iloc[0].semantic_recall_mean   # manual y >= 0 rule
    for f in ("rq2a_paired_ci_mock.csv", "rq2a_admission_errors_mock.csv", "rq2a_predicate_stability_mock.csv",
              "rq2a_generation_cost_mock.csv"):
        assert (out / f).exists(), f


def test_rq2_category_guard_mock(synth):
    from experiments.v2 import rq2_category_guard
    tmp, pr = synth
    rq2_category_guard.main(["--mock", "--scenarios", "G1_novelty_stable"])
    out = tmp / "results_v2" / "rq2_category_guard" / "dev"
    acc = pd.read_csv(out / "rq2b_acceptance_per_value_mock.csv")
    st = acc[(acc.column == "state")]
    rec = st[st.policy == "recurrence"]
    assert rec[rec.kind == "invalid"].accepted.all()                     # PoC rule absorbs the invalid value
    man = st[st.policy == "recurrence_manual"]
    assert not man[man.kind == "invalid"].accepted.any() and man[man.kind == "valid"].accepted.all()
    fro = st[st.policy == "frozen"]
    assert not fro.accepted.any()                                        # frozen never accepts novelty
    val = st[st.policy == "recurrence_llm_value"]                       # per-value LLM check (mock verdicts)
    assert len(val) and val[val.kind == "valid"].accepted.all()
    assert set(val.final_status) <= {"admitted_llm_valid", "rejected", "admitted_llm_unsure", "not_recurring"}
    c = acc[(acc.column == "c") & (acc.policy == "recurrence_manual")]   # pattern-only column: unresolved
    assert c.unresolved.all()


def test_rq3_scheduling_and_runtime(synth):
    from experiments.v2 import rq3_scheduling
    tmp, pr = synth
    rq3_scheduling.main(["--scenarios", "A_stable", "H3_short_incident", "C_degradation", "--runtime", "--seeds", "42"])
    out = tmp / "results_v2" / "rq3_scheduling" / "dev"
    t = pd.read_csv(out / "rq3_detection_cost.csv")
    assert set(t.policy) == set(pr["schedulers"]["policies"])
    allp = t[t.policy == "all"]
    assert (allp.saving_checks_pct.abs() < 1e-9).all()
    inc = pd.read_csv(out / "rq3_incidents.csv")
    assert {"short", "degradation"} <= set(inc.incident)
    rt = pd.read_csv(out / "rq3_runtime.csv")
    assert "ms_profile_median" in rt and "peak_memory_mb" in rt and set(rt.reuse) == {True, False}


def test_ablation_mock(synth):
    from experiments.v2 import ablation
    tmp, pr = synth
    ablation.main(["--mock", "--scenarios", "C_degradation", "S2_semantic_drift", "--seeds", "42"])
    out = tmp / "results_v2" / "ablation" / "dev"
    s = pd.read_csv(out / "ablation_summary_mock.csv")
    assert len(s) == 8 + 1                                               # 8 combinations + fixed_t1 reference
    assert "adaptive|validated|all" in set(s.config)                     # the previously missing condition
    assert (out / "ablation_main_effects_mock.csv").exists()


def test_materialize(synth):
    from experiments.v2 import materialize
    tmp, pr = synth
    materialize.main(["--scenarios", "S1_semantic_stable", "G1_novelty_stable", "--seeds", "42"])
    import json
    a = json.loads((tmp / "v2" / "dev" / "synth" / "seed42" / "S1_semantic_stable" / "audit.json").read_text())
    assert a["all_ok"]


def test_external_benchmark_replay(synth):
    from common.utils import experiment_config, protocol_config
    from experiments.v2.external import run_unit
    from ingestion.benchmarks import BenchmarkReplay, error_cells
    rng = np.random.default_rng(0)
    n = 600
    clean = pd.DataFrame({"index": np.arange(n).astype(str), "a": rng.normal(10, 1, n).round(3).astype(str),
                          "s": rng.choice(["X", "Y", "Z"], n), "note": ""})
    dirty = clean.copy()
    dirty.loc[::17, "a"] = dirty.loc[::17, "a"] + " units"
    dirty.loc[::23, "s"] = "X-*"
    dirty.loc[5, "a"] = "10.0" if clean.loc[5, "a"] == "10" else dirty.loc[5, "a"]
    e = error_cells(dirty, clean)
    assert e["a"].sum() == len(range(0, n, 17)) and e["s"].sum() == len(range(0, n, 23))
    bench = {"n_batches": 20, "label_normalization": "numeric_equal", "empty_as_missing": True,
             "benchmarks": {"toy": {"numeric": ["a"], "categorical": ["s"], "nullable": [], "drop_columns": ["index"],
                                    "files": {"dirty": "x", "clean": "y"}}}}
    rep = BenchmarkReplay("toy", bench, experiment_config(), 1, dirty=dirty, clean=clean)
    out = run_unit("toy", bench, experiment_config(), protocol_config(), 1, methods=["adaptive"], replay=rep)
    ev = out[out.evaluation == 1]
    assert ev.TP.sum() > 0 and ev.cell_TP.sum() > 0
    assert ev["tp_col:a"].sum() == ev["n_col:a"].sum()                   # unit strings fail the type check
    # the corrupted category "X-*" recurs in every batch, so recurrence learns it (the H2b absorption effect)
    assert ev["tp_col:s"].sum() < ev["n_col:s"].sum()


def test_temporal_monthly_replay():
    from common.utils import datasets_config, experiment_config, protocol_config
    from experiments.v2.temporal_chicago import MonthlyReplay
    from monitor.loop import Monitor, MonitorConfig
    rng = np.random.default_rng(0)
    n = 9000
    raw = pd.DataFrame({"Inspection ID": np.arange(n), "Inspection Date": pd.Timestamp("2017-01-01") +
                        pd.to_timedelta(rng.integers(0, 365, n), unit="D"),
                        "Risk": rng.choice(["Risk 1 (High)", "Risk 2 (Medium)"], n), "Results": rng.choice(["Pass", "Fail"], n),
                        "Facility Type": rng.choice(["Restaurant", "School"], n),
                        "Latitude": rng.normal(41.88, 0.08, n), "Longitude": rng.normal(-87.67, 0.06, n)})
    cfg = datasets_config()["chicago_food"]
    rep = MonthlyReplay(raw, cfg, 5)
    assert rep.n_batches == 12 and rep.batch(1).label == "2017-01"
    m = Monitor(cfg, experiment_config(), protocol_config(), MonitorConfig(name="frozen"))
    labels = [m.step(rep.batch(t).df, rep.batch(t).label)["label"].get("label") for t in range(1, 13)]
    assert labels[5:] and all(l in ("stable", "drift", "degradation", "drift+degradation", "insufficient") for l in labels[5:])


def test_benchmark_column_without_any_numeric_value():
    """Beers: every 'ounces' value carries a unit ("12.0 oz"), so the column has no numeric statistics."""
    from common.utils import experiment_config, protocol_config
    from experiments.v2.external import run_unit
    from ingestion.benchmarks import BenchmarkReplay
    rng = np.random.default_rng(1)
    n = 600
    clean = pd.DataFrame({"oz": rng.choice(["12", "16"], n), "abv": rng.normal(0.06, 0.01, n).round(3).astype(str),
                          "s": rng.choice(["X", "Y"], n)})
    dirty = clean.assign(oz=clean.oz + ".0 oz")
    bench = {"n_batches": 20, "label_normalization": "numeric_equal", "empty_as_missing": True,
             "benchmarks": {"toy": {"numeric": ["oz", "abv"], "categorical": ["s"], "nullable": [],
                                    "files": {"dirty": "x", "clean": "y"}}}}
    rep = BenchmarkReplay("toy", bench, experiment_config(), 1, dirty=dirty, clean=clean)
    for method in ("fixed_t1", "fixed_pooled", "rolling_mad", "adaptive"):
        out = run_unit("toy", bench, experiment_config(), protocol_config(), 1, methods=[method], replay=rep)
        ev = out[out.evaluation == 1]
        assert ev["tp_col:oz"].sum() == ev["n_col:oz"].sum()          # every unit string fails the type check


def test_monitor_and_avh_with_statless_numeric_column(env):
    from baselines.avh import AVHMonitor
    from monitor.loop import Monitor, MonitorConfig
    from profiling.profiler import profile_batch
    datasets, exp, pr, src = env
    r = src.replay("synth", "dev", 42, "B_drift")
    m = Monitor(SYNTH_CFG, exp, pr, MonitorConfig(name="x", scheduler="risk"))
    avh = AVHMonitor(["x", "y"], ["c", "state"], 0.05, 4)
    for b in r.batches():
        df = b.df.assign(y="n/a")                                       # no numeric value at all in y
        st = m.step(df, b.label)
        avh.observe(profile_batch(df, ["x", "y"], ["c", "state"]))
    assert st["label"]["label"] and "y" not in [k.split(":")[1] for k in st["label"]["drifted_columns"].split(",") if k]


@pytest.fixture(scope="module")
def env():
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    return datasets, exp, pr, DataSource(datasets, exp, pr, clean_loader=lambda name: make_clean())


def test_cached_units_recomputed_without_or_with_changed_fingerprint(tmp_path, monkeypatch):
    import argparse
    import experiments.v2.common as C
    from common import utils
    datasets, exp, pr = configs()
    for name, val in (("datasets_config", lambda: datasets), ("experiment_config", lambda: exp),
                      ("protocol_config", lambda: pr)):
        monkeypatch.setattr(C, name, val)
    utils.set_workspace(tmp_path)
    try:
        args = argparse.Namespace(split="dev", seeds=[42], datasets=None, scenarios=["A_stable"], jobs=1,
                                  force=False, summary_only=False)
        ctx = C.Context("cache_test", args)
        calls = []
        worker = lambda ds, s, p: (calls.append((ds, s)), pd.DataFrame({"v": [p.get("tag")]}))[1]
        pd.DataFrame({"v": ["old"]}).to_parquet(ctx.unit_path("synth", 42))     # result from before fingerprints
        out = C.run_units(ctx, worker, {"tag": "a"})
        assert calls == [("synth", 42)] and out.v.tolist() == ["a"]             # recomputed, not trusted
        C.run_units(ctx, worker, {"tag": "a"})
        assert len(calls) == 1                                                  # same fingerprint: reused
        C.run_units(ctx, worker, {"tag": "b"})
        assert len(calls) == 2                                                  # changed configuration: recomputed
    finally:
        utils.set_workspace(None)


def test_external_with_llm_rules(synth):
    """Benchmark prompt from the dirty table only, and adaptive + validated LLM rules per generation."""
    from common.utils import data_dir, experiment_config, protocol_config
    from experiments.v2.external import run_unit
    from experiments.v2.llm_rules import benchmark_prompt_and_fingerprint
    from ingestion.benchmarks import BenchmarkReplay
    rng = np.random.default_rng(3)
    n = 800
    clean = pd.DataFrame({"a": rng.normal(10, 1, n).round(3).astype(str), "s": rng.choice(["X", "Y"], n),
                          "name": [f"p{i}" for i in range(n)]})
    dirty = clean.copy()
    dirty.loc[::19, "a"] = "-500"                                          # impossible value, inside no stat bound
    dirty.loc[1::4, "a"] = dirty.loc[1::4, "a"] + " units"               # Beers-like text in a numeric column
    d = data_dir("external", "toy")
    d.mkdir(parents=True, exist_ok=True)
    dirty.to_csv(d / "dirty.csv", index=False)
    clean.to_csv(d / "clean.csv", index=False)
    bench = {"n_batches": 20, "label_normalization": "numeric_equal", "empty_as_missing": True,
             "prompt_pool": {"fraction": 0.1, "min_rows": 100},
             "benchmarks": {"toy": {"numeric": ["a"], "categorical": ["s"], "nullable": [],
                                    "files": {"dirty": "toy/dirty.csv", "clean": "toy/clean.csv"}}}}
    pf = benchmark_prompt_and_fingerprint("toy", bench, experiment_config(), protocol_config(), "m")
    assert "Column: a" in pf["prompt"] and "name" in pf["prompt"] and pf["fingerprint"]["key"]
    assert "not plain numbers" in pf["prompt"] and "units" in pf["prompt"]
    rule = {"column": "a", "rule_type": "range", "min": 0, "max": 1000, "allowed_values": None, "pattern": None,
            "condition": "non-negative", "reason": "", "confidence": 0.9}
    rep = BenchmarkReplay("toy", bench, experiment_config(), 1)
    out = run_unit("toy", bench, experiment_config(), protocol_config(), 1, methods=["adaptive"], replay=rep,
                   llm_gens={"primary": {1: {"rules": [rule]}, 2: {"rules": []}}})
    ev = out[out.evaluation == 1]
    assert set(ev.method) == {"adaptive", "adaptive+llm_primary", "adaptive+deps"} and set(ev.generation) == {0, 1, 2}
    llm1 = ev[(ev.method == "adaptive+llm_primary") & (ev.generation == 1)]
    base = ev[ev.method == "adaptive"]
    assert llm1["tp_col:a"].sum() >= base["tp_col:a"].sum() and llm1.TP.sum() >= base.TP.sum()
