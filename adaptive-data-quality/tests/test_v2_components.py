"""Tests for the v2 components: splits, scenario engine, baselines, AVH, rules, guard, monitor."""
import copy

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import MANUAL, STATES, SYNTH_CFG, configs, make_clean


@pytest.fixture(scope="module")
def env():
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    src = DataSource(datasets, exp, pr, clean_loader=lambda name: make_clean())
    return datasets, exp, pr, src


# ----------------------------------------------------------------------------- splits
def test_splits_disjoint_and_test_guard():
    from ingestion.splits import check_split_allowed, pool_overlap, split_pools
    df = make_clean(5000)
    p = split_pools(df, 0.3, 1)
    assert len(p["dev"]) + len(p["test"]) == len(df)
    assert pool_overlap(p["dev"], p["test"], ["key"]) == 0
    _, _, pr = configs()
    pr = copy.deepcopy(pr)
    pr["protocol"]["frozen"] = False
    with pytest.raises(PermissionError):
        check_split_allowed("test", pr)
    check_split_allowed("dev", pr)
    pr["protocol"]["frozen"] = True
    check_split_allowed("test", pr)


# ----------------------------------------------------------------------------- scenario engine
def test_every_scenario_builds_with_audited_labels(env):
    datasets, exp, pr, src = env
    for scen in pr["scenarios"]:
        r = src.replay("synth", "dev", 42, scen)
        gt = r.ground_truth()
        assert all(a["ok"] for a in r.audits.values()), scen
        assert gt.record_id.is_unique, scen


def test_replay_is_deterministic_and_paired(env):
    _, _, _, src = env
    a = src.replay("synth", "dev", 42, "D_drift_degradation").batch(15)
    b = src.replay("synth", "dev", 42, "D_drift_degradation").batch(15)
    assert a.df.equals(b.df) and a.gt.equals(b.gt)
    # paired scenarios share base records
    c = src.replay("synth", "dev", 42, "A_stable").batch(15)
    assert set(c.df.record_id) & set(a.df.record_id)


def test_novelty_absent_from_warmup_and_matched(env):
    _, _, _, src = env
    r = src.replay("synth", "dev", 42, "G1_novelty_stable")
    for t in range(1, 6):
        assert not r.batch(t).df["state"].isin(["FF", "QZ"]).any()
    b = r.batch(10)
    nov = b.novelty
    cnt = nov.groupby("value").size()
    assert cnt["FF"] == cnt["QZ"] and cnt["J"] == cnt["Q"]            # matched frequency
    assert set(b.gt.loc[b.gt.defect_type == "invalid_novel", "record_id"]) == set(nov[nov.kind == "invalid"].record_id)
    assert not set(nov[nov.kind == "valid"].record_id) & b.defective   # valid novelty is not a defect


def test_incident_and_contamination_labels(env):
    _, _, _, src = env
    r = src.replay("synth", "dev", 42, "H3_short_incident")
    assert r.batch(9).meta["degradation_label"] == 1 and r.batch(11).meta["degradation_label"] == 0
    assert (r.batch(9).gt.incident == "short").sum() > 0
    r = src.replay("synth", "dev", 42, "F1_contam_stable")
    assert (r.batch(8).gt.defect_type == "contamination").sum() == round(0.05 * r.batch_size)
    assert (r.batch(7).gt.defect_type == "contamination").sum() == 0


# ----------------------------------------------------------------------------- baselines
def _profiles(n=6, shift=0.0):
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(0)
    out = []
    for i in range(n):
        df = pd.DataFrame({"x": rng.normal(100 * (1 + shift * i), 10, 2000), "c": rng.choice(list("AB"), 2000)})
        out.append(profile_batch(df, ["x"], ["c"]))
    return out


def test_pooled_moments_exact():
    from adaptation.baselines import pooled_moments
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(1)
    parts = [pd.DataFrame({"x": rng.normal(m, s, n)}) for m, s, n in [(0, 1, 300), (5, 2, 500), (-3, 0.5, 200)]]
    mu, sd = pooled_moments([profile_batch(p, ["x"], []) for p in parts], "x")
    allx = pd.concat(parts).x
    assert mu == pytest.approx(allx.mean()) and sd == pytest.approx(allx.std(ddof=1))


def test_every_numeric_policy_produces_sane_bounds():
    from adaptation.baselines import NUMERIC_POLICIES, numeric_rules
    ps = _profiles()
    for pol in NUMERIC_POLICIES:
        r = numeric_rules(pol, ["x"], 3, ps[:5], ps[:5], ps[:5], ps[5], {"ewma_lambda": 0.3, "iqr_k": 3, "mad_k": 3})[0]
        assert 40 < r["low"] < 90 and 110 < r["high"] < 160, pol


def test_rolling_and_adaptive_follow_drift_fixed_does_not():
    from adaptation.baselines import numeric_rules
    ps = _profiles(10, shift=0.05)
    pr = {"ewma_lambda": 0.3, "iqr_k": 3, "mad_k": 3}
    fixed = numeric_rules("fixed_pooled", ["x"], 3, ps[:5], ps[4:9], ps[:9], ps[9], pr)[0]
    roll = numeric_rules("rolling_mad", ["x"], 3, ps[:5], ps[4:9], ps[:9], ps[9], pr)[0]
    adapt = numeric_rules("adaptive", ["x"], 3, ps[:5], ps[4:9], ps[:9], ps[9], pr)[0]
    assert fixed["high"] < roll["high"] < adapt["high"] + 1e-9 or fixed["high"] < adapt["high"]


def test_avh_flags_shift_not_noise():
    from baselines.avh import AVHMonitor
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(2)
    m = AVHMonitor(["x"], ["c"], 0.05, 4)
    res, alarms, ready = None, 0, 0
    for i in range(40):                                # false-alarm rate on pure noise stays near the 5% budget
        df = pd.DataFrame({"x": rng.normal(100, 10, 3000), "c": rng.choice(list("AB"), 3000)})
        res = m.observe(profile_batch(df, ["x"], ["c"]))
        ready += res["ready"]; alarms += res["alarm"]
    assert ready >= 30 and alarms / ready <= 0.15
    df = pd.DataFrame({"x": rng.normal(140, 10, 3000), "c": rng.choice(list("AB"), 3000)})
    res = m.observe(profile_batch(df, ["x"], ["c"]))
    assert res["alarm"] and res["drift"]
    df = pd.DataFrame({"x": rng.normal(100, 10, 3000), "c": rng.choice(list("AB"), 3000)})
    df.loc[:300, "c"] = "NEW"
    res = m.observe(profile_batch(df, ["x"], ["c"]))
    assert res["degradation"]


# ----------------------------------------------------------------------------- rules
def test_logical_redundancy_keeps_state_list_empirical_drops_it():
    from rules.validator import validate
    rng = np.random.default_rng(3)
    cfg = {"numeric": ["x"], "categorical": ["state"]}
    warm = [pd.DataFrame({"record_id": [f"r{i}" for i in range(1000)], "x": rng.normal(0, 1, 1000),
                          "state": rng.choice(STATES, 1000)}) for _ in range(5)]
    for w in warm:
        w.loc[:2, "state"] = "lowercase"              # malformed tokens both rules reject
    cands = [{"column": "state", "rule_type": "pattern", "pattern": "[A-Z]{2}", "confidence": 0.9},
             {"column": "state", "rule_type": "category", "allowed_values": STATES, "confidence": 0.8}]
    emp = [v["status"] for v in validate(cands, cfg, warm, 0.01, "empirical")]
    log = [v["status"] for v in validate(cands, cfg, warm, 0.01, "logical")]
    assert emp == ["accepted", "redundant"]           # the proof-of-concept failure
    assert log == ["accepted", "accepted"]
    # a pattern IS implied by a list whose values all match it
    cands2 = [dict(cands[0], confidence=0.5), cands[1]]   # the list is now processed first
    assert [v["status"] for v in validate(cands2, cfg, warm, 0.01, "logical")] == ["redundant", "accepted"]


def test_manual_rules_load():
    from rules.manual_rules import domain_constraints, manual_rules
    rules = manual_rules("synth", SYNTH_CFG, ["record_id", "key", "x", "y", "c", "state"], MANUAL)
    assert all(r["key"].startswith("manual") for r in rules)
    dc = domain_constraints(rules)
    assert set(dc) == {"state", "c"}


def test_real_manual_rules_are_valid():
    from common.utils import datasets_config
    from rules.manual_rules import manual_rules
    for ds, cfg in datasets_config().items():
        cols = ["record_id"] + list(dict.fromkeys(cfg["key_columns"] + cfg["numeric"] + cfg["categorical"]))
        assert manual_rules(ds, cfg, cols)


def test_category_guard_policies():
    from adaptation.category_guard import CategoryGuard
    from rules.manual_rules import domain_constraints, manual_rules
    warm = [{"columns": {"state": {"frequencies": {"AA": .5, "BB": .5}}}}] * 5
    hist = [{"columns": {"state": {"frequencies": {"AA": .4, "BB": .4, "FF": .1, "QZ": .1}}}}] * 2
    cons = domain_constraints(manual_rules("synth", SYNTH_CFG, ["record_id", "x", "y", "c", "state"], MANUAL))
    allowed = {}
    for pol in ("frozen", "recurrence", "recurrence_manual"):
        rules, status = CategoryGuard(pol, ["state"], 2, cons).rules(warm, hist)
        allowed[pol] = set(rules[0]["allowed"])
        if pol == "recurrence_manual":
            assert status["state"]["QZ"] == "rejected" and status["state"]["FF"] == "admitted_by_list"
    assert allowed["frozen"] == {"AA", "BB"}
    assert {"FF", "QZ"} <= allowed["recurrence"]
    assert "FF" in allowed["recurrence_manual"] and "QZ" not in allowed["recurrence_manual"]
    # pattern-only column: well-formed invalid value is compatible, not validated
    from adaptation.category_guard import admission_status
    assert admission_status("Q", [{"type": "pattern", "pattern": "[A-Z]"}]) == "admitted_compatible"
    assert admission_status("Q", []) == "admitted_unconstrained"


# ----------------------------------------------------------------------------- executor reuse
def test_profile_reuse_gives_identical_results(env):
    from execution.local_executor import execute
    from profiling.profiler import profile_batch
    from rules.static_rules import base_rules
    _, _, _, src = env
    b = src.replay("synth", "dev", 42, "S2_semantic_drift").batch(12)
    prof, art = profile_batch(b.df, SYNTH_CFG["numeric"], SYNTH_CFG["categorical"], return_artifacts=True)
    rules = base_rules(list(b.df.columns), SYNTH_CFG["numeric"], SYNTH_CFG["categorical"]) + [
        {"type": "range", "column": "x", "low": 0, "high": 10 ** 9}, {"type": "range", "column": "y", "low": 0, "high": 70},
        {"type": "category", "column": "state", "allowed": STATES}, {"type": "pattern", "column": "c", "pattern": "[A-I]"}]
    plain = execute(rules, b.df, return_masks=True)
    fast = execute(rules, b.df, reuse={"profile": prof, **art}, return_masks=True)
    assert plain["failed_ids"] == fast["failed_ids"]
    for k in plain["masks"]:
        assert (plain["masks"][k] == fast["masks"][k]).all(), k
    assert fast["bytes_scanned"] < plain["bytes_scanned"]


# ----------------------------------------------------------------------------- integrated monitor
def _run(env, scenario, mc, seed=42):
    from monitor.loop import Monitor
    datasets, exp, pr, src = env
    r = src.replay("synth", "dev", seed, scenario)
    m = Monitor(SYNTH_CFG, exp, pr, mc, seed=seed)
    return [m.step(b.df, b.label) for b in r.batches()], r


def test_monitor_runs_every_scheduler_and_labels(env):
    from monitor.loop import MonitorConfig
    for pol in ("all", "periodic", "random", "failure_only", "drift_only", "risk"):
        steps, r = _run(env, "D_drift_degradation",
                        MonitorConfig(name=pol, scheduler=pol, degradation_rule="per_check", reuse_profile_stats=True))
        ev = [s for s in steps if not s["warmup"]]
        assert len(ev) == 15
        assert all(s["label"]["label"] in ("stable", "drift", "degradation", "drift+degradation", "insufficient") for s in ev)
        if pol == "all":
            assert all(s["scheduler"]["coverage"] == 1.0 for s in ev)
        # basic checks always run
        for s in ev:
            types = {p["type"] for p in s["per_rule"]}
            assert {"completeness", "type", "range"} <= types


def test_monitor_detects_drift_and_degradation(env):
    from monitor.loop import MonitorConfig
    steps, r = _run(env, "D_drift_degradation", MonitorConfig(name="p", method="adaptive"))
    last = steps[-1]["label"]
    assert last["drift"] and last["degradation"]
    steps, r = _run(env, "A_stable", MonitorConfig(name="p", method="adaptive"))
    labels = [s["label"]["label"] for s in steps if not s["warmup"]]
    assert labels.count("stable") >= 10


def test_reference_pass_does_not_influence_monitor(env):
    from execution.local_executor import execute
    from monitor.loop import Monitor, MonitorConfig
    datasets, exp, pr, src = env
    r = src.replay("synth", "dev", 42, "C_degradation")
    mc = MonitorConfig(name="risk", scheduler="risk")
    a, b = Monitor(SYNTH_CFG, exp, pr, mc), Monitor(SYNTH_CFG, exp, pr, mc)
    for batch in r.batches():
        sa = a.step(batch.df, batch.label)
        execute(a.last_suite, batch.df)                      # evaluation-only reference pass
        sb = b.step(batch.df, batch.label)
        assert sa["failed_ids"] == sb["failed_ids"]
        assert sa["scheduler"].get("standard_executed") == sb["scheduler"].get("standard_executed")


def test_monitor_phase_costs_and_persist(env):
    from monitor.loop import PHASES, MonitorConfig
    steps, _ = _run(env, "A_stable", MonitorConfig(name="p", persist=True))
    s = steps[-1]
    assert set(s["ms"]) == set(PHASES) and s["ms"]["profile"] > 0 and s["ms"]["persist"] > 0
    assert s["ms_total"] >= s["ms_total_upstream_profile"]
    assert steps[4]["calibrate_ms"] > 0


def test_llm_only_suite_has_no_statistical_rules(env):
    from monitor.loop import MonitorConfig
    extra = [{"type": "range", "column": "y", "low": 0, "high": None, "key": "llm0:range:y"}]
    steps, _ = _run(env, "S1_semantic_stable", MonitorConfig(name="llm", extra_rules=extra, include_statistical=False,
                                                             include_base=False, classify=False))
    assert {p["key"] for p in steps[-1]["per_rule"]} == {"llm0:range:y"}


# ----------------------------------------------------------------------------- dev-run fixes (5 Oct 2026)
def test_periodic_runs_at_exact_fractional_rate():
    from monitor.schedulers import Scheduler
    from common.utils import experiment_config
    exp = experiment_config()
    rules = [{"type": "category", "column": f"c{i}", "allowed": ["A"]} for i in range(40)]
    s = Scheduler("periodic", exp, ["_records"], {}, rate=0.75)
    n = 0
    for t in range(6, 46):
        chosen, info = s.select(rules, t, {}, {})
        n += info["standard_executed"]
    assert abs(n / (40 * 40) - 0.75) < 0.03          # an integer period would have run everything (rate 1.0)


def test_avh_student_bound_alarms_where_chebyshev_does_not():
    from baselines.avh import AVHMonitor
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(5)
    mons = {b: AVHMonitor(["x"], ["c"], 0.05, 4, bound=b) for b in ("student", "chebyshev")}
    for i in range(8):
        df = pd.DataFrame({"x": rng.normal(100, 10, 3000), "c": rng.choice(list("AB"), 3000)})
        p = profile_batch(df, ["x"], ["c"])
        for m in mons.values():
            m.observe(p)
    df = pd.DataFrame({"x": rng.normal(102, 10, 3000), "c": rng.choice(list("AB"), 3000)})   # moderate shift
    p = profile_batch(df, ["x"], ["c"])
    res = {b: m.observe(p) for b, m in mons.items()}
    assert res["student"]["alarm"] and not res["chebyshev"]["alarm"]


def test_t_quantile_approximation():
    from baselines.avh import t_quantile
    for p, nu, exact in [(0.975, 4, 2.776), (0.99833, 8, 4.121), (0.9995, 14, 4.140), (0.999, 4, 7.173)]:
        assert abs(t_quantile(p, nu) - exact) / exact < 0.01


def test_semantic_and_invalid_novelty_count_as_degradation(env):
    _, _, _, src = env
    s = src.replay("synth", "dev", 42, "S1_semantic_stable")
    assert s.batch(5).meta["degradation_label"] == 0 and s.batch(6).meta["degradation_label"] == 1
    g = src.replay("synth", "dev", 42, "G1_novelty_stable")
    assert g.batch(5).meta["degradation_label"] == 0 and g.batch(6).meta["degradation_label"] == 1
    assert src.replay("synth", "dev", 42, "B_drift").batch(20).meta["degradation_label"] == 0


def test_drift_redraws_only_common_categories():
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    df = make_clean()
    df.loc[:4, "c"] = "RARE"                            # 5 of 20,000 records
    src = DataSource(datasets, exp, pr, clean_loader=lambda n: df)
    r = src.replay("synth", "dev", 42, "B_drift")
    assert "RARE" not in r.top["c"] and len(r.top["c"]) >= 5


def test_rolling_drift_reference_runs(env):
    from monitor.loop import MonitorConfig
    steps, _ = _run(env, "B_drift", MonitorConfig(name="r", drift_reference="rolling"))
    assert all(s["label"]["label"] for s in steps if not s["warmup"])


# ----------------------------------------------------------------------------- dev run 2 fixes
def test_drift_threshold_multipliers():
    from monitor.classifier import BatchClassifier
    _, exp, _ = configs()
    z = BatchClassifier(SYNTH_CFG, exp, drift_threshold={"method": "zscore"})
    pb = BatchClassifier(SYNTH_CFG, exp, drift_threshold={"method": "prediction_bound", "alpha": 0.05})
    assert z.drift_multiplier(5, 7) == exp["classification"]["calibration_z"]
    k = pb.drift_multiplier(5, 7)                       # t(1 - 0.05/7, 4) * sqrt(1.2): about 5
    assert 4.5 < k < 5.8
    assert pb.drift_multiplier(5, 2) < k                 # fewer columns -> smaller Bonferroni correction
    with pytest.raises(ValueError):
        BatchClassifier(SYNTH_CFG, exp, drift_threshold={"method": "nope"})


def test_prediction_bound_false_drift_near_alpha():
    """Simulated LOO calibration: 5 normal scores per column, 7 columns, max over columns of a new stable score."""
    from monitor.classifier import BatchClassifier
    _, exp, _ = configs()
    rng = np.random.default_rng(0)
    rates = {}
    for method in ("zscore", "prediction_bound"):
        c = BatchClassifier(SYNTH_CFG, exp, drift_threshold={"method": method, "alpha": 0.05})
        k = c.drift_multiplier(5, 7)
        cal = rng.normal(size=(20000, 5, 7))
        tau = cal.mean(1) + k * cal.std(1, ddof=1)
        new = rng.normal(size=(20000, 7))
        rates[method] = float((new > tau).any(1).mean())
    assert 0.12 < rates["zscore"] < 0.25                 # the dev-run-2 rate
    assert rates["prediction_bound"] < 0.07


def test_min_effect_floor(env):
    from monitor.loop import Monitor, MonitorConfig
    datasets, exp, pr, src = env
    pr2 = copy.deepcopy(pr)
    pr2["classification"]["drift_threshold"] = {"method": "zscore", "min_effect": {"numeric": 0.5, "categorical": 0.5}}
    m = Monitor(SYNTH_CFG, exp, pr2, MonitorConfig(name="floor"), seed=42)
    for b in src.replay("synth", "dev", 42, "A_stable").batches():
        m.step(b.df, b.label)
    assert all(v >= 0.5 for v in m.classifier.tau_drift.values())


def test_avh_accepted_history_keeps_alarming_after_step():
    from baselines.avh import AVHMonitor
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(3)
    mons = {h: AVHMonitor(["x"], ["c"], 0.05, 4, history=h) for h in ("accepted", "all")}
    alarms = {h: 0 for h in mons}
    for i in range(16):
        mu = 100 if i < 9 else 130                    # persistent step change from batch 10
        df = pd.DataFrame({"x": rng.normal(mu, 10, 3000), "c": rng.choice(list("AB"), 3000)})
        p = profile_batch(df, ["x"], ["c"])
        for h, m in mons.items():
            r = m.observe(p)
            alarms[h] += int(i >= 9 and r["alarm"])
    assert alarms["accepted"] == 7 and alarms["all"] < 7


def test_dataset_drift_column_override():
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    pr["dataset_scenarios"]["synth"]["drift_numeric"] = ["y"]
    src = DataSource(datasets, exp, pr, clean_loader=lambda name: make_clean())
    r = src.replay("synth", "dev", 42, "B_drift")
    a = src.replay("synth", "dev", 42, "A_stable")
    assert r.drift_numeric == ["y"]
    bd, ba = r.batch(20).df, a.batch(20).df
    assert abs(pd.to_numeric(bd.x, errors="coerce").median() / pd.to_numeric(ba.x, errors="coerce").median() - 1) < 0.02
    assert pd.to_numeric(bd.y, errors="coerce").median() > 1.1 * pd.to_numeric(ba.y, errors="coerce").median()


def test_monthly_replay_drops_partial_last_month():
    from experiments.v2.temporal_chicago import MonthlyReplay
    days = pd.date_range("2020-01-01", "2020-03-31", freq="D").append(pd.date_range("2020-04-01", "2020-04-03"))
    raw = pd.DataFrame({"Inspection Date": days, "Inspection ID": np.arange(len(days))})
    rep = MonthlyReplay(raw, {}, warm=1)
    assert rep.months == ["2020-01", "2020-02", "2020-03"] and rep.dropped_months == ["2020-04"]
    full = raw[raw["Inspection Date"] < "2020-04-01"]
    full = pd.concat([full, pd.DataFrame({"Inspection Date": pd.to_datetime(["2020-04-28"]), "Inspection ID": [999]})])
    assert MonthlyReplay(full, {}, warm=1).dropped_months == []


# ----------------------------------------------------------------------------- final-run additions
def test_calibrated_robust_k_matches_pooled_flag_rate():
    from adaptation.baselines import calibrate_robust_k, numeric_rules, pooled_moments
    from profiling.profiler import profile_batch
    rng = np.random.default_rng(0)
    dfs = [pd.DataFrame({"x": rng.lognormal(3, 1, 5000)}) for _ in range(5)]
    profs = [profile_batch(d, ["x"], []) for d in dfs]
    for pol in ("rolling_mad_cal", "rolling_iqr_cal"):
        k = calibrate_robust_k(pol, dfs, profs, ["x"], 3.0)["x"]
        assert k > 3.0                                       # skewed data: wider than the textbook k = 3
        new = pd.DataFrame({"x": rng.lognormal(3, 1, 20000)})
        r = numeric_rules(pol, ["x"], 3.0, profs, profs, profs, None, {"k_cal": {"x": k}, "mad_k": 3, "iqr_k": 3})[0]
        mu, sd = pooled_moments(profs, "x")
        rate_cal = ((new.x < r["low"]) | (new.x > r["high"])).mean()
        rate_pooled = ((new.x < mu - 3 * sd) | (new.x > mu + 3 * sd)).mean()
        assert abs(rate_cal - rate_pooled) < 0.01            # same false-positive level as the primary comparator


def test_calibrated_policy_runs_in_monitor(env):
    from monitor.loop import MonitorConfig
    steps, _ = _run(env, "A_stable", MonitorConfig(name="c", method="rolling_mad_cal", classify=False))
    rng_rules = [r for r in steps[-1]["per_rule"] if r["type"] == "range"]
    assert rng_rules


def test_value_verdict_guard():
    from adaptation.category_guard import CategoryGuard
    warm = [{"columns": {"c": {"frequencies": {"A": .5, "B": .5}}}}] * 5
    hist = [{"columns": {"c": {"frequencies": {"A": .4, "B": .4, "NEWOK": .1, "NEWBAD": .05, "NEW?": .05}}}}] * 5
    g = CategoryGuard("recurrence_llm_value", ["c"], 2, {"c": {"NEWOK": "valid", "NEWBAD": "invalid"}})
    rules, st = g.rules(warm, hist)
    assert set(rules[0]["allowed"]) == {"A", "B", "NEWOK", "NEW?"}
    assert st["c"]["NEWBAD"] == "rejected" and st["c"]["NEW?"] == "admitted_llm_unsure"
    assert st["c"]["A"] == "admitted_recurrence"


def test_value_check_parsing_and_candidates(env):
    from experiments.v2.llm_value_checks import candidates, parse_verdict
    assert parse_verdict('```json\n{"verdict": "Invalid", "reason": "typo"}\n```') == "invalid"
    assert parse_verdict("no idea") == "unsure"
    _, _, pr, src = env
    cand = candidates(src, "synth", "dev", seeds=[42])
    assert "FF" in cand["state"] and "QZ" in cand["state"]                 # matched valid / invalid novelty
    assert not any(v.startswith("INVALID_") for vs in cand.values() for v in vs)   # random defect tokens never recur


def test_min_effect_override_in_monitor(env):
    from monitor.loop import MonitorConfig
    _, _, pr, _ = env
    floor = dict(pr["classification"]["drift_threshold"], min_effect={"numeric": 9.0, "categorical": 9.0})
    steps, _ = _run(env, "B_drift", MonitorConfig(name="f", drift_threshold=floor))
    assert not any(s["label"]["drift"] for s in steps if not s["warmup"])   # an unreachable floor: no drift labels
