"""Deequ / TFDV integration: exchange export, output validation, rule translation, batch labels, runners.
The TFDV end-to-end test runs the real sidecar when a TFDV environment is available (ADQ_TFDV_PYTHON)."""
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.synthetic import SYNTH_CFG, configs, make_clean

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def exchange(tmp_path_factory):
    from experiments.v2.tool_exchange import export
    from injection.scenario import DataSource
    datasets, exp, pr = configs()
    src = DataSource(datasets, exp, pr, clean_loader=lambda n: make_clean())
    root = tmp_path_factory.mktemp("xch")
    man = export(root, src, datasets, [42], "dev", ["A_stable", "B_drift"],
                 ["A_stable", "B_drift", "C_degradation", "D_drift_degradation"], pr["external_tools"],
                 pr["classification"]["drift_threshold"], "k", verify=["A_stable", "B_drift"])
    return root, man, src, (datasets, exp, pr)


def test_export_layout_and_dedup(exchange):
    root, man, src, _ = exchange
    assert len(man["units"]) == 4
    assert len({u["warm_hash"] for u in man["units"]}) == 1          # identical warm-ups exported once
    u = man["units"][0]
    files = sorted((root / "series" / "synth" / "seed42" / u["scenario"]).glob("*.parquet"))
    assert len(files) == 20
    df = pd.read_parquet(files[7])
    assert list(df.columns) == SYNTH_CFG["numeric"] + SYNTH_CFG["categorical"]   # monitored columns only, no labels
    assert str(df["x"].dtype) == "float64"
    b = src.replay("synth", "dev", 42, u["scenario"]).batch(8).df
    assert len(df) == len(b)
    assert u["rows"][7] == len(b)


def test_clean_warmup_export_drops_erroneous_records(exchange):
    root, man, src, _ = exchange
    u = [x for x in man["units"] if x["scenario"] == "A_stable"][0]
    assert u["verify"] and u["suggest"] and u["warm_clean_hash"] != u["warm_hash"]
    rep = src.replay("synth", "dev", 42, "A_stable")
    for t in range(1, 6):
        clean = pd.read_parquet(root / "warm_clean" / u["warm_clean_hash"] / f"T{t:02d}.parquet")
        obs = pd.read_parquet(root / "warm" / u["warm_hash"] / f"T{t:02d}.parquet")
        b = rep.batch(t)
        assert len(clean) == len(b.df) - b.df.record_id.isin(b.defective).sum() < len(obs)
        assert not clean.state.astype(str).str.startswith("INVALID_").any()


def test_output_validity_uses_content_hash(exchange):
    from experiments.v2.tool_exchange import output_valid
    root, man, _, _ = exchange
    u = man["units"][0]
    p = root / "tfdv" / "series" / "synth" / "seed42" / f"{u['scenario']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"unit": dict(u, series_hash="stale")}))
    assert not output_valid(root, "tfdv", u)
    p.write_text(json.dumps({"unit": u}))
    assert output_valid(root, "tfdv", u)
    p.unlink()


def test_deequ_constraint_translation():
    from baselines.external_tools import deequ_rules, translate_constraint
    cols = {"Facility_Type": "Facility Type"}
    assert translate_constraint('.isComplete("Facility_Type")', cols, []) == {"type": "completeness", "column": "Facility Type"}
    assert translate_constraint('.isNonNegative("y")', {}, [])["low"] == 0.0
    r = translate_constraint('.isContainedIn("Risk", Array("Risk 1 (High)", "say \\"hi\\""))', {}, [])
    assert r == {"type": "category", "column": "Risk", "allowed": ["Risk 1 (High)", 'say "hi"']}
    # real Deequ output (Chicago warm-up): the fractional variant is a batch-level tolerance, not a record rule
    frac = '.isContainedIn("Results", Array("Pass", "Fail", "Pass w/ Conditions", "Out of Business"), _ >= 0.93, Some("It should be above 0.93!"))'
    assert translate_constraint(frac, {}, []) is None
    assert deequ_rules({"suggestions": [{"code_for_constraint": frac}]}, [])[1]["batch_level_only"] == [frac]
    assert translate_constraint('.hasDataType("y", ConstrainableDataTypes.Integral)', {}, [])["type"] == "type"
    assert translate_constraint('.hasDataType("c", ConstrainableDataTypes.String)', {}, []) is None
    sug = {"columns": {}, "suggestions": [{"code_for_constraint": c} for c in
           ['.isComplete("x")', '.isComplete("x")', '.hasCompleteness("y", _ >= 0.9, Some("a"))', '.satisfies("x > 0")']]}
    rules, rep = deequ_rules(sug, ["x"])
    assert len(rules) == 1 and len(rep["batch_level_only"]) == 1 and len(rep["untranslated"]) == 1


def _deequ_fixture(root, unit, fire_t):
    """Deequ output in the sidecar's format (constraint strings as Deequ prints them)."""
    batches = []
    for t in range(1, 21):
        checks = [{"metric": "Mean:x", "group": "distribution", "constraint_status": "Failure" if t in fire_t else "Success",
                   "message": "Value 3 does not meet the constraint" if t in fire_t else ""},
                  {"metric": "Completeness:y", "group": "quality",
                   "constraint_status": "Failure" if t == 1 else "Success",
                   "message": "requirement failed: There have to be previous results in the MetricsRepository!"}]
        batches.append({"t": t, "checks": checks, "metrics": []})
    p = root / "deequ" / "series" / "synth" / "seed42" / f"{unit['scenario']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"unit": unit, "batches": batches, "columns": {}}))
    return p


def test_deequ_labels(exchange):
    from baselines.external_tools import deequ_labels, load_series
    root, man, _, _ = exchange
    u = [x for x in man["units"] if x["scenario"] == "B_drift"][0]
    p = _deequ_fixture(root, u, {8, 9})
    lab = deequ_labels(load_series(root, "deequ", "synth", 42, "B_drift"))
    assert lab[8]["drift"] and not lab[8]["degradation"] and not lab[7]["drift"]
    assert not lab[1]["degradation"] and lab[1]["not_ready"] == "Completeness:y"   # no history is not an anomaly
    # current format: one result list per anomaly configuration; failed suggested constraints add degradation
    ser = json.loads(p.read_text())
    for b in ser["batches"]:
        b["checks"] = {"default": b["checks"], "calibrated": [dict(c, constraint_status="Success") for c in b["checks"]]}
    assert deequ_labels(ser, "default")[8]["drift"] and not deequ_labels(ser, "calibrated")[8]["drift"]
    lab = deequ_labels(ser, "calibrated", {8: 2})
    assert lab[8]["degradation"] and "constraints:2" in lab[8]["fired"] and not lab[9]["degradation"]
    p.unlink()


def _verify_fixture(root, unit, src):
    """Deequ verification output in the sidecar's format: two variants, positions of violating rows."""
    rep = src.replay("synth", "dev", 42, unit["scenario"])
    batches = []
    for t in range(6, 21):
        b = rep.batch(t)
        miss = [int(i) for i in np.flatnonzero(b.df["x"].isna().to_numpy())]
        batches.append({"t": t, "rows": len(b.df),
                        "status": {"observed_default": ["Success" if t % 2 else "Failure"],
                                   "clean_extended": ["Failure", "Success"]},
                        "fails": {"observed_default": [miss], "clean_extended": [miss, [0]]},
                        "no_row_level": {"observed_default": [], "clean_extended": []}})
    out = {"unit": unit, "batches": batches,
           "variants": {"observed_default": {"warm_hash": unit["warm_hash"],
                                             "constraints": ['.hasCompleteness("x", _ >= 0.99)']},
                        "clean_extended": {"warm_hash": unit["warm_clean_hash"],
                                           "constraints": ['.isComplete("x")', '.isNonNegative("y")']}}}
    p = root / "deequ" / "verify" / "synth" / "seed42" / f"{unit['scenario']}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out))
    return p


def test_deequ_flags_readings(exchange):
    from baselines.external_tools import deequ_flags, load_verify
    root, man, src, _ = exchange
    u = [x for x in man["units"] if x["scenario"] == "A_stable"][0]
    p = _verify_fixture(root, u, src)
    ver = load_verify(root, "synth", 42, "A_stable")
    failed = deequ_flags(ver, "observed_default", "failed")
    anyr = deequ_flags(ver, "observed_default", "any")
    assert failed[7]["positions"] == set() and failed[8]["positions"] == anyr[8]["positions"]   # T7 passed
    assert anyr[7]["positions"] and failed[7]["failed"] == 0 and failed[8]["failed"] == 1
    clean = deequ_flags(ver, "clean_extended", "failed")
    assert 0 not in clean[8]["positions"] and clean[8]["failed_constraints"] == '.isComplete("x")'
    with pytest.raises(ValueError):
        deequ_flags(ver, "observed_default", "nope")
    p.unlink()


def test_tfdv_labels_documented_and_calibrated():
    from baselines.external_tools import tfdv_labels
    rng = np.random.default_rng(0)
    cal_d = [{"x": float(v), "c": float(w)} for v, w in zip(rng.normal(.02, .002, 5), rng.normal(.01, .001, 5))]
    cal_q = [{"records": 1000, "present_fraction": {"x": 1.0}, "unexpected_fraction": {"c": 0.01 + 0.001 * i}} for i in range(5)]
    def batch(t, dx, pres, unexp, anomalies):
        return {"t": t, "distance_previous": {"x": dx, "c": 0.0}, "distance_warmup": {"x": dx, "c": 0.0},
                "present_fraction": {"x": pres}, "unexpected_fraction": {"c": unexp}, "records": 1000,
                "schema_anomalies": anomalies}
    enum = [{"feature": "c", "type": "ENUM_TYPE_UNEXPECTED_STRING_VALUES"}]
    series = {"feature_kind": {"x": "numeric", "c": "categorical"}, "columns": {},
              "loo_warmup": cal_d, "consecutive_warmup": cal_d[:4], "loo_warmup_quality": cal_q,
              "batches": [batch(6, 0.02, 1.0, 0.011, enum), batch(7, 0.20, 1.0, 0.012, enum),
                          batch(8, 0.02, 0.95, 0.10, enum + [{"feature": "x", "type": "FEATURE_TYPE_LOW_FRACTION_PRESENT"}])]}
    doc = {"categorical_linf": 0.01, "numeric_jsd": 0.01}
    d = tfdv_labels(series, "warmup", "documented", doc)
    assert d[6]["drift"] and d[6]["degradation"]           # documented: 0.02 > 0.01, any unseen string -> anomaly
    c = tfdv_labels(series, "warmup", "calibrated", doc)
    assert not c[6]["drift"] and not c[6]["degradation"]   # within the warm-up's own variation
    assert c[7]["drift"] and not c[7]["degradation"]
    assert c[8]["degradation"] and "present<" in c[8]["fired"] and "unexpected>" in c[8]["fired"]
    with pytest.raises(ValueError):
        tfdv_labels(series, "nope", "documented", doc)


def test_deequ_methods_score_deequs_own_flags(exchange, monkeypatch):
    import experiments.v2.rq1_thresholds as rq1
    root, man, src, (datasets, exp, pr) = exchange
    u = [x for x in man["units"] if x["scenario"] == "A_stable"][0]
    p = _verify_fixture(root, u, src)
    monkeypatch.setattr(rq1, "data_source", lambda: src)
    methods = {"dq_any": {"variant": "observed_default", "reading": "any"},
               "dq_failed": {"variant": "observed_default", "reading": "failed"},
               "dq_any+base": {"variant": "observed_default", "reading": "any", "base": True}}
    payload = {"methods": [], "tool_methods": methods, "exchange": str(root), "split": "dev", "scenarios": ["A_stable"]}
    df = rq1.worker("synth", 42, payload)
    assert set(df.method) == set(methods) and (df.evaluation == 1).all() and len(df) == 3 * 15
    tot = df.groupby("method")[["TP", "FP", "P"]].sum()
    # every flagged position is a missing x, which is a labelled defect: Deequ's flags are all true positives
    assert tot.loc["dq_any", "TP"] > 0 and tot.loc["dq_any", "FP"] == 0
    assert tot.loc["dq_failed", "TP"] < tot.loc["dq_any", "TP"]
    assert tot.loc["dq_any+base", "TP"] >= tot.loc["dq_any", "TP"]
    # a row-count mismatch (batches changed after export) is an error, never silently misaligned
    ver = json.loads(p.read_text())
    ver["batches"][0]["rows"] += 1
    p.write_text(json.dumps(ver))
    with pytest.raises(RuntimeError):
        rq1.worker("synth", 42, payload)
    p.unlink()


def test_wsl_paths_and_commands(monkeypatch):
    from experiments.v2 import run_tools
    assert run_tools.to_wsl(Path("C:/Users/mujta/Desktop/x y")) in ("/mnt/c/Users/mujta/Desktop/x y", "C:/Users/mujta/Desktop/x y")
    from pathlib import PureWindowsPath
    monkeypatch.setattr(run_tools, "PureWindowsPath", PureWindowsPath)
    assert run_tools.to_wsl("C:\\Users\\mujta\\Desktop\\DataQuality") == "/mnt/c/Users/mujta/Desktop/DataQuality"
    assert run_tools.to_wsl("/home/a/b") == "/home/a/b"
    cmd = run_tools.command("deequ", "dev", False, {"runner": "wsl", "wsl_distribution": "Ubuntu-24.04",
                                                     "venvs": {"deequ": "~/adq-tools/deequ"}})
    assert cmd[:4] == ["wsl.exe", "-d", "Ubuntu-24.04", "-e"] and "SPARK_VERSION=3.5" in cmd[-1]
    assert run_tools.command("tfdv", "dev", False, {"runner": "wsl"})[:3] == ["wsl.exe", "-e", "bash"]


@pytest.mark.skipif(not os.environ.get("ADQ_TFDV_PYTHON"), reason="set ADQ_TFDV_PYTHON to a Python with TFDV")
def test_tfdv_sidecar_end_to_end(exchange):
    from baselines.external_tools import load_series, tfdv_labels
    from experiments.v2.tool_exchange import output_valid
    root, man, src, (datasets, exp, pr) = exchange
    r = subprocess.run([os.environ["ADQ_TFDV_PYTHON"], str(ROOT / "tools" / "external_validators" / "run_tfdv.py"),
                        "--exchange", str(root)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
    for u in man["units"]:
        assert output_valid(root, "tfdv", u)
    doc = pr["external_tools"]["tfdv"]["documented_thresholds"]
    stable = tfdv_labels(load_series(root, "tfdv", "synth", 42, "A_stable"), "warmup", "calibrated", doc)
    drift = tfdv_labels(load_series(root, "tfdv", "synth", 42, "B_drift"), "warmup", "calibrated", doc)
    assert sum(stable[t]["drift"] for t in range(6, 21)) <= 3
    assert sum(drift[t]["drift"] for t in range(12, 21)) >= 7


def test_deequ_sidecar_pure_helpers():
    """The parts of the Deequ sidecar that need no Spark: metric-key parsing and the calibrated deviation factor."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_deequ", ROOT / "tools" / "external_validators" / "run_deequ.py")
    rd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rd)
    assert rd.constraint_key("AnomalyConstraint(Mean(DepDelay,None))") == "Mean:DepDelay"
    assert rd.constraint_key("AnomalyConstraint(Uniqueness(List(a, b),None))") == "Uniqueness:*"
    k = rd.calibrated_factor(0.05, 20, 5)            # t(1 - 0.05/40, 4) * sqrt(1.2): about 7.4
    assert 6.5 < k < 8.5 and rd.calibrated_factor(0.05, 2, 5) < k
    from baselines.avh import t_quantile as tq
    from tools.external_validators.exchange_io import t_quantile
    assert abs(t_quantile(0.999, 4) - tq(0.999, 4)) < 1e-12
