"""Deequ sidecar (Amazon Deequ through PyDeequ; Schelter et al., VLDB 2018). Runs in its own Linux / WSL environment
(Java + PySpark 3.5 + PyDeequ; setup_wsl.sh), reads the exchange folder and writes RAW tool output. Deequ does all
the work itself: it suggests the constraints, evaluates them on every batch, reports which records violate them
(its row-level results) and runs its own anomaly detection on its own metrics. Nothing here sees labels.

  deequ/suggestions/<variant>/<warm hash>.json
      Deequ's constraint suggestions (ConstraintSuggestionRunner) on the five warm-up batches together, with Deequ's
      own Scala code for each constraint. One file per suggestion variant (protocol.yaml external_tools.deequ.
      suggestion_variants): the warm-up as observed (with its ~1% erroneous records) and DEFAULT rules = what a
      practitioner gets; the warm-up without its erroneous records and EXTENDED rules without the NUMERICAL
      exact-equality rules (DEFAULT+STRING+COMMON) = the tool's best case.
  deequ/verify/<ds>/seed<s>/<scenario>.json                                    (H1a, record level)
      Every evaluation batch T6..T20 verified against the suggested constraints of EVERY variant in one
      VerificationSuite run (one Check per constraint): the status of each constraint on the batch (Deequ's
      batch-level verdict) and, from VerificationResult.rowLevelResultsAsDataFrame, the positions of the records
      that violate each constraint (Deequ's record-level verdict). Positions are row numbers of the exported batch,
      whose row order is the replay's; the experiment maps them back to record ids (and checks the row counts).
  deequ/series/<ds>/seed<s>/<scenario>.json                                    (H1b, batch level)
      For T1..T20 in order, one anomaly check per Deequ metric under each anomaly configuration
      (protocol.yaml external_tools.deequ.anomaly_configs):
        history cumulative   metrics of every earlier batch in an InMemoryMetricsRepository (Deequ's documented
                             set-up), e.g. OnlineNormalStrategy with Deequ's default parameters
        history warmup       metrics of T1..T5 only (later batches are checked against the repository but never
                             saved), e.g. BatchNormalStrategy with the deviation factor set by the SAME calibration
                             as the proposed method: one-sided Student-t prediction bound for a new value from n
                             warm-up values, Bonferroni-split over the metrics (factor "calibrated")
      Mapped to the four batch classes by baselines/external_tools.py (H1b comparators "deequ_*").

Column names are made Spark-safe; the outputs carry the mapping back.

    export SPARK_VERSION=3.5
    python run_deequ.py --exchange /mnt/c/.../results/v2/tool_exchange/dev [--force] [--only suggestions|verify|series]
    python run_deequ.py --selftest            # 1-minute check of every Deequ call used here, on a toy table
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exchange_io import (output_current, Progress, load_manifest, series_out, series_paths, series_units,  # noqa: E402
                         suggestion_out, suggestion_variants, t_quantile, variant_hash, verify_units, warm_paths,
                         write_json)

os.environ.setdefault("SPARK_VERSION", "3.5")
# Spark's Python workers must use this interpreter (the venv's): otherwise they take `python3` from PATH, which on
# macOS is the system Python, and PySpark aborts with PYTHON_VERSION_MISMATCH.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

DISTRIBUTION, QUALITY = "distribution", "quality"
ROW = "__adq_row"


def spark_session(memory: str):
    import pydeequ
    from pyspark.sql import SparkSession
    return (SparkSession.builder.master("local[*]").appName("adq-deequ")
            .config("spark.jars.packages", pydeequ.deequ_maven_coord)
            .config("spark.jars.excludes", pydeequ.f2j_maven_coord)
            .config("spark.driver.memory", memory)
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.enabled", "false")
            .getOrCreate())


def read_batches(spark, paths, safe):
    df = spark.read.parquet(*[str(p) for p in paths])
    for orig, s in safe.items():
        if orig != s:
            df = df.withColumnRenamed(orig, s)
    return df


def with_row_numbers(df):
    """Row number of every record in file order (0..n-1): monotonically_increasing_id follows file order inside and
    across partitions; row_number over it makes the numbers contiguous."""
    from pyspark.sql import Window, functions as F
    df = df.withColumn("__mid", F.monotonically_increasing_id())
    return df.withColumn(ROW, F.row_number().over(Window.orderBy("__mid")) - 1).drop("__mid")


# ------------------------------------------------------------------------------------------- suggestions
def scala_rule_set(spark, name: str):
    """Deequ's own rule set com.amazon.deequ.suggestions.Rules.<name> (a Scala Seq), read from the JVM.
    PyDeequ 1.7 wraps only DEFAULT; Deequ defines EXTENDED = DEFAULT ++ STRING ++ NUMERICAL ++ COMMON."""
    jvm = spark._jvm
    try:
        return getattr(jvm.com.amazon.deequ.suggestions.Rules, name)()          # static forwarder of the object
    except Exception:
        try:
            module = jvm.java.lang.Class.forName("com.amazon.deequ.suggestions.Rules$").getField("MODULE$").get(None)
            return getattr(module, name)()
        except Exception as e:
            raise SystemExit(f"unknown Deequ suggestion rule set {name!r} (neither PyDeequ nor "
                             f"com.amazon.deequ.suggestions.Rules defines it): {e}")


# Constraints a rule set may suggest that cannot hold on any batch other than the profiled one: Deequ's NUMERICAL
# rules assert EQUALITY with the warm-up value (hasMin/hasMax/hasMean/hasStandardDeviation(_ == v)), and its HasMean
# rule takes the column MAXIMUM as the mean (suggestions/rules/HasMean.scala). The best-case variant therefore uses
# EXTENDED without NUMERICAL (protocol.yaml: DEFAULT+STRING+COMMON); the self-test checks that none appears.
EXACT_NUMERIC = (".hasMin(", ".hasMax(", ".hasMean(", ".hasStandardDeviation(")


def add_rules(spark, builder, name: str):
    """A single name: PyDeequ's rule set when PyDeequ defines it (DEFAULT: what a Python user gets, which already
    includes the uniqueness rule), else Deequ's Scala set of that name. "A+B+...": the union of Deequ's Scala sets
    com.amazon.deequ.suggestions.Rules.A, .B, ... (e.g. DEFAULT+STRING+COMMON = EXTENDED without NUMERICAL)."""
    from pydeequ import suggestions as S
    parts = [p.strip() for p in name.split("+") if p.strip()]
    if len(parts) == 1 and hasattr(S, parts[0]):
        return builder.addConstraintRule(getattr(S, parts[0])())
    for p in parts:
        builder._ConstraintSuggestionRunBuilder.addConstraintRules(scala_rule_set(spark, p))
    return builder


def suggest(spark, df, rules_name: str):
    """(suggestions as JSON with Deequ's ORIGINAL Scala code, [(code, JVM Constraint)]).
    PyDeequ's run() rewrites the code into lossy pseudo-Python, so the JVM result is read directly."""
    from pydeequ.suggestions import ConstraintSuggestionRunner
    b = add_rules(spark, ConstraintSuggestionRunner(spark).onData(df), rules_name)
    res = b._ConstraintSuggestionRunBuilder.run()
    jvm = spark._jvm
    raw = jvm.com.amazon.deequ.suggestions.ConstraintSuggestionResult.getConstraintSuggestionsAsJson(res)
    by_col = {}
    it = res.constraintSuggestions().iterator()          # Scala Map[String, Seq[ConstraintSuggestion]]; plain
    while it.hasNext():                                  # iterators work on Scala 2.12 and 2.13 alike
        pair = it.next()
        seq_it = pair._2().iterator()
        items = []
        while seq_it.hasNext():
            sg = seq_it.next()
            items.append((str(sg.codeForConstraint()), sg.constraint()))
        by_col[str(pair._1())] = items
    objs = [x for col in sorted(by_col) for x in by_col[col]]
    return json.loads(raw)["constraint_suggestions"], objs


# ------------------------------------------------------------------------------------------- verification
def make_checks(spark, constraints):
    """One Deequ Check per constraint, so that both the status and the row-level result are per constraint.
    constraints: [(name, JVM Constraint)] -> [(name, JVM Check)]."""
    from pydeequ.scala_utils import to_scala_seq
    jvm = spark._jvm
    level = jvm.com.amazon.deequ.checks.CheckLevel.Error()
    return [(name, jvm.com.amazon.deequ.checks.Check(level, name, to_scala_seq(jvm, [c]))) for name, c in constraints]


def verify(spark, df, checks):
    """Run one VerificationSuite with all checks on ``df`` (which carries the row-number column).
    Returns ({check name: constraint status}, {check name: [row numbers violating it]}, rows)."""
    from pyspark.sql import functions as F
    from pydeequ.verification import VerificationResult
    jvm = spark._jvm
    builder = jvm.com.amazon.deequ.VerificationRunBuilder(df._jdf)
    for _, ch in checks:
        builder = builder.addCheck(ch)
    res = VerificationResult(spark, builder.run())
    status = {}
    for c in VerificationResult.checkResultsAsJson(spark, res):
        status[str(c.get("check"))] = str(c.get("constraint_status"))
    rl = VerificationResult.rowLevelResultsAsDataFrame(spark, res, df)
    names = [n for n, _ in checks if n in rl.columns]
    fails = {n: [] for n, _ in checks}
    if names:
        ok = [F.coalesce(F.col(f"`{n}`").cast("boolean"), F.lit(True)) for n in names]
        bad = ok[0]
        for c in ok[1:]:
            bad = bad & c
        rows = rl.where(~bad).select(ROW, *[F.col(f"`{n}`").alias(f"c{i}") for i, n in enumerate(names)]).collect()
        for r in rows:
            for i, n in enumerate(names):
                if r[f"c{i}"] is False:
                    fails[n].append(int(r[ROW]))
    missing = [n for n, _ in checks if n not in rl.columns]
    return status, fails, missing


def run_verify(spark, exchange: Path, unit, meta, variants, cache) -> dict:
    checks, codes = [], {}
    for v, spec in variants.items():
        h = variant_hash(unit, spec)
        objs = suggestions_for(spark, exchange, v, spec, h, unit["dataset"], meta, cache)
        names = [(f"{v}__{i:03d}", c) for i, (_, c) in enumerate(objs)]
        codes[v] = {"warm_hash": h, "constraints": [code for code, _ in objs]}
        checks += make_checks(spark, names)
    batches = []
    paths = series_paths(exchange, unit)
    for t, path in enumerate(paths, 1):
        if t <= unit["warmup"]:
            continue
        df = with_row_numbers(read_batches(spark, [path], meta["safe_names"])).cache()
        n = df.count()
        status, fails, missing = verify(spark, df, checks) if checks else ({}, {}, [])
        out = {"t": t, "rows": n, "status": {}, "fails": {}, "no_row_level": {}}
        for v in variants:
            pre = f"{v}__"
            out["status"][v] = [status.get(nm, "Missing") for nm, _ in checks if nm.startswith(pre)]
            out["fails"][v] = [fails.get(nm, []) for nm, _ in checks if nm.startswith(pre)]
            out["no_row_level"][v] = [nm.split("__")[1] for nm in missing if nm.startswith(pre)]
        batches.append(out)
        df.unpersist()
    return {"unit": unit, "variants": codes, "batches": batches}


def same_spec(path: Path, key: str, value) -> bool:
    """True if the output at ``path`` exists and was computed with the same setting ``key`` (rule set, variant specs,
    anomaly configs): a configuration change recomputes even when the exported batches are unchanged."""
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get(key) == json.loads(json.dumps(value))
    except ValueError:
        return False


def suggestions_for(spark, exchange, variant, spec, warm_hash, dataset, meta, cache):
    """JVM constraint objects of one (variant, warm-up); computed once per process and written to
    deequ/suggestions/<variant>/<hash>.json (the JSON is the reported artefact; the objects are what Deequ verifies)."""
    key = (variant, warm_hash)
    if key not in cache:
        clean = spec.get("warmup") == "clean"
        df = read_batches(spark, warm_paths(exchange, warm_hash, clean), meta["safe_names"])
        rules = spec.get("rules", "DEFAULT")
        sug, objs = suggest(spark, df, rules)
        write_json(suggestion_out(exchange, variant, warm_hash),
                   {"dataset": dataset, "warm_hash": warm_hash, "variant": variant, "warmup": spec.get("warmup"),
                    "rows": df.count(), "rule_set": rules, "columns": {s: o for o, s in meta["safe_names"].items()},
                    "suggestions": sug, "constraints_verified": [c for c, _ in objs]})
        cache[key] = objs
    return cache[key]


# ------------------------------------------------------------------------------------------- anomaly detection
def metric_specs(meta):
    """(key, analyzer, group). Groups decide the four-class mapping (same split as the AVH-style comparator):
    distribution metrics -> drift, quality metrics -> degradation."""
    from pydeequ.analyzers import (ApproxCountDistinct, ApproxQuantile, Completeness, Entropy, Maximum, Mean,
                                   Minimum, Size, StandardDeviation, Uniqueness)
    safe = meta["safe_names"]
    num = [safe[c] for c in meta["numeric"]]
    cat = [safe[c] for c in meta["categorical"]]
    specs = [("Size:*", lambda: Size(), QUALITY),
             ("Uniqueness:*", lambda: Uniqueness(num + cat), QUALITY)]
    for c in num + cat:
        specs.append((f"Completeness:{c}", (lambda c=c: Completeness(c)), QUALITY))
    for c in num:
        specs += [(f"Mean:{c}", (lambda c=c: Mean(c)), DISTRIBUTION),
                  (f"StandardDeviation:{c}", (lambda c=c: StandardDeviation(c)), DISTRIBUTION),
                  (f"Minimum:{c}", (lambda c=c: Minimum(c)), DISTRIBUTION),
                  (f"Maximum:{c}", (lambda c=c: Maximum(c)), DISTRIBUTION),
                  (f"ApproxQuantile:{c}", (lambda c=c: ApproxQuantile(c, 0.5)), DISTRIBUTION)]
    for c in cat:
        specs += [(f"ApproxCountDistinct:{c}", (lambda c=c: ApproxCountDistinct(c)), DISTRIBUTION),
                  (f"Entropy:{c}", (lambda c=c: Entropy(c)), DISTRIBUTION)]
    return specs


def calibrated_factor(alpha: float, n_metrics: int, n_hist: int) -> float:
    """Two-sided Student-t prediction factor for a new value from n_hist values, Bonferroni over n_metrics."""
    return t_quantile(1.0 - alpha / (2.0 * max(n_metrics, 1)), max(n_hist - 1, 1)) * (1.0 + 1.0 / n_hist) ** 0.5


def strategy(cfg, n_metrics: int, n_hist: int):
    from pydeequ import anomaly_detection as A
    params = dict(cfg.get("params") or {})
    for k, v in list(params.items()):
        if v == "calibrated":
            params[k] = calibrated_factor(float(cfg.get("alpha", 0.05)), n_metrics, n_hist)
    return getattr(A, cfg["strategy"])(**params), params


def constraint_key(text: str):
    """Metric key of a Deequ anomaly constraint, e.g. "AnomalyConstraint(Mean(DepDelay,None))" -> "Mean:DepDelay",
    "AnomalyConstraint(Uniqueness(List(a, b),None))" -> "Uniqueness:*", "AnomalyConstraint(Size(None))" -> "Size:*".
    Deequ returns check results as an unordered map, so results are matched to metrics by this key, never by order."""
    m = re.search(r"AnomalyConstraint\((\w+)\((.*)\)\)\s*$", text.strip())
    if not m:
        return None
    name, args = m.group(1), m.group(2)
    if name in ("Size", "Uniqueness"):
        return f"{name}:*"
    depth, first = 0, ""
    for ch in args:
        if ch == "," and depth == 0:
            break
        depth += ch == "("
        depth -= ch == ")"
        first += ch
    return f"{name}:{first.strip()}"


def run_series(spark, exchange: Path, unit, meta, configs) -> dict:
    from pydeequ.repository import InMemoryMetricsRepository, ResultKey
    from pydeequ.verification import VerificationResult, VerificationSuite
    specs = metric_specs(meta)
    paths = series_paths(exchange, unit)
    out = {"unit": unit, "configs": {}, "batches": [{"t": t, "checks": {}} for t in range(1, len(paths) + 1)]}
    for name, cfg in configs.items():
        repo = InMemoryMetricsRepository(spark)
        warm_only = cfg.get("history", "cumulative") == "warmup"
        used = {}
        for t, path in enumerate(paths, 1):
            df = read_batches(spark, [path], meta["safe_names"])
            n_hist = min(t - 1, unit["warmup"]) if warm_only else t - 1
            strat, used = strategy(cfg, len(specs), max(n_hist, 2))
            vs = VerificationSuite(spark).onData(df).useRepository(repo)
            if not warm_only or t <= unit["warmup"]:
                vs = vs.saveOrAppendResult(ResultKey(spark, t * 86_400_000, {"t": str(t)}))
            for _, make, _ in specs:
                vs = vs.addAnomalyCheck(strat, make())
            res = vs.run()
            checks = {}
            for c in VerificationResult.checkResultsAsJson(spark, res):
                k = constraint_key(str(c.get("constraint", ""))) or \
                    constraint_key(f"AnomalyConstraint({str(c.get('check', '')).replace('Anomaly check for ', '')})")
                checks[k] = c
            rows = []
            for key, _, group in specs:
                chk = checks.get(key)
                if chk is None:
                    raise RuntimeError(f"no Deequ anomaly result for {key}; got {sorted(k for k in checks if k)}")
                rows.append({"metric": key, "group": group, "constraint_status": chk.get("constraint_status"),
                             "message": chk.get("constraint_message")})
            out["batches"][t - 1]["checks"][name] = rows
        out["configs"][name] = {**cfg, "params_used": used, "n_metrics": len(specs)}
    return out


def versions(spark) -> dict:
    import pydeequ
    import pyspark
    return {"pydeequ": pydeequ.__version__, "pyspark": pyspark.__version__, "deequ": pydeequ.deequ_maven_coord,
            "java": spark._jvm.System.getProperty("java.version")}


# ------------------------------------------------------------------------------------------- self-test
def selftest(memory: str) -> None:
    """Exercise every Deequ call of this sidecar on a toy table and print what came back."""
    import random
    from pyspark.sql import Row
    spark = spark_session(memory)
    spark.sparkContext.setLogLevel("ERROR")
    print("versions:", versions(spark))
    rnd = random.Random(1)
    good = [Row(x=float(rnd.gauss(10, 2)), c=rnd.choice(["a", "b", "c"])) for _ in range(400)]
    df = spark.createDataFrame(good)
    sug, objs = suggest(spark, df, "DEFAULT")
    print(f"suggestions: {len(sug)} (DEFAULT), JVM constraints: {len(objs)}")
    for code, _ in objs:
        print("   ", code)
    _, objs_e = suggest(spark, df, "EXTENDED")
    print(f"suggestions: {len(objs_e)} (EXTENDED, for reference; its NUMERICAL rules are exact equalities):")
    for code, _ in objs_e:
        print("   ", code)
    assert any(code.startswith(EXACT_NUMERIC) for code, _ in objs_e), "EXTENDED should contain the NUMERICAL rules"
    best = "DEFAULT+STRING+COMMON"
    _, objs_x = suggest(spark, df, best)
    print(f"suggestions: {len(objs_x)} ({best}, the best-case variant)")
    assert objs_x and not any(code.startswith(EXACT_NUMERIC) for code, _ in objs_x), \
        "the best-case rule set must not contain exact-equality numeric constraints"
    assert any(code.startswith(".isUnique(") for code, _ in objs_x), "COMMON (uniqueness) should be included"
    fresh = with_row_numbers(spark.createDataFrame(
        [Row(x=float(rnd.gauss(10, 2)), c=rnd.choice(["a", "b", "c"])) for _ in range(200)], schema=df.schema))
    xs, xf, xm = verify(spark, fresh, make_checks(spark, [(f"x__{i:03d}", c) for i, (_, c) in enumerate(objs_x)]))
    print(f"    {best} verified on a NEW clean batch from the same distribution (200 rows):")
    print(f"    {'constraint':48s} {'status':8s} violating rows")
    for i, (code, _) in enumerate(objs_x):
        n = f"x__{i:03d}"
        print(f"    {code:48s} {xs.get(n, '?'):8s} {'no row-level result' if n in xm else len(xf[n])}")
    bad = good[:200] + [Row(x=None, c="a"), Row(x=-5.0, c="zzz"), Row(x=12.0, c=None)] + good[200:]
    bdf = with_row_numbers(spark.createDataFrame(bad, schema=df.schema))
    checks = make_checks(spark, [(f"t__{i:03d}", c) for i, (_, c) in enumerate(objs)])
    status, fails, missing = verify(spark, bdf, checks)
    print("statuses:", status)
    print("violating rows per constraint:", {k: v for k, v in fails.items() if v})
    print("constraints without row-level results:", missing)
    assert any(200 in v for v in fails.values()), "the missing x at row 200 should violate a completeness constraint"
    meta = {"safe_names": {"x": "x", "c": "c"}, "numeric": ["x"], "categorical": ["c"]}
    from pydeequ.repository import InMemoryMetricsRepository, ResultKey
    from pydeequ.verification import VerificationResult, VerificationSuite
    for cfg in ({"strategy": "OnlineNormalStrategy", "params": {}},
                {"strategy": "BatchNormalStrategy", "params": {"lowerDeviationFactor": "calibrated",
                                                               "upperDeviationFactor": "calibrated"}}):
        repo = InMemoryMetricsRepository(spark)
        for t in range(1, 8):
            part = spark.createDataFrame(good[t * 40:(t + 1) * 40] if t < 7 else bad[190:260], schema=df.schema)
            strat, used = strategy(cfg, 10, 5)
            vs = VerificationSuite(spark).onData(part).useRepository(repo)
            if t <= 5:
                vs = vs.saveOrAppendResult(ResultKey(spark, t * 86_400_000, {"t": str(t)}))
            for _, make, _ in metric_specs(meta):
                vs = vs.addAnomalyCheck(strat, make())
            res = VerificationResult.checkResultsAsJson(spark, vs.run())
        print(cfg["strategy"], used, "last batch:",
              sorted({(constraint_key(str(c.get("constraint"))), c.get("constraint_status")) for c in res})[:6])
    spark.stop()
    print("SELFTEST OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", type=Path)
    ap.add_argument("--only", choices=["suggestions", "verify", "series"], default=None)
    ap.add_argument("--force", action="store_true", help="recompute outputs that exist")
    ap.add_argument("--memory", default="6g", help="Spark driver memory")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest(args.memory)
        return
    if args.exchange is None:
        ap.error("--exchange is required")
    man = load_manifest(args.exchange)
    dcfg = man.get("tools", {}).get("deequ", {})
    variants = suggestion_variants(man)
    spark = spark_session(args.memory)
    spark.sparkContext.setLogLevel("ERROR")
    root = args.exchange / "deequ"
    write_json(root / "manifest.json", {"versions": versions(spark), "settings": dcfg, "export_key": man["export_key"]})
    cache = {}

    if args.only in (None, "suggestions"):
        todo = sorted({(v, variant_hash(u, spec), u["dataset"]) for u in man["units"] if u["suggest"]
                       for v, spec in variants.items()})
        todo = [x for x in todo if args.force or not same_spec(suggestion_out(args.exchange, x[0], x[1]),
                                                              "rule_set", variants[x[0]].get("rules", "DEFAULT"))]
        prog = Progress(len(todo), "deequ suggestions")
        for v, h, ds in todo:
            objs = suggestions_for(spark, args.exchange, v, variants[v], h, ds, man["datasets"][ds], cache)
            prog.step(f"{ds} {v} {h}: {len(objs)} constraints")

    if args.only in (None, "verify"):
        todo = [u for u in verify_units(man)
                if args.force or not (output_current(series_out(args.exchange, "deequ", u, "verify"), u)
                                      and same_spec(series_out(args.exchange, "deequ", u, "verify"),
                                                    "variant_specs", variants))]
        prog = Progress(len(todo), "deequ verification (row level)")
        for u in todo:
            meta = man["datasets"][u["dataset"]]
            res = run_verify(spark, args.exchange, u, meta, variants, cache)
            res["columns"] = {s: o for o, s in meta["safe_names"].items()}
            res["variant_specs"] = variants
            write_json(series_out(args.exchange, "deequ", u, "verify"), res)
            prog.step(f"{u['dataset']} seed {u['seed']} {u['scenario']}")

    if args.only in (None, "series"):
        configs = dcfg.get("anomaly_configs") or {"default": {"strategy": "OnlineNormalStrategy", "params": {}}}
        todo = [u for u in series_units(man) if args.force or not (output_current(series_out(args.exchange, "deequ", u), u)
                                                                 and same_spec(series_out(args.exchange, "deequ", u),
                                                                               "anomaly_configs", configs))]
        prog = Progress(len(todo), "deequ anomaly detection")
        for u in todo:
            meta = man["datasets"][u["dataset"]]
            res = run_series(spark, args.exchange, u, meta, configs)
            res["columns"] = {s: o for o, s in meta["safe_names"].items()}
            res["anomaly_configs"] = configs
            write_json(series_out(args.exchange, "deequ", u), res)
            prog.step(f"{u['dataset']} seed {u['seed']} {u['scenario']}")
    spark.stop()


if __name__ == "__main__":
    main()
