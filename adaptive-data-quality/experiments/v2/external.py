"""External validation on dirty/clean benchmarks (supervisor experiment 7, part 1).

The FROZEN method (protocol and experiment configs as fixed before testing) and its baselines run on at least
two external benchmarks (configs/benchmarks.yaml: tax, beers; hospital optional). Benchmark labels are used
only for evaluation. Rows are replayed as shuffled batches (no credible chronology): this tests error
detection beyond the synthetic error generator, not temporal behaviour.

Methods   fixed_t1, fixed_pooled, rolling_mad, adaptive (all checks every batch), and on top of adaptive:
            adaptive+llm_<m>          validated single-column LLM rules of model m (every generation, validated on
                                      the run's own warm-up exactly as in rq2_coverage)
            adaptive+deps             cross-column rules DISCOVERED on the run's warm-up (rules/crosscolumn.py:
                                      functional dependencies such as zip -> city and conditional constants such as
                                      has_child = N -> child_exemp = 0), no LLM
            adaptive+xcol_<m>         cross-column rules PROPOSED by model m (structure only; mappings learned from
                                      the warm-up; same admission test on the warm-up)
            adaptive+llm_<m>+xcol_<m> both kinds of LLM rules
          The LLM sees only rows reserved for prompts (configs/benchmarks.yaml prompt_pool), which never occur in a
          replay batch, never the clean table or a label (experiments/v2/llm_rules.py benchmark_training_sample).
Measures  record level (a record is erroneous if any cell differs from the clean table) and cell level over the
          checkable columns = monitored + context columns (a single-column rule flags the cell of its column, a
          cross-column rule every cell of the violating tuple, as Raha's FD-violation detector does; uniqueness
          flags are record-level and not counted as cell detections); coverage = share of error cells that lie in
          checkable columns.

    python scripts/download_benchmarks.py
    python -m experiments.v2.external --split dev
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import data_dir, load_yaml  # noqa: E402
from evaluation.stats import compare_methods  # noqa: E402
from execution.local_executor import execute  # noqa: E402
from evaluation.metrics import record_counts  # noqa: E402
from experiments.v2.common import Context, ProfiledReplay, parser, summarize_counts  # noqa: E402
from ingestion.benchmarks import BenchmarkReplay  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402

EXPERIMENT = "external"
METHODS = ["fixed_t1", "fixed_pooled", "rolling_mad", "adaptive"]


def _cell_counts(rules, df, error_mask, monitored):
    from rules.static_rules import involved_columns
    res = execute(rules, df, return_masks=True)
    flagged = pd.DataFrame(False, index=df.index, columns=monitored)
    for r in rules:
        if r["type"] == "uniqueness":
            continue
        for c in involved_columns(r):
            if c in monitored:
                flagged[c] |= res["masks"][r["key"]]
    err = error_mask[monitored].to_numpy()
    f = flagged.to_numpy()
    return {"cell_TP": int((f & err).sum()), "cell_FP": int((f & ~err).sum()), "cell_FN": int((~f & err).sum())}


def run_unit(name, bench, exp, protocol, seed, methods=METHODS, replay=None, llm_gens=None,
             xcol_gens=None) -> pd.DataFrame:
    """``llm_gens`` / ``xcol_gens``: {model key: {generation: response record}} of this benchmark, single-column
    and cross-column proposals (validated per run on the run's warm-up)."""
    from rules.crosscolumn import discover_rules, validate_candidates
    from rules.validator import accepted_rules, validate
    rep = replay or BenchmarkReplay(name, bench, exp, seed, split_seed=protocol["splits"]["split_seed"])
    prr = ProfiledReplay(rep)
    cfg = rep.cfg
    context = [c for c in getattr(rep, "context", []) if c in rep.batch(1).df.columns]
    monitored = cfg["numeric"] + cfg["categorical"]
    checkable = monitored + context
    max_rate = exp["validation"]["max_warmup_failure_rate"]
    configs = [(method, MonitorConfig(name=method, method=method, classify=False), 0) for method in methods]
    warm = [rep.batch(t).df for t in range(1, rep.warm + 1)]
    single, xcol = {}, {}
    for mk, gens in (llm_gens or {}).items():
        for g, rec in sorted(gens.items()):
            val = validate(rec["rules"] or [], cfg, warm, max_rate, "empirical")
            single[(mk, g)] = [dict(r, source="llm") for r in accepted_rules(val, prefix=f"{mk}{g}e_")]
            label = f"adaptive+llm_{mk}"
            configs.append((label, MonitorConfig(name=label, method="adaptive", classify=False,
                                                 extra_rules=single[(mk, g)]), g))
    if context or len(checkable) > 1:
        learned = discover_rules(pd.concat(warm, ignore_index=True), checkable, max_rate)
        configs.append(("adaptive+deps", MonitorConfig(name="adaptive+deps", method="adaptive", classify=False,
                                                       extra_rules=learned), 0))
    for mk, gens in (xcol_gens or {}).items():
        for g, rec in sorted(gens.items()):
            val = validate_candidates(rec["rules"] or [], warm, checkable, max_rate, prefix=f"{mk}{g}x_")
            xcol[(mk, g)] = [r["rule"] for r in val if r["status"] == "accepted"]
            label = f"adaptive+xcol_{mk}"
            configs.append((label, MonitorConfig(name=label, method="adaptive", classify=False,
                                                 extra_rules=xcol[(mk, g)]), g))
            if (mk, g) in single:
                label = f"adaptive+llm_{mk}+xcol_{mk}"
                configs.append((label, MonitorConfig(name=label, method="adaptive", classify=False,
                                                     extra_rules=single[(mk, g)] + xcol[(mk, g)]), g))
    rows = []
    for method, mc, g in configs:
        m = Monitor(cfg, exp, protocol, mc, seed=seed)
        for t, b, (prof, art) in prr:
            step = m.step(b.df, b.label, prof, art)
            row = {**b.meta, "benchmark": name, "dataset": name, "seed": seed, "method": method, "generation": g,
                   **record_counts(step["failed_ids"], b.gt, b.df["record_id"])}
            row.update(_cell_counts([dict(r) for r in m.last_suite], b.df, b.error_mask, checkable))
            row["checkable_error_cells"] = int(b.error_mask[checkable].to_numpy().sum())
            rows.append(row)
    return pd.DataFrame(rows)


def llm_generations(names, bench, exp, protocol, kind: str = "single"):
    """{benchmark: {model key: {generation: record}}} for the current prompts (fingerprint-matched);
    kind "single" = single-column proposals, "xcol" = cross-column proposals."""
    from experiments.v2.llm_rules import load_benchmark_generations, load_benchmark_xcol_generations, models
    load = load_benchmark_generations if kind == "single" else load_benchmark_xcol_generations
    out = {}
    for n in names:
        out[n] = {}
        for mk, mid in models(protocol).items():
            gens = load(n, bench, exp, protocol, mid, protocol["rq2"]["generations"])
            if gens:
                out[n][mk] = gens
            else:
                print(f"[external] {n}: no {mk} {kind} LLM generations (python -m experiments.v2.llm_generate); "
                      f"methods using them skipped", flush=True)
    return out


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--benchmarks", nargs="*", default=None)
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    bench = load_yaml("benchmarks")
    names = args.benchmarks or [n for n, b in bench["benchmarks"].items() if not b.get("optional")]
    frames = []
    import json
    from experiments.v2.common import cache_key
    present = [n for n in names if data_dir("external", bench["benchmarks"][n]["files"]["dirty"]).exists()]
    gens = llm_generations(present, bench, ctx.exp, ctx.protocol)
    xgens = llm_generations(present, bench, ctx.exp, ctx.protocol, "xcol")
    fps = lambda G: {n: {mk: sorted((g, r["fingerprint"]["key"]) for g, r in gm.items()) for mk, gm in d.items()}
                     for n, d in G.items()}
    key = cache_key(ctx, {"benchmarks": bench, "methods": METHODS, "llm": fps(gens), "xcol": fps(xgens)})
    key_file = ctx.raw / "_cache_key.json"
    fresh = key_file.exists() and json.loads(key_file.read_text()).get("key") == key
    if not fresh:                                    # code / configuration changed: cached units are stale
        for old in ctx.raw.glob("*__seed*.parquet"):  # remove them first, so an interrupted run never reuses them
            old.unlink()
        key_file.write_text(json.dumps({"key": key}))
    for name in names:
        if not data_dir("external", bench["benchmarks"][name]["files"]["dirty"]).exists():
            print(f"[external] {name}: files missing, run python scripts/download_benchmarks.py")
            continue
        for seed in ctx.seeds:
            path = ctx.raw / f"{name}__seed{seed}.parquet"
            if path.exists() and fresh and not args.force:
                frames.append(pd.read_parquet(path))
                continue
            df = run_unit(name, bench, ctx.exp, ctx.protocol, seed, llm_gens=gens.get(name),
                          xcol_gens=xgens.get(name))
            df.to_parquet(path, index=False)
            frames.append(df)
            print(f"[external] {name} seed {seed} done", flush=True)
    if not frames:
        return
    raw = pd.concat(frames, ignore_index=True)
    ev = raw[raw.evaluation == 1]
    t = summarize_counts(ev, ["benchmark", "method"])
    cells = ev.groupby(["benchmark", "method"])[["cell_TP", "cell_FP", "cell_FN", "monitored_error_cells",
                                                 "checkable_error_cells", "all_error_cells"]].sum().reset_index()
    cells["cell_precision"] = cells.cell_TP / (cells.cell_TP + cells.cell_FP).where(lambda s: s > 0)
    cells["cell_recall"] = cells.cell_TP / (cells.cell_TP + cells.cell_FN).where(lambda s: s > 0)
    cells["monitored_error_coverage"] = cells.monitored_error_cells / cells.all_error_cells.where(lambda s: s > 0)
    cells["checkable_error_coverage"] = cells.checkable_error_cells / cells.all_error_cells.where(lambda s: s > 0)
    t = t.merge(cells, on=["benchmark", "method"])
    # record level is uninformative when (almost) every record holds an error (Beers: every record has a
    # formatting error, so N = 0, FPR undefined and recall trivially 1): report cell level as the headline there
    t["record_level_informative"] = (t.N >= 0.05 * t.records) & (t.P >= 0.05 * t.records)
    t.to_csv(ctx.out / "external_detection.csv", index=False)
    # which erroneous columns are caught (record-level, by the column holding the error); errors in columns the
    # monitor does not check (names, city, zip) cannot be found by any method
    col_rows = []
    for (bm, meth), g in ev.groupby(["benchmark", "method"]):
        for c in [c[6:] for c in g.columns if c.startswith("n_col:")]:
            n = g[f"n_col:{c}"].sum()
            if n:
                col_rows.append({"benchmark": bm, "method": meth, "column": c, "error_records": int(n),
                                 "detected": int(g[f"tp_col:{c}"].sum()), "recall": g[f"tp_col:{c}"].sum() / n})
    pd.DataFrame(col_rows).to_csv(ctx.out / "external_by_column.csv", index=False)
    ps = ev.groupby(["benchmark", "method", "seed"])[["TP", "P", "FP", "N", "cell_TP", "cell_FP", "cell_FN"]].sum().reset_index()
    ps["cell_P"] = ps.cell_TP + ps.cell_FN
    ps["cell_flagged"] = ps.cell_TP + ps.cell_FP
    compare_methods(ps, ["benchmark"], "method", "fixed_pooled", "seed",
                    {"recall": ("TP", "P"), "fpr": ("FP", "N"), "cell_recall": ("cell_TP", "cell_P"),
                     "cell_precision": ("cell_TP", "cell_flagged")},
                    B=ctx.criteria["bootstrap_replicates"], level=ctx.criteria["confidence_level"]) \
        .to_csv(ctx.out / "external_paired_ci.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"benchmarks": names})
    print(t[["benchmark", "method", "precision", "recall", "fpr", "cell_precision", "cell_recall"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
