"""RQ2 / H2a: additional coverage from validated LLM rules (supervisor experiment 3).

Question   Do validated LLM rules detect independently defined semantic errors that statistical rules miss,
           at an acceptable false-positive increase? Is the gain due to domain knowledge (manual rules) or
           to LLM-assisted authoring?
Configs    stat_fixed (pooled warm-up fixed rules) | stat_adaptive (proposed adaptive rules)
           llm_validated | llm_raw (executable proposals without statistical validation / redundancy removal)
           fixed+llm_validated | adaptive+llm_validated | fixed+llm_raw | adaptive+llm_raw
           manual | fixed+manual | adaptive+manual    (configs/manual_rules.yaml, same permitted information;
           v2.2 with validate_manual: true these are the manual rules ADMITTED by the same warm-up test as the LLM
           rules, and manual_raw | fixed+manual_raw | adaptive+manual_raw are the rules as written)
           LLM configurations for every model (protocol.yaml rq2.models), every generation (rq2.generations)
           and both redundancy modes (empirical = PoC, logical = predicate-level refinement).
Admission  per run, on that run's own warm-up batches T1..T5 (no evaluation data).
Measures   semantic-error recall (overall, per violated column), standard-defect recall, FPR on valid records,
           semantic errors uniquely detected (missed by the statistical suite), incorrectly admitted /
           rejected rules, predicate-level stability across generations and models, generation cost
           (tokens, USD) and validation time; paired seed-level CIs of (with LLM - statistical suite).

All checks run on every batch here, so a configuration's flagged set is exactly the union of its parts;
each distinct rule is executed once per batch and shared by every configuration that contains it.

Requires generations from: python -m experiments.v2.llm_generate   (use --mock for an offline pipeline test)

    python -m experiments.v2.rq2_coverage --split dev --jobs 4
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.metrics import record_counts  # noqa: E402
from evaluation.rule_analysis import admission_outcomes, predicate_stability  # noqa: E402
from evaluation.stats import compare_methods  # noqa: E402
from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_units,  # noqa: E402
                                   summarize_counts)
from experiments.v2.llm_rules import load_generations, models, training_sample  # noqa: E402
from monitor.classifier import _sig  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402
from rules.manual_rules import admitted_manual_rules, manual_config, manual_rules, validate_manual  # noqa: E402
from rules.static_rules import check  # noqa: E402
from rules.validator import accepted_rules, executable_rules, normalise, validate  # noqa: E402

EXPERIMENT = "rq2_coverage"
STAT = {"fixed": "fixed_pooled", "adaptive": "adaptive"}


def _semantic(gt: pd.DataFrame) -> pd.DataFrame:
    return gt[gt["defect_type"].astype(str).str.startswith("semantic")]


def rule_sets_for_run(cfg, exp, warm_dfs, gens, modes, manual):
    """Admitted / raw rule sets of every (kind, model, generation, redundancy) on this run's warm-up."""
    cols = list(warm_dfs[0].columns)
    sets, validations, vcost = {}, {}, []
    for mk, gmap in gens.items():
        for g, rec in gmap.items():
            cands = rec["rules"] or []
            val = None
            for red in modes:
                t0 = time.perf_counter()
                val = validate(cands, cfg, warm_dfs, exp["validation"]["max_warmup_failure_rate"], red)
                vcost.append({"model": mk, "generation": g, "redundancy": red,
                              "validation_ms": (time.perf_counter() - t0) * 1000, "candidates": len(cands)})
                sets[("llm_validated", mk, g, red)] = [dict(r, source="llm") for r in
                                                      accepted_rules(val, prefix=f"{mk}{g}{red[0]}_")]
                validations[(mk, g, red)] = (val, cands)
            sets[("llm_raw", mk, g, "")] = [dict(r, source="llm_raw") for r in
                                           executable_rules(val, cands, cfg, cols, prefix=f"{mk}{g}raw_")]
    if manual:
        sets[("manual", "", 0, "")] = manual
    return sets, validations, vcost


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    gens = {mk: load_generations(src, ds, mid, payload["generations"], payload["mock"])
            for mk, mid in payload["models"].items()}
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        warm_dfs = [rep.batch(t).df for t in range(1, exp["warmup_batches"] + 1)]
        man_raw = manual_rules(ds, cfg, list(warm_dfs[0].columns), payload["manual_cfg"])
        man_val = None
        if validate_manual(payload["manual_cfg"]):
            # v2.2: "manual" = manual rules after the LLM admission test on this run's warm-up; "manual_raw" = as written
            man, man_val, man_cands = admitted_manual_rules(ds, cfg, warm_dfs, exp["validation"]["max_warmup_failure_rate"],
                                                            payload["manual_cfg"])
        else:
            man = man_raw
        sets, validations, vcost = rule_sets_for_run(cfg, exp, warm_dfs, gens, payload["redundancy_modes"], man)
        if man_val is not None:
            sets[("manual_raw", "", 0, "")] = man_raw
            validations[("manual", 0, "logical")] = (man_val, man_cands)
        for v in vcost:
            rows.append({"row_kind": "validation_cost", "dataset": ds, "seed": seed, "scenario": scen, **v})

        stat_flag = {k: {} for k in STAT}
        for k, method in STAT.items():
            m = Monitor(cfg, exp, pr, MonitorConfig(name=method, method=method, classify=False), seed=seed)
            for t, b, (prof, art) in rep:
                stat_flag[k][t] = m.step(b.df, b.label, prof, art)["failed_ids"]

        uniq = {}
        for rs in sets.values():
            for r in rs:
                uniq.setdefault(_sig(r), r)
        eval_batches = []
        for t in range(exp["warmup_batches"] + 1, exp["n_batches"] + 1):
            b = rep.batch(t)
            ids = b.df["record_id"].to_numpy()
            flagged_by = {s: set(ids[check(r, b.df).fillna(False).to_numpy(dtype=bool)]) for s, r in uniq.items()}
            sem = _semantic(b.gt)
            sem_ids = set(sem.record_id)
            eval_batches.append((b.df, b.gt, stat_flag["adaptive"][t]))

            def emit(config, flagged, model="", gen=0, red="", base=None):
                row = {"row_kind": "batch", **b.meta, "dataset": ds, "seed": seed, "scenario": scen,
                       "config": config, "model": model, "generation": gen, "redundancy": red,
                       **record_counts(flagged, b.gt, ids)}
                row["sem_n"], row["sem_tp"] = len(sem_ids), len(flagged & sem_ids)
                for k in STAT:
                    row[f"sem_unique_vs_{k}"] = len((flagged & sem_ids) - stat_flag[k][t])
                for col, g in sem.groupby("column"):
                    s_ids = set(g.record_id)
                    row[f"n_semcol:{col}"], row[f"tp_semcol:{col}"] = len(s_ids), len(flagged & s_ids)
                rows.append(row)

            for k in STAT:
                emit(f"stat_{k}", stat_flag[k][t])
            for (kind, mk, g, red), rs in sets.items():
                f = set().union(*[flagged_by[_sig(r)] for r in rs]) if rs else set()
                emit(kind, f, mk, g, red)
                for k in STAT:
                    emit(f"{k}+{kind}", f | stat_flag[k][t], mk, g, red)

        if any(len(_semantic(gt)) for _, gt, _ in eval_batches):
            for (mk, g, red), (val, cands) in validations.items():
                ao = admission_outcomes(val, cands, cfg, eval_batches, payload["rule_fp_tolerance"])
                for r in ao.to_dict("records"):
                    rows.append({"row_kind": "admission", "dataset": ds, "seed": seed, "scenario": scen,
                                 "model": mk, "generation": g, "redundancy": red, **r})
    return pd.DataFrame(rows)


def generation_cost(gens_by_model, prices) -> list:
    """Tokens and USD per generation. Bedrock reported inputTokens = 2 for every GPT-6 Astra call (4 Oct 2026)
    although the full prompt was used, so the logged cost omits the input charge. When the reported input is
    implausibly small (< half the character-based estimate), the table also gives the estimate (characters / 3.2,
    as in rules/llm_generator.py) and the cost with it, and flags the row; the AWS bill is the authority."""
    from rules.llm_generator import SYSTEM_PROMPT
    rows = []
    for ds, gm in gens_by_model.items():
        for mk, gg in gm.items():
            for g, r in gg.items():
                est_in = int(len(SYSTEM_PROMPT + (r.get("prompt") or "")) / 3.2) + 20
                rep_in, out_tok = int(r.get("input_tokens") or 0), int(r.get("output_tokens") or 0)
                implausible = rep_in < 0.5 * est_in
                p_in, p_out = prices.get(r.get("model_id"), [np.nan, np.nan])
                rows.append({"dataset": ds, "model": mk, "model_id": r.get("model_id"), "generation": g,
                             "rules": len(r.get("rules") or []), "input_tokens_reported": rep_in,
                             "input_tokens_estimated": est_in, "input_tokens_implausible": implausible,
                             "output_tokens": out_tok, "cost_usd_logged": r.get("cost_usd"),
                             "cost_usd_corrected": ((est_in if implausible else rep_in) * p_in + out_tok * p_out) / 1e6})
    return rows


def stability_tables(src, ctx, gens_by_model, probe_n: int) -> pd.DataFrame:
    out = []
    for ds, cfg in ctx.datasets.items():
        sample = training_sample(src, ds)
        cols = list(sample.columns)
        rule_sets = {}
        for mk, gmap in gens_by_model[ds].items():
            for g, rec in gmap.items():
                rs = []
                for c in rec["rules"] or []:
                    r, err = normalise(c, cfg, cols)
                    if not err:
                        rs.append(r)
                rule_sets[f"{mk}:{g}"] = rs
        if len(rule_sets) < 2:
            continue
        st = predicate_stability(rule_sets, sample, cfg, probe_n)
        st["pair"] = np.where(st.gen_a.str.split(":").str[0] == st.gen_b.str.split(":").str[0], "within_model", "across_models")
        st["model_a"], st["model_b"] = st.gen_a.str.split(":").str[0], st.gen_b.str.split(":").str[0]
        out.append(st.assign(dataset=ds))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--mock", action="store_true", help="use the offline mock generations (pipeline test)")
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    rq2 = ctx.protocol["rq2"]
    mdl = models(ctx.protocol)
    src = data_source()
    gens_by_model = {ds: {mk: load_generations(src, ds, mid, rq2["generations"], args.mock) for mk, mid in mdl.items()}
                     for ds in ctx.datasets}
    missing = [(ds, mk) for ds, g in gens_by_model.items() for mk, x in g.items() if not x]
    if missing and not args.summary_only:
        print(f"WARNING: no LLM generations for {missing}; run python -m experiments.v2.llm_generate first. "
              "Those configurations are skipped.")
    payload = {"models": mdl, "generations": rq2["generations"], "redundancy_modes": rq2["redundancy_modes"],
               "rule_fp_tolerance": rq2["rule_fp_tolerance"], "mock": args.mock, "manual_cfg": manual_config()}
    raw = run_units(ctx, worker, payload, suffix="_mock" if args.mock else "")
    crit = ctx.criteria
    sfx = "_mock" if args.mock else ""

    b = raw[(raw.row_kind == "batch") & (raw.evaluation == 1)].copy()
    keys = ["dataset", "scenario", "config", "model", "redundancy"]
    per_gen = summarize_counts(b, keys + ["generation"])
    sem = b.groupby(keys + ["generation"])[["sem_n", "sem_tp", "sem_unique_vs_fixed", "sem_unique_vs_adaptive"]].sum().reset_index()
    per_gen = per_gen.merge(sem, on=keys + ["generation"])
    per_gen["semantic_recall"] = per_gen.sem_tp / per_gen.sem_n.where(per_gen.sem_n > 0)
    per_gen.to_csv(ctx.out / f"rq2a_per_generation{sfx}.csv", index=False)
    metrics = ["semantic_recall", "recall", "precision", "fpr", "f1", "sem_unique_vs_fixed", "sem_unique_vs_adaptive"]
    agg = per_gen.groupby(keys)[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg.reset_index().to_csv(ctx.out / f"rq2a_summary{sfx}.csv", index=False)

    # per violated column
    semcols = [c for c in b.columns if c.startswith("n_semcol:")]
    if semcols:
        g = b.groupby(keys)[semcols + [c.replace("n_", "tp_", 1) for c in semcols]].sum()
        long = []
        for idx, r in g.iterrows():
            for c in semcols:
                if r[c]:
                    long.append({**dict(zip(keys, idx)), "column": c.split(":", 1)[1], "n": r[c],
                                 "recall": r[c.replace("n_", "tp_", 1)] / r[c]})
        pd.DataFrame(long).to_csv(ctx.out / f"rq2a_semantic_by_column{sfx}.csv", index=False)

    # paired CIs: each "<stat>+X" vs "stat_<stat>", counts pooled over generations within each seed
    b["sem_fn"] = b.sem_n - b.sem_tp
    cis = []
    for k in STAT:
        for (kind, mk, red), _ in b[b.config.str.startswith(f"{k}+")].groupby(["config", "model", "redundancy"]):
            sub = b[((b.config == kind) & (b.model == mk) & (b.redundancy == red)) | (b.config == f"stat_{k}")]
            ps = sub.groupby(["dataset", "scenario", "config", "seed"])[["TP", "P", "FP", "N", "sem_tp", "sem_n"]].sum().reset_index()
            c = compare_methods(ps, ["dataset", "scenario"], "config", f"stat_{k}", "seed",
                                {"semantic_recall": ("sem_tp", "sem_n"), "recall": ("TP", "P"), "fpr": ("FP", "N")},
                                B=crit["bootstrap_replicates"], level=crit["confidence_level"])
            cis.append(c.assign(model=mk, redundancy=red))
    ci = pd.concat(cis, ignore_index=True) if cis else pd.DataFrame()
    if len(ci):
        tol = crit["fpr_increase_tolerance"]
        ci["fpr_within_tolerance"] = np.where(ci.metric == "fpr", ci.ci_high <= tol, np.nan)
    ci.to_csv(ctx.out / f"rq2a_paired_ci{sfx}.csv", index=False)

    adm = raw[raw.row_kind == "admission"]
    if len(adm):
        a = adm.groupby(["dataset", "model", "redundancy"]).agg(
            candidates=("index", "count"), admitted=("admitted", "sum"), incorrect=("correct", lambda s: int(s.eq(False).sum())),
            incorrectly_admitted=("incorrectly_admitted", "sum"), incorrectly_rejected=("incorrectly_rejected", "sum")).reset_index()
        a.to_csv(ctx.out / f"rq2a_admission_errors{sfx}.csv", index=False)
        adm.to_csv(ctx.out / f"rq2a_admission_detail{sfx}.csv", index=False)
    vc = raw[raw.row_kind == "validation_cost"]
    if len(vc):
        vc.groupby(["dataset", "model", "redundancy"])["validation_ms"].describe().to_csv(ctx.out / f"rq2a_validation_cost{sfx}.csv")
    pd.DataFrame(generation_cost(gens_by_model, ctx.exp["llm"]["prices"])).to_csv(
        ctx.out / f"rq2a_generation_cost{sfx}.csv", index=False)
    stability_tables(src, ctx, gens_by_model, rq2["probe_values"]).to_csv(ctx.out / f"rq2a_predicate_stability{sfx}.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"models": mdl, "mock": args.mock})
    show = agg.reset_index()
    print(show[[c for c in show.columns if c in keys or c.endswith("_mean")]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
