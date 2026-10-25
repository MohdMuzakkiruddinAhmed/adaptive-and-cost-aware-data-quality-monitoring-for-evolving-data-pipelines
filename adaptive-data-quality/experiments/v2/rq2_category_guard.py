"""RQ2 / H2b: preventing invalid categories from becoming accepted (supervisor experiment 4).

Design     Scenarios G1 (stable) and G2 (drift): previously unseen VALID categories (real values removed from
           the record pool, introduced from T6 or T11) and INVALID categories (well-formed values outside the
           domain), introduced at the same batches, with the same share per batch and matched format, so
           recurrence alone cannot reveal which is valid (protocol.yaml dataset_scenarios.*.novelty).
Policies   frozen | recurrence (PoC) | recurrence_manual | recurrence_llm (per model and generation) |
           recurrence_llm_value (per model: the LLM is asked about each new recurring value itself;
           experiments/v2/llm_value_checks.py)
           + honesty controls: recurrence with the manual / LLM domain rules executed as ordinary checks but
           WITHOUT guarding admission (does the guard add anything beyond running the rules?), and the guarded
           policies with their domain rules also executed.
Measures   per novel value: batches until an invalid value is accepted (censored if never), delay until a
           legitimate value is accepted, admission status (admitted by list / rejected / compatible-only /
           unconstrained = unresolved); per batch: invalid-value detection and false positives on legitimate
           novelty, for the category check alone and for the whole suite; record-level FPR / recall of the
           suite. "Rejecting every new category" (frozen) is the trivial control.

    python -m experiments.v2.rq2_category_guard --split dev --jobs 4
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.v2.common import (Context, ProfiledReplay, data_source, parser, run_monitor_over,  # noqa: E402
                                   run_units, summarize_counts)
from experiments.v2.llm_rules import load_generations, models  # noqa: E402
from experiments.v2.llm_value_checks import load_verdicts  # noqa: E402
from monitor.loop import Monitor, MonitorConfig  # noqa: E402
from rules.manual_rules import (admitted_manual_rules, domain_constraints, manual_config, manual_rules,  # noqa: E402
                                validate_manual)
from rules.static_rules import check  # noqa: E402
from rules.validator import accepted_rules, validate  # noqa: E402

EXPERIMENT = "rq2_category_guard"


def policies_for_run(cfg, exp, warm_dfs, ds, payload, src):
    """(name, model, generation, category_policy, guard constraints, executed domain rules)."""
    man_raw = manual_rules(ds, cfg, list(warm_dfs[0].columns), payload["manual_cfg"])
    if validate_manual(payload["manual_cfg"]):
        # v2.2: the manual constraints are the manual rules admitted on this run's warm-up (as the LLM constraints
        # below); recurrence_manual_raw keeps the rules as written (sensitivity)
        man = admitted_manual_rules(ds, cfg, warm_dfs, exp["validation"]["max_warmup_failure_rate"],
                                    payload["manual_cfg"])[0]
    else:
        man = man_raw
    man_dom = [r for r in man if r["type"] in ("category", "pattern")]
    out = [("frozen", "", 0, "frozen", None, []),
           ("recurrence", "", 0, "recurrence", None, []),
           ("recurrence_manual", "", 0, "recurrence_manual", domain_constraints(man), []),
           ("recurrence+manual_rules", "", 0, "recurrence", None, man_dom),
           ("recurrence_manual+manual_rules", "", 0, "recurrence_manual", domain_constraints(man), man_dom)]
    if man is not man_raw:
        out.append(("recurrence_manual_raw", "", 0, "recurrence_manual", domain_constraints(man_raw), []))
    for mk, mid in payload["models"].items():
        for g, rec in load_generations(src, ds, mid, payload["generations"], payload["mock"]).items():
            val = validate(rec["rules"] or [], cfg, warm_dfs, exp["validation"]["max_warmup_failure_rate"], "logical")
            llm = [dict(r, source="llm") for r in accepted_rules(val, prefix=f"{mk}{g}_")]
            dom = [r for r in llm if r["type"] in ("category", "pattern")]
            out += [("recurrence_llm", mk, g, "recurrence_llm", domain_constraints(llm), []),
                    ("recurrence+llm_rules", mk, g, "recurrence", None, dom),
                    ("recurrence_llm+llm_rules", mk, g, "recurrence_llm", domain_constraints(llm), dom)]
        verdicts = load_verdicts(src, ds, mid, payload["mock"])
        if any(verdicts.values()):
            out.append(("recurrence_llm_value", mk, 1, "recurrence_llm_value", verdicts, []))
    return out


def value_check_digest(src, ctx, mock: bool) -> dict:
    """Fingerprint of the cached value verdicts (new answers -> recompute); warns when none exist."""
    import hashlib
    out = {}
    for mk, mid in models(ctx.protocol).items():
        h = hashlib.sha256()
        n = 0
        for ds in ctx.datasets:
            for col, vals in load_verdicts(src, ds, mid, mock).items():
                for v, verdict in sorted(vals.items()):
                    h.update(f"{ds}|{col}|{v}|{verdict}".encode())
                    n += 1
        out[mk] = (n, h.hexdigest()[:12])
        if not n:
            print(f"[{EXPERIMENT}] no {mk} value checks (python -m experiments.v2.llm_generate): "
                  f"recurrence_llm_value skipped", flush=True)
    return out


def worker(ds: str, seed: int, payload) -> pd.DataFrame:
    src = data_source()
    cfg, exp, pr = src.datasets[ds], src.exp, src.protocol
    novelty = pr["dataset_scenarios"][ds]["novelty"]
    rows = []
    for scen in payload["scenarios"]:
        rep = ProfiledReplay(src.replay(ds, payload["split"], seed, scen))
        warm_dfs = [rep.batch(t).df for t in range(1, exp["warmup_batches"] + 1)]
        for name, mk, g, pol, cons, dom in policies_for_run(cfg, exp, warm_dfs, ds, payload, src):
            mc = MonitorConfig(name=name, method="adaptive", category_policy=pol, guard_constraints=cons,
                               extra_rules=dom, classify=False, dedupe_extra=False)
            m = Monitor(cfg, exp, pr, mc, seed=seed)
            for t, b, (prof, art) in rep:
                step = m.step(b.df, b.label, prof, art)
                if t <= exp["warmup_batches"]:
                    continue
                suite_flag = step["failed_ids"]
                cat_rules = {r["column"]: r for r in m.last_suite if r["type"] == "category" and r.get("source") == "stat"}
                ids = b.df["record_id"].to_numpy()
                # record-level suite counts (with valid novelty FP)
                valid_ids = set(b.novelty.loc[b.novelty.kind == "valid", "record_id"]) if len(b.novelty) else set()
                inv_ids = set(b.gt.loc[b.gt.defect_type == "invalid_novel", "record_id"])
                from evaluation.metrics import record_counts
                rows.append({"row_kind": "suite", **b.meta, "dataset": ds, "seed": seed, "scenario": scen,
                             "policy": name, "model": mk, "generation": g,
                             **record_counts(suite_flag, b.gt, ids, valid_ids),
                             "inv_n": len(inv_ids), "inv_tp_suite": len(inv_ids & suite_flag)})
                for nv in novelty:
                    c = nv["column"]
                    cat_mask = check(cat_rules[c], b.df).fillna(False).to_numpy(dtype=bool) if c in cat_rules else \
                        np.zeros(len(b.df), dtype=bool)
                    allowed = set(cat_rules[c]["allowed"]) if c in cat_rules else set()
                    status = (step["admissions"] or {}).get(c, {})
                    for kind in ("valid", "invalid"):
                        for v in nv[kind]:
                            sel = (b.df[c].astype(str) == v).to_numpy()
                            rows.append({"row_kind": "value", "dataset": ds, "seed": seed, "scenario": scen,
                                         "policy": name, "model": mk, "generation": g, "t": t, "column": c,
                                         "value": v, "kind": kind, "start": nv["start"],
                                         "present": int(sel.sum()), "flagged_category": int((sel & cat_mask).sum()),
                                         "flagged_suite": int(sum(1 for r in ids[sel] if r in suite_flag)),
                                         "admitted": v in allowed, "status": status.get(v, "not_recurring")})
    return pd.DataFrame(rows)


def acceptance_table(values: pd.DataFrame) -> pd.DataFrame:
    """Per (dataset, seed, scenario, policy, model, generation, value): first batch it was accepted."""
    out = []
    keys = ["dataset", "seed", "scenario", "policy", "model", "generation", "column", "value", "kind", "start"]
    for k, g in values.groupby(keys):
        g = g.sort_values("t")
        acc = g[(g.t >= g.start) & g.admitted.fillna(False).astype(bool)]
        last = g.iloc[-1]
        out.append({**dict(zip(keys, k)), "accepted": bool(len(acc)),
                    "batches_to_acceptance": float(acc.t.iloc[0] - k[-1]) if len(acc) else np.nan,
                    "final_status": last.status,
                    "unresolved": last.status in ("admitted_compatible", "admitted_unconstrained"),
                    "detected_share_category": g.flagged_category.sum() / max(g.present.sum(), 1),
                    "detected_share_suite": g.flagged_suite.sum() / max(g.present.sum(), 1)})
    return pd.DataFrame(out)


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--mock", action="store_true", help="use the offline mock LLM generations")
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    payload = {"models": models(ctx.protocol), "generations": ctx.protocol["rq2"]["generations"], "mock": args.mock,
               "manual_cfg": manual_config(), "value_checks": value_check_digest(data_source(), ctx, args.mock)}
    sfx = "_mock" if args.mock else ""
    raw = run_units(ctx, worker, payload, suffix=sfx)
    values = raw[raw.row_kind == "value"]
    suite = raw[raw.row_kind == "suite"]

    acc = acceptance_table(values)
    acc.to_csv(ctx.out / f"rq2b_acceptance_per_value{sfx}.csv", index=False)
    g = acc.groupby(["dataset", "policy", "model", "kind"])
    summ = g.agg(values=("value", "count"), accepted_share=("accepted", "mean"),
                 median_batches_to_acceptance=("batches_to_acceptance", "median"),
                 unresolved_share=("unresolved", "mean"),
                 detected_share_category=("detected_share_category", "mean"),
                 detected_share_suite=("detected_share_suite", "mean")).reset_index()
    summ.to_csv(ctx.out / f"rq2b_summary{sfx}.csv", index=False)

    # detection over time (batch-level curves) and false positives on legitimate novelty
    curve = values.groupby(["dataset", "policy", "model", "kind", "t"])[["present", "flagged_category", "flagged_suite"]].sum().reset_index()
    curve["share_flagged_category"] = curve.flagged_category / curve.present.where(curve.present > 0)
    curve["share_flagged_suite"] = curve.flagged_suite / curve.present.where(curve.present > 0)
    curve.to_csv(ctx.out / f"rq2b_detection_over_time{sfx}.csv", index=False)
    rec = summarize_counts(suite, ["dataset", "scenario", "policy", "model"])
    if "fp_valid_novel" in rec:
        rec["fpr_valid_novelty"] = rec.fp_valid_novel / rec.n_valid_novel.where(rec.n_valid_novel > 0)
    inv = suite.groupby(["dataset", "scenario", "policy", "model"])[["inv_n", "inv_tp_suite"]].sum().reset_index()
    rec = rec.merge(inv, on=["dataset", "scenario", "policy", "model"])
    rec["invalid_recall_suite"] = rec.inv_tp_suite / rec.inv_n.where(rec.inv_n > 0)
    rec.to_csv(ctx.out / f"rq2b_suite_record_level{sfx}.csv", index=False)
    ctx.manifest({"tables": str(ctx.out)}, {"mock": args.mock})
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
