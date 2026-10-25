"""Write the v2 ground truth, labels, label audits and profiles of every replay (manuscript Section 6.6).

Batches are regenerated deterministically in memory by every experiment; this step stores the small, auditable
artefacts so that no table depends on an untraceable aggregate: per (dataset, split, seed, scenario)
  ground_truth.parquet   every labelled defect (record, column, type, severity, batch, incident)
  labels.csv             per batch: drift level, nominal and realised defect rate, drift / degradation labels
  novelty.parquet        valid and invalid novel categories (H2b scenarios)
  audit.json             label audit of every batch (must all be ok)
  profiles/T01.json ...  batch profiles
and, with --batches, the batches themselves (large). Output: data/v2/<split>/<dataset>/seed<seed>/<scenario>/

    python -m experiments.v2.materialize --split dev
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import data_dir, write_json  # noqa: E402
from experiments.v2.common import Context, data_source, parser  # noqa: E402
from profiling.profiler import profile_batch  # noqa: E402


def main(argv=None) -> None:
    ap = parser(__doc__)
    ap.add_argument("--batches", action="store_true", help="also write every batch as parquet (large)")
    args = ap.parse_args(argv)
    ctx = Context("materialize", args)
    scen = args.scenarios or list(ctx.protocol["scenarios"])
    src = data_source()
    from experiments.v2.common import cache_key
    key = cache_key(ctx, {"materialize": True})
    summary = []
    for ds, cfg in ctx.datasets.items():
        for seed in ctx.seeds:
            for sc in scen:
                out = data_dir("v2", ctx.split, ds, f"seed{seed}", sc)
                if (out / "audit.json").exists() and not args.force:
                    import json
                    if json.loads((out / "audit.json").read_text()).get("cache_key") == key:
                        continue                         # unchanged code and configuration
                rep = src.replay(ds, ctx.split, seed, sc, cache=False)
                gts, novs, labels = [], [], []
                (out / "profiles").mkdir(parents=True, exist_ok=True)
                for t in range(1, rep.n_batches + 1):
                    b = rep.batch(t)
                    gts.append(b.gt); novs.append(b.novelty); labels.append(b.meta)
                    write_json(profile_batch(b.df, cfg["numeric"], cfg["categorical"], batch=b.label),
                               out / "profiles" / f"{b.label}.json")
                    if args.batches or ctx.protocol["protocol"].get("materialize_batches"):
                        b.df.to_parquet(out / f"{b.label}.parquet", index=False)
                import pandas as pd
                pd.concat(gts, ignore_index=True).astype({"severity": str, "incident": str}).to_parquet(out / "ground_truth.parquet", index=False)
                pd.concat(novs, ignore_index=True).to_parquet(out / "novelty.parquet", index=False)
                pd.DataFrame(labels).to_csv(out / "labels.csv", index=False)
                ok = all(a["ok"] for a in rep.audits.values())
                write_json({"all_ok": ok, "batch_size": rep.batch_size, "batches": rep.audits, "cache_key": key},
                           out / "audit.json")
                summary.append({"dataset": ds, "seed": seed, "scenario": sc, "audit_ok": ok,
                                "defects": int(sum(len(g) for g in gts))})
                print(f"[materialize] {ds} seed {seed} {sc}: audit {'ok' if ok else 'FAILED'}", flush=True)
    ctx.manifest({"data": str(data_dir("v2", ctx.split))}, {"materialized": summary})


if __name__ == "__main__":
    main()
