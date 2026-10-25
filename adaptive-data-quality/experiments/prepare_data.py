"""Build the experimental data: clean data, T1..T20 batches per scenario, ground truth, profiles.

    python -m experiments.prepare_data                  # all datasets
    python -m experiments.prepare_data --dataset bts_airline
    python -m experiments.prepare_data --clean-only     # only data/clean (enough for the v2 experiments)

Outputs
    data/clean/<dataset>.parquet                         relatively clean source data
    data/clean/<dataset>_clean_report.json               rows kept at each cleaning step
    data/batches/<dataset>/<scenario>/T01.parquet ...    experimental batches
    data/ground_truth/<dataset>/<scenario>/T01.csv ...   injected defects (record_id, column, defect_type, severity, batch)
    data/ground_truth/<dataset>/<scenario>/labels.csv    per batch: drift level, defect rate, drift / degradation labels
    results/profiles/<dataset>/<scenario>/T01.json ...   batch profiles (Section 5)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import batch_label, data_dir, datasets_config, experiment_config, write_json  # noqa: E402
from ingestion.loaders import load_raw  # noqa: E402
from ingestion.prepare import clean, make_batches  # noqa: E402
from injection.defect_injector import (append_duplicates, inject_defects, inject_distribution_shift,  # noqa: E402
                                       inject_semantic_defects)
from injection.label_audit import audit_batch  # noqa: E402
from profiling.profiler import profile_batch  # noqa: E402
from profiling.store import ProfileStore  # noqa: E402


def prepare_dataset(name: str, cfg: dict, exp: dict, raw_root: Path, clean_only: bool = False) -> None:
    t0 = time.perf_counter()
    print(f"[{name}] loading raw data")
    raw = load_raw(name, cfg, raw_root)
    clean_df, report = clean(raw, cfg)
    clean_df.to_parquet(data_dir("clean", f"{name}.parquet"), index=False)
    write_json(report, data_dir("clean", f"{name}_clean_report.json"))
    print(f"[{name}] clean data: {report}")
    if not clean_only:              # proof-of-concept batches; the v2 experiments only need data/clean
        build_scenarios(name, cfg, exp, clean_df)
    print(f"[{name}] done in {time.perf_counter() - t0:.0f}s")


def build_scenarios(name: str, cfg: dict, exp: dict, clean_df: pd.DataFrame, scenarios: dict | None = None,
                    verbose: bool = True) -> None:
    """Batches, ground truth, labels and profiles for every scenario, from the clean data.
    Also used by the Section 14 sensitivity analysis with modified settings (seed, defect rate, severity)."""
    ds_index = list(datasets_config()).index(name)
    seed = exp["seed"]
    base = make_batches(clean_df, exp["n_batches"], cfg["batch_size"], seed=seed + ds_index)
    top = {c: clean_df[c].astype(str).value_counts().index[:10].tolist() for c in cfg["drift_categorical"]}
    numeric, categorical = cfg["numeric"], cfg["categorical"]
    warm = exp["warmup_batches"]

    if scenarios is None:
        scenarios = {**exp["scenarios"], **(exp.get("semantic_scenarios") or {})}
    for scen, spec in scenarios.items():
        drift = exp["schedules"][spec["drift"]]
        rates = exp["schedules"][spec["defects"]]
        sem = exp["semantic_schedules"][spec["semantic"]] if spec.get("semantic") else None
        bdir = data_dir("batches", name, scen); bdir.mkdir(parents=True, exist_ok=True)
        gdir = data_dir("ground_truth", name, scen); gdir.mkdir(parents=True, exist_ok=True)
        store = ProfileStore(name, scen)
        labels = []
        for t in range(1, exp["n_batches"] + 1):
            b = batch_label(t)
            # same seeds in every scenario, so scenarios differ only in drift level / defect rate
            df = inject_distribution_shift(base[t - 1], drift[t - 1], cfg["drift_numeric"], cfg["drift_categorical"],
                                           top, np.random.default_rng([seed, ds_index, t, 1]))
            reference = df
            # duplicates are deferred: appended after every other mutation (benchmark-integrity fix,
            # manuscript Section 6.2); their source records are reserved from later mutations
            df, gt, dup_src = inject_defects(df, rates[t - 1], numeric, categorical,
                                             np.random.default_rng([seed, ds_index, t, 2]), b, exp["defects"],
                                             defer_duplicates=True)
            if sem is not None:   # Experiment 3b: realistic domain violations on other records
                df, gs = inject_semantic_defects(df, sem[t - 1], cfg["semantic_defects"],
                                                 np.random.default_rng([seed, ds_index, t, 3]), b,
                                                 set(gt["record_id"]) | set(dup_src))
                gt = pd.concat([gt, gs], ignore_index=True)
            df, gd = append_duplicates(df, dup_src, b, exp["defects"]["severity"]["duplicate"])
            gt = pd.concat([x for x in (gt, gd) if len(x)], ignore_index=True)
            audit = audit_batch(df, gt, reference)
            if not audit["ok"]:
                raise RuntimeError(f"[{name}] {scen} {b}: ground truth does not match the batch: {audit['issues']}")
            df.to_parquet(bdir / f"{b}.parquet", index=False)
            gt.to_csv(gdir / f"{b}.csv", index=False)
            store.save(profile_batch(df, numeric, categorical, batch=b))
            labels.append({
                "batch": b, "drift_level": drift[t - 1], "defect_rate": rates[t - 1],
                "records": len(df), "defects": len(gt),
                "realized_defect_fraction": len(gt) / len(df),   # nominal r -> about r / (1 + r/4)
                "semantic_rate": sem[t - 1] if sem is not None else 0.0,
                "drift_label": int(drift[t - 1] > 0),
                "degradation_label": int(rates[t - 1] > rates[0]),
                "evaluation": int(t > warm),
            })
        pd.DataFrame(labels).to_csv(gdir / "labels.csv", index=False)
        if verbose:
            print(f"[{name}] {scen}: {exp['n_batches']} batches, {sum(l['defects'] for l in labels):,} defects injected")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="*", default=None)
    ap.add_argument("--raw-root", default=str(data_dir("original")))
    ap.add_argument("--clean-only", action="store_true",
                    help="only build data/clean (all the v2 experiments need), not the proof-of-concept batches")
    args = ap.parse_args()
    cfgs, exp = datasets_config(), experiment_config()
    for name in args.dataset or list(cfgs):
        prepare_dataset(name, cfgs[name], exp, Path(args.raw_root), clean_only=args.clean_only)


if __name__ == "__main__":
    main()
