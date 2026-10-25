"""Exchange folder between this pipeline and the external validation tools (Deequ, TensorFlow Data Validation).

Why a separate step: the two tools need their own runtimes (Deequ = Scala on Spark with a JVM; TFDV publishes no
Windows build and pins its own TensorFlow / Arrow stack), so they run as SIDECARS in a Linux / WSL environment
(tools/external_validators/). The pipeline writes the replay batches here, the sidecars write their raw outputs
next to them, and the experiment runners translate those outputs into rules and batch labels
(baselines/external_tools.py). The batches are the SAME deterministic replays every other method sees, so all
comparisons stay paired.

    results/v2/tool_exchange/<split>/
      manifest.json                        units, column roles, tool settings, export fingerprint
      warm/<hash>/T01..T05.parquet          warm-up batches, deduplicated by content (Deequ constraint suggestion)
      warm_clean/<hash>/T01..T05.parquet    the same warm-up batches WITHOUT their erroneous records (the labels are
                                            used only to drop records; nothing labelled is exported): Deequ's
                                            suggestions assume trustworthy profiling data, so the "clean warm-up"
                                            variant is the tool's best case (an optimistic upper bound)
      series/<ds>/seed<s>/<scenario>/T01..T20.parquet   whole runs: batch-level anomaly detection (H1b, "series"
                                            units) and Deequ's verification of its suggested constraints with
                                            row-level results (H1a, "verify" units)
      deequ/...  tfdv/...                  written by the sidecars

Only the monitored columns are exported (numeric as float, categorical as string), so every tool judges the
same columns as the monitor. No labels are exported.

    python -m experiments.v2.tool_exchange --split dev          (run_all step "export_tools")
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.v2.common import Context, cache_key, data_source, parser  # noqa: E402
from common.utils import protocol_config, v2_results_dir  # noqa: E402

EXPERIMENT = "tool_exchange"


def exchange_dir(split: str) -> Path:
    return v2_results_dir(EXPERIMENT, split)


def safe_name(column: str) -> str:
    """Column name usable in Spark SQL / TFDV feature names (the sidecars rename; outputs are mapped back)."""
    import re
    return re.sub(r"\W", "_", column)


def export_frame(df: pd.DataFrame, numeric: List[str], categorical: List[str]) -> pd.DataFrame:
    """Monitored columns only. A numeric column whose values do not all parse as numbers stays text, so the tools
    see the same values the monitor sees (no silent coercion)."""
    out = {}
    for c in numeric:
        x = pd.to_numeric(df[c], errors="coerce")
        lost = int(x.isna().sum() - df[c].isna().sum())
        out[c] = x.astype("float64") if lost == 0 else df[c].map(lambda v: None if pd.isna(v) else str(v)).astype("string")
    for c in categorical:
        out[c] = df[c].map(lambda v: None if pd.isna(v) else str(v)).astype("string")
    return pd.DataFrame(out)


def frame_hash(frames: List[pd.DataFrame], dataset: str) -> str:
    h = hashlib.sha256(dataset.encode())
    for f in frames:
        h.update(pd.util.hash_pandas_object(f, index=False).to_numpy().tobytes())
        h.update("|".join(f.columns).encode())
    return h.hexdigest()[:16]


def tool_outputs_digest(split: str, tools=("deequ", "tfdv")) -> Optional[str]:
    """Fingerprint of the exchange manifest and every sidecar output; part of the cache keys of the runners that
    use them, so new tool outputs recompute the affected experiments. None when nothing was exported."""
    root = exchange_dir(split)
    man = root / "manifest.json"
    if not man.exists():
        return None
    h = hashlib.sha256(man.read_bytes())
    for tool in tools:
        for p in sorted((root / tool).rglob("*.json")) if (root / tool).exists() else []:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def output_valid(root: Path, tool: str, unit: Dict[str, Any]) -> bool:
    """A sidecar output counts only if it was computed on the batches exported now (content hash), so a changed
    replay can never be scored with stale tool results."""
    if tool == "deequ_suggestions":                         # file name IS the content hash of the warm-up
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        variants = (man.get("tools", {}).get("deequ", {}) or {}).get("suggestion_variants") or {}
        return all(suggestion_path(root, v, unit, spec).exists() for v, spec in variants.items())
    if tool == "deequ_verify":
        p = root / "deequ" / "verify" / unit["dataset"] / f"seed{unit['seed']}" / f"{unit['scenario']}.json"
    else:
        p = root / tool / "series" / unit["dataset"] / f"seed{unit['seed']}" / f"{unit['scenario']}.json"
    if not p.exists():
        return False
    try:
        return json.loads(p.read_text(encoding="utf-8"))["unit"].get("series_hash") == unit["series_hash"]
    except (ValueError, KeyError):
        return False


def suggestion_path(root: Path, variant: str, unit: Dict[str, Any], spec: Dict[str, Any]) -> Path:
    h = unit["warm_clean_hash"] if spec.get("warmup") == "clean" else unit["warm_hash"]
    return root / "deequ" / "suggestions" / variant / f"{h}.json"


def missing_outputs(split: str, tool: str, datasets, seeds, scenarios) -> Optional[int]:
    """Number of required units without a valid output (None when nothing was exported)."""
    man = load_manifest(split)
    if man is None:
        return None
    root = exchange_dir(split)
    kind = {"deequ_suggestions": "suggest", "deequ_verify": "verify"}.get(tool, "series")
    need = [u for u in man["units"] if u["dataset"] in datasets and int(u["seed"]) in set(seeds)
            and u["scenario"] in scenarios and u.get(kind)]
    want = {(d, int(s), sc) for d in datasets for s in seeds for sc in scenarios}
    have = {(u["dataset"], int(u["seed"]), u["scenario"]) for u in need}
    return len(want - have) + sum(not output_valid(root, tool, u) for u in need)


def load_manifest(split: str) -> Optional[Dict[str, Any]]:
    p = exchange_dir(split) / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def unit_entry(manifest: Dict[str, Any], dataset: str, seed: int, scenario: str) -> Optional[Dict[str, Any]]:
    for u in manifest.get("units", []):
        if u["dataset"] == dataset and int(u["seed"]) == int(seed) and u["scenario"] == scenario:
            return u
    return None


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def main(argv=None) -> None:
    ap = parser(__doc__)
    args = ap.parse_args(argv)
    ctx = Context(EXPERIMENT, args)
    pr = ctx.protocol
    tcfg = pr.get("external_tools", {})
    suggest = sorted({s for e in tcfg.get("suggest_experiments", []) for s in pr["experiments"].get(e, [])})
    series = sorted({s for e in tcfg.get("series_experiments", []) for s in pr["experiments"].get(e, [])})
    verify = sorted({s for e in tcfg.get("verify_experiments", []) for s in pr["experiments"].get(e, [])})
    if args.scenarios:
        suggest = [s for s in suggest if s in args.scenarios]
        series = [s for s in series if s in args.scenarios]
        verify = [s for s in verify if s in args.scenarios]
    root = exchange_dir(ctx.split)
    key = cache_key(ctx, {"suggest": suggest, "series": series, "verify": verify, "seeds": ctx.seeds,
                          "datasets": sorted(ctx.datasets), "tools": tcfg})
    old = load_manifest(ctx.split)
    if old and old.get("export_key") == key and not args.force:
        print(f"[{EXPERIMENT}] batches already exported for this code and configuration ({root})")
        return
    manifest = export(root, data_source(), ctx.datasets, ctx.seeds, ctx.split, suggest, series, tcfg,
                      pr["classification"].get("drift_threshold", {}), key, verify)
    n_series = sum(u["series"] or u["verify"] for u in manifest["units"])
    n_warm = len({u["warm_hash"] for u in manifest["units"]})
    print(f"[{EXPERIMENT}] {len(manifest['units'])} units ({n_series} whole runs, {n_warm} distinct warm-ups) -> {root}")


def export(root: Path, src, datasets: Dict[str, Any], seeds, split: str, suggest, series, tcfg, drift_threshold,
           key: str, verify=()) -> Dict[str, Any]:
    """Write the batches and manifest.json; returns the manifest."""
    verify = list(verify or [])
    for sub in ("warm", "warm_clean", "series"):           # old batches; sidecar outputs are kept (they are
        if (root / sub).exists():                          # validated against the batch content hash)
            shutil.rmtree(root / sub)
    units, warm_seen = [], set()
    datasets_meta = {}
    for ds, cfg in datasets.items():
        num, cat = list(cfg["numeric"]), list(cfg["categorical"])
        datasets_meta[ds] = {"numeric": num, "categorical": cat, "safe_names": {c: safe_name(c) for c in num + cat}}
        for seed in seeds:
            for scen in sorted(set(suggest) | set(series) | set(verify)):
                rep = src.replay(ds, split, seed, scen)
                warm_frames = [export_frame(rep.batch(t).df, num, cat) for t in range(1, rep.warm + 1)]
                h = frame_hash(warm_frames, ds)
                if h not in warm_seen:
                    for t, f in enumerate(warm_frames, 1):
                        _write(f, root / "warm" / h / f"T{t:02d}.parquet")
                    warm_seen.add(h)
                clean_frames = []
                for t in range(1, rep.warm + 1):
                    b = rep.batch(t)
                    keep = ~b.df["record_id"].isin(b.defective).to_numpy()
                    clean_frames.append(export_frame(b.df[keep].reset_index(drop=True), num, cat))
                hc = frame_hash(clean_frames, ds + "|clean")
                if hc not in warm_seen:
                    for t, f in enumerate(clean_frames, 1):
                        _write(f, root / "warm_clean" / hc / f"T{t:02d}.parquet")
                    warm_seen.add(hc)
                is_series = scen in series
                is_verify = scen in verify
                sh = None
                if is_series or is_verify:
                    frames = [warm_frames[t - 1] if t <= rep.warm else export_frame(rep.batch(t).df, num, cat)
                              for t in range(1, rep.n_batches + 1)]
                    for t, f in enumerate(frames, 1):
                        _write(f, root / "series" / ds / f"seed{seed}" / scen / f"T{t:02d}.parquet")
                    sh = frame_hash(frames, ds)
                units.append({"dataset": ds, "seed": seed, "scenario": scen, "warm_hash": h, "warm_clean_hash": hc,
                              "series_hash": sh, "suggest": scen in suggest or is_verify, "series": is_series,
                              "verify": is_verify, "n_batches": rep.n_batches, "warmup": rep.warm,
                              "rows": [len(rep.batch(t).df) for t in range(1, rep.n_batches + 1)]
                              if (is_series or is_verify) else None})
            print(f"[{EXPERIMENT}] {ds} seed {seed}: exported", flush=True)
    manifest = {"split": split, "export_key": key, "datasets": datasets_meta, "units": units,
                "tools": tcfg, "classification": {"drift_threshold": drift_threshold}}
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    main()
