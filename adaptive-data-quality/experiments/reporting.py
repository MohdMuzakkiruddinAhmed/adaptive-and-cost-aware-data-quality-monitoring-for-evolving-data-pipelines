"""Shared output helpers: aggregation, run logs (Section 17), figure style."""
from __future__ import annotations

import platform
import subprocess
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from common.utils import now_iso, results_dir, write_json

# categorical slots 1-2 (validated adjacent pair); marker shape is the secondary encoding
COLORS = {"adaptive": "#2a78d6", "static": "#eb6834"}
MARKERS = {"adaptive": "o", "static": "s"}
LABELS = {"adaptive": "Adaptive thresholds", "static": "Static thresholds"}
DATASET_LABELS = {"bts_airline": "BTS Airline", "online_retail": "Online Retail II", "chicago_food": "Chicago Food Inspections"}


def aggregate(df: pd.DataFrame, by) -> pd.DataFrame:
    """Micro-averaged metrics (sum TP/FP/FN/TN over batches, then compute)."""
    g = df.groupby(by)[["TP", "FP", "FN", "TN", "records", "defects", "rules_executed", "runtime_ms"]].sum()
    g["precision"] = g.TP / (g.TP + g.FP).where(lambda s: s > 0)
    g["recall"] = g.TP / (g.TP + g.FN).where(lambda s: s > 0)
    g["f1"] = 2 * g.precision * g.recall / (g.precision + g.recall)
    g["fpr"] = g.FP / (g.FP + g.TN)
    g["fnr"] = g.FN / (g.FN + g.TP).where(lambda s: s > 0)
    return g.reset_index()


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parents[1], timeout=5).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def write_run_log(experiment: str, datasets: Dict[str, Any], exp: Dict[str, Any], batch_df: pd.DataFrame,
                  summary: Dict[str, Any], outputs: Dict[str, str], runtime_s: float) -> Path:
    """Section 17 fields for every run. Logs are never overwritten (timestamped file names)."""
    stamp = now_iso().replace(":", "").replace("-", "")
    log = {
        "experiment_id": f"{experiment}_{stamp}",
        "date": now_iso(),
        "datasets": {k: {"name": v["name"], "batch_size": v["batch_size"]} for k, v in datasets.items()},
        "dataset_version": "raw files listed in data/original/download_manifest.json",
        "random_seed": exp["seed"],
        "number_of_records": int(batch_df.drop_duplicates(["dataset", "scenario", "batch"])["records"].sum()),
        "injected_defects": {"types": exp["defects"]["types"], "rates": exp["schedules"]},
        "drift_levels": {s: exp["schedules"][v["drift"]] for s, v in exp["scenarios"].items()},
        "rules_executed": int(batch_df["rules_executed"].sum()),
        "parameters": {k: exp[k] for k in ("range_k", "adaptive", "classification", "warmup_batches", "n_batches")},
        "runtime_s": runtime_s,
        "detection_results": summary,
        "cost": {"total_rule_runtime_ms": float(batch_df["runtime_ms"].sum()),
                 "records_scanned": int(batch_df["records"].sum()),
                 "cloud_cost": "not measured (local run)"},
        "output_paths": outputs,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "git_commit": _git_commit()},
    }
    path = results_dir("raw", "run_logs", f"{log['experiment_id']}.json")
    write_json(log, path)
    return path


def style_axes(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#9a9a93")
    ax.tick_params(colors="#55554f", labelsize=9)
    ax.grid(axis="y", color="#e6e5df", linewidth=0.8)
    ax.set_axisbelow(True)
