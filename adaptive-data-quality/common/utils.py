"""Shared helpers: config loading, paths, JSON conversion, run logging."""
from __future__ import annotations

import datetime as _dt
import json
import math
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> Dict[str, Any]:
    """Load configs/<name>.yaml"""
    with open(REPO_ROOT / "configs" / f"{name}.yaml", "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def datasets_config() -> Dict[str, Dict[str, Any]]:
    return load_yaml("datasets")["datasets"]


def experiment_config() -> Dict[str, Any]:
    return load_yaml("experiment")


# Section 14 sensitivity runs build their own batches, ground truth and profiles. set_workspace()
# redirects exactly those three locations to a separate folder, so the main experimental data and
# results are never touched. Everything else (clean data, tables, logs) keeps its normal location.
_WORKSPACE: Optional[Path] = None


def set_workspace(path: Optional[Path]) -> None:
    global _WORKSPACE
    _WORKSPACE = Path(path) if path is not None else None


@contextmanager
def workspace(path: Optional[Path]) -> Iterator[None]:
    previous = _WORKSPACE
    set_workspace(path)
    try:
        yield
    finally:
        set_workspace(previous)


def data_dir(*parts: str) -> Path:
    if _WORKSPACE is not None and parts and parts[0] in ("batches", "ground_truth", "v2", "external"):
        return _WORKSPACE.joinpath(*parts)
    return REPO_ROOT.joinpath("data", *parts)


def results_dir(*parts: str) -> Path:
    if _WORKSPACE is not None and parts and parts[0] == "profiles":
        return _WORKSPACE.joinpath(*parts)
    return REPO_ROOT.joinpath("results", *parts)


def to_jsonable(obj: Any) -> Any:
    """Convert numpy / pandas values to plain JSON types (NaN -> None)."""
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, int):
        return obj
    if isinstance(obj, (pd.Timestamp, _dt.datetime, _dt.date)):
        return obj.isoformat()
    try:
        if pd.isna(obj):
            return None
    except (TypeError, ValueError):
        pass
    return str(obj)


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_jsonable(obj), fh, indent=2)


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def batch_label(t: int) -> str:
    return f"T{t:02d}"


def protocol_config() -> Dict[str, Any]:
    """configs/protocol.yaml: the v2 evaluation protocol (splits, seeds, scenarios, baselines, criteria)."""
    return load_yaml("protocol")


def v2_results_dir(*parts: str) -> Path:
    """results/v2/...: every v2 output, kept apart from the proof-of-concept results."""
    if _WORKSPACE is not None:
        return _WORKSPACE.joinpath("results_v2", *parts)
    return REPO_ROOT.joinpath("results", "v2", *parts)
