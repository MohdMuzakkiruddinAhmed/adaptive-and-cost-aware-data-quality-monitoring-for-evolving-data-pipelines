"""Shared helpers of the external-tool sidecars (standard library + pandas only, so each tool keeps its own
environment). Reads the exchange folder written by experiments/v2/tool_exchange.py."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List


def load_manifest(exchange: Path) -> Dict[str, Any]:
    p = exchange / "manifest.json"
    if not p.exists():
        sys.exit(f"{p} not found: run  python -m experiments.v2.tool_exchange --split <split>  first")
    return json.loads(p.read_text(encoding="utf-8"))


def warm_paths(exchange: Path, warm_hash: str, clean: bool = False) -> List[Path]:
    return sorted((exchange / ("warm_clean" if clean else "warm") / warm_hash).glob("T*.parquet"))


def series_paths(exchange: Path, unit: Dict[str, Any]) -> List[Path]:
    return sorted((exchange / "series" / unit["dataset"] / f"seed{unit['seed']}" / unit["scenario"]).glob("T*.parquet"))


def series_out(exchange: Path, tool: str, unit: Dict[str, Any], kind: str = "series") -> Path:
    return exchange / tool / kind / unit["dataset"] / f"seed{unit['seed']}" / f"{unit['scenario']}.json"


def verify_units(manifest: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    return (u for u in manifest["units"] if u.get("verify"))


def suggestion_variants(manifest: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    d = manifest.get("tools", {}).get("deequ", {}) or {}
    return d.get("suggestion_variants") or {"observed_default": {"warmup": "observed", "rules": "DEFAULT"}}


def variant_hash(unit: Dict[str, Any], spec: Dict[str, Any]) -> str:
    return unit["warm_clean_hash"] if spec.get("warmup") == "clean" else unit["warm_hash"]


def suggestion_out(exchange: Path, variant: str, warm_hash: str) -> Path:
    return exchange / "deequ" / "suggestions" / variant / f"{warm_hash}.json"


def output_current(path: Path, unit: Dict[str, Any]) -> bool:
    """True if an output exists and was computed on the currently exported batches (same content hash)."""
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8"))["unit"].get("series_hash") == unit["series_hash"]
    except (ValueError, KeyError):
        return False


def series_units(manifest: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    return (u for u in manifest["units"] if u["series"])


def warm_hashes(manifest: Dict[str, Any], suggest_only: bool = True, clean: bool = False) -> Dict[str, str]:
    """warm hash -> dataset, for the units that need a warm-up artefact."""
    out = {}
    for u in manifest["units"]:
        if u["suggest"] or not suggest_only:
            out[u["warm_clean_hash"] if clean else u["warm_hash"]] = u["dataset"]
    return out


def write_json(path: Path, obj: Any) -> None:
    """Atomic write: an interrupted sidecar never leaves a half-written output that looks complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=str)
    os.replace(tmp, path)


class Progress:
    def __init__(self, total: int, label: str) -> None:
        self.total, self.label, self.done, self.t0 = total, label, 0, time.time()

    def step(self, what: str) -> None:
        self.done += 1
        el = time.time() - self.t0
        eta = el / self.done * (self.total - self.done)
        print(f"[{self.label}] {self.done}/{self.total} {what}  ({el / 60:.1f} min, ~{eta / 60:.1f} min left)", flush=True)


def t_quantile(p: float, nu: float) -> float:
    """Student-t quantile by a Cornish-Fisher expansion around the normal quantile (standard library only; the same
    function as baselines/avh.py, accurate to <1% for nu >= 4 up to p = 0.999)."""
    from statistics import NormalDist
    z = NormalDist().inv_cdf(p)
    g = [(z ** 3 + z) / 4, (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96,
         (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384,
         (79 * z ** 9 + 776 * z ** 7 + 1482 * z ** 5 - 1920 * z ** 3 - 945 * z) / 92160]
    return z + sum(gi / nu ** (i + 1) for i, gi in enumerate(g))
