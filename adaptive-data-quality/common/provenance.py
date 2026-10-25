"""Run provenance (manuscript Section 6.6): code, configuration and data fingerprints.

Every v2 run writes a manifest with the source version (a hash of every project .py file), the full
configuration and its hash, seeds, data hashes, prompt hashes, environment and outputs. Caches (LLM
answers, generated rules) are keyed by these fingerprints rather than by a setting name, so a cached
artefact can never be silently reused after the code, data or configuration it came from changed.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from common.utils import REPO_ROOT, now_iso, to_jsonable, write_json

CODE_DIRS = ("adaptation", "baselines", "common", "drift", "execution", "experiments", "evaluation", "ingestion",
             "injection", "monitor", "optimizer", "profiling", "risk", "rules")


def sha256_bytes(b: bytes, n: int = 16) -> str:
    return hashlib.sha256(b).hexdigest()[:n]


def config_hash(obj: Any, n: int = 16) -> str:
    return sha256_bytes(json.dumps(to_jsonable(obj), sort_keys=True).encode(), n)


def file_hash(path: Path, n: int = 16, chunk: int = 1 << 20) -> Optional[str]:
    path = Path(path)
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:n]


def code_fingerprint(files: Optional[Iterable[Path]] = None, n: int = 16) -> str:
    """Hash of the project source (paths and contents), independent of file timestamps."""
    if files is None:
        files = sorted(p for d in CODE_DIRS for p in (REPO_ROOT / d).rglob("*.py") if "__pycache__" not in p.parts)
    h = hashlib.sha256()
    for p in files:
        h.update(str(Path(p).relative_to(REPO_ROOT)).replace("\\", "/").encode())
        h.update(Path(p).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:n]


def config_files_hashes() -> Dict[str, Optional[str]]:
    return {p.name: file_hash(p) for p in sorted((REPO_ROOT / "configs").glob("*.yaml"))}


def environment() -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    return {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "pandas": pd.__version__, "numpy": np.__version__,
            "executable": sys.executable}


def write_manifest(path: Path, experiment: str, split: str, seeds, config: Dict[str, Any],
                   data_hashes: Dict[str, Any], outputs: Dict[str, str], extra: Optional[Dict[str, Any]] = None) -> Path:
    manifest = {"experiment": experiment, "created_at": now_iso(), "split": split, "seeds": list(seeds),
                "code_fingerprint": code_fingerprint(), "config_hash": config_hash(config),
                "config_files": config_files_hashes(), "config": config, "data": data_hashes,
                "environment": environment(), "outputs": outputs, **(extra or {})}
    write_json(manifest, path)
    return path
