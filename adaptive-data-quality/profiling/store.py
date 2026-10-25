"""Profile storage for historical comparison.

results/profiles/<dataset>/<scenario>/<batch>.json   one profile per batch
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from common.utils import results_dir, write_json


class ProfileStore:
    def __init__(self, dataset: str, scenario: str, root: Path | None = None) -> None:
        self.dir = (root or results_dir("profiles")) / dataset / scenario

    def save(self, profile: Dict[str, Any]) -> Path:
        path = self.dir / f"{profile['batch']}.json"
        write_json(profile, path)
        return path

    def load(self, batch: str) -> Dict[str, Any]:
        with open(self.dir / f"{batch}.json", "r", encoding="utf-8") as fh:
            return json.load(fh)

    def history(self, batches: List[str]) -> List[Dict[str, Any]]:
        """Profiles of the given batches, in order (H_{t-k:t})."""
        return [self.load(b) for b in batches]
