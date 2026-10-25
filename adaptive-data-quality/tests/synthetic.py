"""Small synthetic dataset for testing the v2 pipeline end to end without the real data (fast)."""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from common.utils import experiment_config, protocol_config

SYNTH_CFG = {
    "name": "Synthetic", "batch_size": 400,
    "key_columns": ["key"], "numeric": ["x", "y"], "categorical": ["c", "state"],
    "drift_numeric": ["x", "y"], "drift_categorical": ["c"],
    "importance": {"x": 1.0, "y": 0.5, "c": 1.0, "state": 0.5, "_records": 0.5},
    "description": "synthetic", "column_descriptions": {"x": "a positive amount", "y": "a duration", "c": "a code",
                                                         "state": "a state code"},
    "semantic_defects": [{"column": "y", "uniform": [-10, -1], "what": "negative duration"},
                         {"column": "state", "values": ["QQ"], "what": "non-existent state"}],
}
STATES = ["AA", "BB", "CC", "DD", "EE", "FF", "GG"]


def make_clean(n: int = 20000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"key": np.arange(n), "x": rng.gamma(4, 25, n) + 1, "y": rng.normal(50, 8, n).clip(1),
                         "c": rng.choice(list("ABCDEFGHIJ"), n, p=[.3, .2, .1, .1, .08, .07, .05, .04, .03, .03]),
                         "state": rng.choice(STATES, n, p=[.3, .2, .15, .15, .1, .06, .04])})


def configs():
    exp = experiment_config()
    pr = copy.deepcopy(protocol_config())
    pr["dataset_scenarios"] = {"synth": {
        "contamination_columns": ["x"], "sparse_category_column": "state",
        "novelty": [{"column": "state", "start": 6, "share": 0.02, "valid": ["FF"], "invalid": ["QZ"]},
                    {"column": "c", "start": 8, "share": 0.02, "valid": ["J"], "invalid": ["Q"]}]}}
    datasets = {"synth": SYNTH_CFG}
    return datasets, exp, pr


MANUAL = {"rules": {"synth": [
    {"column": "y", "rule_type": "range", "min": 0, "max": None},
    {"column": "state", "rule_type": "category", "allowed_values": STATES},
    {"column": "c", "rule_type": "pattern", "pattern": "[A-Z]"},
    {"column": "_records", "rule_type": "uniqueness"},
]}}
