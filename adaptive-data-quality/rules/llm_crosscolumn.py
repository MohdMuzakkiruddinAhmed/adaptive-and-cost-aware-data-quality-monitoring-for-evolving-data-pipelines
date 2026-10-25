"""LLM proposals of CROSS-COLUMN rules (v2.1; rules/crosscolumn.py executes and validates them).

A separate prompt and system prompt from rules/llm_generator.py (whose single-column prompt and cache stay
unchanged): the model sees the column names, descriptions, value statistics and a few example records, and names
dependencies ("zip determines city") and conditional constraints ("if has_child is N then child_exemp is 0").
It is never asked for the mapping itself (which zip belongs to which city): mappings are learned from the warm-up.

Same client, budget, cost ledger and fingerprinted cache as the single-column proposals (CrossColumnClient only
swaps the system prompt).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from rules.llm_generator import LLMClient, prompt_hash

XCOL_SYSTEM_PROMPT = """You are a data-quality engineer. You design CROSS-COLUMN validation rules for batches of records that arrive continuously in a data pipeline: rules that relate two or more fields of the same record. Single-column checks (missing values, ranges, allowed categories, formats) already exist; do not repeat them.

Use the meaning of the columns in their domain. A good rule holds for every legitimate record and is broken by records in which one of the related fields is wrong. Do not propose a rule that only happens to hold in the examples.

Answer with JSON only, no prose, in exactly this form:
{"rules": [ {rule}, ... ]}

Two rule types:
  dependency    the value(s) of the determinant column(s) determine the value of the dependent column (a
                functional dependency, e.g. a postal code determines its city). Fields:
                  "rule_type": "dependency", "determinant": [column, ...], "dependent": column
                The mapping itself is learned from historical batches; only name the columns.
  conditional   when a column takes one of the given values, another column is restricted. Fields:
                  "rule_type": "conditional",
                  "if":   {"column": column, "values": [value, ...]},
                  "then": {"column": column, "allowed_values": [value, ...]}   or
                          {"column": column, "min": number or null, "max": number or null}
Every rule also has:
  "condition":  the rule in plain words
  "reason":     why it holds in this domain
  "confidence": a number between 0 and 1"""


def build_xcol_prompt(name: str, cfg: Dict[str, Any], sample: pd.DataFrame, columns: List[str], n_examples: int,
                      seed: int) -> str:
    """Columns usable in cross-column rules (monitored + context columns), their statistics on the training sample
    and ``n_examples`` example records (training sample only: never an evaluation record)."""
    rng = np.random.default_rng(seed)
    lines = [f"Dataset: {cfg['name']}", f"Description: {cfg.get('description', '')}",
             f"Statistics below come from a training sample of {len(sample):,} records.", "", "Columns:"]
    desc = cfg.get("column_descriptions", {})
    for c in columns:
        s = sample[c]
        x = pd.to_numeric(s, errors="coerce")
        if c in cfg.get("numeric", []) and x.notna().mean() > 0.5:
            q = x.quantile([0, 0.5, 1]).to_numpy()
            stats = f"numeric, min {q[0]:.6g}, median {q[1]:.6g}, max {q[2]:.6g}, share of zeros {(x == 0).mean():.1%}"
        else:
            vc = s.astype(str).where(s.notna()).value_counts(normalize=True)
            top = ", ".join(f"{k} ({v:.1%})" for k, v in vc.head(8).items())
            stats = f"text, {s.nunique():,} distinct values; most common: {top}"
        lines.append(f"  {c}: {desc.get(c, 'n/a')}; {stats}; missing {s.isna().mean():.1%}")
    idx = np.sort(rng.choice(len(sample), size=min(n_examples, len(sample)), replace=False))
    ex = sample.iloc[idx][columns]
    # lineterminator fixed: to_csv defaults to os.linesep, which put CRLF into the prompt on Windows and made the
    # prompt (and its cache fingerprint) depend on the operating system
    lines += ["", "Example records:", ex.to_csv(index=False, lineterminator="\n").strip(), "",
              "Propose cross-column rules for this dataset. Return JSON only."]
    return "\n".join(lines)


def parse_xcol(text: str):
    from rules.llm_generator import parse_rules
    return parse_rules(text)


class CrossColumnClient(LLMClient):
    """LLMClient with the cross-column system prompt (cache, budget and ledger unchanged)."""

    def _request(self, prompt: str) -> Dict[str, Any]:
        req = super()._request(prompt)
        req["system"] = [{"text": XCOL_SYSTEM_PROMPT}]
        return req

    def generate(self, dataset, run, prompt, est_input_tokens, cache_dir=None, fingerprint=None):
        import rules.llm_generator as G
        saved_system, saved_mock = G.SYSTEM_PROMPT, G._mock_answer
        G.SYSTEM_PROMPT = XCOL_SYSTEM_PROMPT            # prompt hash, record and request use the cross-column prompt
        G._mock_answer = _mock_xcol_answer
        try:
            return super().generate(dataset, run, prompt, est_input_tokens, cache_dir, fingerprint)
        finally:
            G.SYSTEM_PROMPT, G._mock_answer = saved_system, saved_mock


def xcol_prompt_hash(model_id: str, prompt: str, max_tokens: int, temperature) -> str:
    return prompt_hash(model_id, XCOL_SYSTEM_PROMPT, prompt, max_tokens, temperature)


def _mock_xcol_answer(prompt: str) -> str:
    """Offline stand-in (pipeline tests only): one plausible dependency between the first two columns, one
    conditional on the first text column, one malformed rule."""
    cols = [ln.split(":")[0].strip() for ln in prompt.split("Columns:\n")[1].split("\n\n")[0].splitlines()]
    rules = []
    if len(cols) >= 2:
        rules.append({"rule_type": "dependency", "determinant": [cols[-1]], "dependent": cols[0],
                      "condition": "mock", "reason": "mock", "confidence": 0.6})
        rules.append({"rule_type": "conditional", "if": {"column": cols[0], "values": ["x"]},
                      "then": {"column": cols[1], "min": 0, "max": None}, "condition": "mock", "reason": "mock",
                      "confidence": 0.5})
    rules.append({"rule_type": "dependency", "determinant": ["no_such_column"], "dependent": cols[0] if cols else "x",
                  "confidence": 0.1})
    return json.dumps({"rules": rules})
