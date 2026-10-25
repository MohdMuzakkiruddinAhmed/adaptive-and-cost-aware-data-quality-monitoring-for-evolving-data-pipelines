"""LLM-assisted rule generation (project description, Section 8.3) on AWS Bedrock.

The LLM receives the dataset description, and for every monitored column its name,
description, type, example values and summary statistics from the batch profile. It must
answer with structured JSON rules, each with a reason and a confidence. Its output is
never activated directly: every rule goes through rules/validator.py (Section 9).

Cost control
------------
* Every response is cached in results/llm_responses/<dataset>/run_<r>.json together with
  its token usage and cost. Re-running the experiments reuses the cache and costs nothing.
  A new call is made only if the prompt, the model or the run number changes.
* Before each new call the worst-case cost (input tokens + max_tokens output) is checked
  against llm.max_cost_usd. The run stops instead of exceeding the budget.
* Every paid call is appended to results/llm_responses/cost_ledger.csv.

Credentials come from the environment (AWS_BEARER_TOKEN_BEDROCK or IAM keys, and AWS_REGION),
never from this repository. ADQ_LLM_MOCK=1 replaces Bedrock with a deterministic offline
stand-in for testing the pipeline (results are marked as mock and stored separately).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from common.utils import now_iso, results_dir

RULE_TYPES = ["completeness", "range", "category", "pattern", "uniqueness", "type"]

SYSTEM_PROMPT = """You are a data-quality engineer. You design validation rules for batches of records that arrive continuously in a data pipeline. The rules are executed automatically on every new batch, and a record that fails any rule is reported as a data-quality defect.

Use the semantics of each column (its name, description and meaning in the domain) as well as the statistics provided. A good rule catches values that are genuinely invalid or implausible for the domain. It should not fire on legitimate records, including legitimate values that are rare, or legitimate changes in the data over time (for example seasonal shifts or growth).

Answer with JSON only, no prose, in exactly this form:
{"rules": [ {rule}, ... ]}

Each rule has these fields:
  "column":         the column name, or "_records" for a rule on the whole record
  "rule_type":      one of "completeness", "range", "category", "pattern", "uniqueness", "type"
  "min", "max":     numbers for "range" rules (either may be null for a one-sided range), otherwise null
  "allowed_values": list of strings for "category" rules, otherwise null
  "pattern":        a regular expression the whole value must match, for "pattern" rules, otherwise null
  "condition":      the rule in plain words
  "reason":         why this rule is valid for this data
  "confidence":     a number between 0 and 1

Meaning of the rule types:
  completeness  the value must not be missing
  range         a numeric value must lie in [min, max]
  category      the value must be one of allowed_values (use only for small closed sets)
  pattern       the text value must fully match the regular expression
  uniqueness    use with column "_records": the same record must not appear twice
  type          the value must be numeric"""


# --------------------------------------------------------------------------- prompt
def _num(v: Any) -> str:
    if v is None:
        return "null"
    v = float(v)
    return f"{v:.6g}"


def column_block(name: str, col_type: str, description: Optional[str], profile: Dict[str, Any],
                 samples: List[Any]) -> str:
    lines = [f"Column: {name}", f"  description: {description or 'n/a'}", f"  type: {col_type}"]
    lines.append(f"  missing: {_num(profile.get('missing_pct'))}% of records")
    if col_type == "numerical":
        q = profile.get("quantiles", {})
        lines.append("  statistics: " + ", ".join(
            [f"min {_num(profile.get('min'))}", f"max {_num(profile.get('max'))}", f"mean {_num(profile.get('mean'))}",
             f"median {_num(profile.get('median'))}", f"std {_num(profile.get('std'))}",
             f"unique values {profile.get('unique_count')}"]))
        lines.append("  quantiles: " + ", ".join(f"{k} {_num(v)}" for k, v in q.items()))
    else:
        freq = sorted(profile.get("frequencies", {}).items(), key=lambda kv: -kv[1])
        lines.append(f"  number of distinct values: {profile.get('cardinality')}")
        lines.append("  most common values (share of records): " +
                     ", ".join(f"{k} ({v:.3%})" for k, v in freq[:20]))
    lines.append("  example values: " + ", ".join(str(s) for s in samples))
    return "\n".join(lines)


def build_prompt(name: str, cfg: Dict[str, Any], profile: Dict[str, Any], example_df: pd.DataFrame,
                 n_samples: int, seed: int,
                 stats_source: str = "Statistics below come from one recent batch.") -> str:
    """Prompt for one dataset.

    LEAKAGE RULE (manuscript Sections 5.2 and 8.3): ``profile`` and ``example_df`` must contain only
    records the monitor is allowed to see before evaluation: the warm-up batches of the run or the
    development (training) pool. The proof of concept drew examples from the whole cleaned source,
    which overlaps later evaluation records; callers now pass warm-up/training records only.
    """
    rng = np.random.default_rng(seed)
    blocks = []
    for c in cfg["numeric"] + cfg["categorical"]:
        vals = example_df[c].dropna().to_numpy()
        pick = vals[rng.choice(len(vals), size=min(n_samples, len(vals)), replace=False)]
        samples = [(_num(v) if c in cfg["numeric"] else str(v)) for v in pick]
        blocks.append(column_block(c, "numerical" if c in cfg["numeric"] else "categorical",
                                   cfg.get("column_descriptions", {}).get(c), profile["columns"][c], samples))
    others = [c for c in cfg["key_columns"] if c not in cfg["numeric"] + cfg["categorical"]]
    return (f"Dataset: {cfg['name']}\n"
            f"Description: {cfg.get('description', '')}\n"
            f"Batch size: about {profile['dataset']['record_count']:,} records. {stats_source}\n"
            f"The records also contain identifying columns ({', '.join(others)}); together with the columns below "
            f"they make up the whole record.\n\n"
            + "\n\n".join(blocks)
            + "\n\nPropose the validation rules for the columns above (and for whole records where useful). "
              "Return JSON only.")


def prompt_hash(model_id: str, system: str, prompt: str, max_tokens: int, temperature) -> str:
    return hashlib.sha256(json.dumps([model_id, system, prompt, max_tokens, temperature]).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- response parsing
def parse_rules(text: str) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Extract the rule list from the model's answer. Returns (rules, error)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    start, end = t.find("{"), t.rfind("}")
    if start < 0 or end < 0:
        return None, "no JSON object in response"
    try:
        obj = json.loads(t[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f"JSON error: {exc}"
    rules = obj.get("rules") if isinstance(obj, dict) else None
    if not isinstance(rules, list):
        return None, "JSON has no 'rules' list"
    return rules, None


# --------------------------------------------------------------------------- Bedrock client
class BudgetExceeded(RuntimeError):
    pass


class LLMClient:
    def __init__(self, llm_cfg: Dict[str, Any], mock: Optional[bool] = None) -> None:
        self.cfg = dict(llm_cfg)
        prices = self.cfg.get("prices") or {}
        if self.cfg["model_id"] in prices:
            self.cfg["price_input_per_mtok"], self.cfg["price_output_per_mtok"] = prices[self.cfg["model_id"]]
        if "price_input_per_mtok" not in self.cfg:
            raise ValueError(f"No price configured for {self.cfg['model_id']}: add it to llm.prices "
                             "in configs/experiment.yaml so costs can be tracked")
        self.mock = bool(int(os.environ.get("ADQ_LLM_MOCK", "0"))) if mock is None else mock
        self.spent_usd = 0.0          # new spend in this process
        self.calls = 0
        self._runtime = None

    @property
    def cache_root(self) -> Path:
        return results_dir("llm_responses_mock" if self.mock else "llm_responses")

    def cost(self, in_tok: int, out_tok: int) -> float:
        return (in_tok * self.cfg["price_input_per_mtok"] + out_tok * self.cfg["price_output_per_mtok"]) / 1e6

    def _client(self):
        if self._runtime is None:
            import boto3
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if not region:
                raise RuntimeError("AWS_REGION is not set (see scripts/check_bedrock.py)")
            self._runtime = boto3.client("bedrock-runtime", region_name=region)
        return self._runtime

    def _request(self, prompt: str) -> Dict[str, Any]:
        req = {"modelId": self.cfg["model_id"], "system": [{"text": SYSTEM_PROMPT}],
               "messages": [{"role": "user", "content": [{"text": prompt}]}],
               "inferenceConfig": {"maxTokens": int(self.cfg["max_tokens"])}}
        if self.cfg.get("temperature") is not None:
            req["inferenceConfig"]["temperature"] = float(self.cfg["temperature"])
        return req

    def count_input_tokens(self, prompt: str) -> Tuple[int, str]:
        """Exact count via Bedrock CountTokens (free) when available, else a character-based estimate."""
        if not self.mock:
            req = self._request(prompt)
            mid = self.cfg["model_id"]
            base = mid.split(".", 1)[1] if mid.split(".", 1)[0] in ("us", "eu", "apac", "global") else mid
            for model in dict.fromkeys([mid, base]):     # profile ID first, then the base model ID
                try:
                    resp = self._client().count_tokens(
                        modelId=model, input={"converse": {"system": req["system"], "messages": req["messages"]}})
                    return int(resp["inputTokens"]), "Bedrock CountTokens (exact)"
                except Exception:  # noqa: BLE001 - not supported for this model/SDK: try next / fall back
                    continue
        return int(len(SYSTEM_PROMPT + prompt) / 3.2) + 20, "estimate (characters / 3.2)"

    def generate(self, dataset: str, run: int, prompt: str, est_input_tokens: int,
                 cache_dir: Optional[Path] = None, fingerprint: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return the cached response for (dataset, run, prompt, model) or make one paid call.

        ``cache_dir`` overrides the cache location (v2 keeps its own cache per model and fingerprint).
        ``fingerprint`` (code/data/configuration hashes, Section 6.6) is stored with the answer, and a
        cached answer is reused only if its fingerprint is identical."""
        h = prompt_hash(self.cfg["model_id"], SYSTEM_PROMPT, prompt, self.cfg["max_tokens"], self.cfg.get("temperature"))
        path = (Path(cache_dir) if cache_dir is not None else self.cache_root / dataset) / f"run_{run}.json"
        if path.exists():
            cached = json.loads(path.read_text(encoding="utf-8"))
            if cached.get("prompt_hash") == h and (fingerprint is None or cached.get("fingerprint") == fingerprint):
                cached["from_cache"] = True
                return cached

        worst = self.cost(est_input_tokens, int(self.cfg["max_tokens"]))
        if not self.mock and self.spent_usd + worst > self.cfg["max_cost_usd"]:
            raise BudgetExceeded(f"next call could cost up to ${worst:.2f}; spent ${self.spent_usd:.2f} of "
                                 f"the ${self.cfg['max_cost_usd']:.2f} budget (llm.max_cost_usd)")
        if self.mock:
            text = _mock_answer(prompt)
            usage, stop = {"inputTokens": est_input_tokens, "outputTokens": int(len(text) / 3.2)}, "end_turn"
        else:
            resp = self._client().converse(**self._request(prompt))
            text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
            usage, stop = resp["usage"], resp.get("stopReason")
        cost = 0.0 if self.mock else self.cost(usage["inputTokens"], usage["outputTokens"])
        self.spent_usd += cost
        self.calls += 1
        rules, err = parse_rules(text)
        rec = {"dataset": dataset, "run": run, "model_id": self.cfg["model_id"], "mock": self.mock,
               "prompt_hash": h, "created_at": now_iso(), "stop_reason": stop,
               "input_tokens": usage["inputTokens"], "output_tokens": usage["outputTokens"], "cost_usd": cost,
               "prompt": prompt, "response_text": text, "rules": rules, "parse_error": err, "from_cache": False,
               "fingerprint": fingerprint, "temperature": self.cfg.get("temperature"),
               "max_tokens": self.cfg["max_tokens"], "system_prompt_hash": prompt_hash("", SYSTEM_PROMPT, "", 0, None)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        ledger = (path.parent.parent if cache_dir is not None else self.cache_root) / "cost_ledger.csv"
        new = not ledger.exists()
        with open(ledger, "a", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["created_at", "dataset", "run", "model_id", "input_tokens", "output_tokens", "cost_usd"])
            w.writerow([rec["created_at"], dataset, run, rec["model_id"], rec["input_tokens"], rec["output_tokens"],
                        f"{cost:.6f}"])
        return rec


# --------------------------------------------------------------------------- offline stand-in
def _mock_answer(prompt: str) -> str:
    """Deterministic stand-in used ONLY to test the pipeline offline (ADQ_LLM_MOCK=1).
    Produces a mix of sensible, too-strict, redundant and malformed rules."""
    rules: List[Dict[str, Any]] = [{"column": "_records", "rule_type": "uniqueness", "min": None, "max": None,
                                    "allowed_values": None, "pattern": None, "condition": "no duplicate records",
                                    "reason": "mock", "confidence": 0.9}]
    for block in prompt.split("\n\nColumn: ")[1:]:
        name = block.split("\n", 1)[0]
        rules.append({"column": name, "rule_type": "completeness", "min": None, "max": None, "allowed_values": None,
                      "pattern": None, "condition": f"{name} not null", "reason": "mock", "confidence": 0.95})
        rules.append(dict(rules[-1], confidence=0.5))                      # redundant copy
        if "type: numerical" in block:
            q = dict(re.findall(r"(p\d\d) (-?[\d.e+-]+)", block))
            lo, hi = float(q["p01"]), float(q["p99"])
            span = hi - lo
            rules.append({"column": name, "rule_type": "range", "min": lo - 3 * span, "max": hi + 3 * span,
                          "allowed_values": None, "pattern": None, "condition": "plausible range", "reason": "mock",
                          "confidence": 0.8})
            rules.append({"column": name, "rule_type": "range", "min": float(q["p05"]), "max": float(q["p95"]),
                          "allowed_values": None, "pattern": None, "condition": "too strict", "reason": "mock",
                          "confidence": 0.4})
        else:
            common = re.findall(r"(?:: |, )([^,(]+?) \(\d", block.split("most common values")[1].split("\n")[0])
            if len(common) < 15:
                rules.append({"column": name, "rule_type": "category", "min": None, "max": None,
                              "allowed_values": [c.strip() for c in common], "pattern": None,
                              "condition": "known categories", "reason": "mock", "confidence": 0.7})
            rules.append({"column": name, "rule_type": "pattern", "min": None, "max": None, "allowed_values": None,
                          "pattern": r"[^\n]{1,60}", "condition": "short text", "reason": "mock", "confidence": 0.6})
    rules.append({"column": "no_such_column", "rule_type": "range", "min": 0, "max": 1, "confidence": 0.3})
    rules.append({"column": "_records", "rule_type": "sql", "condition": "SELECT 1", "confidence": 0.2})
    return json.dumps({"rules": rules})
