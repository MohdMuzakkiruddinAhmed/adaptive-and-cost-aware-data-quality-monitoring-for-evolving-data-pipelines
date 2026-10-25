"""Per-value LLM checks for the category guard (H2b policy "recurrence_llm_value"). THIS CAN SPEND MONEY
(about 1-2 US cents per value and model; called from experiments.v2.llm_generate).

Why: in dev run 2 the rule-based LLM guard could reject a new value only if an LLM RULE enumerated the column's
domain (US states, inspection results). For open domains (airports, facility types, countries) the rules are
format patterns, so typos such as "Swizterland" or "Bakkery" passed. Here the LLM is asked directly about each
NEW value that recurrence would admit: is it a legitimate real-world value of this column, or an error?

Candidates: values absent from the warm-up that occur in at least category_min_batches evaluation batches of the
guard scenarios (protocol.yaml experiments.rq2_category_guard) for the split's seeds, in the columns that carry
the H2b novelty. These are exactly the values the recurrence rule would learn. Random defect tokens never recur.
Context given to the LLM: dataset and column description and the column's known values with their shares in the
TRAINING sample (experiments/v2/llm_rules.py, development pool without novelty values). No labels, no batch data.

One call per (model, dataset, column, value); answers are cached with a fingerprint (prompt, model settings, this
module's code) under results/v2/llm_value_checks/<model>/<dataset>/<column>/<value hash>.json and reused for every
run and split. Verdicts: valid / invalid / unsure (unsure or missing -> the guard falls back to recurrence and
reports the value as unresolved).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from common.provenance import config_hash, file_hash
from common.utils import REPO_ROOT, now_iso, v2_results_dir
from experiments.v2.llm_rules import llm_settings, model_slug, training_sample
from rules.llm_generator import BudgetExceeded, LLMClient

VALUE_SYSTEM_PROMPT = (
    "You are a data-quality expert. You judge whether a single categorical value is a legitimate real-world value "
    "of a column, using the column description and the values already known to be in the data. A legitimate value "
    "is one that can correctly appear in this column (for example a real airport code, a real country, a real "
    "facility type), even if it is rare or new. An error is a misspelling, an invalid or non-existent code, a "
    "placeholder, or a value of the wrong kind. Answer with JSON only: "
    '{"verdict": "valid" | "invalid" | "unsure", "reason": "<one sentence>"}')
MAX_TOKENS = 2000
KNOWN_VALUES = 40


class ValueCheckClient(LLMClient):
    """LLMClient (pricing, budget, Bedrock) with the value-check system prompt and a small output budget."""

    def __init__(self, llm_cfg: Dict[str, Any], mock: Optional[bool] = None) -> None:
        cfg = dict(llm_cfg)
        cfg["max_tokens"] = MAX_TOKENS
        super().__init__(cfg, mock)

    def _request(self, prompt: str) -> Dict[str, Any]:
        req = super()._request(prompt)
        req["system"] = [{"text": VALUE_SYSTEM_PROMPT}]
        return req

    def ask(self, prompt: str, path: Path, fingerprint: Dict[str, Any]) -> Dict[str, Any]:
        if path.exists():
            rec = json.loads(path.read_text(encoding="utf-8"))
            if rec.get("fingerprint") == fingerprint:
                rec["from_cache"] = True
                return rec
        est_in = int(len(VALUE_SYSTEM_PROMPT + prompt) / 3.2) + 20
        worst = self.cost(est_in, MAX_TOKENS)
        if not self.mock and self.spent_usd + worst > self.cfg["max_cost_usd"]:
            raise BudgetExceeded(f"value checks: next call could cost up to ${worst:.3f}; spent ${self.spent_usd:.2f}")
        if self.mock:
            v = prompt.split("New value: ", 1)[1].split("\n", 1)[0].strip().strip('"')
            text = json.dumps({"verdict": "invalid" if any(ch.isdigit() for ch in v) or v.isupper() and len(v) > 2 and
                               v[0] in "QXZ" else "valid", "reason": "mock"})
            usage = {"inputTokens": est_in, "outputTokens": 20}
        else:
            resp = self._client().converse(**self._request(prompt))
            text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
            usage = resp["usage"]
        cost = 0.0 if self.mock else self.cost(usage["inputTokens"], usage["outputTokens"])
        self.spent_usd += cost
        self.calls += 1
        rec = {"model_id": self.cfg["model_id"], "mock": self.mock, "created_at": now_iso(), "prompt": prompt,
               "response_text": text, "verdict": parse_verdict(text), "input_tokens": usage["inputTokens"],
               "output_tokens": usage["outputTokens"], "cost_usd": cost, "fingerprint": fingerprint,
               "from_cache": False}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        return rec


def parse_verdict(text: str) -> str:
    t = text.strip()
    s, e = t.find("{"), t.rfind("}")
    if s >= 0 and e > s:
        try:
            v = str(json.loads(t[s:e + 1]).get("verdict", "")).strip().lower()
            if v in ("valid", "invalid", "unsure"):
                return v
        except (ValueError, AttributeError):
            pass
    return "unsure"


def value_prompt(cfg: Dict[str, Any], column: str, value: str, known: Dict[str, float]) -> str:
    desc = cfg.get("column_descriptions", {}).get(column) or ""
    known_s = "; ".join(f"{k} ({v:.1%})" for k, v in list(known.items())[:KNOWN_VALUES])
    return (f"Dataset: {cfg['name']}\nDescription: {cfg.get('description', '')}\n"
            f"Column: {column}" + (f" ({desc})" if desc else "") + "\n"
            f"Most common known values with their share of records: {known_s}\n"
            f"New value: \"{value}\"\n"
            "Is the new value a legitimate value of this column? Return JSON only.")


def value_fingerprint(model_id: str, prompt: str, exp: Dict[str, Any]) -> Dict[str, Any]:
    llm = llm_settings(exp, model_id)
    return {"prompt": hashlib.sha256((VALUE_SYSTEM_PROMPT + prompt).encode()).hexdigest()[:16],
            "llm_settings": config_hash({"model_id": model_id, "max_tokens": MAX_TOKENS,
                                         "temperature": llm.get("temperature")}),
            "code": file_hash(Path(__file__))}


def value_path(model_id: str, dataset: str, column: str, value: str) -> Path:
    h = hashlib.sha256(value.encode()).hexdigest()[:16]
    return v2_results_dir("llm_value_checks", model_slug(model_id), dataset, column.replace(" ", "_"), f"{h}.json")


def guard_columns(protocol: Dict[str, Any], dataset: str) -> List[str]:
    return sorted({nv["column"] for nv in protocol.get("dataset_scenarios", {}).get(dataset, {}).get("novelty", [])})


def candidates(src, dataset: str, split: str, seeds=None) -> Dict[str, List[str]]:
    """New values the recurrence rule would admit in the guard scenarios (see module docstring)."""
    pr, exp = src.protocol, src.exp
    cols = guard_columns(pr, dataset)
    seeds = seeds or pr["splits"]["seeds"][split]
    min_b = exp["adaptive"]["category_min_batches"]
    out: Dict[str, set] = {c: set() for c in cols}
    for seed in seeds:
        for scen in pr["experiments"].get("rq2_category_guard", []):
            rep = src.replay(dataset, split, seed, scen)
            sets = [{c: set(rep.batch(t).df[c].dropna().astype(str)) for c in cols} for t in range(1, rep.n_batches + 1)]
            warm = {c: set().union(*[s[c] for s in sets[:rep.warm]]) for c in cols}
            for c in cols:
                counts: Dict[str, int] = {}
                for s in sets[rep.warm:]:
                    for v in s[c] - warm[c]:
                        counts[v] = counts.get(v, 0) + 1
                out[c] |= {v for v, n in counts.items() if n >= min_b}
    return {c: sorted(v) for c, v in out.items()}


def known_values(src, dataset: str) -> Dict[str, Dict[str, float]]:
    sample = training_sample(src, dataset)
    return {c: sample[c].dropna().astype(str).value_counts(normalize=True).to_dict()
            for c in guard_columns(src.protocol, dataset)}


def run_value_checks(src, model_key: str, model_id: str, split: str = "dev", mock: bool = False,
                     dry_run: bool = False) -> Dict[str, Any]:
    """Ask the model about every candidate of every dataset with novelty (cached answers cost nothing).
    ``dry_run``: only count the uncached candidates and print the worst-case cost."""
    llm = llm_settings(src.exp, model_id)
    llm["max_cost_usd"] = float(src.protocol["rq2"].get("value_check_max_cost_usd", llm["max_cost_usd"]))
    client = ValueCheckClient(llm, mock=mock)
    n_new = 0
    if dry_run:
        todo, worst = 0, 0.0
        for ds in src.datasets:
            if not guard_columns(src.protocol, ds):
                continue
            cfg, known = src.datasets[ds], known_values(src, ds)
            for col, values in candidates(src, ds, split).items():
                for v in values:
                    prompt = value_prompt(cfg, col, v, known[col])
                    path = value_path(model_id, ds, col, v)
                    cached = path.exists() and json.loads(path.read_text(encoding="utf-8")).get("fingerprint") == \
                        value_fingerprint(model_id, prompt, src.exp)
                    if not cached:
                        todo += 1
                        worst += client.cost(int(len(VALUE_SYSTEM_PROMPT + prompt) / 3.2) + 20, MAX_TOKENS)
        typical = client.cost(700, 300) * todo
        print(f"[value checks {model_key}] {todo} uncached value(s): typical ${typical:.2f}, worst case ${worst:.2f} "
              f"(cap ${llm['max_cost_usd']:.2f})")
        return {"todo": todo, "worst_usd": worst, "typical_usd": typical}
    for ds in src.datasets:
        if not guard_columns(src.protocol, ds):
            continue
        cfg, known = src.datasets[ds], known_values(src, ds)
        cand = candidates(src, ds, split)
        for col, values in cand.items():
            for v in values:
                prompt = value_prompt(cfg, col, v, known[col])
                fp = value_fingerprint(model_id, prompt, src.exp)
                try:
                    rec = client.ask(prompt, value_path(model_id, ds, col, v) if not mock else
                                     value_path(model_id + "_mock", ds, col, v), fp)
                except BudgetExceeded as exc:
                    print(f"[value checks {model_key}] STOPPED: {exc}")
                    return {"spent": client.spent_usd, "calls": client.calls}
                n_new += not rec["from_cache"]
        print(f"[value checks {model_key}] {ds}: {sum(len(v) for v in cand.values())} candidate value(s)", flush=True)
    print(f"[value checks {model_key}] new spend ${client.spent_usd:.4f} over {client.calls} call(s)")
    return {"spent": client.spent_usd, "calls": client.calls}


def load_verdicts(src, dataset: str, model_id: str, mock: bool = False) -> Dict[str, Dict[str, str]]:
    """{column: {value: verdict}} for the cached answers whose fingerprint matches the current prompt."""
    known = known_values(src, dataset)
    out: Dict[str, Dict[str, str]] = {}
    mid = model_id + ("_mock" if mock else "")
    for col in guard_columns(src.protocol, dataset):
        d = value_path(mid, dataset, col, "x").parent
        out[col] = {}
        for p in d.glob("*.json") if d.exists() else []:
            rec = json.loads(p.read_text(encoding="utf-8"))
            v = rec["prompt"].split("New value: ", 1)[1].split("\n", 1)[0].strip()[1:-1]
            if rec.get("fingerprint") == value_fingerprint(model_id, value_prompt(src.datasets[dataset], col, v,
                                                                                   known[col]), src.exp):
                out[col][v] = rec["verdict"]
    return out
