"""Independent author of the "manual" domain rules (v2.1).

The manual-rule condition (RQ2 / H2a, H2b manual guard) must be written by someone who has not seen the injected
errors, the novelty values or the evaluation data. The v2.0 draft (configs/manual_rules_draft_v2.0.yaml) was
written by the same assistant that wrote the error generator, which is a bias risk. Until a human expert writes the
rules, they are written by a model of a DIFFERENT family from the rule-generating LLMs, given only what a domain
expert gets from a data dictionary:

  * dataset name, description and source, and each column's name, type and description (configs/datasets.yaml)
  * the instruction to use public reference lists where they exist (state codes, country names, carrier lists)

and NOT: statistics, example values, the injected error types, the novelty values or any record. The author writes
one rule set per dataset (temperature 0, one generation) in the same JSON format as the LLM proposals, so the rules
run through the same code (rules/validator.py normalise). experiments/v2/manual_author.py calls the model and
writes configs/manual_rules.yaml with the author and the prompt hash recorded.
"""
from __future__ import annotations

from typing import Any, Dict

from rules.llm_generator import LLMClient

AUTHOR_SYSTEM_PROMPT = """You are a senior data-quality engineer with domain expertise. You are writing the validation rules for a data pipeline from its DATA DICTIONARY ONLY, before any data arrives. You have not seen any records.

Write the rules a careful domain expert would write: rules that every legitimate record satisfies and that catch values that are impossible or invalid in the domain. Use public reference knowledge where it exists (for example official code lists, the list of countries, the list of US states and territories, published category lists of the data provider). Do not guess statistical limits: a range rule needs a domain reason (a duration cannot be negative, a price must be positive). Prefer an open format rule (pattern) to a closed list when you cannot know the complete list.

Answer with JSON only, no prose, in exactly this form:
{"rules": [ {rule}, ... ]}

Each rule has these fields:
  "column":         the column name, or "_records" for a rule on the whole record
  "rule_type":      one of "completeness", "range", "category", "pattern", "uniqueness", "type"
  "min", "max":     numbers for "range" rules (either may be null for a one-sided range), otherwise null
  "allowed_values": list of strings for "category" rules, otherwise null
  "pattern":        a regular expression the whole value must match, for "pattern" rules, otherwise null
  "condition":      the rule in plain words
  "provenance":     the domain knowledge or reference list the rule comes from

Meaning of the rule types:
  completeness  the value must not be missing
  range         a numeric value must lie in [min, max]
  category      the value must be one of allowed_values (a closed list you can justify from a reference)
  pattern       the text value must fully match the regular expression
  uniqueness    use with column "_records": the same record must not appear twice
  type          the value must be numeric"""


def author_prompt(name: str, cfg: Dict[str, Any]) -> str:
    lines = [f"Dataset: {cfg.get('name', name)}",
             f"Domain: {cfg.get('domain', 'n/a')}",
             f"Source: {cfg.get('source', 'n/a')}",
             f"Description: {cfg.get('description', '')}", "", "Columns to validate:"]
    desc = cfg.get("column_descriptions", {})
    for c in cfg["numeric"]:
        lines.append(f"  {c} (numeric): {desc.get(c, 'n/a')}")
    for c in cfg["categorical"]:
        lines.append(f"  {c} (text / categorical): {desc.get(c, 'n/a')}")
    others = [c for c in cfg.get("key_columns", []) if c not in cfg["numeric"] + cfg["categorical"]]
    if others:
        lines.append(f"Identifying columns (not validated): {', '.join(others)}")
    lines += ["", "Write the validation rules for the columns above (and for whole records where useful). "
                  "Return JSON only."]
    return "\n".join(lines)


class AuthorClient(LLMClient):
    """LLMClient with the rule-author system prompt (cache, budget and ledger unchanged)."""

    def _request(self, prompt: str) -> Dict[str, Any]:
        req = super()._request(prompt)
        req["system"] = [{"text": AUTHOR_SYSTEM_PROMPT}]
        return req

    def generate(self, dataset, run, prompt, est_input_tokens, cache_dir=None, fingerprint=None):
        import rules.llm_generator as G
        saved = G.SYSTEM_PROMPT
        G.SYSTEM_PROMPT = AUTHOR_SYSTEM_PROMPT
        try:
            return super().generate(dataset, run, prompt, est_input_tokens, cache_dir, fingerprint)
        finally:
            G.SYSTEM_PROMPT = saved
