"""Write the "manual" domain rules with an independent author model (AWS Bedrock). THIS SCRIPT CAN SPEND MONEY
(a few cents: one short call per dataset).

    python scripts/check_bedrock.py --probe                    # is protocol.yaml manual_author.model_id callable?
    python -m experiments.v2.manual_author --dry-run           # prompts only, no call
    python -m experiments.v2.manual_author                     # call, then write configs/manual_rules.yaml

Why: the v2.0 manual rules were drafted by the assistant that also wrote the error generator (bias risk; the file
said so). rules/llm_author.py explains what the author sees (the data dictionary only) and what it does not (any
record, statistic, injected error type or novelty value). The author is a model of a different family from the
two rule-generating LLMs (protocol.yaml manual_author.model_id).

Output
  results/v2/manual_author/<model>/<dataset>/run_1.json   raw answer, prompt, cost (fingerprinted cache)
  configs/manual_rules.yaml                               the rules that pass the syntax check (rules/validator.py
                                                          normalise), with author, date and prompt hashes; rules
                                                          that fail it are listed in the file's "rejected" section
  The previous file is kept as configs/manual_rules_draft_v2.0.yaml (first run only).
A human expert can later replace or edit configs/manual_rules.yaml; record the change in its "author" field.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.provenance import config_hash, file_hash  # noqa: E402
from common.utils import REPO_ROOT, datasets_config, experiment_config, protocol_config, v2_results_dir  # noqa: E402
from experiments.v2.llm_rules import llm_settings, model_slug  # noqa: E402
from rules.llm_author import AUTHOR_SYSTEM_PROMPT, AuthorClient, author_prompt  # noqa: E402
from rules.llm_generator import BudgetExceeded, prompt_hash  # noqa: E402
from rules.validator import normalise  # noqa: E402

CONFIG = REPO_ROOT / "configs" / "manual_rules.yaml"
DRAFT = REPO_ROOT / "configs" / "manual_rules_draft_v2.0.yaml"
FIELDS = ("column", "rule_type", "min", "max", "allowed_values", "pattern", "condition", "provenance")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mock", action="store_true", help="offline stand-in (pipeline test; does not write the config)")
    args = ap.parse_args(argv)
    datasets, exp, pr = datasets_config(), experiment_config(), protocol_config()
    ma = pr["manual_author"]
    settings = llm_settings(exp, ma["model_id"])
    settings["temperature"] = ma.get("temperature", 0.0)
    settings["max_tokens"] = int(ma.get("max_tokens", settings["max_tokens"]))
    client = AuthorClient(settings, mock=True if args.mock else None)
    out_rules, rejected, meta = {}, {}, {}
    for ds, cfg in datasets.items():
        prompt = author_prompt(ds, cfg)
        fp = {"prompt": prompt_hash(ma["model_id"], AUTHOR_SYSTEM_PROMPT, prompt, settings["max_tokens"],
                                    settings.get("temperature")),
              "author_code": file_hash(REPO_ROOT / "rules" / "llm_author.py")}
        fp["key"] = config_hash(fp, 12)
        d = v2_results_dir("manual_author", model_slug(ma["model_id"]), ds, fp["key"] + ("_mock" if client.mock else ""))
        pdir = v2_results_dir("manual_author", model_slug(ma["model_id"]))
        pdir.mkdir(parents=True, exist_ok=True)
        (pdir / f"{ds}_prompt.txt").write_text(AUTHOR_SYSTEM_PROMPT + "\n\n---\n\n" + prompt, encoding="utf-8")
        if args.dry_run:
            print(f"[manual_author] {ds}: prompt written to {pdir / (ds + '_prompt.txt')} ({len(prompt)} characters)")
            continue
        n_in, _ = client.count_input_tokens(prompt)
        try:
            rec = client.generate(ds, 1, prompt, n_in, cache_dir=d, fingerprint=fp)
        except BudgetExceeded as exc:
            sys.exit(f"STOPPED: {exc}")
        cols = ["record_id"] + list(cfg.get("key_columns", [])) + cfg["numeric"] + cfg["categorical"]
        keep, bad = [], []
        for r in rec["rules"] or []:
            rule, err = normalise(r, cfg, cols)
            item = {k: r.get(k) for k in FIELDS if r.get(k) is not None}
            (bad if err else keep).append(item if not err else dict(item, error=err))
        out_rules[ds], rejected[ds] = keep, bad
        meta[ds] = {"prompt_hash": fp["prompt"], "cost_usd": rec["cost_usd"], "from_cache": rec["from_cache"],
                    "parse_error": rec.get("parse_error")}
        print(f"[manual_author] {ds}: {len(keep)} rules, {len(bad)} rejected by the syntax check "
              f"({'cache' if rec['from_cache'] else '$%.4f' % rec['cost_usd']})")
    if args.dry_run or args.mock:
        return
    if CONFIG.exists() and not DRAFT.exists():
        shutil.copy(CONFIG, DRAFT)
    old = yaml.safe_load(CONFIG.read_text(encoding="utf-8")) if CONFIG.exists() else {}
    doc = {"author": f"{ma['model_id']} (independent author model, data dictionary only; "
                     f"{dt.date.today().isoformat()}; rules/llm_author.py)",
           "validate_manual": bool(old.get("validate_manual", False)),
           "author_details": {"model_id": ma["model_id"], "temperature": settings.get("temperature"),
                              "system_prompt_hash": prompt_hash("", AUTHOR_SYSTEM_PROMPT, "", 0, None), "datasets": meta},
           "rules": out_rules, "rejected": rejected}
    header = ("# Manual domain rules (RQ2 / H2a manual condition, H2b manual category guard).\n"
              "# WRITTEN BY experiments/v2/manual_author.py: an independent author model saw only the data dictionary\n"
              "# (rules/llm_author.py). Edit by hand only to replace them with a human expert's rules, and record that in\n"
              "# 'author'. The v2.0 draft is kept in configs/manual_rules_draft_v2.0.yaml.\n")
    CONFIG.write_text(header + yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=110), encoding="utf-8")
    print(f"[manual_author] wrote {CONFIG}")


if __name__ == "__main__":
    main()
