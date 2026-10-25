"""LLM rule proposals for the v2 experiments: training-only prompts and fingerprinted caches.

Leakage control (manuscript Sections 5.2, 6.3, 8.3; supervisor: "restrict LLM examples to training/warm-up
data"): prompt statistics and example values come from a TRAINING SAMPLE drawn from the development pool
only, with the H2b novelty values removed. The test pool, every evaluation batch and every novelty value
are therefore invisible to the LLM. The same generations are used for dev and test runs.

Cache key (Section 6.6): a response is reused only if its fingerprint matches: hash of the training sample,
prompt hash, LLM settings (model, max tokens, temperature, system prompt) and the prompt-building code.
Responses live in results/v2/llm_responses/<model>/<dataset>/<fingerprint>/run_<g>.json, so a changed
prompt or data never silently reuses an old answer; old answers stay on disk for provenance.

Admission (validation) of the proposals is done per RUN on that run's own warm-up batches T1..T5, by the
experiment runners, with either redundancy mode.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from common.provenance import config_hash, file_hash, sha256_bytes
from common.utils import REPO_ROOT, v2_results_dir
from ingestion.splits import exclude_values
from profiling.profiler import profile_batch
from rules.llm_generator import SYSTEM_PROMPT, build_prompt, prompt_hash

STATS_NOTE = "Statistics below come from a training sample of earlier batches."


def model_slug(model_id: str) -> str:
    return model_id.replace(":", "_").replace("/", "_")


def holdout_values(protocol: Dict[str, Any], dataset: str) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for nv in protocol.get("dataset_scenarios", {}).get(dataset, {}).get("novelty", []):
        out.setdefault(nv["column"], []).extend(nv["valid"])
    return out


def training_sample(src, dataset: str) -> pd.DataFrame:
    """Development-pool records without novelty values; one batch worth, fixed by the split seed."""
    cfg, pr = src.datasets[dataset], src.protocol
    pool = exclude_values(src.pools(dataset)["dev"], holdout_values(pr, dataset))
    n = min(cfg["batch_size"], len(pool))
    rng = np.random.default_rng([pr["splits"]["split_seed"], 77])
    return pool.iloc[np.sort(rng.choice(len(pool), size=n, replace=False))].reset_index(drop=True)


def llm_settings(exp: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    llm = copy.deepcopy(exp["llm"])
    llm["model_id"] = model_id
    return llm


def prompt_and_fingerprint(src, dataset: str, model_id: str) -> Dict[str, Any]:
    cfg, exp = src.datasets[dataset], src.exp
    sample = training_sample(src, dataset)
    profile = profile_batch(sample, cfg["numeric"], cfg["categorical"], batch="training")
    idx = list(src.datasets).index(dataset)
    prompt = build_prompt(dataset, cfg, profile, sample, exp["llm"]["sample_values"], seed=exp["seed"] + idx,
                          stats_source=STATS_NOTE)
    llm = llm_settings(exp, model_id)
    fp = {"training_sample": sha256_bytes(pd.util.hash_pandas_object(sample, index=False).values.tobytes()),
          "prompt": prompt_hash(model_id, SYSTEM_PROMPT, prompt, llm["max_tokens"], llm.get("temperature")),
          "llm_settings": config_hash({k: llm.get(k) for k in ("model_id", "max_tokens", "temperature")}),
          "generator_code": file_hash(REPO_ROOT / "rules" / "llm_generator.py")}
    fp["key"] = config_hash(fp, 12)
    return {"prompt": prompt, "fingerprint": fp, "sample": sample, "profile": profile}


def cache_dir(model_id: str, dataset: str, fp_key: str) -> Path:
    return v2_results_dir("llm_responses", model_slug(model_id), dataset, fp_key)


def load_generations(src, dataset: str, model_id: str, max_generations: Optional[int] = None,
                     mock: bool = False) -> Dict[int, Dict[str, Any]]:
    """{generation: response record} for responses whose fingerprint matches the current prompt/data/code.
    ``mock=True`` reads the offline stand-in answers (pipeline tests only)."""
    pf = prompt_and_fingerprint(src, dataset, model_id)
    d = cache_dir(model_id, dataset, pf["fingerprint"]["key"] + ("_mock" if mock else ""))
    out = {}
    for p in sorted(d.glob("run_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("fingerprint") == pf["fingerprint"] and rec.get("rules") is not None:
            out[int(rec["run"])] = rec
    if max_generations:
        out = {g: r for g, r in out.items() if g <= max_generations}
    return out


def models(protocol: Dict[str, Any], keys: Optional[List[str]] = None) -> Dict[str, str]:
    m = protocol["llm_models"]
    keys = keys or protocol["rq2"]["models"]
    return {k: m[k] for k in keys}


# ----------------------------------------------------------------------------- external benchmarks
def benchmark_training_sample(name: str, bench: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any]):
    """The rows of the DIRTY benchmark table reserved for prompts (configs/benchmarks.yaml prompt_pool, chosen by the
    split seed). They never occur in a replay batch (ingestion/benchmarks.py), so the LLM sees no evaluated record,
    and it never sees the clean table or any label."""
    from ingestion.benchmarks import BenchmarkReplay
    seed = protocol["splits"]["split_seed"]
    rep = BenchmarkReplay(name, bench, exp, seed, split_seed=seed)
    sample = rep.prompt_rows.reset_index(drop=True)
    if not len(sample):
        raise ValueError(f"{name}: configs/benchmarks.yaml has no prompt_pool")
    cfg = dict(rep.cfg)
    monitored = cfg["numeric"] + cfg["categorical"]
    cfg["key_columns"] = [c for c in sample.columns if c not in monitored]
    cfg["context_columns"] = [c for c in rep.context if c in sample.columns]
    return cfg, sample


def benchmark_prompt_and_fingerprint(name: str, bench: Dict[str, Any], exp: Dict[str, Any], protocol: Dict[str, Any],
                                     model_id: str) -> Dict[str, Any]:
    cfg, sample = benchmark_training_sample(name, bench, exp, protocol)
    profile = profile_batch(sample, cfg["numeric"], cfg["categorical"], batch="training")
    try:
        prompt = build_prompt(name, cfg, profile, sample, exp["llm"]["sample_values"], seed=exp["seed"] + 100,
                              stats_source=STATS_NOTE)
    except ValueError:
        # a "numeric" column holds values that are not plain numbers (Beers: "12.0 ounce"): the shared prompt builder
        # prints numeric examples as numbers, so it gets the parsable values as examples and the prompt states the
        # other formats explicitly (the LLM must see them: they are what a type or pattern rule would catch)
        prompt = _prompt_with_text_values(name, cfg, profile, sample, exp)
    llm = llm_settings(exp, model_id)
    fp = {"training_sample": sha256_bytes(pd.util.hash_pandas_object(sample, index=False).values.tobytes()),
          "prompt": prompt_hash(model_id, SYSTEM_PROMPT, prompt, llm["max_tokens"], llm.get("temperature")),
          "llm_settings": config_hash({k: llm.get(k) for k in ("model_id", "max_tokens", "temperature")}),
          "generator_code": file_hash(REPO_ROOT / "rules" / "llm_generator.py")}
    fp["key"] = config_hash(fp, 12)
    return {"prompt": prompt, "fingerprint": fp, "sample": sample, "profile": profile, "cfg": cfg}


def _prompt_with_text_values(name, cfg, profile, sample, exp) -> str:
    clean = sample.copy()
    notes = []
    rng = np.random.default_rng(exp["seed"] + 101)
    for c in cfg["numeric"]:
        x = sample[c]
        num = pd.to_numeric(x, errors="coerce")
        bad = x.notna() & num.isna()
        if bad.any():
            clean[c] = x.where(~bad)
            vals = x[bad].astype(str).to_numpy()
            pick = vals[rng.choice(len(vals), size=min(exp["llm"]["sample_values"], len(vals)), replace=False)]
            notes.append(f"Column {c}: {bad.mean():.1%} of the non-missing values are not plain numbers, e.g. "
                         + ", ".join(f'"{v}"' for v in pick))
    prompt = build_prompt(name, cfg, profile, clean, exp["llm"]["sample_values"], seed=exp["seed"] + 100,
                          stats_source=STATS_NOTE)
    marker = "\n\nPropose the validation rules"
    return prompt.replace(marker, "\n\nValues of numeric columns that are not plain numbers:\n" + "\n".join(notes)
                          + marker, 1)


def load_benchmark_generations(name: str, bench, exp, protocol, model_id: str,
                               max_generations: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
    pf = benchmark_prompt_and_fingerprint(name, bench, exp, protocol, model_id)
    d = cache_dir(model_id, name, pf["fingerprint"]["key"])
    out = {}
    for p in sorted(d.glob("run_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("fingerprint") == pf["fingerprint"] and rec.get("rules") is not None:
            out[int(rec["run"])] = rec
    if max_generations:
        out = {g: r for g, r in out.items() if g <= max_generations}
    return out


# ----------------------------------------------------------------------------- cross-column proposals (v2.1)
def benchmark_xcol_prompt_and_fingerprint(name: str, bench: Dict[str, Any], exp: Dict[str, Any],
                                          protocol: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    from rules.llm_crosscolumn import XCOL_SYSTEM_PROMPT, build_xcol_prompt
    cfg, sample = benchmark_training_sample(name, bench, exp, protocol)
    columns = cfg["numeric"] + cfg["categorical"] + cfg["context_columns"]
    prompt = build_xcol_prompt(name, cfg, sample, columns, int(exp["llm"].get("xcol_examples", 15)),
                               seed=exp["seed"] + 200)
    llm = llm_settings(exp, model_id)
    fp = {"training_sample": sha256_bytes(pd.util.hash_pandas_object(sample, index=False).values.tobytes()),
          "prompt": prompt_hash(model_id, XCOL_SYSTEM_PROMPT, prompt, llm["max_tokens"], llm.get("temperature")),
          "llm_settings": config_hash({k: llm.get(k) for k in ("model_id", "max_tokens", "temperature")}),
          "generator_code": file_hash(REPO_ROOT / "rules" / "llm_crosscolumn.py")}
    fp["key"] = config_hash(fp, 12)
    return {"prompt": prompt, "fingerprint": fp, "sample": sample, "cfg": cfg, "columns": columns}


def xcol_cache_dir(model_id: str, name: str, fp_key: str) -> Path:
    return v2_results_dir("llm_responses", model_slug(model_id), f"{name}__xcol", fp_key)


def load_benchmark_xcol_generations(name: str, bench, exp, protocol, model_id: str,
                                    max_generations: Optional[int] = None) -> Dict[int, Dict[str, Any]]:
    pf = benchmark_xcol_prompt_and_fingerprint(name, bench, exp, protocol, model_id)
    out = {}
    for p in sorted(xcol_cache_dir(model_id, name, pf["fingerprint"]["key"]).glob("run_*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("fingerprint") == pf["fingerprint"] and rec.get("rules") is not None:
            out[int(rec["run"])] = rec
    if max_generations:
        out = {g: r for g, r in out.items() if g <= max_generations}
    return out
