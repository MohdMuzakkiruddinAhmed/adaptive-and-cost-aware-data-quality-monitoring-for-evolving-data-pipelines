"""TensorFlow Data Validation sidecar (TFDV; Breck et al., SysML 2019). Runs in its own Linux / WSL environment
(TFDV publishes no Windows build; setup_wsl.sh), reads the exchange folder and writes RAW tool output:

  tfdv/series/<ds>/seed<s>/<scenario>.json, for T1..T20:
    schema_anomalies   tfdv.validate_statistics(batch, schema) with the schema TFDV infers from the five warm-up
                       batches together (tfdv.infer_schema, default arguments): presence, type, shape and string
                       domains. These are TFDV's data-quality anomalies.
    distance_previous  TFDV's own drift distance per feature against the PREVIOUS batch (its documented drift
                       set-up: consecutive spans): L-infinity for categorical features, approximate Jensen-Shannon
                       divergence for numeric features.
    distance_warmup    the same distances against the warm-up batches together (change since the warm-up, the
                       definition of the drift labels).
  and calibration distances from the warm-up only (no labels):
    loo_warmup         each warm-up batch against the other four together
    consecutive_warmup T2..T5 against the batch before
    loo_warmup_quality share of present values and of values outside the string domain inferred from the other
                       four warm-up batches (calibrates TFDV's presence and domain-mass tolerances)
  Per batch also present_fraction and unexpected_fraction (against the warm-up schema), the quantities behind
  TFDV's presence and string-domain anomalies, so that their tolerances can be calibrated too.

The distances are read from TFDV's anomaly descriptions with the comparator thresholds set to 1e-9, so TFDV
reports every non-zero distance; the decision thresholds are applied afterwards (baselines/external_tools.py),
either TFDV's documented example value or a threshold calibrated on the warm-up like the proposed method's.

    python run_tfdv.py --exchange /mnt/c/.../results/v2/tool_exchange/dev [--force]
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from exchange_io import Progress, output_current, load_manifest, series_out, series_paths, series_units, write_json  # noqa: E402

warnings.filterwarnings("ignore")
MEASURE = 1e-9
_DIST = re.compile(r"between current and previous is ([0-9.eE+-]+)")


def read(path: Path, safe: dict):
    import pandas as pd
    return pd.read_parquet(path).rename(columns=safe)


def stats_of(df):
    import tensorflow_data_validation as tfdv
    return tfdv.generate_statistics_from_dataframe(df)


def measuring_schema(schema, meta):
    """Copy of the schema with drift comparators that report every non-zero distance."""
    import tensorflow_data_validation as tfdv
    m = copy.deepcopy(schema)
    safe = meta["safe_names"]
    for c in meta["categorical"]:
        tfdv.get_feature(m, safe[c]).drift_comparator.infinity_norm.threshold = MEASURE
    for c in meta["numeric"]:
        tfdv.get_feature(m, safe[c]).drift_comparator.jensen_shannon_divergence.threshold = MEASURE
    return m


def _type_name(t) -> str:
    from tensorflow_metadata.proto.v0 import anomalies_pb2
    return anomalies_pb2.AnomalyInfo.Type.Name(t)


def distances(current, previous, mschema, features) -> dict:
    import tensorflow_data_validation as tfdv
    a = tfdv.validate_statistics(current, mschema, previous_statistics=previous)
    out = {f: 0.0 for f in features}
    for f, info in a.anomaly_info.items():
        for r in info.reason:
            if _type_name(r.type).startswith("COMPARATOR_"):
                m = _DIST.search(r.description)
                if m:
                    out[f] = float(m.group(1))
    return out


def schema_anomalies(current, schema) -> list:
    import tensorflow_data_validation as tfdv
    a = tfdv.validate_statistics(current, schema)
    rows = []
    for f, info in a.anomaly_info.items():
        for r in info.reason:
            rows.append({"feature": f, "type": _type_name(r.type), "short": r.short_description,
                         "description": r.description})
    for r in a.dataset_anomaly_info.reason:
        rows.append({"feature": "*", "type": _type_name(r.type), "short": r.short_description,
                     "description": r.description})
    return rows


def present_fraction(stats) -> dict:
    """Share of records with a value, per feature, from TFDV's statistics."""
    out = {}
    for f in stats.datasets[0].features:
        name = f.path.step[0] if f.path.step else f.name
        cs = f.num_stats.common_stats if f.HasField("num_stats") else f.string_stats.common_stats
        n = cs.num_non_missing + cs.num_missing
        out[name] = float(cs.num_non_missing / n) if n else 1.0
    return out


def unexpected_fraction(df, schema, cat_feats) -> dict:
    """Share of non-missing values outside the string domain of the schema (the quantity TFDV's
    ENUM_TYPE_UNEXPECTED_STRING_VALUES anomaly tests against distribution_constraints.min_domain_mass)."""
    import tensorflow_data_validation as tfdv
    out = {}
    for f in cat_feats:
        try:
            dom = set(tfdv.get_domain(schema, f).value)
        except (ValueError, AttributeError):
            continue                                     # no domain inferred (too many distinct values)
        x = df[f].dropna().astype(str)
        out[f] = float((~x.isin(dom)).mean()) if len(x) else 0.0
    return out


def run_unit(exchange: Path, unit: dict, meta: dict) -> dict:
    import pandas as pd
    import tensorflow_data_validation as tfdv
    safe = meta["safe_names"]
    feats = [safe[c] for c in meta["numeric"] + meta["categorical"]]
    frames = [read(p, safe) for p in series_paths(exchange, unit)]
    w = unit["warmup"]
    stats = [stats_of(f) for f in frames]
    warm_stats = stats_of(pd.concat(frames[:w], ignore_index=True))
    schema = tfdv.infer_schema(warm_stats)
    msch = measuring_schema(schema, meta)
    cats = [safe[c] for c in meta["categorical"]]
    loo, loo_quality = [], []
    for j in range(w):
        others = stats_of(pd.concat([frames[i] for i in range(w) if i != j], ignore_index=True))
        loo.append(distances(stats[j], others, msch, feats))
        sch_j = tfdv.infer_schema(others)
        loo_quality.append({"records": len(frames[j]), "present_fraction": present_fraction(stats[j]),
                            "unexpected_fraction": unexpected_fraction(frames[j], sch_j, cats)})
    consec = [distances(stats[j], stats[j - 1], msch, feats) for j in range(1, w)]
    batches = []
    for t, st in enumerate(stats, 1):
        batches.append({"t": t, "schema_anomalies": schema_anomalies(st, schema),
                        "records": len(frames[t - 1]), "present_fraction": present_fraction(st),
                        "unexpected_fraction": unexpected_fraction(frames[t - 1], schema, cats),
                        "distance_previous": distances(st, stats[t - 2], msch, feats) if t > 1 else None,
                        "distance_warmup": distances(st, warm_stats, msch, feats)})
    from google.protobuf import text_format
    return {"unit": unit, "schema": text_format.MessageToString(schema),
            "feature_kind": {**{safe[c]: "numeric" for c in meta["numeric"]},
                             **{safe[c]: "categorical" for c in meta["categorical"]}},
            "columns": {s: o for o, s in safe.items()},
            "loo_warmup": loo, "consecutive_warmup": consec, "loo_warmup_quality": loo_quality, "batches": batches}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exchange", required=True, type=Path)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    import tensorflow_data_validation as tfdv
    man = load_manifest(args.exchange)
    root = args.exchange / "tfdv"
    write_json(root / "manifest.json", {"versions": {"tensorflow_data_validation": tfdv.__version__},
                                        "settings": man.get("tools", {}).get("tfdv", {}),
                                        "export_key": man["export_key"], "measure_threshold": MEASURE})
    todo = [u for u in series_units(man) if args.force or not output_current(series_out(args.exchange, "tfdv", u), u)]
    prog = Progress(len(todo), "tfdv")
    for u in todo:
        write_json(series_out(args.exchange, "tfdv", u), run_unit(args.exchange, u, man["datasets"][u["dataset"]]))
        prog.step(f"{u['dataset']} seed {u['seed']} {u['scenario']}")


if __name__ == "__main__":
    main()
