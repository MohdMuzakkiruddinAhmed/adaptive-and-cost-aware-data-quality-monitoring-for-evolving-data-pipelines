# Established-tool comparators: Amazon Deequ and TensorFlow Data Validation

The monitor is compared with two widely used data-validation tools, run **as the tools themselves** (not
re-implementations) on exactly the same replay batches as every other method:

| Tool | What is used | Experiment | Comparator |
|---|---|---|---|
| Deequ (PyDeequ 1.7, Deequ 2.0 on Spark 3.5) | constraint suggestions from the warm-up, **verified by Deequ itself** on every evaluation batch; the records Deequ reports as violating (row-level results) are its verdict | H1a (`rq1_thresholds`) | `deequ_observed`, `deequ_observed_rows`, `deequ_clean`, `deequ_clean_rows`, `deequ_clean_rows+base` |
| Deequ | batch-level status of the suggested constraints (share of batches with a failed constraint) | H1a | `rq1a_deequ_batch_alarms.csv` |
| Deequ | one anomaly check per batch metric: as documented (`OnlineNormalStrategy`, defaults, cumulative history) and calibrated (warm-up history, `BatchNormalStrategy`, Student-t factor) | H1b (`rq1_classification`) | `deequ_anomaly`, `deequ_anomaly_calibrated`, `deequ_calibrated_checks` |
| TFDV 1.21 | schema inferred from the warm-up + drift comparators (L-infinity / Jensen-Shannon) | H1b | `tfdv_previous`, `tfdv_warmup`, `tfdv_warmup_calibrated` |

Deequ suggestion variants: `observed_default` = the warm-up as it arrives (with its ~1% erroneous records), DEFAULT
rules, i.e. what a practitioner gets; `clean_extended` = the warm-up without its erroneous records, Deequ's EXTENDED
rules without NUMERICAL (`DEFAULT+STRING+COMMON`), i.e. the tool's best case (no other method gets clean data, so this is an optimistic bound). Readings: `failed` =
records violating a constraint whose batch-level check failed; `rows` (any) = records violating any suggested
constraint's row-level predicate.

**Before the first run after an update, run the Deequ self-test** (one minute; it exercises every Deequ call used):

```
SPARK_VERSION=3.5 ~/adq-tools/deequ/bin/python tools/external_validators/run_deequ.py --selftest
```

It must end with `SELFTEST OK`. If it fails, send the output: nothing else needs to be re-run.

How the raw outputs become rules and labels, and why, is documented in `baselines/external_tools.py`.

## Why sidecars

Deequ is a Scala library on Spark (needs a JVM) and TFDV ships no Windows build. Both therefore run in their own
Linux environments (WSL on Windows). The pipeline exports the batches to an exchange folder, the tools write
their raw outputs next to them, and the experiments read those outputs:

```
python -m experiments.v2.tool_exchange --split dev     # 1. export batches (Windows, project environment)
python -m experiments.v2.run_tools --split dev         # 2. run both tools (starts them inside WSL)
python -m experiments.v2.rq1_thresholds --split dev    # 3. experiments pick the outputs up
```

`python -m experiments.v2.run_all` does all of this (steps `export_tools` and `tools`). If a tool is not set up
or fails, the pipeline continues and the experiments skip that tool's comparators with a warning.

Tool outputs are tied to the content of the exported batches (hash): when a code or configuration change alters
the batches, the old outputs are ignored and the tools recompute; when the batches are unchanged, the tools skip
work already done.

## One-time set-up (Windows)

1. PowerShell: `wsl --install -d Ubuntu-24.04`, restart, create the Linux user.
2. Ubuntu terminal:
   `bash /mnt/c/Users/<you>/Desktop/DataQuality/adaptive-data-quality/tools/external_validators/setup_wsl.sh`
   (installs Java 17 and two virtual environments under `~/adq-tools`, then runs a smoke test of each tool).
3. Nothing else: `protocol.yaml` `external_tools.runner: wsl` makes `run_all` start the tools in WSL.
   On Linux set `runner: local`; with `runner: none` the commands are printed for running by hand.

## One-time set-up (macOS, Apple Silicon)

Deequ runs natively; TFDV has no arm64 Linux build, so it runs in an Ubuntu 24.04 `linux/amd64` container
(Docker Desktop, Rosetta).

1. `brew install openjdk@17 python@3.11` and start Docker Desktop.
2. `bash tools/external_validators/setup_macos.sh` (or `... deequ` / `... tfdv` for one tool). It creates
   `~/adq-tools/deequ` (PySpark 3.5.9, PyDeequ 1.7.0 on Python 3.11), runs the Deequ self-test, builds the image
   `adq-tfdv:1.21.0` from `Dockerfile.tfdv` and runs a TFDV smoke test.
3. `protocol.yaml` already has `runner: {deequ: local, tfdv: docker}`, `java_home` and `docker` set for this;
   on Windows set `runner: wsl` instead (see the main README).

Deequ self-test on macOS:
`JAVA_HOME=/opt/homebrew/opt/openjdk@17 ~/adq-tools/deequ/bin/python tools/external_validators/run_deequ.py --selftest`

## Run time and disk

The dev split exports whole runs of the nine H1a scenarios (Deequ verifies their evaluation batches) and of the four
H1b scenarios (anomaly detection, TFDV), plus deduplicated warm-ups (observed and clean). BTS batches dominate:
expect roughly 0.5 GB for dev and 1-2 GB for test. TFDV takes a few seconds per run. Deequ starts one Spark job per
batch: verification of about 2,000 dev batches and two anomaly configurations take a few hours on dev, roughly twice
that on test. Everything is resumable: an interrupted run continues where it stopped.

## Fairness choices (state them in the paper)

* Deequ does its own work end to end: it suggests, verifies and reports the violating records; nothing is
  translated into this project's rule engine. It is run as a practitioner would (observed warm-up, default rules,
  default anomaly strategy) and in its best case (clean warm-up, extended rules, calibrated anomaly factor).
  Deequ suggests no numeric bounds beyond non-negativity, so it finds few outliers: that is the tool's behaviour.
* The best case leaves out Deequ's NUMERICAL suggestion rules: `hasMin`, `hasMax`, `hasMean` and
  `hasStandardDeviation` are suggested as exact equalities with the warm-up value (`_ == v`), which fail on any
  other batch, and Deequ's HasMean rule takes the column maximum as the mean (`suggestions/rules/HasMean.scala`).
  PyDeequ's DEFAULT (used for `observed_default`) already includes the uniqueness rule, unlike Deequ's Scala DEFAULT.
* Deequ and TFDV metrics are mapped to drift (distribution) and degradation (quality) the same way as the
  AVH-style comparator.
* TFDV: `tfdv_previous` is the documented set-up (drift between consecutive spans, threshold 0.01). Its guide gives
  no Jensen-Shannon example value for numeric features; 0.01 is used (an assumption). TFDV's default schema flags
  any missing value in an always-present feature and any unseen string, so with a normal base error rate it labels
  every batch "degradation"; `tfdv_warmup_calibrated` therefore calibrates the drift thresholds and the presence
  and domain-mass tolerances on the warm-up with the same prediction bound as the proposed method.
