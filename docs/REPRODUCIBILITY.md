# Reproducing the research artifact

This release supports three levels of verification: checking archived evidence, testing the implementation on synthetic data, and regenerating the original experiments. They answer different questions.

Software version **0.1.0** and confirmatory **protocol v2.2** identify different things. The frozen pipeline fingerprint is **fd749de031ca4f54**.

## 1. Verify the released evidence

From the repository root, with Python 3.12 or newer:

    python release_tools/verify_release.py

This standard-library check does not require research data, AWS credentials, Java, Docker, or scientific Python packages. It checks recorded evidence and provenance. Passing is an artifact-consistency result, not a fresh execution of the scientific experiments.

The source archive includes code, configurations, saved summaries and per-run results, cached model responses, manuscript source, and the paper PDF. Large source/cleaned data and Deequ/TFDV exchange outputs are distributed in the separate research-assets archive.

## 2. Run the synthetic test suite

Create an isolated environment from the repository root:

    python -m venv .venv

Activate it:

    # macOS / Linux
    source .venv/bin/activate

    # Windows PowerShell
    .venv\Scripts\Activate.ps1

Install the portable dependencies and test:

    python -m pip install --upgrade pip
    python -m pip install -e "./adaptive-data-quality[dev]"
    cd adaptive-data-quality
    python -m pytest -q -p no:cacheprovider

Tests use small synthetic tables, temporary workspaces and mock model responses. They do not require the research-assets archive or contact an LLM service. The real TFDV test is skipped unless ADQ_TFDV_PYTHON explicitly identifies a TFDV environment.

If a restricted Windows environment has an inaccessible default temporary directory, set process-local paths before starting Python:

    New-Item -ItemType Directory -Force .test-temp | Out-Null
    $env:TEMP = (Resolve-Path .test-temp).Path
    $env:TMP = $env:TEMP
    python -m pytest -q -p no:cacheprovider --basetemp .test-temp/pytest

This changes temporary-file placement, not the monitor.

## 3. Restore companion research assets

Obtain **adaptive-dq-research-assets-v0.1.0.zip** from the matching release. From the repository root:

    python release_tools/restore_assets.py /path/to/adaptive-dq-research-assets-v0.1.0.zip

The helper restores data under adaptive-data-quality/data and tool outputs under adaptive-data-quality/results/v2/tool_exchange. Review DATA_SOURCES.md and THIRD_PARTY_NOTICES.md for original licensing terms. MIT does not replace those terms.

Original download scripts remain available, but retrieve live upstream data. Chicago's dataset evolves; a new download need not reproduce the archived extract or hashes.

## 4. Rebuild confirmatory summaries

With the matching assets restored, run from adaptive-data-quality:

    python -m experiments.v2.run_all --split test --jobs 4 --steps materialize rq1_thresholds rq1_classification rq2_coverage rq2_category_guard validator_stress rq3_scheduling ablation external temporal_chicago

This explicit list omits export_tools and tools. Saved Deequ/TFDV results come from the companion archive; these commands do not launch the sidecars.

Cached unit results are reused only when code, configuration and payload keys match. Preserve frozen pipeline files and configurations byte-for-byte. A mismatch can trigger substantial recomputation instead of a quick summary rebuild. Data preparation and chronological replay still require the restored data.

The frozen configuration contains platform-specific sidecar settings. Do not edit them merely to inspect cached results. Use a separate copy with documented settings for a new environment or run.

## 5. Deliberate full recomputation

From adaptive-data-quality:

    python -m experiments.v2.run_all --split test --jobs 4 --runtime --force

The recorded confirmatory run took about 113 minutes on the study machine; this is not a duration promise for other hardware. Deequ and TFDV need their own environments. See adaptive-data-quality/tools/external_validators/README.md.

This default command does not create new model generations. The optional flag --with-llm-generation, direct non-dry-run generation commands, or changed generator inputs can contact a paid service. Credentials are unnecessary for evidence verification and synthetic tests.

## Environments and reporting

The retained requirements-lock.txt records the study's Python 3.14 environment. It is provenance, not a promise that every pin is available on every platform.

The portable installation uses Python 3.12 or newer. Separate validation used Python 3.12.10, pandas 2.3.3, NumPy 2.4.3 and pytest 9.0.2. Portable test success is not bit-for-bit reproduction of the original environment.

Reporting corrections remain separate from original measurements: runtime totals are divided by fifteen for milliseconds per batch; secondary RQ1a Holm adjustments use declared comparison families; Tax recall denominators are explicit. See paper/latex/reporting_corrections and EXPERIMENTS.md. Primary H1a decisions remain unchanged.



## Regenerate the separate reporting correction

From the repository root, with standard-library Python only:

    python release_tools/recompute_reporting_corrections.py

The command reads the archived paired-comparison CSV under adaptive-data-quality/results/v2/rq1_thresholds/test. It writes a fresh corrected CSV and verification report under .artifacts/reporting-corrections and checks exact agreement with the supplied corrected CSV. It does not overwrite archived research results or change primary H1a decisions. No research-assets archive, external service or additional Python package is needed.
