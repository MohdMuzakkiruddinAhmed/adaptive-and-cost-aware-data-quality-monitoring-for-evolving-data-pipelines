"""Compatibility entry point for the repository's reporting-only correction."""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "release_tools" / "recompute_reporting_corrections.py"
if not SCRIPT.is_file():
    raise SystemExit(
        "Run this helper from the complete research release. The canonical command is "
        "python release_tools/recompute_reporting_corrections.py from its repository root.")
runpy.run_path(str(SCRIPT), run_name="__main__")
