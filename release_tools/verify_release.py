"""Verify source integrity, release metadata and frozen saved evidence without services."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
AUTHORS = [
    "Muzakkiruddin Ahmed Mohammed", "Ahsan Shahid", "Selimhan Dagtas",
    "John Talburt", "Abdelrahman Elfikky",
]

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def run():
    checks = []
    def check(name, passed, **detail):
        checks.append({"name": name, "passed": bool(passed), **detail})
    manifest = ROOT / "MANIFEST.sha256"
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        sha, name = line.split("  ", 1)
        target = (ROOT / name).resolve()
        if not re.fullmatch(r"[a-f0-9]{64}", sha) or not target.is_relative_to(ROOT) or name in expected:
            raise ValueError("Invalid integrity manifest.")
        expected[name] = sha
    missing, mismatched, oversize = [], [], []
    for name, sha in expected.items():
        path = ROOT / name
        if not path.is_file():
            missing.append(name)
        elif digest(path) != sha:
            mismatched.append(name)
        elif path.stat().st_size >= 100 * 1024 * 1024:
            oversize.append(name)
    check("manifest_integrity", not missing and not mismatched, files=len(expected), missing=missing, mismatched=mismatched)
    check("regular_git_file_sizes", not oversize, files_at_or_above_100_MiB=oversize)
    main = (ROOT / "paper" / "latex" / "main.tex").read_text(encoding="utf-8-sig")
    found = re.findall(r"\\author\{([^}]+)\}", main)
    check("manuscript_author_order", found == AUTHORS, authors=found)
    check("third_author_email", r"\author{Selimhan Dagtas}" in main and r"\email{sedagtas@ualr.edu}" in main)
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    families = re.findall(r'family-names: "([^"]+)"', citation)
    check("citation_author_order", families == ["Mohammed", "Shahid", "Dagtas", "Talburt", "Elfikky"])
    check("citation_version", 'version: "0.1.0"' in citation and "cff-version: 1.2.0" in citation)
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    check("mit_license", license_text.startswith("MIT License\n") and "Permission is hereby granted" in license_text)
    check("third_party_sources", (ROOT / "third_party" / "acmart" / "acmart.dtx").is_file()
          and (ROOT / "third_party" / "acmart" / "LPPL-1.3c.txt").is_file()
          and (ROOT / "THIRD_PARTY_NOTICES.md").is_file())
    asset_manifest = json.loads((ROOT / "assets_manifest.json").read_text(encoding="utf-8"))
    check("companion_manifest", asset_manifest["schema_version"] == 1 and len(asset_manifest["files"]) > 0)
    forbidden = []
    secret_patterns = [
        ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
        ("github_token", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}\b")),
        ("github_fine_grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{40,}\b")),
        ("openai_token", re.compile(r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{32,}\b")),
        ("private_key", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ]
    for name in expected:
        path = ROOT / name
        if path.name in {".env", "credentials", "id_rsa", "id_ed25519"} or path.suffix.lower() in {".pem", ".key"}:
            forbidden.append({"path": name, "rule": "sensitive_filename"})
            continue
        if path.suffix.lower() in {".py", ".json", ".csv", ".yaml", ".yml", ".toml", ".md", ".txt", ".cff", ".example"}:
            data = path.read_text(encoding="utf-8", errors="ignore")
            for rule, pattern in secret_patterns:
                if pattern.search(data):
                    forbidden.append({"path": name, "rule": rule})
    check("high_confidence_secret_scan", not forbidden, redacted_findings=forbidden)
    child = subprocess.run([sys.executable, str(ROOT / "release_tools" / "verify_results.py")],
                           cwd=ROOT, capture_output=True, text=True)
    check("saved_metric_verification", child.returncode == 0)
    if child.returncode:
        print(child.stdout)
        print(child.stderr, file=sys.stderr)
    else:
        print(child.stdout.strip())
    report = {"all_checks_passed": all(x["passed"] for x in checks),
              "checks_passed": sum(x["passed"] for x in checks), "checks_total": len(checks), "checks": checks}
    output = ROOT / ".artifacts" / "release-verification.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("all_checks_passed", "checks_passed", "checks_total")}, indent=2))
    if not report["all_checks_passed"]:
        raise SystemExit(1)

if __name__ == "__main__":
    run()
