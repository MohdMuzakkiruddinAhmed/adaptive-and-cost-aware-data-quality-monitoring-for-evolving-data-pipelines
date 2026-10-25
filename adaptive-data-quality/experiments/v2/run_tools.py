"""Start the Deequ and TFDV sidecars on the exported batches (run_all step "tools").

protocol.yaml external_tools.runner decides how, either one value for both tools or one per tool
({deequ: local, tfdv: docker}):
  wsl     Windows: run inside WSL (distribution external_tools.wsl_distribution); project paths are translated to
          /mnt/<drive>/... automatically
  local   Linux / macOS: run in a login shell (external_tools.java_home, if set, becomes JAVA_HOME for Deequ)
  docker  run in the image external_tools.docker.image; the project folder is mounted at the same path, so all
          paths stay valid (macOS: TFDV has no arm64 Linux build, see tools/external_validators/Dockerfile.tfdv)
  none    print the commands only (run them by hand)
A failing or missing tool is reported and the pipeline continues: the experiments then skip that tool's
comparators with a warning. One-time set-up: tools/external_validators/setup_wsl.sh (Windows / Linux) or
setup_macos.sh (macOS).

    python -m experiments.v2.run_tools --split dev [--tools deequ tfdv] [--force]
"""
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path, PureWindowsPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.utils import protocol_config  # noqa: E402
from experiments.v2.tool_exchange import exchange_dir, load_manifest  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools" / "external_validators"
SCRIPTS = {"deequ": "run_deequ.py", "tfdv": "run_tfdv.py"}


def runner_for(tool: str, cfg: dict) -> str:
    """external_tools.runner: one value for every tool, or a mapping tool -> runner."""
    r = cfg.get("runner", "none")
    return (r.get(tool, "none") if isinstance(r, dict) else r) or "none"


def docker_command(tool: str, inner: list, cfg: dict) -> list:
    """docker run with the project mounted at its host path (paths in the exchange manifest stay valid)."""
    d = cfg.get("docker", {}) or {}
    root = str(REPO_ROOT)
    cmd = ["docker", "run", "--rm", "-v", f"{root}:{root}", "-w", root, "-e", "HOME=/tmp"]
    if d.get("platform"):
        cmd += ["--platform", str(d["platform"])]
    if hasattr(os, "getuid"):
        cmd += ["--user", f"{os.getuid()}:{os.getgid()}"]   # outputs owned by the host user
    return cmd + [str(d.get("image", f"adq-{tool}"))] + inner


def to_wsl(path: Path) -> str:
    """C:\\Users\\x\\y -> /mnt/c/Users/x/y (paths already POSIX are returned unchanged)."""
    s = str(path)
    w = PureWindowsPath(s)
    if not w.drive or ":" not in w.drive:
        return s
    return "/mnt/" + w.drive[0].lower() + "/" + "/".join(w.parts[1:])


def command(tool: str, split: str, force: bool, cfg: dict) -> list:
    runner = runner_for(tool, cfg)
    script, exchange = TOOLS_DIR / SCRIPTS[tool], exchange_dir(split)
    if runner == "docker":
        venv = (cfg.get("docker", {}) or {}).get("venvs", {}).get(tool, f"/opt/adq-tools/{tool}")
        env = ["env", "SPARK_VERSION=3.5"] if tool == "deequ" else []
        inner = env + [f"{venv}/bin/python", str(script), "--exchange", str(exchange)] + (["--force"] if force else [])
        return docker_command(tool, inner, cfg)
    venv = cfg.get("venvs", {}).get(tool, f"~/adq-tools/{tool}")
    if runner == "wsl":
        script_s, exchange_s = to_wsl(script), to_wsl(exchange)
    else:
        script_s, exchange_s = str(script), str(exchange)
    env = "SPARK_VERSION=3.5 " if tool == "deequ" else ""
    if tool == "deequ" and runner == "local" and cfg.get("java_home"):
        env = f"JAVA_HOME={shlex.quote(str(cfg['java_home']))} " + env
    inner = f"{env}{venv}/bin/python {shlex.quote(script_s)} --exchange {shlex.quote(exchange_s)}" + (" --force" if force else "")
    if runner == "wsl":
        dist = cfg.get("wsl_distribution") or ""
        return ["wsl.exe"] + (["-d", dist] if dist else []) + ["-e", "bash", "-lc", inner]
    return ["bash", "-lc", inner]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--tools", nargs="*", default=list(SCRIPTS), choices=list(SCRIPTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    cfg = protocol_config().get("external_tools", {})
    if load_manifest(args.split) is None:
        print("[tools] nothing exported yet: python -m experiments.v2.tool_exchange --split", args.split)
        return 1
    failed = []
    for tool in args.tools:
        cmd = command(tool, args.split, args.force, cfg)
        print(f"[tools] {tool}: {' '.join(shlex.quote(c) for c in cmd)}", flush=True)
        if runner_for(tool, cfg) == "none":
            continue
        try:
            rc = subprocess.run(cmd).returncode
        except FileNotFoundError as e:
            rc = f"not found ({e})"
        if rc != 0:
            failed.append(tool)
            print(f"[tools] {tool} FAILED ({rc}). Its comparators will be skipped. Set-up: "
                  f"tools/external_validators/README.md", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
