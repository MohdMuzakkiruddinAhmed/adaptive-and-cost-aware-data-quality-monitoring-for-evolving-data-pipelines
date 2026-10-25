"""Download the external dirty/clean benchmarks listed in configs/benchmarks.yaml (Raha/Baran collection).

    python scripts/download_benchmarks.py            # tax and beers (hospital with --all)
"""
import argparse
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.utils import data_dir, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="also optional benchmarks")
    args = ap.parse_args()
    cfg = load_yaml("benchmarks")
    for name, b in cfg["benchmarks"].items():
        if b.get("optional") and not args.all:
            continue
        for kind in ("dirty", "clean"):
            dest = data_dir("external", b["files"][kind])
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                print(f"{dest} exists")
                continue
            url = f"{b['url']}/{kind}.csv"
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, dest)
    print("done")


if __name__ == "__main__":
    main()
