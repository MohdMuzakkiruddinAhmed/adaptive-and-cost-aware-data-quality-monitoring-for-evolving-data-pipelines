"""Download the three study datasets into data/original/ (or --dest).

    python scripts/download_data.py                 # all three
    python scripts/download_data.py --only retail   # one dataset

Sources
  BTS On-Time Performance, Jan-Jun 2024 (PREZIP monthly files, ~28 MB each, zipped)
  UCI Online Retail II (id 502), CC BY 4.0 (~45 MB)
  City of Chicago Food Inspections, 4ijn-s7e5 (~350 MB CSV, full current snapshot)

The Chicago portal serves the live dataset, so record the download date: the
snapshot changes as new inspections are added.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

BTS_URL = ("https://transtats.bts.gov/PREZIP/"
           "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip")
UCI_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
CHI_URL = "https://data.cityofchicago.org/api/views/4ijn-s7e5/rows.csv?accessType=DOWNLOAD"


def fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  exists, skipped: {dest.name}")
        return
    print(f"  downloading {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh, length=1 << 20)
    tmp.replace(dest)
    print(f"  saved {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(Path(__file__).resolve().parents[1] / "data" / "original"))
    ap.add_argument("--only", choices=["bts", "retail", "chicago"], nargs="*")
    ap.add_argument("--bts-year", type=int, default=2024)
    ap.add_argument("--bts-months", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6])
    args = ap.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    only = set(args.only or ["bts", "retail", "chicago"])
    manifest = {"downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "files": []}

    if "bts" in only:
        print("BTS On-Time Performance")
        for m in args.bts_months:
            url = BTS_URL.format(year=args.bts_year, month=m)
            f = dest / url.rsplit("/", 1)[1]
            fetch(url, f)
            manifest["files"].append({"dataset": "bts_airline", "url": url, "file": f.name})
    if "retail" in only:
        print("UCI Online Retail II")
        z = dest / "online_retail_ii.zip"
        fetch(UCI_URL, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        manifest["files"].append({"dataset": "online_retail", "url": UCI_URL, "file": "online_retail_II.xlsx"})
    if "chicago" in only:
        print("Chicago Food Inspections")
        f = dest / "Food_Inspections.csv"
        fetch(CHI_URL, f)
        manifest["files"].append({"dataset": "chicago_food", "url": CHI_URL, "file": f.name})

    with open(dest / "download_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"Done. Manifest: {dest / 'download_manifest.json'}")


if __name__ == "__main__":
    main()
