"""Load the three raw datasets from data/original (Section 4)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def find_files(root: Path, patterns: List[str]) -> List[Path]:
    """Files matching the first pattern that matches anything, in natural sort order."""
    key = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(p))]
    for pat in patterns:
        hits = sorted({p for p in Path(root).glob(pat) if p.is_file()}, key=key)
        if hits:
            return hits
    return []


def _read_csv_or_zip(path: Path, **kw) -> pd.DataFrame:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            member = next(m for m in zf.namelist() if m.lower().endswith(".csv"))
            with zf.open(member) as fh:
                return pd.read_csv(fh, **kw)
    return pd.read_csv(path, **kw)


def load_bts(files: List[Path], cfg: Dict[str, Any]) -> pd.DataFrame:
    frames = [_read_csv_or_zip(f, usecols=cfg["load_columns"], low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["Flight_Number_Reporting_Airline"] = df["Flight_Number_Reporting_Airline"].astype(str)
    for c in ("Reporting_Airline", "Origin", "Dest", "OriginState"):
        df[c] = df[c].astype(str)
    return df


def load_retail(files: List[Path], cfg: Dict[str, Any]) -> pd.DataFrame:
    path = files[0]
    str_cols = {"Invoice": str, "StockCode": str, "Description": str, "Country": str}
    if path.suffix.lower() in (".xlsx", ".xls"):
        df = pd.concat(pd.read_excel(path, sheet_name=None, dtype=str_cols).values(), ignore_index=True)
    else:
        df = pd.read_csv(path, dtype=str_cols, encoding="latin-1")
    df = df[["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Country"]]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Invoice"] = df["Invoice"].astype(str).str.strip()
    df["StockCode"] = df["StockCode"].astype(str).str.strip()
    return df


def load_chicago(files: List[Path], cfg: Dict[str, Any]) -> pd.DataFrame:
    cols = ["Inspection ID", "Inspection Date", "Risk", "Results", "Facility Type", "Latitude", "Longitude"]
    df = pd.read_csv(files[0], usecols=cols, dtype={"Risk": str, "Results": str, "Facility Type": str})
    df["Inspection Date"] = pd.to_datetime(df["Inspection Date"], format="%m/%d/%Y")
    return df[cols]


LOADERS = {"bts": load_bts, "retail": load_retail, "chicago": load_chicago}


def load_raw(name: str, cfg: Dict[str, Any], root: Path) -> pd.DataFrame:
    files = find_files(root, cfg["files"])
    if not files:
        raise FileNotFoundError(f"{name}: no files under {root} matching {cfg['files']}. "
                                "Run: python scripts/download_data.py")
    return LOADERS[cfg["loader"]](files, cfg)
