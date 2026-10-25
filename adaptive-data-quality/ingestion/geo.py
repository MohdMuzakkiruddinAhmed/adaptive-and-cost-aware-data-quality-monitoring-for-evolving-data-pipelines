"""City of Chicago boundary: objective validity of coordinates (no extra dependencies).

Source: City of Chicago, "Boundaries - City" (data.cityofchicago.org, dataset qqq8-j68g), downloaded by
scripts/download_data.py to data/original/Chicago_City_Boundary.csv (WKT MULTIPOLYGON, WGS84 longitude/latitude).

Used for two things:
  * the domain range of the Chicago coordinate columns (bounding box of the city, configs/datasets.yaml
    domain_ranges), which replaces the k-sigma statistical range for those columns in every method: a k-sigma band
    around the mean has no meaning for positions in a city (dev run 3 Chicago replay: 130 of 146 degraded-check
    alarms were legitimate establishments at the city's edges, e.g. O'Hare and the far South Side);
  * objective adjudication of coordinate alarms in the chronological replay: a flagged position inside the city
    limits is NOT an error, one outside them (or 0 / missing) is, so no human judgement is needed for these checks.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

BOUNDARY_FILE = "Chicago_City_Boundary.csv"
BOUNDARY_URL = "https://data.cityofchicago.org/api/views/qqq8-j68g/rows.csv?accessType=DOWNLOAD"


def parse_multipolygon(wkt: str) -> List[List[np.ndarray]]:
    """WKT (MULTI)POLYGON -> list of polygons, each a list of rings (outer ring first) as (n, 2) lon/lat arrays."""
    body = wkt.strip()
    m = re.match(r"^(MULTIPOLYGON|POLYGON)\s*\((.*)\)\s*$", body, re.S | re.I)
    if not m:
        raise ValueError("not a WKT polygon")
    kind, inner = m.group(1).upper(), m.group(2)
    polys_txt = re.findall(r"\(\((.*?)\)\)", inner, re.S) if kind == "MULTIPOLYGON" else [inner.strip()[1:-1]]
    polys = []
    for p in polys_txt:
        rings = []
        for r in re.split(r"\)\s*,\s*\(", p):
            pts = [tuple(map(float, xy.split())) for xy in r.replace("(", "").replace(")", "").split(",") if xy.strip()]
            rings.append(np.asarray(pts, dtype=float))
        polys.append(rings)
    return polys


def _in_ring(x: np.ndarray, y: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Even-odd ray casting, vectorised over the points."""
    inside = np.zeros(x.shape, dtype=bool)
    xs, ys = ring[:, 0], ring[:, 1]
    xj, yj = np.roll(xs, 1), np.roll(ys, 1)
    for x1, y1, x2, y2 in zip(xs, ys, xj, yj):
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = (x2 - x1) * (y - y1) / (y2 - y1) + x1
        inside ^= cond & (x < xint)
    return inside


def points_in_polygons(lon, lat, polys: List[List[np.ndarray]]) -> np.ndarray:
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    out = np.zeros(x.shape, dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    for rings in polys:
        lo, hi = rings[0].min(axis=0), rings[0].max(axis=0)
        cand = ok & (x >= lo[0]) & (x <= hi[0]) & (y >= lo[1]) & (y <= hi[1])
        if not cand.any():
            continue
        inside = _in_ring(x[cand], y[cand], rings[0])
        for hole in rings[1:]:
            inside &= ~_in_ring(x[cand], y[cand], hole)
        out[np.flatnonzero(cand)[inside]] = True
    return out


class CityBoundary:
    def __init__(self, polys: List[List[np.ndarray]]) -> None:
        self.polys = polys
        allpts = np.vstack([r[0] for r in polys])
        self.lon_range: Tuple[float, float] = (float(allpts[:, 0].min()), float(allpts[:, 0].max()))
        self.lat_range: Tuple[float, float] = (float(allpts[:, 1].min()), float(allpts[:, 1].max()))

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "CityBoundary":
        from common.utils import data_dir
        path = Path(path) if path else data_dir("original", BOUNDARY_FILE)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: python scripts/download_data.py --only chicago")
        df = pd.read_csv(path)
        polys = []
        for wkt in df["the_geom"].dropna():
            polys += parse_multipolygon(str(wkt))
        return cls(polys)

    def contains(self, lon, lat) -> np.ndarray:
        return points_in_polygons(lon, lat, self.polys)
