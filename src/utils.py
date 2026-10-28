from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd


def minmax(s: pd.Series, reverse: bool = False) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    lo, hi = np.nanmin(x.to_numpy()), np.nanmax(x.to_numpy())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        out = pd.Series(np.zeros(len(x)), index=s.index, dtype=float)
    else:
        out = (x - lo) / (hi - lo)
    if reverse:
        out = 1.0 - out
    return out.clip(0, 1).fillna(0.0)


def entropy_weights(frame: pd.DataFrame) -> np.ndarray:
    """客观熵权。输入需已归一化到 0–1。"""
    x = frame.to_numpy(dtype=float)
    x = np.clip(x, 1e-12, 1.0)
    p = x / x.sum(axis=0, keepdims=True)
    k = 1.0 / np.log(len(frame))
    e = -k * np.nansum(p * np.log(p), axis=0)
    d = 1.0 - e
    if d.sum() <= 0:
        return np.full(frame.shape[1], 1.0 / frame.shape[1])
    return d / d.sum()


def table_file(settings, stem: str, tag: str = "", ext: str = "json") -> Path:
    name = f"{stem}_{tag}.{ext}" if tag else f"{stem}.{ext}"
    return settings.out_tables / name


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def gdf_to_metric(gdf: gpd.GeoDataFrame, crs: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if out.crs is None:
        out = out.set_crs(4326)
    return out.to_crs(crs)


def add_xy(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()
    c = out.geometry.centroid
    out["x"] = c.x
    out["y"] = c.y
    return out


def safe_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out
