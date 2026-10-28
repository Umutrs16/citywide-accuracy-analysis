from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import Settings
from src.geoai.spatial_cv import regression_metrics, spatial_block_ids, spatial_group_kfold
from src.utils import save_json


def available_features(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def make_model(settings: Settings) -> LGBMRegressor:
    m = settings.raw["model"]
    return LGBMRegressor(
        n_estimators=int(m["n_estimators"]),
        learning_rate=float(m["learning_rate"]),
        num_leaves=int(m["num_leaves"]),
        subsample=float(m["subsample"]),
        colsample_bytree=float(m["colsample_bytree"]),
        min_child_samples=int(m["min_child_samples"]),
        random_state=int(m["random_state"]),
        verbosity=-1,
    )


def fit_spatial_cv(train: pd.DataFrame, settings: Settings, features: list[str] | None = None) -> dict:
    feats = available_features(train, features or settings.feature_cols)
    if "LST_C" not in train.columns:
        raise ValueError("训练表缺少 LST_C 标签")
    df = train.dropna(subset=feats + ["LST_C", "x", "y"]).copy()
    X = df[feats]
    y = df["LST_C"]
    groups = spatial_block_ids(df["x"], df["y"], settings.block_cv_m)
    oof = np.full(len(df), np.nan)
    fold_metrics = []
    for i, (tr, va) in enumerate(spatial_group_kfold(X, y, groups, settings.raw["model"]["n_splits"]), 1):
        model = make_model(settings)
        model.fit(X.iloc[tr], y.iloc[tr])
        pred = model.predict(X.iloc[va])
        oof[va] = pred
        met = regression_metrics(y.iloc[va], pred)
        met["fold"] = i
        fold_metrics.append(met)
    overall = regression_metrics(y, oof)
    overall["block_m"] = settings.block_cv_m
    overall["features"] = feats

    # 分位误差
    q = pd.qcut(y, q=4, duplicates="drop")
    q_err = []
    for cat in q.cat.categories:
        mask = q == cat
        q_err.append({"bin": str(cat), **regression_metrics(y[mask], oof[mask])})

    final = make_model(settings)
    final.fit(X, y)
    return {
        "model": final,
        "features": feats,
        "oof": oof,
        "y": y.to_numpy(),
        "overall": overall,
        "folds": fold_metrics,
        "quantile_error": q_err,
        "index": df.index.to_numpy(),
    }


def predict_frame(model, df: pd.DataFrame, features: list[str], col: str = "LST_pred") -> pd.DataFrame:
    out = df.copy()
    use = out.dropna(subset=features)
    pred = model.predict(use[features])
    out.loc[use.index, col] = pred
    if "LST_C" in out.columns:
        out["LST_residual"] = out[col] - out["LST_C"]
    return out


def aggregation_check(grid: pd.DataFrame) -> dict:
    """10 m/分析网格预测回看 30 m 标签的守恒性。"""
    if "LST_C" not in grid.columns or "LST_pred" not in grid.columns:
        return {}
    d = grid.dropna(subset=["LST_C", "LST_pred"])
    return regression_metrics(d["LST_C"], d["LST_pred"])


def save_model_bundle(result: dict, settings: Settings, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    path = settings.out_models / f"lgbm_lst{suffix}.joblib"
    joblib.dump({"model": result["model"], "features": result["features"]}, path)
    save_json(
        {
            "overall": result["overall"],
            "folds": result["folds"],
            "quantile_error": result["quantile_error"],
        },
        settings.out_tables / f"geoai_spatial_cv{suffix}.json",
    )
    if not tag:
        joblib.dump({"model": result["model"], "features": result["features"]}, settings.out_models / "lgbm_lst.joblib")
        save_json(
            {
                "overall": result["overall"],
                "folds": result["folds"],
                "quantile_error": result["quantile_error"],
            },
            settings.out_tables / "geoai_spatial_cv.json",
        )
    return path
