from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold


def spatial_block_ids(x: pd.Series, y: pd.Series, block_m: float) -> pd.Series:
    bx = np.floor(x.to_numpy(dtype=float) / block_m).astype(int)
    by = np.floor(y.to_numpy(dtype=float) / block_m).astype(int)
    return pd.Series([f"{a}_{b}" for a, b in zip(bx, by)], index=x.index)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    return {
        "n": int(mask.sum()),
        "R2": float(r2_score(y_true, y_pred)),
        "RMSE": rmse,
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "Bias": float(np.mean(y_pred - y_true)),
    }


def spatial_group_kfold(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 5):
    n_groups = groups.nunique()
    splits = max(2, min(n_splits, n_groups))
    gkf = GroupKFold(n_splits=splits)
    return gkf.split(X, y, groups)
