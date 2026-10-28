from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.utils import save_json, table_file


def _xy(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)])


def coverage_matrix(demand_xy: np.ndarray, cand_xy: np.ndarray, radius: float) -> np.ndarray:
    # Boolean [n_demand, n_cand]
    n_d = len(demand_xy)
    n_c = len(cand_xy)
    cov = np.zeros((n_d, n_c), dtype=bool)
    chunk = 250
    r2 = radius**2
    for i in range(0, n_d, chunk):
        sl = demand_xy[i : i + chunk]
        d2 = ((sl[:, None, :] - cand_xy[None, :, :]) ** 2).sum(axis=2)
        cov[i : i + chunk] = d2 <= r2
    return cov


def greedy_mclp(weights: np.ndarray, cov: np.ndarray, k: int) -> list[int]:
    chosen = []
    uncovered = np.ones(len(weights), dtype=bool)
    remain = set(range(cov.shape[1]))
    for _ in range(min(k, cov.shape[1])):
        best_j, best_gain = None, -1.0
        for j in remain:
            gain = float(weights[uncovered & cov[:, j]].sum())
            if gain > best_gain:
                best_gain, best_j = gain, j
        if best_j is None or best_gain <= 0:
            break
        chosen.append(best_j)
        remain.remove(best_j)
        uncovered[cov[:, best_j]] = False
    return chosen


def pulp_mclp(weights: np.ndarray, cov: np.ndarray, k: int) -> list[int] | None:
    try:
        import pulp
    except ImportError:
        return None
    n_d, n_c = cov.shape
    if n_c == 0 or n_d == 0:
        return []
    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y_{j}", cat="Binary") for j in range(n_c)]
    x = [pulp.LpVariable(f"x_{i}", cat="Binary") for i in range(n_d)]
    prob += pulp.lpDot(weights.tolist(), x)
    prob += pulp.lpSum(y) <= k
    for i in range(n_d):
        js = np.where(cov[i])[0]
        if len(js) == 0:
            prob += x[i] == 0
        else:
            prob += x[i] <= pulp.lpSum(y[j] for j in js)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=30))
    if pulp.LpStatus[status] not in {"Optimal", "Not Solved", "Undefined"} and pulp.value(prob.objective) is None:
        return None
    return [j for j, v in enumerate(y) if pulp.value(v) and pulp.value(v) > 0.5]


def baseline_random(weights: np.ndarray, cov: np.ndarray, k: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    idx = np.arange(cov.shape[1])
    rng.shuffle(idx)
    return idx[:k].tolist()


def baseline_weight(cand_w: np.ndarray, k: int) -> list[int]:
    return np.argsort(-cand_w)[:k].tolist()


def covered_weight(weights: np.ndarray, cov: np.ndarray, chosen: list[int]) -> float:
    if not chosen:
        return 0.0
    mask = cov[:, chosen].any(axis=1)
    return float(weights[mask].sum())


def run_mclp(settings: Settings, grid: pd.DataFrame, facilities: pd.DataFrame | None = None, tag: str = "") -> dict:
    g = grid.dropna(subset=["x", "y"]).copy()
    speed = float(settings.raw["access"]["walk_speed_kmh"])
    minutes = float(settings.raw["access"]["minutes"])
    radius = speed * 1000.0 * minutes / 60.0
    max_d = int(settings.raw["mclp"]["max_demand_points"])
    max_c = int(settings.raw["mclp"]["max_candidates"])

    from src.gis.osm_network import RETROFIT, classify_facility

    g["demand_w"] = g.get("high_risk_pop", g.get("pop_use", 1)) * (0.4 + 0.6 * g.get("Gap", 0).clip(0, 1))
    demand = g.sort_values("demand_w", ascending=False).head(max_d)
    # 候选优先：现有公共建筑改造，而不是任意格网中心新建
    cand = None
    exist_xy = np.empty((0, 2))
    if facilities is not None and len(facilities):
        fac = facilities.copy()
        if "service_class" not in fac.columns and "fclass" in fac.columns:
            fac["service_class"] = fac["fclass"].map(classify_facility)
        if "x" not in fac.columns and "geometry" in fac.columns:
            fac["x"] = fac.geometry.centroid.x
            fac["y"] = fac.geometry.centroid.y
        extra = fac.dropna(subset=["x", "y"]).head(80)
        exist_xy = _xy(extra) if len(extra) else exist_xy
        retrofit = fac.copy()
        if "fclass" in retrofit.columns:
            retrofit = retrofit[retrofit["fclass"].astype(str).str.lower().isin({x.lower() for x in RETROFIT})]
        if retrofit.empty:
            retrofit = fac[fac.get("service_class", "C").isin(["A", "B"])] if "service_class" in fac.columns else fac
        retrofit = retrofit.dropna(subset=["x", "y"]).head(max_c)
        if len(retrofit) >= 8:
            cand = retrofit.copy()
            cand["pop_use"] = 1.0
            cand["UTHI"] = 0.5
            cand["demand_w"] = 1.0
            cand["Gap"] = 0.5
    if cand is None:
        cand = g.sort_values("Gap", ascending=False).head(max_c)

    demand_xy = _xy(demand)
    cand_xy = _xy(cand)
    weights = demand["demand_w"].to_numpy(dtype=float)
    cov = coverage_matrix(demand_xy, cand_xy, radius)
    exist_cov = coverage_matrix(demand_xy, exist_xy, radius) if len(exist_xy) else np.zeros((len(demand), 0), dtype=bool)
    base_w = float(weights[exist_cov.any(axis=1)].sum()) if exist_cov.size else 0.0
    total_w = float(weights.sum())

    results = []
    selected = {}
    for k in settings.raw["mclp"]["n_new"]:
        exact = pulp_mclp(weights, cov, int(k))
        greedy = greedy_mclp(weights, cov, int(k))
        chosen = exact if exact else greedy
        rnd_shares = []
        for seed in range(200):
            rnd = baseline_random(weights, cov, int(k), seed=seed)
            rnd_shares.append(covered_weight(weights, cov, rnd) / max(total_w, 1e-6))
        rnd_mean = float(np.mean(rnd_shares))
        rnd_sd = float(np.std(rnd_shares))
        pop_first = baseline_weight(cand.get("pop_use", cand["demand_w"]).to_numpy(dtype=float), int(k))
        risk_first = baseline_weight(cand.get("UTHI", cand["demand_w"]).to_numpy(dtype=float), int(k))
        gap_first = baseline_weight(cand.get("Gap", cand["demand_w"]).to_numpy(dtype=float), int(k))
        rec = {
            "k": int(k),
            "solver": "pulp" if exact else "greedy",
            "mclp_cover": covered_weight(weights, cov, chosen),
            "greedy_cover": covered_weight(weights, cov, greedy),
            "random_cover": rnd_mean * total_w,
            "random_share_mean": rnd_mean,
            "random_share_sd": rnd_sd,
            "random_n": 200,
            "pop_priority_cover": covered_weight(weights, cov, pop_first),
            "risk_priority_cover": covered_weight(weights, cov, risk_first),
            "gap_priority_cover": covered_weight(weights, cov, gap_first),
            "baseline_existing_cover": base_w,
            "demand_total": total_w,
        }
        rec["mclp_share"] = rec["mclp_cover"] / max(total_w, 1e-6)
        rec["pop_priority_share"] = rec["pop_priority_cover"] / max(total_w, 1e-6)
        rec["gap_priority_share"] = rec["gap_priority_cover"] / max(total_w, 1e-6)
        rec["lift_vs_random"] = rec["mclp_share"] - rnd_mean
        results.append(rec)
        selected[str(k)] = cand.iloc[chosen][["x", "y"]].assign(k=int(k)).to_dict(orient="records")

    payload = {
        "radius_m": radius,
        "n_demand": int(len(demand)),
        "n_candidates": int(len(cand)),
        "candidate_mode": "retrofit_public_buildings"
        if cand is not None and facilities is not None and "fclass" in getattr(cand, "columns", [])
        else "high_gap_grid",
        "results": results,
        "selected": selected,
    }
    save_json(payload, table_file(settings, "mclp_results", tag))
    # 便于前端绘图
    pts = []
    for k, rows in selected.items():
        for r in rows:
            pts.append({**r, "k": int(k)})
    pd.DataFrame(pts).to_csv(table_file(settings, "mclp_sites", tag, "csv"), index=False, encoding="utf-8-sig")
    return payload
