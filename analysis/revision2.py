"""Second-round revision: population geometry, residential decision domain, calibration curve.

Reads archived tables and models. Does not write competition outputs.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis import (  # noqa: E402
    CENSUS,
    FIG,
    RES,
    coverage_matrix,
    covered_weight,
    cv_predict,
    greedy_mclp,
    jaccard_cols,
)
from revision import extra  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import spatial_block_ids, spatial_group_kfold  # noqa: E402
from src.geoai.train_geoai import available_features, make_model  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

OUT = RES / "revision2_summary.json"


def census_people(pop: np.ndarray, mask: np.ndarray) -> float:
    return float((pop * mask).sum() / max(float(np.nansum(pop)), 1e-9) * CENSUS)


def norm_on(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = np.zeros(len(values), dtype=float)
    vv = values[mask]
    vv = vv[np.isfinite(vv)]
    if len(vv) == 0:
        return out
    lo, hi = float(np.min(vv)), float(np.max(vv))
    if hi <= lo:
        out[mask] = 0.0
        return out
    out[mask] = np.clip((values[mask] - lo) / (hi - lo), 0, 1)
    return out


def top_within(score: np.ndarray, universe: np.ndarray, q: float = 0.80) -> np.ndarray:
    mask = np.zeros(len(score), dtype=bool)
    idx = np.flatnonzero(universe & np.isfinite(score))
    if len(idx) == 0:
        return mask
    thr = np.quantile(score[idx], q)
    mask[idx[score[idx] >= thr]] = True
    return mask


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    return float(jaccard_cols(a[:, None], b)[0])


def square_coverage(x: np.ndarray, y: np.ndarray, half: float, res: float = 100.0) -> dict:
    xmin = float(np.min(x) - half)
    ymin = float(np.min(y) - half)
    nx = int(np.ceil((np.max(x) + half - xmin) / res))
    ny = int(np.ceil((np.max(y) + half - ymin) / res))
    used = res
    if nx * ny > 25_000_000:
        used = 200.0
        nx = int(np.ceil((np.max(x) + half - xmin) / used))
        ny = int(np.ceil((np.max(y) + half - ymin) / used))
    cover = np.zeros((ny, nx), dtype=np.uint16)
    j0 = np.clip(np.floor((x - half - xmin) / used).astype(np.int32), 0, nx - 1)
    i0 = np.clip(np.floor((y - half - ymin) / used).astype(np.int32), 0, ny - 1)
    j1 = np.clip(np.ceil((x + half - xmin) / used).astype(np.int32), 1, nx)
    i1 = np.clip(np.ceil((y + half - ymin) / used).astype(np.int32), 1, ny)
    for a, b, c, d in zip(i0, i1, j0, j1):
        if b > a and d > c:
            cover[a:b, c:d] += 1
    pix = used * used / 1e6
    return {
        "raster_m": used,
        "nominal_square_km2": float(len(x) * (2 * half) ** 2 / 1e6),
        "union_km2": float((cover > 0).sum() * pix),
        "overlap_pixel_km2": float((cover >= 2).sum() * pix),
        "exclusive_km2": float((cover == 1).sum() * pix),
        "max_overlap_count": int(cover.max()) if cover.size else 0,
        "mean_overlap_where_covered": float(cover[cover > 0].mean()) if (cover > 0).any() else None,
        "stacked_km2": float(cover.sum() * pix),
    }


def concentration(pop: np.ndarray, mask: np.ndarray) -> dict:
    vals = pop[mask & (pop > 0)]
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {"n": 0}
    order = np.argsort(-vals)
    ranked = vals[order]
    cum = np.cumsum(ranked) / ranked.sum()

    def share(k: int) -> float:
        return float(cum[min(k, len(cum)) - 1])

    return {
        "n_positive": int(len(vals)),
        "n_below_1": int((vals < 1).sum()),
        "n_below_10": int((vals < 10).sum()),
        "min": float(vals.min()),
        "median": float(np.median(vals)),
        "p90": float(np.quantile(vals, 0.90)),
        "max": float(vals.max()),
        "mean": float(vals.mean()),
        "share_top10": share(10),
        "share_top50": share(50),
        "share_top100": share(100),
        "share_top450": share(450),
        "total_people_raw": float(vals.sum()),
    }


def exact_mclp(weights: np.ndarray, cov: np.ndarray, k: int, time_limit: int = 12):
    import pulp

    n_d, n_c = cov.shape
    prob = pulp.LpProblem("mclp", pulp.LpMaximize)
    y = [pulp.LpVariable(f"y{j}", cat="Binary") for j in range(n_c)]
    x = [pulp.LpVariable(f"x{i}", cat="Binary") for i in range(n_d)]
    prob += pulp.lpDot(weights.tolist(), x)
    prob += pulp.lpSum(y) <= k
    for i in range(n_d):
        js = np.flatnonzero(cov[i])
        if len(js) == 0:
            prob += x[i] == 0
        else:
            prob += x[i] <= pulp.lpSum(y[int(j)] for j in js)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.005))
    chosen = [j for j, v in enumerate(y) if v.value() is not None and v.value() > 0.5]
    obj = pulp.value(prob.objective)
    return chosen, None if obj is None else float(obj), pulp.LpStatus[status]


def recall_block_bootstrap(y, pred, blocks, n_boot: int = 300, seed: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    point = extra(y, pred)["top20_recall"]
    stats = []
    for _ in range(n_boot):
        take = rng.choice(uniq, size=len(uniq), replace=True)
        parts = [np.flatnonzero(blocks == b) for b in take]
        idx = np.concatenate(parts) if parts else np.array([], dtype=int)
        if len(idx) < 30:
            continue
        stats.append(extra(y[idx], pred[idx])["top20_recall"])
    arr = np.asarray(stats, dtype=float)
    return {
        "point": float(point),
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "n_boot": int(len(arr)),
    }


def style():
    for candidate in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if Path(candidate).exists():
            font_manager.fontManager.addfont(candidate)
            break
    plt.rcParams.update({
        "font.family": "Microsoft YaHei",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    })


def main() -> None:
    import geopandas as gpd
    import joblib

    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    yining = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Yiningshi_2024.parquet")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    roi = json.loads((ROOT / "outputs" / "tables" / "population_roi_sum.json").read_text(encoding="utf-8"))
    prev = json.loads((RES / "revision_summary.json").read_text(encoding="utf-8"))

    x = grid["x"].to_numpy(dtype=float)
    y = grid["y"].to_numpy(dtype=float)
    lon = grid.geometry.x.to_numpy(dtype=float)
    lat = grid.geometry.y.to_numpy(dtype=float)
    pop = grid["population"].to_numpy(dtype=float)
    pop_w = grid["worldpop"].to_numpy(dtype=float)
    built = grid["built"].to_numpy(dtype=float) >= 0.15
    lst = grid["LST_pred"].to_numpy(dtype=float)
    p90 = grid["LST_P90"].to_numpy(dtype=float)
    uthi = grid["UTHI"].to_numpy(dtype=float)
    gap_stored = np.clip(grid["Gap"].to_numpy(dtype=float), 0, 1)
    access = grid["access"].to_numpy(dtype=float) if "access" in grid.columns else np.full(len(grid), 0.35)
    city_top = uthi >= np.quantile(uthi, 0.80)

    print("population geometry", flush=True)
    geom = square_coverage(x, y, half=200.0, res=100.0)
    yx = yining["x"].to_numpy(dtype=float)
    yy = yining["y"].to_numpy(dtype=float)
    geom_y = square_coverage(yx, yy, half=100.0, res=50.0)
    ghsl_sum = float(np.nansum(pop))
    wpop_sum = float(np.nansum(pop_w))
    roi_g = float(roi["Wulumuqishi"]["ghsl_roi_sum"])
    roi_w = float(roi["Wulumuqishi"]["worldpop_roi_sum"])

    demand_w = pop * city_top * (0.4 + 0.6 * gap_stored)
    order450 = np.argsort(-demand_w)[:450]
    in450 = np.zeros(len(grid), dtype=bool)
    in450[order450] = True
    pos = city_top & (pop > 0)
    complement = pos & ~in450
    conc = concentration(pop, city_top)
    pop_audit = {
        "assignment": (
            "Each sample point is the center of a nominal square in the local UTM projection "
            "(400 m for Urumqi, 200 m for Yining). 100 m population counts are summed by "
            "rasterizing those squares. A pixel covered by more than one square is assigned "
            "to one square only, so people are not double-counted. The squares are not a "
            "partition of the administrative polygon."
        ),
        "raster_files_local": False,
        "roi_method": roi["Wulumuqishi"]["method"],
        "geometry_urumqi": geom,
        "geometry_yining": geom_y,
        "ghsl_grid_sum": ghsl_sum,
        "ghsl_roi_sum": roi_g,
        "ghsl_grid_over_roi": ghsl_sum / roi_g,
        "worldpop_grid_sum": wpop_sum,
        "worldpop_roi_sum": roi_w,
        "worldpop_grid_over_roi": wpop_sum / roi_w,
        "census": CENSUS,
        "priority_positive": conc,
        "demand450_share_of_priority_pop": float(pop[in450].sum() / max(pop[city_top].sum(), 1e-9)),
        "complement_n": int(complement.sum()),
        "complement_people_raw": float(pop[complement].sum()),
        "complement_mean": float(pop[complement].mean()) if complement.any() else None,
        "complement_median": float(np.median(pop[complement])) if complement.any() else None,
        "complement_max": float(pop[complement].max()) if complement.any() else None,
        "census_scaled_complement": census_people(pop, complement),
    }
    print(
        "  grid/roi GHSL",
        round(pop_audit["ghsl_grid_over_roi"], 4),
        "overlap km2",
        round(geom["overlap_pixel_km2"], 1),
        "union",
        round(geom["union_km2"], 1),
        "complement people",
        round(pop_audit["complement_people_raw"], 1),
        flush=True,
    )

    print("models for residuals and learning curve", flush=True)
    built_rows = work["built"].to_numpy(dtype=float) >= 0.15
    city_met, oof, _, _, y_all = cv_predict(work, feats, "spatial", 3000, "lgbm")
    resid = y_all - oof
    built_frame = work.loc[built_rows]
    built_met, built_oof, _, _, built_y = cv_predict(built_frame, feats, "spatial", 3000, "lgbm")
    blocks_all = spatial_block_ids(work["x"], work["y"], 3000).to_numpy()
    boot_city = recall_block_bootstrap(y_all[built_rows], oof[built_rows], blocks_all[built_rows])
    blocks_b = spatial_block_ids(built_frame["x"], built_frame["y"], 3000).to_numpy()
    boot_built = recall_block_bootstrap(built_y, built_oof, blocks_b, seed=4)
    print("  recall city-on-built", boot_city, "refit", boot_built, flush=True)

    # Fixed built-up test blocks; vary the number of training rows.
    rng_l = np.random.default_rng(5)
    uniq_b = np.unique(blocks_b)
    n_hold = max(1, int(round(0.2 * len(uniq_b))))
    hold = set(rng_l.choice(uniq_b, size=n_hold, replace=False).tolist())
    is_test = np.isin(blocks_b, list(hold))
    pool = np.flatnonzero(~is_test)
    test = np.flatnonzero(is_test)
    Xb = built_frame[feats]
    yb = built_y
    curve = []
    for n_train in (80, 160, 320, 600, int(len(pool))):
        reps = 1 if n_train >= len(pool) else 6
        rows = []
        for rep in range(reps):
            take = pool if n_train >= len(pool) else rng_l.choice(pool, size=n_train, replace=False)
            model = make_model(settings)
            model.fit(Xb.iloc[take], yb[take])
            pred = model.predict(Xb.iloc[test])
            rows.append(extra(yb[test], pred))
        curve.append({
            "n_train": int(min(n_train, len(pool))),
            "n_test": int(len(test)),
            "repeats": reps,
            "R2_mean": float(np.mean([r["R2"] for r in rows])),
            "RMSE_mean": float(np.mean([r["RMSE"] for r in rows])),
            "recall_mean": float(np.mean([r["top20_recall"] for r in rows])),
            "recall_sd": float(np.std([r["top20_recall"] for r in rows])),
        })
        print("  curve", curve[-1]["n_train"], "R2", round(curve[-1]["R2_mean"], 3), "recall", round(curve[-1]["recall_mean"], 3), flush=True)

    # Residual structure on the citywide out-of-fold errors.
    rng_v = np.random.default_rng(6)
    from sklearn.neighbors import NearestNeighbors
    xy_all = np.column_stack([work["x"].to_numpy(dtype=float), work["y"].to_numpy(dtype=float)])
    origins = rng_v.choice(len(resid), size=1800, replace=False)
    nn = NearestNeighbors(n_neighbors=30).fit(xy_all)
    dist, ind = nn.kneighbors(xy_all[origins])
    gamma = 0.5 * (resid[origins][:, None] - resid[ind]) ** 2
    dist = dist[:, 1:]
    gamma = gamma[:, 1:]
    edges = np.arange(0, 15001, 500)
    variogram = []
    sill = float(np.var(resid))
    practical = None
    flat = dist.ravel()
    gflat = gamma.ravel()
    bin_id = np.digitize(flat, edges) - 1
    for b in range(len(edges) - 1):
        sel = bin_id == b
        if int(sel.sum()) < 40:
            continue
        gmean = float(gflat[sel].mean())
        center = float(0.5 * (edges[b] + edges[b + 1]))
        variogram.append({"distance_m": center, "gamma": gmean, "n": int(sel.sum())})
        if practical is None and gmean >= 0.95 * sill:
            practical = center
    print("  variogram bins", [(round(v["distance_m"]), round(v["gamma"], 2)) for v in variogram[:12]], "sill", round(sill, 2), "range", practical, flush=True)
    elev_tr = work["elevation"].to_numpy(dtype=float)
    hetero = {}
    for name, values in (("observed_LST", y_all), ("elevation", elev_tr), ("built", work["built"].to_numpy(dtype=float))):
        cuts = np.quantile(values, [0.25, 0.5, 0.75])
        bins = np.digitize(values, cuts)
        hetero[name] = [
            {"bin": int(b), "n": int((bins == b).sum()), "residual_sd": float(resid[bins == b].std())}
            for b in range(4)
        ]
    print("  variogram 95% sill m", practical, "sill", round(sill, 2), flush=True)

    # Two decision domains.
    print("residential domain", flush=True)
    universe = built & (pop > 0) & np.isfinite(lst) & np.isfinite(p90)

    def component_matrix(lst_values, population, ref):
        thermal = 0.7 * norm_on(lst_values, ref) + 0.3 * norm_on(p90, ref)
        exposure = norm_on(population, ref)
        cool_parts = []
        for name in ("NDVI", "trees", "water", "ALBEDO"):
            if name in grid.columns:
                cool_parts.append(norm_on(grid[name].to_numpy(dtype=float), ref))
        cooling = np.mean(cool_parts, axis=0) if cool_parts else norm_on(grid["CoolingCapacity"].to_numpy(dtype=float), ref)
        service = norm_on(access, ref)
        return np.column_stack([thermal, exposure, 1 - cooling, 1 - service])

    base_X = component_matrix(lst, pop, universe)
    base_score = base_X @ np.full(4, 0.25)
    resid_top = top_within(base_score, universe)
    # Built-up top 20% even where population is zero, using citywide-scaled dynamic score from the previous definition.
    built_scores_city_scale = uthi.copy()
    built_only = np.zeros(len(grid), dtype=bool)
    bidx = np.flatnonzero(built)
    built_only[bidx[built_scores_city_scale[bidx] >= np.quantile(built_scores_city_scale[bidx], 0.80)]] = True
    inter_built = int((built_only & city_top).sum())
    inter_resid = int((resid_top & city_top).sum())
    p90_only = top_within(norm_on(p90, universe), universe)
    # Citywide priority using only the observed 90th percentile as the thermal term.
    city_p90_score = np.column_stack([
        norm_on(p90, np.ones(len(grid), dtype=bool)),
        norm_on(pop, np.ones(len(grid), dtype=bool)),
        1 - grid["CoolingCapacity"].to_numpy(dtype=float),
        1 - grid["ServiceAdaptation"].to_numpy(dtype=float),
    ]) @ np.full(4, 0.25)
    city_p90_top = city_p90_score >= np.quantile(city_p90_score, 0.80)

    weights = pd.read_csv(ROOT / "outputs" / "tables" / "uthi_weight_mc_draws.csv")
    W = weights[["w_TS", "w_EXP", "w_COOL", "w_SERV"]].to_numpy(dtype=float)
    W_use = W[::25]  # 40 vectors inside the archived constrained set

    tr_block = blocks_all
    resid_by = {}
    for b, r, is_b in zip(tr_block, resid, built_rows):
        resid_by.setdefault(int(b), []).append(float(r))
    resid_built = resid[built_rows]
    resid_open = resid[~built_rows]
    g_block = spatial_block_ids(grid["x"], grid["y"], 3000).to_numpy()
    codes, uniques = pd.factorize(g_block)
    block_pool = []
    block_has_built = np.zeros(len(uniques), dtype=bool)
    for code, flag in zip(codes, built):
        if flag:
            block_has_built[code] = True
    for b, is_b in zip(uniques, block_has_built):
        pool = resid_by.get(int(b)) or (resid_built if is_b else resid_open)
        block_pool.append(np.asarray(pool, dtype=float))
    rng_e = np.random.default_rng(11)
    fields = []
    for _ in range(25):
        fields.append(np.array([rng_e.choice(pool) for pool in block_pool])[codes])

    def run_slice(label: str, scenarios: list[tuple[np.ndarray, np.ndarray]]) -> dict:
        hits = np.zeros(len(grid), dtype=np.int32)
        jacs, people, stable_people = [], [], []
        masks = []
        for lst_s, pop_s in scenarios:
            X = component_matrix(lst_s, pop_s, universe)
            # weights applied outside; this helper is equal-weight. overwritten below
            score = X @ np.full(4, 0.25)
            mask = top_within(score, universe)
            masks.append(mask)
            hits += mask
            jacs.append(jaccard(mask, resid_top))
            people.append(census_people(pop_s if pop_s is pop else pop, mask))
        freq = hits / max(len(scenarios), 1)
        stable = freq >= 0.90
        return {
            "label": label,
            "n": len(scenarios),
            "jaccard_mean": float(np.mean(jacs)),
            "jaccard_p05": float(np.quantile(jacs, 0.05)),
            "jaccard_p50": float(np.quantile(jacs, 0.50)),
            "people_p05": float(np.quantile(people, 0.05)),
            "people_p50": float(np.quantile(people, 0.50)),
            "people_p95": float(np.quantile(people, 0.95)),
            "baseline_still_ge_90": float(stable[resid_top].mean()) if resid_top.any() else None,
            "stable90_n": int(stable.sum()),
            "stable90_people": census_people(pop, stable),
            "freq": freq,
        }

    # Weight-only, population-only, temperature-only, and the joint cross, all ranked inside universe.
    weight_jacs = []
    whits = np.zeros(universe.sum(), dtype=np.int32)
    u_index = np.flatnonzero(universe)
    for w in W_use:
        score = base_X @ w
        mask = top_within(score, universe)
        weight_jacs.append(jaccard(mask, resid_top))
        whits += mask[u_index]
    weight_only = {
        "n": int(len(W_use)),
        "jaccard_mean": float(np.mean(weight_jacs)),
        "jaccard_p05": float(np.quantile(weight_jacs, 0.05)),
        "baseline_still_ge_90": float(((whits / len(W_use)) >= 0.90)[resid_top[u_index]].mean()),
    }

    def priority_for(lst_values, population, w):
        X = component_matrix(lst_values, population, universe)
        return top_within(X @ w, universe)

    temp_jacs, temp_people = [], []
    temp_hits = np.zeros(len(grid), dtype=np.int32)
    temp_masks = []
    for delta in fields:
        mask = priority_for(lst + delta, pop, np.full(4, 0.25))
        temp_masks.append(mask)
        temp_hits += mask
        temp_jacs.append(jaccard(mask, resid_top))
        temp_people.append(census_people(pop, mask))
    temp_only = {
        "n": 25,
        "jaccard_mean": float(np.mean(temp_jacs)),
        "jaccard_p05": float(np.quantile(temp_jacs, 0.05)),
        "people_p50": float(np.median(temp_people)),
        "baseline_still_ge_90": float(((temp_hits / 25) >= 0.90)[resid_top].mean()),
        "stable90_n": int(((temp_hits / 25) >= 0.90).sum()),
        "stable90_people": census_people(pop, (temp_hits / 25) >= 0.90),
    }
    pop_mask = priority_for(lst, pop_w, np.full(4, 0.25))
    # WorldPop universe is the same points; exposure uses WorldPop, including zeros.
    pop_source = {
        "jaccard": jaccard(pop_mask, resid_top),
        "people_ghsl": census_people(pop, resid_top),
        "people_worldpop_on_ghsl_set": census_people(pop_w, resid_top) * (CENSUS / 4_054_369),
    }
    # census_people already scales by CENSUS using GHSL denominator. WorldPop headcount on the same cells:
    pop_source["people_worldpop_recount"] = float(pop_w[resid_top].sum() / max(wpop_sum, 1e-9) * CENSUS)

    joint_hits = np.zeros(len(grid), dtype=np.int32)
    joint_jacs, joint_people = [], []
    # 25 fields x 20 weights x 2 populations = 1000 scenarios. Enough to see the residential domain.
    W_joint = W[::50]
    scenario_masks_ghsl = []
    for delta in fields:
        for population, tag in ((pop, "ghsl"), (pop_w, "worldpop")):
            for w in W_joint:
                mask = priority_for(lst + delta, population, w)
                joint_hits += mask
                joint_jacs.append(jaccard(mask, resid_top))
                joint_people.append(census_people(pop if tag == "ghsl" else pop_w, mask) if tag == "ghsl"
                                    else float(population[mask].sum() / max(wpop_sum, 1e-9) * CENSUS))
                if tag == "ghsl" and np.allclose(w, 0.25):
                    scenario_masks_ghsl.append(mask)
    n_joint = len(joint_jacs)
    joint_freq = joint_hits / n_joint
    joint_stable = joint_freq >= 0.90
    residential = {
        "universe_n": int(universe.sum()),
        "universe_people": census_people(pop, universe),
        "priority_n": int(resid_top.sum()),
        "priority_unpopulated_share": float((pop[resid_top] <= 0).mean()) if resid_top.any() else None,
        "priority_people": census_people(pop, resid_top),
        "built_only_top20_n": int(built_only.sum()),
        "built_only_unpopulated_share": float((pop[built_only] <= 0).mean()) if built_only.any() else None,
        "intersection_built_only_and_citywide_n": inter_built,
        "share_of_built_only_also_citywide": inter_built / max(int(built_only.sum()), 1),
        "intersection_residential_and_citywide_n": inter_resid,
        "share_of_residential_also_citywide": inter_resid / max(int(resid_top.sum()), 1),
        "jaccard_residential_vs_citywide": jaccard(resid_top, city_top),
        "jaccard_built_only_vs_citywide": jaccard(built_only, city_top),
        "p90_only_jaccard_vs_model": jaccard(p90_only, resid_top),
        "citywide_p90_only_jaccard_vs_model": jaccard(city_p90_top, city_top),
        "p90_finite_share": float(np.isfinite(p90).mean()),
        "weight_only": weight_only,
        "temperature_only": temp_only,
        "population_source": pop_source,
        "joint": {
            "n_scenarios": n_joint,
            "n_temperature_fields": 25,
            "n_weights": int(len(W_joint)),
            "jaccard_mean": float(np.mean(joint_jacs)),
            "jaccard_p05": float(np.quantile(joint_jacs, 0.05)),
            "jaccard_p50": float(np.quantile(joint_jacs, 0.50)),
            "people_p05": float(np.quantile(joint_people, 0.05)),
            "people_p50": float(np.quantile(joint_people, 0.50)),
            "people_p95": float(np.quantile(joint_people, 0.95)),
            "baseline_still_ge_90": float(joint_stable[resid_top].mean()) if resid_top.any() else None,
            "stable90_n": int(joint_stable.sum()),
            "stable90_people": census_people(pop, joint_stable),
        },
    }
    print(
        "  universe", residential["universe_n"],
        "priority", residential["priority_n"],
        "also citywide", round(residential["share_of_residential_also_citywide"], 3),
        "temp jaccard", round(temp_only["jaccard_mean"], 3),
        "joint", round(residential["joint"]["jaccard_mean"], 3),
        flush=True,
    )

    print("siting", flush=True)
    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    names = {n.lower() for n in RETROFIT}
    cand = fac[fac["fclass"].astype(str).str.lower().isin(names)].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])
    xy = np.column_stack([x, y])

    def demand_of(mask, population):
        # Rank inside the scenario priority. Keep every inhabited priority point up to 450.
        score = population * mask
        idx = np.flatnonzero(score > 0)
        if len(idx) > 450:
            idx = idx[np.argsort(-score[idx])[:450]]
        return idx

    def solve_pair(mask, population):
        idx = demand_of(mask, population)
        weights_d = population[idx]
        cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
        greedy = greedy_mclp(weights_d, cov, 10)
        g_share = covered_weight(weights_d, cov, greedy) / max(float(weights_d.sum()), 1e-9)
        return idx, weights_d, cov, greedy, g_share

    idx0, w0, cov0, greedy0, share0 = solve_pair(resid_top, pop)
    t0 = time.perf_counter()
    exact0, exact_obj, exact_status = exact_mclp(w0, cov0, 10, time_limit=20)
    exact_share0 = covered_weight(w0, cov0, exact0) / max(float(w0.sum()), 1e-9)
    print("  baseline exact", exact_status, "share", round(exact_share0, 4), "greedy", round(share0, 4), "sec", round(time.perf_counter() - t0, 1), flush=True)

    fixed_g, re_g, reg_g = [], [], []
    fixed_e, re_e, reg_e, statuses = [], [], [], []
    hit = np.zeros(len(cand), dtype=int)
    # 25 GHSL temperature fields, equal weight. Exact on each, with a short limit.
    do_exact = exact_status == "Optimal" or exact_share0 + 1e-6 >= share0
    for delta in fields:
        mask = priority_for(lst + delta, pop, np.full(4, 0.25))
        idx, weights_d, cov, greedy, g_share = solve_pair(mask, pop)
        fixed = covered_weight(weights_d, cov, greedy0) / max(float(weights_d.sum()), 1e-9)
        # Map baseline candidate indices. greedy0 indexes candidates, cov columns are candidates. Good.
        fixed_g.append(fixed)
        re_g.append(g_share)
        reg_g.append(g_share - fixed)
        hit[np.asarray(greedy, dtype=int)] += 1
        if do_exact:
            chosen_e, _, status = exact_mclp(weights_d, cov, 10, time_limit=8)
            e_share = covered_weight(weights_d, cov, chosen_e) / max(float(weights_d.sum()), 1e-9)
            fixed_exact = covered_weight(weights_d, cov, exact0) / max(float(weights_d.sum()), 1e-9)
            fixed_e.append(fixed_exact)
            re_e.append(e_share)
            reg_e.append(e_share - fixed_exact)
            statuses.append(status)
    reg_g = np.asarray(reg_g)
    cand_ll = cand.to_crs(4326)
    siting = {
        "demand_n": int(len(idx0)),
        "demand_share_of_residential_priority_pop": float(pop[idx0].sum() / max(pop[resid_top].sum(), 1e-9)),
        "baseline_greedy_share": float(share0),
        "baseline_exact_share": float(exact_share0),
        "baseline_exact_status": exact_status,
        "scenario_n": int(len(reg_g)),
        "greedy_fixed_p50": float(np.median(fixed_g)),
        "greedy_reopt_p50": float(np.median(re_g)),
        "greedy_regret_p50": float(np.median(reg_g)),
        "greedy_regret_p05": float(np.quantile(reg_g, 0.05)),
        "greedy_regret_p95": float(np.quantile(reg_g, 0.95)),
        "greedy_regret_min": float(reg_g.min()),
        "greedy_regret_values": [float(v) for v in reg_g],
        "always_selected": int(sum(hit[i] == len(reg_g) for i in greedy0)),
        "n_ever": int((hit > 0).sum()),
        "sites": [
            {
                "name": None if pd.isna(cand.iloc[i]["name"]) else str(cand.iloc[i]["name"]),
                "fclass": str(cand.iloc[i]["fclass"]),
                "x": float(cand.iloc[i]["x"]),
                "y": float(cand.iloc[i]["y"]),
                "lon": float(cand_ll.geometry.iloc[i].x),
                "lat": float(cand_ll.geometry.iloc[i].y),
                "frequency": float(hit[i] / len(reg_g)),
                "in_baseline": bool(i in set(greedy0)),
            }
            for i in np.argsort(-hit) if hit[i] > 0
        ],
    }
    if reg_e:
        reg_e = np.asarray(reg_e)
        siting.update({
            "exact_scenario_n": int(len(reg_e)),
            "exact_status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
            "exact_fixed_p50": float(np.median(fixed_e)),
            "exact_reopt_p50": float(np.median(re_e)),
            "exact_regret_p50": float(np.median(reg_e)),
            "exact_regret_p05": float(np.quantile(reg_e, 0.05)),
            "exact_regret_p95": float(np.quantile(reg_e, 0.95)),
            "exact_regret_min": float(reg_e.min()),
            "n_negative_exact_regret": int((reg_e < -1e-9).sum()),
        })
    print("  regret greedy p50", round(siting["greedy_regret_p50"], 4), "min", round(siting["greedy_regret_min"], 4), flush=True)

    # One robust plan: maximize the worst coverage of the baseline residential demand
    # under the 25 GHSL temperature scenarios. Points that leave the priority get weight 0.
    print("robust plan", flush=True)
    try:
        import pulp
        base_idx = idx0
        cov_b = cov0
        n_d, n_c = cov_b.shape
        prob = pulp.LpProblem("robust", pulp.LpMaximize)
        yvar = [pulp.LpVariable(f"ry{j}", cat="Binary") for j in range(n_c)]
        xvar = [[pulp.LpVariable(f"rx{s}_{i}", cat="Binary") for i in range(n_d)] for s in range(len(fields))]
        t = pulp.LpVariable("t", lowBound=0)
        prob += t
        prob += pulp.lpSum(yvar) <= 10
        for s, delta in enumerate(fields):
            mask = priority_for(lst + delta, pop, np.full(4, 0.25))
            ww = np.array([pop[base_idx[i]] if mask[base_idx[i]] else 0.0 for i in range(n_d)])
            prob += pulp.lpDot(ww.tolist(), xvar[s]) >= t
            for i in range(n_d):
                js = np.flatnonzero(cov_b[i])
                if len(js) == 0 or ww[i] <= 0:
                    prob += xvar[s][i] == 0
                else:
                    prob += xvar[s][i] <= pulp.lpSum(yvar[int(j)] for j in js)
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=40, gapRel=0.01))
        robust_chosen = [j for j, v in enumerate(yvar) if v.value() is not None and v.value() > 0.5]
        # Worst and median coverage of this plan, the baseline plan, on the same 25 demands.
        def cover_share(chosen, mask):
            idx = demand_of(mask, pop)
            # Evaluate on the scenario demand, not only the baseline points.
            weights_d = pop[idx]
            cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
            return covered_weight(weights_d, cov, chosen) / max(float(weights_d.sum()), 1e-9)
        rob_shares, base_shares = [], []
        for delta in fields:
            mask = priority_for(lst + delta, pop, np.full(4, 0.25))
            rob_shares.append(cover_share(robust_chosen, mask))
            base_shares.append(cover_share(exact0, mask))
        siting["robust"] = {
            "status": pulp.LpStatus[status],
            "n_sites": len(robust_chosen),
            "worst_share": float(np.min(rob_shares)) if rob_shares else None,
            "median_share": float(np.median(rob_shares)) if rob_shares else None,
            "baseline_exact_worst_share": float(np.min(base_shares)) if base_shares else None,
            "baseline_exact_median_share": float(np.median(base_shares)) if base_shares else None,
        }
        print("  robust", siting["robust"], flush=True)
    except Exception as exc:  # noqa: BLE001
        siting["robust"] = {"status": f"failed: {exc}"}
        print("  robust failed", exc, flush=True)

    print("yining few-shot", flush=True)
    bundle = joblib.load(ROOT / "outputs" / "models" / "lgbm_lst.joblib")
    yf = available_features(yining, bundle["features"])
    yuse = yining.dropna(subset=yf + ["LST_C", "x", "y"]).copy()
    pred = bundle["model"].predict(yuse[yf])
    obs = yuse["LST_C"].to_numpy(dtype=float)
    yb_id = spatial_block_ids(yuse["x"], yuse["y"], 3000).to_numpy()
    uniq = np.unique(yb_id)
    rng_c = np.random.default_rng(9)
    few = []
    for n_cal in (20, 50, 100, 200, 500):
        reps = []
        for _ in range(12):
            order_b = rng_c.permutation(uniq)
            chosen_blocks = []
            cal_idx = []
            for b in order_b:
                members = np.flatnonzero(yb_id == b)
                if len(cal_idx) >= n_cal:
                    break
                need = n_cal - len(cal_idx)
                take = members if len(members) <= need else rng_c.choice(members, size=need, replace=False)
                cal_idx.extend(np.atleast_1d(take).tolist())
                chosen_blocks.append(b)
            cal_idx = np.asarray(cal_idx[:n_cal], dtype=int)
            test_blocks = [b for b in uniq if b not in set(chosen_blocks)]
            test_idx = np.flatnonzero(np.isin(yb_id, test_blocks))
            if len(test_idx) < 50:
                continue
            bias = float(np.mean(pred[cal_idx] - obs[cal_idx]))
            raw_m = extra(obs[test_idx], pred[test_idx])
            cal_m = extra(obs[test_idx], pred[test_idx] - bias)
            reps.append({"raw_R2": raw_m["R2"], "raw_RMSE": raw_m["RMSE"], "cal_R2": cal_m["R2"], "cal_RMSE": cal_m["RMSE"], "cal_Bias": cal_m["Bias"], "n_test": int(len(test_idx))})
        few.append({
            "n_cal": n_cal,
            "repeats": len(reps),
            "raw_R2_mean": float(np.mean([r["raw_R2"] for r in reps])),
            "cal_R2_mean": float(np.mean([r["cal_R2"] for r in reps])),
            "cal_R2_sd": float(np.std([r["cal_R2"] for r in reps])),
            "cal_RMSE_mean": float(np.mean([r["cal_RMSE"] for r in reps])),
            "raw_RMSE_mean": float(np.mean([r["raw_RMSE"] for r in reps])),
        })
        print("  n", n_cal, "cal R2", round(few[-1]["cal_R2_mean"], 3), "RMSE", round(few[-1]["cal_RMSE_mean"], 2), flush=True)

    # Held-out block intercept residuals for the map: reuse 5-fold, same as revision.
    groups = pd.Series(yb_id, index=yuse.index)
    cal = pred.copy()
    Xy = yuse[yf]
    for tr, va in spatial_group_kfold(Xy, yuse["LST_C"], groups, 5):
        bias = float(np.mean(pred[tr] - obs[tr]))
        cal[va] = pred[va] - bias
    raw_resid = pred - obs
    cal_resid = cal - obs

    payload = {
        "population": pop_audit,
        "no_terrain_full_precision": {
            "citywide_model_on_built": prev["domains"]["no_terrain_citywide_model_built_test"],
            "built_refit": prev["domains"]["no_terrain_built_model"],
            "note": "R2 and RMSE agree through the third and second reported digits. They are not the same prediction vector. Spearman and hot-fifth recall differ.",
        },
        "recall_bootstrap": {"citywide_model_on_built_rows": boot_city, "built_refit": boot_built},
        "learning_curve": curve,
        "variogram": {"sill": sill, "distance_m_at_95pct_sill": practical, "bins": variogram},
        "heteroscedasticity": hetero,
        "residential": {k: v for k, v in residential.items()},
        "siting": {k: v for k, v in siting.items() if k != "sites"},
        "siting_sites": siting["sites"],
        "few_shot": few,
        "equal_n_wording": "The equal-size comparison suggests that sample size alone does not explain the lower built-up performance; differences in predictor coverage and outcome variability also contribute.",
    }
    # freq arrays saved separately
    np.savez_compressed(
        RES / "revision2_maps.npz",
        lon=lon.astype(np.float32),
        lat=lat.astype(np.float32),
        city_top=city_top.astype(np.uint8),
        resid_top=resid_top.astype(np.uint8),
        universe=universe.astype(np.uint8),
        pop_pos=(pop > 0).astype(np.uint8),
        joint_freq=joint_freq.astype(np.float32),
        yining_lon=yuse.geometry.x.to_numpy(dtype=np.float32),
        yining_lat=yuse.geometry.y.to_numpy(dtype=np.float32),
        yining_raw_resid=raw_resid.astype(np.float32),
        yining_cal_resid=cal_resid.astype(np.float32),
    )
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved json", flush=True)
    make_figures(payload, lon, lat, city_top, resid_top, pop > 0, joint_freq, universe, yuse, raw_resid, cal_resid)
    print("saved figures", flush=True)


def make_figures(payload, lon, lat, city_top, resid_top, inhabited, freq, universe, yuse, raw_resid, cal_resid) -> None:
    import geopandas as gpd

    style()
    blue, verm, green, orange = "#0072B2", "#D55E00", "#009E73", "#E69F00"
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson")

    def add_boundary(ax):
        boundary.boundary.plot(ax=ax, color="#666666", linewidth=0.4)
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.tick_params(labelsize=7)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))
    panels = (
        (axes[0, 0], city_top & ~inhabited, city_top & inhabited, "A  Citywide top 20%", "No residents", "Has residents"),
        (axes[0, 1], None, resid_top, "B  Inhabited built-up top 20%", None, "Residential priority"),
    )
    ax = axes[0, 0]
    add_boundary(ax)
    ax.scatter(lon[city_top & ~inhabited], lat[city_top & ~inhabited], s=1, c=blue, marker="s", linewidths=0, rasterized=True, label="No residents")
    ax.scatter(lon[city_top & inhabited], lat[city_top & inhabited], s=3, c=verm, marker="s", linewidths=0, rasterized=True, label="Has residents")
    ax.legend(frameon=False, loc="lower left", markerscale=2)
    ax.set_title("A  Citywide top 20%")
    ax.set_ylabel("Latitude")
    ax = axes[0, 1]
    add_boundary(ax)
    ax.scatter(lon[resid_top], lat[resid_top], s=4, c=verm, marker="s", linewidths=0, rasterized=True)
    ax.set_title("B  Inhabited built-up top 20%")
    ax.set_xlim(87.3, 87.85)
    ax.set_ylim(43.65, 44.05)
    core = universe & (lon >= 87.35) & (lon <= 87.85) & (lat >= 43.65) & (lat <= 44.05)
    ax = axes[1, 0]
    sc = ax.scatter(lon[core], lat[core], c=freq[core], s=6, cmap="cividis", vmin=0, vmax=1, marker="s", linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Scenario frequency")
    ax.set_aspect("equal")
    ax.set_title("C  Residential scenario frequency")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax = axes[1, 1]
    add_boundary(ax)
    only_city = city_top & ~resid_top
    only_res = resid_top & ~city_top
    both = city_top & resid_top
    ax.scatter(lon[only_city & inhabited], lat[only_city & inhabited], s=2, c=blue, marker="s", linewidths=0, rasterized=True, label="Citywide only")
    ax.scatter(lon[only_res], lat[only_res], s=3, c=orange, marker="s", linewidths=0, rasterized=True, label="Residential only")
    ax.scatter(lon[both], lat[both], s=4, c=verm, marker="s", linewidths=0, rasterized=True, label="Both")
    ax.legend(frameon=False, loc="lower left", markerscale=2)
    ax.set_xlim(87.3, 87.85)
    ax.set_ylim(43.65, 44.05)
    ax.set_title("D  Where the two priority sets meet")
    ax.set_xlabel("Longitude")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_priority_stability.png", bbox_inches="tight")
    plt.close(fig)

    sites = pd.DataFrame(payload["siting_sites"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.8))
    # Regret is stored as summaries; draw a schematic from percentiles is weak.
    # Recompute is not available. Show the reported percentiles as a text-backed bar of median regret only if arrays missing.
    # The caller did not pass the regret vector. Read it back from a side channel: store was not in payload arrays.
    regret = np.asarray(payload["siting"].get("greedy_regret_values", []), dtype=float)
    if len(regret):
        axes[0].hist(regret, bins=12, color=verm, edgecolor="white")
        axes[0].axvline(0, color="#444444", lw=0.8)
        axes[0].axvline(float(np.median(regret)), color="black", lw=1.0, linestyle="--")
    axes[0].set_xlabel("Greedy re-solve minus fixed plan")
    axes[0].set_ylabel("Scenarios")
    axes[0].set_title("A  Regret of keeping the ten sites")
    base_sites = sites[sites.in_baseline].sort_values(["lat", "lon"]).copy()
    axes[1].scatter(base_sites.lon, base_sites.lat, s=36, c=base_sites.frequency, cmap="cividis", vmin=0, vmax=1, zorder=3)
    for i, (_, row) in enumerate(base_sites.iterrows()):
        label = "" if not isinstance(row["name"], str) else row["name"].split("（")[0].split("(")[0]
        axes[1].annotate(
            label,
            (row.lon, row.lat),
            textcoords="offset points",
            xytext=(6, 8 if i % 2 == 0 else -10),
            fontsize=6,
        )
    axes[1].set_title("B  Baseline sites if the plan is re-solved")
    axes[1].set_xlabel("Longitude")
    axes[1].set_ylabel("Latitude")
    axes[1].set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_siting.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 2 from the previous SHAP table, split so elevation does not hide the rest.
    sh = pd.DataFrame(prev_shap_rows()).head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(sh))
    for ax, cols, title in (
        (axes[0], ["citywide_model_all", "citywide_model_built_rows", "built_model"], "A  All features"),
        (axes[1], ["citywide_model_all", "citywide_model_built_rows", "built_model"], "B  Without the elevation bar"),
    ):
        pass
    show = sh.copy()
    axes[0].barh(y - 0.25, show.citywide_model_all, 0.24, color=blue, label="Citywide model, all rows")
    axes[0].barh(y, show.citywide_model_built_rows, 0.24, color=orange, label="Citywide model, built-up rows")
    axes[0].barh(y + 0.25, show.built_model, 0.24, color=green, label="Refit on built-up rows")
    axes[0].set_yticks(y, show.feature)
    axes[0].set_xlabel("Mean |TreeSHAP| (°C)")
    axes[0].set_title("A  Full scale")
    drop = show[show.feature != "elevation"]
    y2 = np.arange(len(drop))
    axes[1].barh(y2 - 0.25, drop.citywide_model_all, 0.24, color=blue)
    axes[1].barh(y2, drop.citywide_model_built_rows, 0.24, color=orange)
    axes[1].barh(y2 + 0.25, drop.built_model, 0.24, color=green)
    axes[1].set_yticks(y2, drop.feature)
    axes[1].set_xlabel("Mean |TreeSHAP| (°C)")
    axes[1].set_title("B  Elevation removed from the axis")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_shap_domains.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    ylon = yuse.geometry.x.to_numpy(dtype=float)
    ylat = yuse.geometry.y.to_numpy(dtype=float)
    # scatter uses a sample
    rng = np.random.default_rng(1)
    take = rng.choice(len(raw_resid), size=min(4000, len(raw_resid)), replace=False)
    obs = yuse["LST_C"].to_numpy(dtype=float)
    pred = obs + raw_resid
    axes[0, 0].scatter(obs[take], pred[take], s=4, alpha=0.25, c=blue, linewidths=0, rasterized=True)
    lims = [float(np.min(obs[take])), float(np.max(obs[take]))]
    axes[0, 0].plot(lims, lims, color="black", lw=0.8)
    axes[0, 0].set_xlabel("Observed LST (°C)")
    axes[0, 0].set_ylabel("Predicted LST (°C)")
    axes[0, 0].set_title("A  Uncalibrated transfer")
    few = payload["few_shot"]
    axes[0, 1].plot([r["n_cal"] for r in few], [r["cal_RMSE_mean"] for r in few], marker="o", color=verm, label="Held-out RMSE after intercept")
    axes[0, 1].plot([r["n_cal"] for r in few], [r["raw_RMSE_mean"] for r in few], marker="o", color=blue, label="Same points, no intercept")
    axes[0, 1].set_xlabel("Yining points used to estimate the intercept")
    axes[0, 1].set_ylabel("RMSE on other spatial blocks (°C)")
    axes[0, 1].legend(frameon=False, fontsize=6)
    axes[0, 1].set_title("B  Few-sample intercept")
    vmax = float(np.quantile(np.abs(raw_resid), 0.98))
    for ax, resid, title in (
        (axes[1, 0], raw_resid, "C  Uncalibrated residual"),
        (axes[1, 1], cal_resid, "D  After block intercept"),
    ):
        sc = ax.scatter(ylon, ylat, c=resid, s=2, cmap="coolwarm", vmin=-vmax, vmax=vmax, linewidths=0, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel("Longitude")
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Predicted − observed (°C)")
    axes[1, 0].set_ylabel("Latitude")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_transfer_aoa.png", bbox_inches="tight")
    plt.close(fig)


def prev_shap_rows():
    prev = json.loads((RES / "revision_summary.json").read_text(encoding="utf-8"))
    return prev["shap"]["rows"]


if __name__ == "__main__":
    main()
