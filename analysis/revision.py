"""Revision experiments requested by the manuscript review. Reads archived tables only."""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis import (  # noqa: E402
    CENSUS,
    FEATS_ALL,
    FIG,
    RES,
    coverage_matrix,
    covered_weight,
    cv_predict,
    greedy_mclp,
    jaccard_cols,
    mean_abs_shap,
    minmax,
    pack_metrics,
)
from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import spatial_block_ids  # noqa: E402
from src.geoai.train_geoai import available_features  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

CORE = {"天山区", "沙依巴克区", "新市区", "水磨沟区"}
TERRAIN = ["elevation", "slope", "ERA5_T2M"]


def extra(y, pred) -> dict:
    m = pack_metrics(y, pred)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sd = float(np.std(y))
    m["NRMSE"] = float(m["RMSE"] / sd) if sd else None
    m["Spearman"] = float(spearmanr(y, pred).statistic)
    thr_y = np.quantile(y, 0.80)
    thr_p = np.quantile(pred, 0.80)
    hot = y >= thr_y
    m["top20_recall"] = float(((pred >= thr_p) & hot).sum() / max(hot.sum(), 1))
    m["y_sd"] = sd
    return m


def pitch(lon, lat) -> dict:
    def step(v, floor):
        u = np.sort(np.unique(np.round(v, 5)))
        d = np.diff(u)
        d = d[d > floor]
        return float(np.median(d)) if len(d) else None

    dlon = step(lon, 1e-5)
    dlat = step(lat, 1e-5)
    lat0 = float(np.mean(lat))
    m_per_deg_lat = 110540.0
    m_per_deg_lon = 111320.0 * np.cos(np.deg2rad(lat0))
    return {
        "dlat_deg": dlat,
        "dlon_deg": dlon,
        "ns_m": None if dlat is None else dlat * m_per_deg_lat,
        "ew_m": None if dlon is None else dlon * m_per_deg_lon,
        "footprint_km2": None
        if dlat is None or dlon is None
        else dlat * m_per_deg_lat * dlon * m_per_deg_lon / 1e6,
    }


def top_mask(score: np.ndarray, q: float = 0.80) -> np.ndarray:
    return score >= np.quantile(score, q)


def census_people(pop, mask) -> float:
    return float((pop * mask).sum() / max(float(pop.sum()), 1e-9) * CENSUS)


def compose(pop, built, mask) -> dict:
    m = mask.astype(bool)
    return {
        "n": int(m.sum()),
        "share_of_cells": float(m.mean()),
        "unpopulated_share_of_set": float((pop[m] <= 0).mean()) if m.any() else None,
        "built_share_of_set": float(built[m].mean()) if m.any() else None,
        "people_ghsl": census_people(pop, m),
        "people_in_built_cells": census_people(pop, m & built),
        "people_in_unbuilt_cells": census_people(pop, m & ~built),
    }


def main() -> None:
    import geopandas as gpd
    import joblib

    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    yining = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Yiningshi_2024.parquet")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    cent = work.geometry.centroid
    work["lon"] = cent.x.to_numpy()
    work["lat"] = cent.y.to_numpy()
    built = work["built"].to_numpy(dtype=float) >= 0.15

    # --- extent ---
    print("extent", flush=True)
    extent = {}
    for city, frame, scale, boundary_name in [
        ("Wulumuqishi", grid, 400, "boundary_Wulumuqishi_union.geojson"),
        ("Yiningshi", yining, 200, "boundary_Yiningshi_union.geojson"),
    ]:
        b = gpd.read_file(ROOT / "data" / "raw" / boundary_name)
        if b.crs is None:
            b.set_crs(4326, inplace=True)
        admin = float(b.to_crs(32645).geometry.union_all().area / 1e6)
        lon = frame.geometry.centroid.x.to_numpy()
        lat = frame.geometry.centroid.y.to_numpy()
        p = pitch(lon, lat)
        foot = p["footprint_km2"] or (scale / 1000) ** 2
        extent[city] = {
            "n": int(len(frame)),
            "nominal_scale_m": scale,
            "admin_area_km2": admin,
            "pitch": p,
            "naive_square_area_km2": len(frame) * (scale / 1000) ** 2,
            "footprint_sum_km2": len(frame) * foot,
            "footprint_over_admin": (len(frame) * foot) / admin,
        }
        print(city, "admin", round(admin, 1), "pitch", round(p["ew_m"] or -1, 1), round(p["ns_m"] or -1, 1),
              "foot/admin", round(extent[city]["footprint_over_admin"], 3), flush=True)

    # --- fair domain comparison ---
    print("citywide spatial CV", flush=True)
    city_met, oof, models, val_idx, y = cv_predict(work, feats, "spatial", 3000, "lgbm")
    resid = oof - y
    domains = {
        "citywide_model_citywide_test": extra(y, oof),
        "citywide_model_built_test": extra(y[built], oof[built]),
    }
    print("  built-test R2", round(domains["citywide_model_built_test"]["R2"], 3), flush=True)

    cells = grid[["district", "geometry"]].copy()
    joined = gpd.sjoin_nearest(work[["geometry"]], cells, how="left", distance_col="d_deg")
    joined = joined[~joined.index.duplicated(keep="first")]
    district = joined["district"].reindex(work.index).astype(str)
    core = district.isin(CORE).to_numpy()
    domains["citywide_model_core4_test"] = extra(y[core], oof[core])
    print("  core-test R2", round(domains["citywide_model_core4_test"]["R2"], 3), "n", int(core.sum()), flush=True)

    print("built refit", flush=True)
    built_frame = work.loc[built]
    built_met, built_oof, built_models, built_val, built_y = cv_predict(built_frame, feats, "spatial", 3000, "lgbm")
    domains["built_model_built_test"] = extra(built_y, built_oof)

    no_terrain = [f for f in feats if f not in TERRAIN]
    print("no terrain", flush=True)
    nt_met, nt_oof, _, _, nt_y = cv_predict(work, no_terrain, "spatial", 3000, "lgbm")
    domains["no_terrain_citywide"] = extra(nt_y, nt_oof)
    domains["no_terrain_citywide_model_built_test"] = extra(nt_y[built], nt_oof[built])
    nt_b_met, nt_b_oof, _, _, nt_b_y = cv_predict(work.loc[built], no_terrain, "spatial", 3000, "lgbm")
    domains["no_terrain_built_model"] = extra(nt_b_y, nt_b_oof)
    print("  no-terrain city R2", round(nt_met["R2"], 3), "built-test", round(domains["no_terrain_citywide_model_built_test"]["R2"], 3), flush=True)

    print("equal-n control", flush=True)
    rng = np.random.default_rng(7)
    equal = []
    for i in range(8):
        take = rng.choice(len(work), size=int(built.sum()), replace=False)
        met, *_ = cv_predict(work.iloc[take], feats, "spatial", 3000, "lgbm", seed=10 + i)
        equal.append(met["R2"])
        print(f"  draw {i+1} R2 {met['R2']:.3f}", flush=True)
    equal_n = {
        "n": int(built.sum()),
        "repeats": 8,
        "r2_mean": float(np.mean(equal)),
        "r2_sd": float(np.std(equal)),
        "r2_min": float(np.min(equal)),
        "r2_max": float(np.max(equal)),
        "values": [float(v) for v in equal],
    }

    print("SHAP built refit", flush=True)
    shap_built_model = mean_abs_shap(built_models, built_val, built_frame[feats])
    shap_rows = []
    # Citywide-model SHAP on all rows and on built rows, reusing fold models.
    import shap

    acc_all = np.zeros(len(feats))
    acc_built = np.zeros(len(feats))
    n_all = n_b = 0
    X = work[feats]
    for model, va in zip(models, val_idx):
        explainer = shap.TreeExplainer(model)
        values = np.asarray(explainer.shap_values(X.iloc[va]), dtype=float)
        acc_all += np.abs(values).sum(axis=0)
        n_all += len(va)
        take = built[va]
        if take.sum():
            acc_built += np.abs(values[take]).sum(axis=0)
            n_b += int(take.sum())
    shap_all = acc_all / n_all
    shap_on_built = acc_built / n_b
    for i, f in enumerate(feats):
        shap_rows.append({
            "feature": f,
            "citywide_model_all": float(shap_all[i]),
            "citywide_model_built_rows": float(shap_on_built[i]),
            "built_model": float(shap_built_model[i]),
        })
    shap_rows.sort(key=lambda r: r["citywide_model_all"], reverse=True)

    def share(imp, names):
        idx = [feats.index(n) for n in names]
        return float(np.sum(imp[idx]) / np.sum(imp))

    shap_summary = {
        "terrain_share_citywide_model_all": share(shap_all, TERRAIN),
        "terrain_share_citywide_model_built_rows": share(shap_on_built, TERRAIN),
        "terrain_share_built_model": share(shap_built_model, TERRAIN),
        "rows": shap_rows,
    }
    print("  terrain shares", round(shap_summary["terrain_share_citywide_model_all"], 3),
          round(shap_summary["terrain_share_citywide_model_built_rows"], 3),
          round(shap_summary["terrain_share_built_model"], 3), flush=True)

    # --- grid index, error models, composition ---
    print("error models", flush=True)
    g = grid
    lon_g = g.geometry.centroid.x.to_numpy(dtype=float)
    lat_g = g.geometry.centroid.y.to_numpy(dtype=float)
    lst = g["LST_pred"].to_numpy(dtype=float)
    p90 = g["LST_P90"].to_numpy(dtype=float)
    pop = g["population"].to_numpy(dtype=float)
    pop_w = g["worldpop"].to_numpy(dtype=float)
    built_g = g["built"].to_numpy(dtype=float) >= 0.15
    cool_lack = 1.0 - g["CoolingCapacity"].to_numpy(dtype=float)
    serv_lack = 1.0 - g["ServiceAdaptation"].to_numpy(dtype=float)
    gap = np.clip(g["Gap"].to_numpy(dtype=float), 0, 1)
    elev = g["elevation"].to_numpy(dtype=float)
    base_top = g["UTHI"].to_numpy(dtype=float) >= np.quantile(g["UTHI"].to_numpy(dtype=float), 0.80)
    p90_n = minmax(p90)
    expo = minmax(pop)

    def thermal_from(lst_values, use_p90=True, ref=None):
        def norm(v):
            if ref is None:
                return minmax(v)
            lo = np.nanmin(v[ref]) if np.isfinite(v[ref]).any() else np.nanmin(v)
            hi = np.nanmax(v[ref]) if np.isfinite(v[ref]).any() else np.nanmax(v)
            if hi <= lo:
                return np.zeros_like(v)
            return np.clip((v - lo) / (hi - lo), 0, 1)
        t = norm(lst_values)
        if use_p90:
            t = 0.7 * t + 0.3 * norm(p90)
        return t

    def score_of(thermal, exposure):
        risk = np.column_stack([thermal, exposure, cool_lack, serv_lack])
        return risk @ np.full(4, 0.25)

    # HPI specification, no temperature error.
    variants = {}
    base_score = score_of(thermal_from(lst), expo)
    variants["stored_components"] = float(jaccard_cols(base_top[:, None], top_mask(g["UTHI"].to_numpy(dtype=float)))[0])
    specs = {
        "dynamic_with_p90": thermal_from(lst, True),
        "no_p90": thermal_from(lst, False),
        "winsor_2_98": thermal_from(np.clip(lst, *np.quantile(lst, [0.02, 0.98])), True),
        "built_reference_minmax": thermal_from(lst, True, ref=built_g),
    }
    hpi_sets = {}
    for name, thermal in specs.items():
        sc = score_of(thermal, expo)
        hpi_sets[name] = top_mask(sc)
        variants[name] = {
            "jaccard_vs_dynamic_with_p90": float(jaccard_cols(hpi_sets[name][:, None], hpi_sets["dynamic_with_p90"])[0])
            if name != "dynamic_with_p90" else 1.0,
            "people": census_people(pop, hpi_sets[name]),
            **compose(pop, built_g, hpi_sets[name]),
        }
    # Top-k thresholds on the dynamic score.
    dyn = score_of(specs["dynamic_with_p90"], expo)
    for q, label in [(0.70, "top30"), (0.80, "top20"), (0.90, "top10")]:
        variants[label] = compose(pop, built_g, top_mask(dyn, q))

    # Priority recomputed inside built-up cells only.
    built_scores = dyn[built_g]
    built_only_top = np.zeros(len(dyn), dtype=bool)
    built_only_top[np.flatnonzero(built_g)[built_scores >= np.quantile(built_scores, 0.80)]] = True
    variants["top20_within_built_only"] = {
        **compose(pop, built_g, built_only_top),
        "jaccard_vs_citywide_top20": float(jaccard_cols(built_only_top[:, None], hpi_sets["dynamic_with_p90"])[0]),
    }

    tr_block = spatial_block_ids(work["x"], work["y"], 3000).to_numpy()
    resid_by = {}
    for b, r, is_b, el in zip(tr_block, resid, built, work["elevation"].to_numpy(dtype=float)):
        resid_by.setdefault(b, []).append(float(r))
    resid_built = resid[built]
    resid_open = resid[~built]
    elev_cuts = np.quantile(work["elevation"].to_numpy(dtype=float), [1 / 3, 2 / 3])

    def elev_bin(v):
        return np.digitize(v, elev_cuts)

    pools_strat = {}
    for is_b, eb, r in zip(built, elev_bin(work["elevation"].to_numpy(dtype=float)), resid):
        pools_strat.setdefault((bool(is_b), int(eb)), []).append(float(r))
    g_block = spatial_block_ids(g["x"], g["y"], 3000).to_numpy()
    codes, uniques = pd.factorize(g_block)
    block_built = np.zeros(len(uniques), dtype=bool)
    for code, flag in zip(codes, built_g):
        if flag:
            block_built[code] = True
    block_pool = []
    for b, is_b in zip(uniques, block_built):
        pool = resid_by.get(b) or (resid_built if is_b else resid_open)
        block_pool.append(np.asarray(pool, dtype=float))
    g_elev_bin = elev_bin(elev)

    def run_fields(kind: str, n_fields: int, thermal_fn) -> dict:
        rng_e = np.random.default_rng(11)
        hits = np.zeros(len(g), dtype=np.int32)
        jacs = []
        people = []
        base = hpi_sets["dynamic_with_p90"] if thermal_fn is thermal_from else None
        own_base = top_mask(score_of(thermal_fn(lst), expo))
        compare = own_base
        for i in range(n_fields):
            if kind == "block":
                delta = np.array([rng_e.choice(pool) for pool in block_pool])[codes]
            elif kind == "iid":
                delta = rng_e.choice(resid_open, size=len(g))
                n_built = int(built_g.sum())
                if n_built:
                    delta = delta.copy()
                    delta[built_g] = rng_e.choice(resid_built, size=n_built)
            else:
                delta = np.empty(len(g), dtype=float)
                for is_b in (False, True):
                    for eb in (0, 1, 2):
                        sel = (built_g == is_b) & (g_elev_bin == eb)
                        n_sel = int(sel.sum())
                        if n_sel == 0:
                            continue
                        pool = pools_strat.get((is_b, eb)) or (resid_built if is_b else resid_open)
                        delta[sel] = rng_e.choice(np.asarray(pool, dtype=float), size=n_sel)
            sc = score_of(thermal_fn(lst + delta), expo)
            mask = top_mask(sc)
            hits += mask
            jacs.append(float(jaccard_cols(mask[:, None], compare)[0]))
            people.append(census_people(pop, mask))
        freq = hits / n_fields
        stable = freq >= 0.90
        return {
            "kind": kind,
            "n_fields": n_fields,
            "jaccard_mean": float(np.mean(jacs)),
            "jaccard_p05": float(np.quantile(jacs, 0.05)),
            "jaccard_p50": float(np.quantile(jacs, 0.50)),
            "people_p05": float(np.quantile(people, 0.05)),
            "people_p50": float(np.quantile(people, 0.50)),
            "people_p95": float(np.quantile(people, 0.95)),
            "baseline_still_ge_90": float(stable[compare].mean()) if compare.any() else None,
            "stable90": compose(pop, built_g, stable),
        }

    # 40 fields is enough to compare error models; block-25 matches the manuscript design.
    error_models = {
        "block_25": run_fields("block", 25, lambda v: thermal_from(v, True)),
        "block_40": run_fields("block", 40, lambda v: thermal_from(v, True)),
        "iid_40": run_fields("iid", 40, lambda v: thermal_from(v, True)),
        "stratified_40": run_fields("stratified", 40, lambda v: thermal_from(v, True)),
        "block_25_no_p90": run_fields("block", 25, lambda v: thermal_from(v, False)),
    }
    for key, rec in error_models.items():
        print(f"  {key} Jaccard {rec['jaccard_mean']:.3f}", flush=True)

    # Composition of the archived joint-frequency surface, if present.
    freq_path = RES / "priority_probability.npz"
    composition = {"baseline_top20": compose(pop, built_g, base_top)}
    if freq_path.exists():
        z = np.load(freq_path)
        p_top = z["p_top"]
        if len(p_top) == len(g):
            stable = p_top >= 0.90
            composition["joint_stable90"] = compose(pop, built_g, stable)
            north = lat_g >= 44.4
            composition["joint_stable90_north_of_44_4"] = compose(pop, built_g, stable & north)
            composition["joint_stable90_south_of_44_4"] = compose(pop, built_g, stable & ~north)
            composition["baseline_not_stable"] = compose(pop, built_g, base_top & ~stable)
            composition["stable_not_baseline"] = compose(pop, built_g, stable & ~base_top)

    # --- siting: fixed plan versus re-optimization ---
    print("siting", flush=True)
    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    names = {x.lower() for x in RETROFIT}
    cand = fac[fac["fclass"].astype(str).str.lower().isin(names)].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])
    xy = np.column_stack([g["x"].to_numpy(dtype=float), g["y"].to_numpy(dtype=float)])
    demand_factor = 0.4 + 0.6 * gap

    def solve(mask, population):
        demand_w = population * mask * demand_factor
        order = np.argsort(-demand_w)[:450]
        weights_d = demand_w[order]
        cov = coverage_matrix(xy[order], cand_xy, 1125.0)
        chosen = greedy_mclp(weights_d, cov, 10)
        total = float(weights_d.sum())
        share = covered_weight(weights_d, cov, chosen) / max(total, 1e-9)
        return chosen, share, order, weights_d, cov

    base_chosen, base_share, base_order, base_w, base_cov = solve(base_top, pop)
    # How much of the priority population sits in the 450 demand cells.
    demand_pop_share = float(pop[base_order].sum() / max(pop[base_top].sum(), 1e-9))
    # Coverage of all populated priority cells, not only the 450.
    pri_idx = np.flatnonzero(base_top & (pop > 0))
    pri_cov = coverage_matrix(xy[pri_idx], cand_xy, 1125.0)
    pri_w = pop[pri_idx]
    citywide_cover_fixed = covered_weight(pri_w, pri_cov, base_chosen) / max(float(pri_w.sum()), 1e-9)

    rng_s = np.random.default_rng(11)
    # This 25-field ensemble is its own seed stream. Site identity below belongs to it.
    fixed_shares = []
    reopt_shares = []
    regrets = []
    fixed_city_shares = []
    hit = np.zeros(len(cand), dtype=int)
    for i in range(25):
        delta = np.array([rng_s.choice(pool) for pool in block_pool])[codes]
        for population in (pop, pop_w):
            thermal = thermal_from(lst + delta, True)
            mask = top_mask(score_of(thermal, minmax(population)))
            chosen, share, order, weights_d, cov = solve(mask, population)
            # base_chosen indexes candidates. cov columns are those same candidates.
            fixed = covered_weight(weights_d, cov, base_chosen) / max(float(weights_d.sum()), 1e-9)
            hit[np.asarray(chosen, dtype=int)] += 1
            fixed_shares.append(fixed)
            reopt_shares.append(share)
            regrets.append(share - fixed)
            pri = np.flatnonzero(mask & (population > 0))
            if len(pri) == 0:
                fixed_city_shares.append(0.0)
            else:
                cov_all = coverage_matrix(xy[pri], cand_xy, 1125.0)
                fixed_city_shares.append(covered_weight(population[pri], cov_all, base_chosen) / max(float(population[pri].sum()), 1e-9))
    fixed_shares = np.asarray(fixed_shares)
    reopt_shares = np.asarray(reopt_shares)
    regrets = np.asarray(regrets)
    fixed_city_shares = np.asarray(fixed_city_shares)
    siting = {
        "baseline_reopt_share_of_450": base_share,
        "demand450_share_of_baseline_priority_population": demand_pop_share,
        "baseline_fixed_share_of_all_populated_priority_cells": float(citywide_cover_fixed),
        "n_populated_priority_cells": int((base_top & (pop > 0)).sum()),
        "n_priority_cells": int(base_top.sum()),
        "scenario_n": 50,
        "reopt_share_p05": float(np.quantile(reopt_shares, 0.05)),
        "reopt_share_p50": float(np.quantile(reopt_shares, 0.50)),
        "reopt_share_p95": float(np.quantile(reopt_shares, 0.95)),
        "fixed_share_p05": float(np.quantile(fixed_shares, 0.05)),
        "fixed_share_p50": float(np.quantile(fixed_shares, 0.50)),
        "fixed_share_p95": float(np.quantile(fixed_shares, 0.95)),
        "regret_p50": float(np.quantile(regrets, 0.50)),
        "regret_p95": float(np.quantile(regrets, 0.95)),
        "fixed_citywide_pop_cover_p05": float(np.quantile(fixed_city_shares, 0.05)),
        "fixed_citywide_pop_cover_p50": float(np.quantile(fixed_city_shares, 0.50)),
        "fixed_citywide_pop_cover_p95": float(np.quantile(fixed_city_shares, 0.95)),
        "baseline_sites_always_selected": int(sum(hit[i] == 50 for i in base_chosen)),
        "baseline_sites_ge_80": int(sum(hit[i] >= 40 for i in base_chosen)),
        "n_sites_ever": int((hit > 0).sum()),
        "reopt_sites": [
            {
                "name": None if pd.isna(cand.iloc[i]["name"]) else str(cand.iloc[i]["name"]),
                "fclass": str(cand.iloc[i]["fclass"]),
                "frequency": float(hit[i] / 50),
                "in_baseline": bool(i in set(base_chosen)),
            }
            for i in np.argsort(-hit) if hit[i] > 0
        ],
        "note": "Shares of the 450-cell objective are not citywide population coverage. fixed_citywide is the share of populated priority-cell population within 1,125 m of the baseline ten sites. Greedy is exact on the baseline instance and is not claimed to be exact in every scenario.",
    }
    print("  demand pop share", round(demand_pop_share, 3), "fixed p50", round(siting["fixed_share_p50"], 3),
          "reopt p50", round(siting["reopt_share_p50"], 3), "city cover", round(citywide_cover_fixed, 3), flush=True)

    # --- transfer calibration ---
    print("transfer", flush=True)
    bundle = joblib.load(ROOT / "outputs" / "models" / "lgbm_lst.joblib")
    yf = available_features(yining, bundle["features"])
    yuse = yining.dropna(subset=yf + ["LST_C", "x", "y"]).copy()
    pred = bundle["model"].predict(yuse[yf])
    obs = yuse["LST_C"].to_numpy(dtype=float)
    raw = extra(obs, pred)
    groups = spatial_block_ids(yuse["x"], yuse["y"], 3000)
    from src.geoai.spatial_cv import spatial_group_kfold

    cal = np.full(len(yuse), np.nan)
    Xy = yuse[yf]
    for tr, va in spatial_group_kfold(Xy, yuse["LST_C"], groups, 5):
        bias = float(np.mean(pred[tr] - obs[tr]))
        cal[va] = pred[va] - bias
    calibrated = extra(obs, cal)
    # Local model on the Yining grid.
    local_met, local_oof, _, _, local_y = cv_predict(yuse, list(yf), "spatial", 3000, "lgbm")
    transfer = {
        "raw": raw,
        "centered_rmse": float(np.sqrt(max(raw["RMSE"] ** 2 - raw["Bias"] ** 2, 0.0))),
        "block_intercept": calibrated,
        "local_model": local_met,
        "archived_local_sample_r2": 0.7933979598251224,
        "archived_local_sample_n": 8859,
    }
    print("  raw R2", round(raw["R2"], 3), "cal R2", round(calibrated["R2"], 3), "local", round(local_met["R2"], 3), flush=True)

    # Scatter sample
    rng_p = np.random.default_rng(1)
    take = rng_p.choice(len(obs), size=min(4000, len(obs)), replace=False)
    scatter = {"obs": obs[take], "pred": pred[take], "cal": cal[take]}

    out = {
        "extent": extent,
        "domains": domains,
        "equal_n": equal_n,
        "shap": shap_summary,
        "hpi_variants": variants,
        "error_models": error_models,
        "composition": composition,
        "siting": siting,
        "transfer": transfer,
    }
    (RES / "revision_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    make_figures(out, lon_g, lat_g, base_top, built_g, pop, scatter, composition)
    print("saved", RES / "revision_summary.json", flush=True)


def make_figures(out, lon, lat, base_top, built_g, pop, scatter, composition) -> None:
    from matplotlib import font_manager
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
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 200,
    })
    blue, orange, green, verm = "#0072B2", "#E69F00", "#009E73", "#D55E00"
    d = out["domains"]

    def bar_pair(ax, labels, series, ylabel, title):
        x = np.arange(len(labels))
        w = 0.8 / len(series)
        for i, (name, vals, color) in enumerate(series):
            ax.bar(x + (i - (len(series) - 1) / 2) * w, vals, w, label=name, color=color)
        ax.set_xticks(x, labels)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=7)

    labels = ["Citywide", "Built-up\nrows", "Four\ndistricts"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    r2s = [
        d["citywide_model_citywide_test"]["R2"],
        d["citywide_model_built_test"]["R2"],
        d["citywide_model_core4_test"]["R2"],
    ]
    refit = [np.nan, d["built_model_built_test"]["R2"], np.nan]
    nrmse = [
        d["citywide_model_citywide_test"]["NRMSE"],
        d["citywide_model_built_test"]["NRMSE"],
        d["citywide_model_core4_test"]["NRMSE"],
    ]
    bar_pair(axes[0], labels, [("Citywide model", r2s, blue), ("Refit in domain", refit, orange)], "Out-of-fold R²", "A  Same model, different test rows")
    axes[0].set_ylim(0, 1)
    axes[1].bar(np.arange(3), nrmse, color=blue)
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylabel("RMSE / SD of observed LST")
    axes[1].set_title("B  Scale-free error")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_domain_accuracy.png")
    plt.close(fig)

    sh = pd.DataFrame(out["shap"]["rows"]).head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    y = np.arange(len(sh))
    ax.barh(y - 0.25, sh.citywide_model_all, 0.24, color=blue, label="Citywide model, all rows")
    ax.barh(y, sh.citywide_model_built_rows, 0.24, color=orange, label="Citywide model, built-up rows")
    ax.barh(y + 0.25, sh.built_model, 0.24, color=green, label="Model refit on built-up rows")
    ax.set_yticks(y, sh.feature)
    ax.set_xlabel("Mean |TreeSHAP| (°C)")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_shap_domains.png")
    plt.close(fig)

    # Figure 3: empty versus inhabited priority cells, then scenario frequency.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    core = (lon >= 87.48) & (lon <= 87.75) & (lat >= 43.72) & (lat <= 43.98)
    inhabited = pop > 0
    axes[0].scatter(lon[base_top & ~inhabited], lat[base_top & ~inhabited], s=2, c=blue, marker="s", linewidths=0, rasterized=True, label="No residents")
    axes[0].scatter(lon[base_top & inhabited], lat[base_top & inhabited], s=4, c=verm, marker="s", linewidths=0, rasterized=True, label="Has residents")
    axes[0].legend(frameon=False, loc="lower left", markerscale=2, fontsize=7)
    axes[0].set_title("A  Baseline priority cells")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")
    if "joint_stable90" in composition:
        z = np.load(RES / "priority_probability.npz")
        p = z["p_top"]
        sc = axes[1].scatter(lon[core], lat[core], c=p[core], s=8, cmap="cividis", vmin=0, vmax=1, marker="s", linewidths=0, rasterized=True)
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="Scenario frequency")
    axes[1].set_title("B  Core window frequency")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Longitude")
    fig.savefig(FIG / "fig3_priority_stability.png", bbox_inches="tight")
    plt.close(fig)

    s = out["siting"]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    base_sites = pd.DataFrame([r for r in s["reopt_sites"] if r["in_baseline"]]).sort_values("frequency")
    axes[0].bar(
        [0, 1],
        [s["fixed_share_p50"], s["reopt_share_p50"]],
        color=[verm, blue],
        width=0.6,
    )
    axes[0].errorbar(
        [0, 1],
        [s["fixed_share_p50"], s["reopt_share_p50"]],
        yerr=[
            [s["fixed_share_p50"] - s["fixed_share_p05"], s["reopt_share_p50"] - s["reopt_share_p05"]],
            [s["fixed_share_p95"] - s["fixed_share_p50"], s["reopt_share_p95"] - s["reopt_share_p50"]],
        ],
        fmt="none",
        ecolor="black",
        capsize=3,
    )
    axes[0].set_xticks([0, 1], ["Fixed baseline\nten sites", "Re-solved\nin each scenario"])
    axes[0].set_ylabel("Share of the 450-cell objective")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("A  Coverage under temperature error")
    labels = []
    for name, fclass in zip(base_sites["name"], base_sites.fclass):
        text = "" if pd.isna(name) else str(name).split("（")[0].split("(")[0]
        labels.append(text if text and text != "None" else str(fclass))
    axes[1].barh(np.arange(len(base_sites)), base_sites.frequency, color=green)
    axes[1].set_yticks(np.arange(len(base_sites)), labels, fontsize=7)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Selection frequency if re-solved")
    axes[1].set_title("B  Baseline sites")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_siting.png")
    plt.close(fig)

    t = out["transfer"]
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.6))
    axes[0].scatter(scatter["obs"], scatter["pred"], s=4, alpha=0.25, c=blue, linewidths=0, rasterized=True, label="Uncalibrated")
    lims = [min(scatter["obs"].min(), scatter["pred"].min()), max(scatter["obs"].max(), scatter["pred"].max())]
    axes[0].plot(lims, lims, color="black", lw=0.8)
    axes[0].set_xlabel("Observed LST (°C)")
    axes[0].set_ylabel("Predicted LST (°C)")
    axes[0].set_title("A  Urumqi model in Yining")
    names = ["Uncalibrated", "Block intercept", "Local model"]
    rmses = [t["raw"]["RMSE"], t["block_intercept"]["RMSE"], t["local_model"]["RMSE"]]
    biases = [t["raw"]["Bias"], t["block_intercept"]["Bias"], t["local_model"]["Bias"]]
    x = np.arange(3)
    axes[1].bar(x - 0.16, rmses, 0.32, color=blue, label="RMSE")
    axes[1].bar(x + 0.16, biases, 0.32, color=orange, label="Bias")
    axes[1].set_xticks(x, names, fontsize=7)
    axes[1].set_ylabel("°C")
    axes[1].legend(frameon=False)
    axes[1].set_title("B  Held-out calibration")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_transfer_aoa.png")
    plt.close(fig)
    _ = pop


if __name__ == "__main__":
    main()
