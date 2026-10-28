"""Decision-audit analyses. Reads the archived tables and writes only under analysis/.

Does not call any function that saves maps, models, or competition tables.
"""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import regression_metrics, spatial_block_ids, spatial_group_kfold  # noqa: E402
from src.geoai.train_geoai import available_features, make_model  # noqa: E402
from src.gis.mclp import coverage_matrix, covered_weight, greedy_mclp  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

PAPER = Path(__file__).resolve().parent
RES = PAPER / "results"
FIG = PAPER / "figures"
CENSUS = 4_054_369
CORE_LON = (87.48, 87.75)
CORE_LAT = (43.72, 43.98)
FEATS_ALL = [
    "B2", "B3", "B4", "B8", "B11", "B12", "NDVI", "NDBI", "NDWI", "BSI", "ALBEDO",
    "VV", "VH", "VV_minus_VH", "built", "trees", "grass", "water", "built_height",
    "elevation", "slope", "ERA5_T2M", "dist_water", "local_green",
]
WATCH = [
    ROOT / "outputs" / "tables" / "master_results.json",
    ROOT / "outputs" / "models" / "lgbm_lst.joblib",
    ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet",
    ROOT / "data" / "processed" / "heatcheck_train.parquet",
    ROOT / "report" / "正式稿" / "技术方案.md",
]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jaccard_cols(top: np.ndarray, base: np.ndarray) -> np.ndarray:
    base_col = base[:, None]
    inter = np.logical_and(top, base_col).sum(axis=0)
    union = np.logical_or(top, base_col).sum(axis=0)
    return inter / np.maximum(union, 1)


def minmax(x: np.ndarray) -> np.ndarray:
    lo = np.nanmin(x)
    hi = np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=float)
    out = (x - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~np.isfinite(out)] = 0.0
    return out


def pack_metrics(y, pred) -> dict:
    m = regression_metrics(y, pred)
    return {k: (int(v) if k == "n" else float(v)) for k, v in m.items()}


def cv_predict(df: pd.DataFrame, feats: list[str], mode: str, block_m: float, learner: str, seed: int = 42):
    X = df[feats]
    y = df["LST_C"].to_numpy(dtype=float)
    if mode == "spatial":
        groups = spatial_block_ids(df["x"], df["y"], block_m)
        splits = list(spatial_group_kfold(X, df["LST_C"], groups, 5))
    else:
        splits = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))
    oof = np.full(len(df), np.nan)
    models = []
    val_idx = []
    settings = get_settings()
    for tr, va in splits:
        if learner == "lgbm":
            model = make_model(settings)
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
        elif learner == "rf":
            model = RandomForestRegressor(
                n_estimators=200, min_samples_leaf=5, max_features="sqrt",
                random_state=seed, n_jobs=-1,
            )
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
        else:
            model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
        oof[va] = pred
        models.append(model)
        val_idx.append(va)
    met = pack_metrics(y, oof)
    met["n_splits"] = len(splits)
    met["mode"] = mode
    met["learner"] = learner
    met["block_m"] = None if mode == "random" else float(block_m)
    met["y_sd"] = float(np.std(y))
    return met, oof, models, val_idx, y


def mean_abs_shap(models, val_idx, X: pd.DataFrame) -> np.ndarray:
    import shap

    acc = np.zeros(X.shape[1], dtype=float)
    n = 0
    for model, va in zip(models, val_idx):
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X.iloc[va])
        if isinstance(values, list):
            values = values[0]
        acc += np.abs(np.asarray(values, dtype=float)).sum(axis=0)
        n += len(va)
    return acc / max(n, 1)


def aoi_threshold(di_train: np.ndarray) -> float:
    q1, q3 = np.quantile(di_train, [0.25, 0.75])
    fence = q3 + 1.5 * (q3 - q1)
    kept = di_train[di_train <= fence]
    return float(kept.max() if len(kept) else di_train.max())


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    before = {str(p.relative_to(ROOT)): sha(p) for p in WATCH}
    (RES / "watched_hashes_before.json").write_text(json.dumps(before, indent=2), encoding="utf-8")

    import geopandas as gpd

    print("loading tables", flush=True)
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    yining = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Yiningshi_2024.parquet")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    assert feats == FEATS_ALL

    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    cent = work.geometry.to_crs(4326).centroid
    work["lon"] = cent.x.to_numpy()
    work["lat"] = cent.y.to_numpy()
    built_mask = work["built"].to_numpy(dtype=float) >= 0.15
    core_mask = work["lon"].between(*CORE_LON).to_numpy() & work["lat"].between(*CORE_LAT).to_numpy()
    print(f"train {len(work)} built {built_mask.sum()} core {core_mask.sum()}", flush=True)

    # --- RQ1: learners and validation design ---
    print("cross-validation", flush=True)
    specs = [
        ("citywide", "spatial", "lgbm", 3000, work),
        ("citywide", "random", "lgbm", 3000, work),
        ("citywide", "spatial", "rf", 3000, work),
        ("citywide", "spatial", "ridge", 3000, work),
        ("citywide", "spatial", "lgbm", 2000, work),
        ("citywide", "spatial", "lgbm", 5000, work),
        ("built", "spatial", "lgbm", 3000, work.loc[built_mask]),
        ("built", "random", "lgbm", 3000, work.loc[built_mask]),
        ("built", "spatial", "rf", 3000, work.loc[built_mask]),
        ("built", "spatial", "ridge", 3000, work.loc[built_mask]),
    ]
    cv_rows = []
    city_pack = None
    for domain, mode, learner, block, frame in specs:
        print(f"  {domain} {mode} {learner} {block}", flush=True)
        met, oof, models, val_idx, y = cv_predict(frame, feats, mode, block, learner)
        row = {"domain": domain, **met}
        cv_rows.append(row)
        print(f"    R2 {met['R2']:.4f} RMSE {met['RMSE']:.3f}", flush=True)
        if domain == "citywide" and mode == "spatial" and learner == "lgbm" and block == 3000:
            city_pack = (frame, oof, models, val_idx, y)

    frame, oof, models, val_idx, y = city_pack
    resid = oof - y
    print("SHAP on out-of-fold rows", flush=True)
    X = frame[feats]
    shap_all = mean_abs_shap(models, val_idx, X)
    # Rebuild validation membership so domain SHAP uses the same fold models.
    in_val = np.zeros(len(frame), dtype=bool)
    for va in val_idx:
        in_val[va] = True
    assert in_val.all()
    built_local = frame["built"].to_numpy(dtype=float) >= 0.15
    lon = frame["lon"].to_numpy()
    lat = frame["lat"].to_numpy()
    core_local = (lon >= CORE_LON[0]) & (lon <= CORE_LON[1]) & (lat >= CORE_LAT[0]) & (lat <= CORE_LAT[1])

    def subset_shap(mask: np.ndarray) -> np.ndarray:
        acc = np.zeros(len(feats))
        n = 0
        import shap
        for model, va in zip(models, val_idx):
            take = mask[va]
            if take.sum() < 5:
                continue
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(X.iloc[va].iloc[np.flatnonzero(take)])
            if isinstance(values, list):
                values = values[0]
            acc += np.abs(np.asarray(values, dtype=float)).sum(axis=0)
            n += int(take.sum())
        return acc / max(n, 1), n

    shap_built, n_shap_built = subset_shap(built_local)
    shap_core, n_shap_core = subset_shap(core_local)
    from scipy.stats import spearmanr

    rho_built = float(spearmanr(shap_all, shap_built).statistic)
    rho_core = float(spearmanr(shap_all, shap_core).statistic)

    def group_share(imp: np.ndarray, names: list[str]) -> float:
        idx = [feats.index(n) for n in names]
        return float(imp[idx].sum() / imp.sum())

    terrain = ["elevation", "slope", "ERA5_T2M"]
    shap_table = []
    for i, f in enumerate(feats):
        shap_table.append({
            "feature": f,
            "citywide": float(shap_all[i]),
            "built": float(shap_built[i]),
            "core": float(shap_core[i]),
        })
    shap_table.sort(key=lambda r: r["citywide"], reverse=True)

    # --- index reconstruction ---
    print("index and joint uncertainty", flush=True)
    g = grid.dropna(subset=["LST_pred", "LST_P90", "population", "worldpop", "UTHI", "x", "y"]).copy()
    gcent = g.geometry.centroid
    lon_g = gcent.x.to_numpy(dtype=float)
    lat_g = gcent.y.to_numpy(dtype=float)
    lst = g["LST_pred"].to_numpy(dtype=float)
    lst_p90 = g["LST_P90"].to_numpy(dtype=float)
    pop_g = g["population"].to_numpy(dtype=float)
    pop_w = g["worldpop"].to_numpy(dtype=float)
    cool_lack = 1.0 - g["CoolingCapacity"].to_numpy(dtype=float)
    serv_lack = 1.0 - g["ServiceAdaptation"].to_numpy(dtype=float)
    stored = np.column_stack([
        g["ThermalStress"].to_numpy(dtype=float),
        g["Exposure"].to_numpy(dtype=float),
        cool_lack,
        serv_lack,
    ])
    base_score = stored @ np.full(4, 0.25)
    uthi = g["UTHI"].to_numpy(dtype=float)
    recon_corr = float(np.corrcoef(base_score, uthi)[0, 1])
    base_top = uthi >= np.quantile(uthi, 0.80)
    weights = pd.read_csv(ROOT / "outputs" / "tables" / "uthi_weight_mc_draws.csv")
    W_all = weights[["w_TS", "w_EXP", "w_COOL", "w_SERV"]].to_numpy(dtype=float)
    scores_w = stored @ W_all.T
    thr_w = np.quantile(scores_w, 0.80, axis=0)
    top_w = scores_w >= thr_w
    jac_w = jaccard_cols(top_w, base_top)
    rho_w = np.array([spearmanr(base_score, scores_w[:, j]).statistic for j in range(0, W_all.shape[1], 20)])
    p_weight = top_w.mean(axis=1)
    weight_only = {
        "n_draw": int(top_w.shape[1]),
        "jaccard_mean": float(jac_w.mean()),
        "jaccard_sd": float(jac_w.std()),
        "jaccard_p05": float(np.quantile(jac_w, 0.05)),
        "spearman_mean_every_20th": float(np.mean(rho_w)),
        "archived_csv_jaccard_mean": float(weights["jaccard"].mean()),
        "reconstruction_corr_with_UTHI": recon_corr,
        "stable_p75_share_of_baseline": float((p_weight >= 0.75)[base_top].mean()),
    }
    print("  weight-only Jaccard", round(weight_only["jaccard_mean"], 4), "archived", round(weight_only["archived_csv_jaccard_mean"], 4), flush=True)

    # Block-constant residual fields from spatial-CV OOF errors.
    tr_block = spatial_block_ids(frame["x"], frame["y"], 3000).to_numpy()
    resid_by = {}
    built_tr = frame["built"].to_numpy(dtype=float) >= 0.15
    for b, r, is_b in zip(tr_block, resid, built_tr):
        resid_by.setdefault(b, []).append(float(r))
    resid_built = resid[built_tr]
    resid_open = resid[~built_tr]
    g_block = spatial_block_ids(g["x"], g["y"], 3000).to_numpy()
    block_codes, block_uniques = pd.factorize(g_block)
    built_g = g["built"].to_numpy(dtype=float) >= 0.15
    block_is_built = np.zeros(len(block_uniques), dtype=bool)
    for code, flag in zip(block_codes, built_g):
        if flag:
            block_is_built[code] = True
    pools = []
    for b, is_b in zip(block_uniques, block_is_built):
        pool = resid_by.get(b)
        if not pool:
            pool = resid_built if is_b else resid_open
        pools.append(np.asarray(pool, dtype=float))

    # Core four districts: same administrative definition as the archived domain table.
    core_names = {"天山区", "沙依巴克区", "新市区", "水磨沟区"}
    dist_map = grid.assign(id=grid["id"].astype(str)).drop_duplicates("id").set_index("id")["district"].astype(str)
    core4 = frame["id"].astype(str).map(dist_map).isin(core_names).to_numpy()
    print(f"core-four district rows {int(core4.sum())}", flush=True)
    if core4.sum() >= 120:
        met_core, _, _, _, _ = cv_predict(frame.loc[core4], feats, "spatial", 3000, "lgbm")
        cv_rows.append({"domain": "core4", **met_core})
        print(f"  core4 spatial lgbm R2 {met_core['R2']:.4f} RMSE {met_core['RMSE']:.3f}", flush=True)

    def census_people(pop, mask) -> float:
        return float((pop * mask).sum() / max(pop.sum(), 1e-9) * CENSUS)

    p90_n = minmax(lst_p90)
    pop_sources = {"GHSL": pop_g, "WorldPop": pop_w}
    exposure = {k: minmax(v) for k, v in pop_sources.items()}
    thermal0 = 0.7 * minmax(lst) + 0.3 * p90_n
    risk0 = {
        name: np.column_stack([thermal0, expo, cool_lack, serv_lack])
        for name, expo in exposure.items()
    }
    top0 = {}
    for name, risk in risk0.items():
        score0 = risk @ np.full(4, 0.25)
        top0[name] = score0 >= np.quantile(score0, 0.80)
    factor = {
        "zero_residual_ghsl_jaccard_vs_baseline": float(jaccard_cols(top0["GHSL"][:, None], base_top)[0]),
        "population_source_jaccard_equal_weight": float(jaccard_cols(top0["WorldPop"][:, None], top0["GHSL"])[0]),
        "population_source_people_ghsl": census_people(pop_g, top0["GHSL"]),
        "population_source_people_worldpop": census_people(pop_w, top0["WorldPop"]),
    }

    rng = np.random.default_rng(42)
    n_resid = 25
    # Every 5th archived weight vector: same design, smaller ensemble.
    W = W_all[::5]
    n = len(g)
    n_scen = n_resid * 2 * len(W)
    hits = np.zeros(n, dtype=np.int32)
    jac_joint = []
    people = []
    people_by = {"GHSL": [], "WorldPop": []}
    jac_lst = []
    # Equal-weight slice used for siting, stored as high-risk masks later via demand rebuild.
    equal_masks = []  # list of (pop_name, mask)

    for r_i in range(n_resid):
        delta_block = np.array([rng.choice(pool) for pool in pools], dtype=float)
        delta = delta_block[block_codes]
        thermal = 0.7 * minmax(lst + delta) + 0.3 * p90_n
        for pop_name, expo in exposure.items():
            risk = np.column_stack([thermal, expo, cool_lack, serv_lack]).astype(np.float32)
            score_eq = risk @ np.full(4, 0.25, dtype=np.float32)
            mask_eq = score_eq >= np.quantile(score_eq, 0.80)
            equal_masks.append((pop_name, mask_eq))
            if pop_name == "GHSL":
                jac_lst.append(float(jaccard_cols(mask_eq[:, None], base_top)[0]))
            scores = risk @ W.T.astype(np.float32)
            thr = np.quantile(scores, 0.80, axis=0)
            top = scores >= thr
            hits += top.sum(axis=1).astype(np.int32)
            jac_joint.append(jaccard_cols(top, base_top))
            pop = pop_sources[pop_name]
            share = (pop[:, None] * top).sum(axis=0) / max(pop.sum(), 1e-9)
            people.append(share * CENSUS)
            people_by[pop_name].append(share * CENSUS)
        if (r_i + 1) % 5 == 0:
            print(f"  residual field {r_i + 1}/{n_resid}", flush=True)

    jac_joint = np.concatenate(jac_joint)
    people = np.concatenate(people)
    people_ghsl = np.concatenate(people_by["GHSL"])
    people_wpop = np.concatenate(people_by["WorldPop"])
    factor["lst_only_equal_weight_ghsl_jaccard_mean"] = float(np.mean(jac_lst))
    factor["lst_only_equal_weight_ghsl_jaccard_p05"] = float(np.quantile(jac_lst, 0.05))
    factor["lst_only_equal_weight_ghsl_jaccard_p50"] = float(np.quantile(jac_lst, 0.50))
    factor["people_ghsl_p05"] = float(np.quantile(people_ghsl, 0.05))
    factor["people_ghsl_p50"] = float(np.quantile(people_ghsl, 0.50))
    factor["people_ghsl_p95"] = float(np.quantile(people_ghsl, 0.95))
    factor["people_worldpop_p05"] = float(np.quantile(people_wpop, 0.05))
    factor["people_worldpop_p50"] = float(np.quantile(people_wpop, 0.50))
    factor["people_worldpop_p95"] = float(np.quantile(people_wpop, 0.95))
    p_top = hits / n_scen
    stable90 = p_top >= 0.90
    stable75 = p_top >= 0.75

    joint = {
        "n_residual_fields": n_resid,
        "n_weights": int(len(W)),
        "n_population_sources": 2,
        "n_scenarios": int(n_scen),
        "design": "25 block-constant OOF residual fields × 200 archived constrained weights (every 5th of 1000) × GHSL and WorldPop",
        "jaccard_vs_baseline_mean": float(jac_joint.mean()),
        "jaccard_vs_baseline_p05": float(np.quantile(jac_joint, 0.05)),
        "jaccard_vs_baseline_p50": float(np.quantile(jac_joint, 0.50)),
        "jaccard_vs_baseline_p95": float(np.quantile(jac_joint, 0.95)),
        "baseline_top20_still_ge_0.75": float(stable75[base_top].mean()),
        "baseline_top20_still_ge_0.90": float(stable90[base_top].mean()),
        "stable90_n": int(stable90.sum()),
        "stable90_share_of_baseline": float(np.logical_and(stable90, base_top).sum() / base_top.sum()),
        "people_p05": float(np.quantile(people, 0.05)),
        "people_p50": float(np.quantile(people, 0.50)),
        "people_p95": float(np.quantile(people, 0.95)),
        "stable90_people_ghsl": census_people(pop_g, stable90),
        "stable90_people_worldpop": census_people(pop_w, stable90),
        "baseline_people_ghsl": census_people(pop_g, base_top),
        "baseline_people_worldpop": census_people(pop_w, base_top),
        "mean_p_inside_baseline": float(p_top[base_top].mean()),
        "mean_p_outside_baseline": float(p_top[~base_top].mean()),
        "factors": factor,
    }
    print("  joint Jaccard", round(joint["jaccard_vs_baseline_mean"], 3), "people p05-p95",
          round(joint["people_p05"]), round(joint["people_p95"]), flush=True)

    # --- MCLP under equal-weight priority masks ---
    print("siting", flush=True)
    import geopandas as gpd

    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    retrofit_names = {x.lower() for x in RETROFIT}
    cand = fac[fac["fclass"].astype(str).str.lower().isin(retrofit_names)].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])
    gap = g["Gap"].to_numpy(dtype=float).clip(0, 1)
    demand_factor = 0.4 + 0.6 * gap

    def solve(mask: np.ndarray, pop: np.ndarray):
        demand_w = pop * mask * demand_factor
        order = np.argsort(-demand_w)[:450]
        weights_d = demand_w[order]
        dem_xy = np.column_stack([g["x"].to_numpy(dtype=float)[order], g["y"].to_numpy(dtype=float)[order]])
        cov = coverage_matrix(dem_xy, cand_xy, 1125.0)
        chosen = greedy_mclp(weights_d, cov, 10)
        share = covered_weight(weights_d, cov, chosen) / max(float(weights_d.sum()), 1e-9)
        return chosen, float(share)

    base_chosen, base_share = solve(base_top, pop_g)
    base_set = set(base_chosen)
    freq = np.zeros(len(cand), dtype=int)
    shares = []
    set_jac = []
    for pop_name, mask in equal_masks:
        chosen, share = solve(mask, pop_sources[pop_name])
        shares.append(share)
        for j in chosen:
            freq[j] += 1
        set_jac.append(len(base_set & set(chosen)) / max(len(base_set | set(chosen)), 1))
    shares = np.asarray(shares)
    freq_rate = freq / len(equal_masks)
    mclp = {
        "k": 10,
        "radius_m": 1125,
        "n_candidates": int(len(cand)),
        "n_demand": 450,
        "n_scenarios": int(len(equal_masks)),
        "solver": "greedy",
        "baseline_share": base_share,
        "archived_pulp_share": 0.575084887220248,
        "share_p05": float(np.quantile(shares, 0.05)),
        "share_p50": float(np.quantile(shares, 0.50)),
        "share_p95": float(np.quantile(shares, 0.95)),
        "set_jaccard_mean": float(np.mean(set_jac)),
        "set_jaccard_p05": float(np.quantile(set_jac, 0.05)),
        "baseline_sites_always_selected": int((freq_rate[base_chosen] == 1).sum()) if base_chosen else 0,
        "baseline_sites_ge_80": int((freq_rate[base_chosen] >= 0.80).sum()) if base_chosen else 0,
        "n_sites_ge_80": int((freq_rate >= 0.80).sum()),
        "n_sites_ever": int((freq > 0).sum()),
        "baseline_site_frequency": [float(freq_rate[j]) for j in base_chosen],
    }
    cand = cand.reset_index(drop=True)
    site_rows = []
    for j, rate in enumerate(freq_rate):
        if rate <= 0 and j not in base_set:
            continue
        site_rows.append({
            "candidate": j,
            "name": str(cand.loc[j, "name"]) if "name" in cand.columns else "",
            "fclass": str(cand.loc[j, "fclass"]),
            "frequency": float(rate),
            "in_baseline_k10": j in base_set,
            "x": float(cand.loc[j, "x"]),
            "y": float(cand.loc[j, "y"]),
        })
    site_rows.sort(key=lambda r: (-r["in_baseline_k10"], -r["frequency"]))
    print("  MCLP baseline", round(base_share, 3), "scenario p50", round(mclp["share_p50"], 3), flush=True)

    # --- transfer and area of applicability ---
    print("transfer AOA", flush=True)
    import joblib

    bundle = joblib.load(ROOT / "outputs" / "models" / "lgbm_lst.joblib")
    model = bundle["model"]
    yf = available_features(yining, bundle["features"])
    yuse = yining.dropna(subset=yf + ["LST_C"]).copy()
    pred = model.predict(yuse[yf])
    obs = yuse["LST_C"].to_numpy(dtype=float)
    transfer_met = pack_metrics(obs, pred)
    mu = work[feats].mean().to_numpy(dtype=float)
    sd = work[feats].std(ddof=0).to_numpy(dtype=float)
    sd[sd == 0] = 1.0
    Ztr = (work[feats].to_numpy(dtype=float) - mu) / sd
    nn = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(Ztr)
    d_tr = nn.kneighbors(Ztr, return_distance=True)[0][:, 1]
    d_bar = float(d_tr.mean())
    di_tr = d_tr / d_bar
    thr_di = aoi_threshold(di_tr)
    Zy = (yuse[feats].to_numpy(dtype=float) - mu) / sd
    d_new = NearestNeighbors(n_neighbors=1, algorithm="kd_tree").fit(Ztr).kneighbors(Zy, return_distance=True)[0][:, 0]
    di_new = d_new / d_bar
    inside = di_new <= thr_di
    y_built = yuse["built"].to_numpy(dtype=float) >= 0.15

    def slice_met(mask: np.ndarray) -> dict:
        if mask.sum() < 30:
            return {"n": int(mask.sum())}
        out = pack_metrics(obs[mask], pred[mask])
        out["share"] = float(mask.mean())
        return out

    shift = ((yuse[feats].mean().to_numpy(dtype=float) - mu) / sd)
    shift_rows = sorted(
        [{"feature": f, "standardized_mean_difference": float(v)} for f, v in zip(feats, shift)],
        key=lambda r: abs(r["standardized_mean_difference"]), reverse=True,
    )
    aoa = {
        "definition": "Dissimilarity index = distance to nearest Urumqi training point in standardized predictor space, divided by the mean nearest-neighbor distance among training points. Threshold = maximum training DI at or below the upper boxplot fence (Q3 + 1.5 IQR), following Meyer and Pebesma (2021).",
        "threshold": thr_di,
        "train_di_mean": float(di_tr.mean()),
        "yining_inside_share": float(inside.mean()),
        "yining_built_inside_share": float(inside[y_built].mean()) if y_built.any() else None,
        "transfer_all": transfer_met,
        "transfer_inside": slice_met(inside),
        "transfer_outside": slice_met(~inside),
        "transfer_built_inside": slice_met(inside & y_built),
        "transfer_built_outside": slice_met((~inside) & y_built),
        "largest_feature_shifts": shift_rows[:8],
        "archived_transfer_2024": {"R2": -3.0189514499195944, "RMSE": 9.65754724846168, "Bias": 9.313894778689086},
    }
    print("  transfer R2", round(transfer_met["R2"], 3), "inside share", round(aoa["yining_inside_share"], 3), flush=True)

    summary = {
        "cv": cv_rows,
        "shap": {
            "n_citywide": int(len(frame)),
            "n_built": int(n_shap_built),
            "n_core": int(n_shap_core),
            "spearman_citywide_vs_built": rho_built,
            "spearman_citywide_vs_core": rho_core,
            "terrain_share_citywide": group_share(shap_all, terrain),
            "terrain_share_built": group_share(shap_built, terrain),
            "terrain_share_core": group_share(shap_core, terrain),
            "rows": shap_table,
        },
        "weight_only": weight_only,
        "joint": joint,
        "mclp": mclp,
        "aoa": aoa,
        "domains_train": {
            "citywide_n": int(len(work)),
            "built_n": int(built_mask.sum()),
            "core_n": int(core_mask.sum()),
            "citywide_lst_sd": float(work["LST_C"].std()),
            "built_lst_sd": float(work.loc[built_mask, "LST_C"].std()),
            "core_lst_sd": float(work.loc[core_mask, "LST_C"].std()),
        },
    }
    (RES / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(cv_rows).to_csv(RES / "cv_metrics.csv", index=False)
    pd.DataFrame(shap_table).to_csv(RES / "shap_domains.csv", index=False)
    pd.DataFrame(site_rows).to_csv(RES / "mclp_site_frequency.csv", index=False)
    np.savez_compressed(
        RES / "priority_probability.npz",
        lon=lon_g, lat=lat_g, p_top=p_top.astype(np.float32), baseline=base_top.astype(np.uint8),
    )
    make_figures(summary, lon_g, lat_g, p_top, base_top, di_new, thr_di, shares)
    after = {str(p.relative_to(ROOT)): sha(p) for p in WATCH}
    (RES / "watched_hashes_after.json").write_text(json.dumps(after, indent=2), encoding="utf-8")
    changed = [k for k in before if before[k] != after[k]]
    print("WATCHED FILES CHANGED:", changed or "none", flush=True)
    if changed:
        raise SystemExit(2)


def _style():
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
        "savefig.dpi": 300,
    })


def make_figures(summary, lon, lat, p_top, base_top, di_new, thr_di, shares) -> None:
    _style()
    blue, orange, green, verm = "#0072B2", "#E69F00", "#009E73", "#D55E00"
    cv = pd.DataFrame(summary["cv"])

    def pick(domain, mode, learner, block):
        q = cv[(cv.domain == domain) & (cv["mode"] == mode) & (cv.learner == learner)]
        if mode == "spatial":
            q = q[q.block_m == block]
        return q.iloc[0]

    # Figure 1
    labels = ["LightGBM\nspatial", "LightGBM\nrandom", "Random forest\nspatial", "Ridge\nspatial"]
    keys = [("spatial", "lgbm", 3000), ("random", "lgbm", 3000), ("spatial", "rf", 3000), ("spatial", "ridge", 3000)]
    city_r = [pick("citywide", *k).R2 for k in keys]
    built_r = [pick("built", *k).R2 for k in keys]
    city_e = [pick("citywide", *k).RMSE for k in keys]
    built_e = [pick("built", *k).RMSE for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    x = np.arange(len(labels))
    w = 0.36
    axes[0].bar(x - w / 2, city_r, w, color=blue, label="Citywide")
    axes[0].bar(x + w / 2, built_r, w, color=orange, label="Built-up")
    axes[0].set_ylabel("Out-of-fold R²")
    axes[0].set_ylim(0, 1)
    axes[0].set_xticks(x, labels)
    axes[0].legend(frameon=False)
    axes[0].set_title("A  Accuracy")
    axes[1].bar(x - w / 2, city_e, w, color=blue, label="Citywide")
    axes[1].bar(x + w / 2, built_e, w, color=orange, label="Built-up")
    axes[1].set_ylabel("Out-of-fold RMSE (°C)")
    axes[1].set_xticks(x, labels)
    axes[1].set_title("B  Error")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_domain_accuracy.png")
    plt.close(fig)

    # Figure 2
    sh = pd.DataFrame(summary["shap"]["rows"]).head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    y = np.arange(len(sh))
    ax.barh(y - 0.18, sh.citywide, 0.36, color=blue, label="Citywide out-of-fold rows")
    ax.barh(y + 0.18, sh.built, 0.36, color=orange, label="Built-up out-of-fold rows")
    ax.set_yticks(y, sh.feature)
    ax.set_xlabel("Mean |TreeSHAP| contribution (°C)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_shap_domains.png")
    plt.close(fig)

    # Figure 3
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.2))
    core = (lon >= CORE_LON[0]) & (lon <= CORE_LON[1]) & (lat >= CORE_LAT[0]) & (lat <= CORE_LAT[1])
    for ax, mask, title in (
        (axes[0], np.ones(len(lon), dtype=bool), "A  Urumqi municipality"),
        (axes[1], core, "B  Core urban window"),
    ):
        sc = ax.scatter(
            lon[mask], lat[mask], c=p_top[mask], s=2 if mask.sum() > 20000 else 8,
            cmap="cividis", vmin=0, vmax=1, marker="s", linewidths=0, rasterized=True,
        )
        ax.set_aspect("equal")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(title)
    fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02, label="Probability of top-20% priority")
    fig.savefig(FIG / "fig3_priority_stability.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4
    freq = pd.read_csv(RES / "mclp_site_frequency.csv")
    base_sites = freq[freq.in_baseline_k10 == True].sort_values("frequency")  # noqa: E712
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.6))
    axes[0].hist(shares, bins=15, color=blue, edgecolor="white")
    axes[0].axvline(summary["mclp"]["baseline_share"], color=verm, lw=1.2, label="Baseline")
    axes[0].set_xlabel("Share of priority-weighted demand covered (k = 10)")
    axes[0].set_ylabel("Scenarios")
    axes[0].legend(frameon=False)
    axes[0].set_title("A  Coverage under perturbed priority")
    def _short(name, fclass) -> str:
        text = "" if pd.isna(name) else str(name)
        text = text.split("（")[0].split("(")[0].strip()
        if text in {"", "None", "nan"}:
            text = str(fclass)
        return text if len(text) <= 28 else text[:26] + "…"

    labels = [_short(a, b) for a, b in zip(base_sites["name"], base_sites.fclass)]
    axes[1].barh(np.arange(len(base_sites)), base_sites.frequency, color=green)
    axes[1].set_yticks(np.arange(len(base_sites)), labels, fontsize=7)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Selection frequency")
    axes[1].set_title("B  Baseline sites")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_siting.png")
    plt.close(fig)

    # Figure 5
    aoa = summary["aoa"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axes[0].hist(di_new, bins=40, color=blue, edgecolor="white")
    axes[0].axvline(thr_di, color=verm, lw=1.2, label=f"Threshold {thr_di:.2f}")
    axes[0].set_xlabel("Dissimilarity index")
    axes[0].set_ylabel("Yining cells")
    axes[0].legend(frameon=False)
    axes[0].set_title("A  Predictor-space dissimilarity")
    names = ["Inside AOA", "Outside AOA"]
    rmses = [aoa["transfer_inside"]["RMSE"], aoa["transfer_outside"]["RMSE"]]
    biases = [aoa["transfer_inside"]["Bias"], aoa["transfer_outside"]["Bias"]]
    x = np.arange(2)
    axes[1].bar(x - 0.16, rmses, 0.32, color=blue, label="RMSE")
    axes[1].bar(x + 0.16, biases, 0.32, color=orange, label="Bias")
    axes[1].set_xticks(x, names)
    axes[1].set_ylabel("°C")
    axes[1].legend(frameon=False)
    axes[1].set_title("B  Direct-transfer error")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_transfer_aoa.png")
    plt.close(fig)
    _ = base_top


if __name__ == "__main__":
    main()
