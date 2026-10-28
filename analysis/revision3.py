"""Third-round revision. Reads archived rasters and grids. Writes only under analysis/."""
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
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
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
)
from revision import extra  # noqa: E402
from revision2 import (  # noqa: E402
    census_people,
    exact_mclp,
    jaccard,
    norm_on,
    square_coverage,
    style,
    top_within,
)
from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import spatial_block_ids  # noqa: E402
from src.geoai.train_geoai import available_features, make_model  # noqa: E402
from src.gis.mclp import greedy_mclp  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

OUT = RES / "revision3_summary.json"
GHSL_TIF = ROOT / "data" / "raw" / "pop_ghsl_100m_Wulumuqishi.tif"
WPOP_TIF = ROOT / "data" / "raw" / "pop_worldpop_100m_Wulumuqishi.tif"
_SOLVE = {"n": 0}


def solve_exact(weights, cov, k=10, time_limit=6):
    """Unique variable names so repeated solves do not collide in PuLP."""
    import pulp

    _SOLVE["n"] += 1
    tag = f"s{_SOLVE['n']}"
    n_d, n_c = cov.shape
    prob = pulp.LpProblem(f"mclp_{tag}", pulp.LpMaximize)
    y = [pulp.LpVariable(f"{tag}_y{j}", cat="Binary") for j in range(n_c)]
    x = [pulp.LpVariable(f"{tag}_x{i}", cat="Binary") for i in range(n_d)]
    prob += pulp.lpDot(np.asarray(weights, dtype=float).tolist(), x)
    prob += pulp.lpSum(y) <= k
    for i in range(n_d):
        js = np.flatnonzero(cov[i])
        if len(js) == 0:
            prob += x[i] == 0
        else:
            prob += x[i] <= pulp.lpSum(y[int(j)] for j in js)
    status = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.01))
    chosen = [j for j, v in enumerate(y) if v.value() is not None and v.value() > 0.5]
    obj = pulp.value(prob.objective)
    return chosen, None if obj is None else float(obj), pulp.LpStatus[status]


def components(grid, lst_values, population, ref, p90, access):
    thermal = 0.7 * norm_on(lst_values, ref) + 0.3 * norm_on(p90, ref)
    exposure = norm_on(population, ref)
    cool = []
    for name in ("NDVI", "trees", "water", "ALBEDO"):
        if name in grid.columns:
            cool.append(norm_on(grid[name].to_numpy(dtype=float), ref))
    cooling = np.mean(cool, axis=0) if cool else norm_on(grid["CoolingCapacity"].to_numpy(dtype=float), ref)
    service = norm_on(access, ref)
    return np.column_stack([thermal, exposure, 1.0 - cooling, 1.0 - service])


def priority(grid, lst_values, population, ref, p90, access, w):
    return top_within(components(grid, lst_values, population, ref, p90, access) @ np.asarray(w, dtype=float), ref)


def square_touch_area(x, y, half=200.0, res=100.0) -> dict:
    """Two definitions of the overlapping region."""
    base = square_coverage(x, y, half=half, res=res)
    xmin = float(np.min(x) - half)
    ymin = float(np.min(y) - half)
    used = base["raster_m"]
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
    touch = np.zeros(len(x), dtype=bool)
    for i, (a, b, c, d) in enumerate(zip(i0, i1, j0, j1)):
        if b > a and d > c and cover[a:b, c:d].max() >= 2:
            touch[i] = True
    nominal = (2 * half) ** 2 / 1e6
    base["definition"] = (
        "overlap_pixel_km2 is the area of accounting pixels covered by two or more squares, counted once. "
        "squares_that_touch_overlap_km2 is the sum of the nominal areas of squares that contain at least one such pixel."
    )
    base["squares_touching_overlap_n"] = int(touch.sum())
    base["squares_that_touch_overlap_km2"] = float(touch.sum() * nominal)
    return base


def assign_population(raster_path: Path, xy: np.ndarray, boundary) -> dict:
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import xy as rio_xy
    from shapely.geometry import box

    import geopandas as gpd

    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float64)
        arr[~np.isfinite(arr)] = 0.0
        arr[arr < 0] = 0.0
        geoms = [g for g in boundary.to_crs(src.crs).geometry if g is not None and not g.is_empty]
        admin = rasterize([(g, 1) for g in geoms], out_shape=arr.shape, transform=src.transform, fill=0, dtype="uint8")
        arr = np.where(admin == 1, arr, 0.0)
        admin_sum = float(arr.sum())
        cells = gpd.GeoDataFrame(
            geometry=[box(float(px) - 200, float(py) - 200, float(px) + 200, float(py) + 200) for px, py in xy],
            crs=32645,
        ).to_crs(src.crs)
        zone = rasterize(
            [(geom, i + 1) for i, geom in enumerate(cells.geometry) if geom is not None and not geom.is_empty],
            out_shape=arr.shape,
            transform=src.transform,
            fill=0,
            dtype="int32",
        )
        square = np.bincount(zone.ravel(), weights=arr.ravel(), minlength=len(xy) + 1)[1:]
        rows, cols = np.nonzero(arr > 0)
        vals = arr[rows, cols]
        lon, lat = rio_xy(src.transform, rows, cols, offset="center")
        crs = src.crs
    from pyproj import Transformer
    mx, my = Transformer.from_crs(crs, "EPSG:32645", always_xy=True).transform(np.asarray(lon), np.asarray(lat))
    pix = np.column_stack([mx, my])
    tree = cKDTree(xy)
    nearest = tree.query(pix, k=1)[1].astype(np.int32)
    vor = np.bincount(nearest, weights=vals, minlength=len(xy)).astype(np.float64)
    owner = zone[rows, cols] - 1
    in_square = owner >= 0
    reassigned = in_square & (owner != nearest)
    missed = ~in_square
    return {
        "admin_sum": admin_sum,
        "square": square.astype(np.float64),
        "voronoi": vor,
        "people_in_overlap_or_edge": None,
        "people_reassigned_to_a_different_point": float(vals[reassigned].sum()) if reassigned.any() else 0.0,
        "people_inside_admin_but_outside_every_square": float(vals[missed].sum()) if missed.any() else 0.0,
        "pixels_positive": int(len(vals)),
        "pixels_reassigned": int(reassigned.sum()),
    }


def variogram(xy: np.ndarray, resid: np.ndarray, step: float = 1000.0, max_m: float = 20000.0, n_origins: int = 2000, seed: int = 6) -> dict:
    rng = np.random.default_rng(seed)
    n = len(resid)
    origins = rng.choice(n, size=min(n_origins, n), replace=False)
    # Chunk so the distance matrix stays modest.
    sill = float(np.var(resid))
    edges = np.arange(0.0, max_m + step, step)
    acc = np.zeros(len(edges) - 1)
    cnt = np.zeros(len(edges) - 1, dtype=np.int64)
    others = np.arange(n)
    for start in range(0, len(origins), 250):
        take = origins[start:start + 250]
        diff = xy[take, None, :] - xy[None, :, :]
        d = np.sqrt((diff ** 2).sum(axis=2))
        g = 0.5 * (resid[take, None] - resid[None, :]) ** 2
        # Count each unordered pair once. A pair is seen when its lower index is an origin.
        keep = (np.arange(n)[None, :] > take[:, None]) & (d > 0) & (d < max_m)
        d = d[keep]
        g = g[keep]
        bins = np.digitize(d, edges) - 1
        for b in range(len(acc)):
            sel = bins == b
            if sel.any():
                acc[b] += g[sel].sum()
                cnt[b] += int(sel.sum())
    rows = []
    practical = None
    for b, (a, c) in enumerate(zip(acc, cnt)):
        if c < 30:
            continue
        center = float(0.5 * (edges[b] + edges[b + 1]))
        gamma = float(a / c)
        rows.append({"distance_m": center, "gamma": gamma, "n_pairs": int(c)})
        if practical is None and gamma >= 0.95 * sill:
            practical = center
    return {"sill": sill, "n_points": int(n), "n_origins": int(len(origins)), "distance_m_at_95pct_sill": practical, "bins": rows}


def hot_recall(score: np.ndarray, truth: np.ndarray) -> float:
    if len(score) < 20:
        return float("nan")
    pred = score >= np.quantile(score, 0.80)
    hot = truth >= np.quantile(truth, 0.80)
    return float((pred & hot).sum() / max(hot.sum(), 1))


def main() -> None:
    import geopandas as gpd
    import joblib

    RES.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    print("load", flush=True)
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    yining = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Yiningshi_2024.parquet")
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    xy = np.column_stack([grid["x"].to_numpy(dtype=float), grid["y"].to_numpy(dtype=float)])
    lon = grid.geometry.x.to_numpy(dtype=float)
    lat = grid.geometry.y.to_numpy(dtype=float)
    pop = grid["population"].to_numpy(dtype=float)
    pop_w = grid["worldpop"].to_numpy(dtype=float)
    built_p = grid["built"].to_numpy(dtype=float)
    lst = grid["LST_pred"].to_numpy(dtype=float)
    p90 = grid["LST_P90"].to_numpy(dtype=float)
    access = grid["access"].to_numpy(dtype=float) if "access" in grid.columns else np.full(len(grid), 0.35)
    built = built_p >= 0.15
    universe = built & (pop > 0) & np.isfinite(lst) & np.isfinite(p90)
    w_equal = np.full(4, 0.25)

    print("overlap definitions", flush=True)
    overlap = square_touch_area(xy[:, 0], xy[:, 1])
    print(
        "  overlap region", round(overlap["overlap_pixel_km2"], 1),
        "squares touching it", round(overlap["squares_that_touch_overlap_km2"], 1),
        flush=True,
    )

    print("voronoi assignment", flush=True)
    t0 = time.perf_counter()
    ghsl_as = assign_population(GHSL_TIF, xy, boundary)
    print("  ghsl sec", round(time.perf_counter() - t0, 1), "admin", round(ghsl_as["admin_sum"]), flush=True)
    wpop_as = assign_population(WPOP_TIF, xy, boundary)
    vor = ghsl_as["voronoi"]
    square = ghsl_as["square"]

    def match_stats(archived, recomputed):
        both = np.isfinite(archived) & np.isfinite(recomputed)
        return {
            "archived_sum": float(np.nansum(archived)),
            "recomputed_sum": float(np.nansum(recomputed)),
            "ratio_recomputed_over_archived": float(np.nansum(recomputed) / max(np.nansum(archived), 1e-9)),
            "pearson_on_positive": float(np.corrcoef(archived[both & ((archived > 0) | (recomputed > 0))], recomputed[both & ((archived > 0) | (recomputed > 0))])[0, 1])
            if ((archived > 0) | (recomputed > 0)).sum() > 10 else None,
            "max_abs_diff": float(np.nanmax(np.abs(archived - recomputed))),
        }

    base_score = components(grid, lst, pop, universe, p90, access) @ w_equal
    resid_top = top_within(base_score, universe)
    vor_universe = built & (vor > 0) & np.isfinite(lst) & np.isfinite(p90)
    vor_top = priority(grid, lst, vor, vor_universe, p90, access, w_equal)
    # Same points as the archived universe, exposure replaced.
    same_universe_vor_top = priority(grid, lst, vor, universe, p90, access, w_equal)

    def pop_brief(values, mask):
        vals = values[mask & (values > 0)]
        if len(vals) == 0:
            return {"n": 0}
        order = np.argsort(-vals)
        ranked = vals[order]
        return {
            "n_positive": int(len(vals)),
            "max": float(vals.max()),
            "median": float(np.median(vals)),
            "share_top100": float(ranked[: min(100, len(ranked))].sum() / ranked.sum()),
            "total": float(vals.sum()),
            "census_scaled": census_people(values, mask),
        }

    population_partition = {
        "overlap_definition": overlap,
        "ghsl_archived_vs_square_rerun": match_stats(pop, square),
        "worldpop_archived_vs_square_rerun": match_stats(pop_w, wpop_as["square"]),
        "ghsl_admin_raster_sum": ghsl_as["admin_sum"],
        "ghsl_people_reassigned": ghsl_as["people_reassigned_to_a_different_point"],
        "ghsl_people_outside_squares": ghsl_as["people_inside_admin_but_outside_every_square"],
        "ghsl_pixels_reassigned": ghsl_as["pixels_reassigned"],
        "ghsl_positive_pixels": ghsl_as["pixels_positive"],
        "worldpop_people_reassigned": wpop_as["people_reassigned_to_a_different_point"],
        "worldpop_people_outside_squares": wpop_as["people_inside_admin_but_outside_every_square"],
        "voronoi_total": float(vor.sum()),
        "square_rerun_total": float(square.sum()),
        "archived_priority": pop_brief(pop, resid_top),
        "voronoi_priority": pop_brief(vor, vor_top),
        "jaccard_priority_rebuilt_universe": jaccard(resid_top, vor_top),
        "jaccard_priority_same_universe_new_exposure": jaccard(resid_top, same_universe_vor_top),
        "share_of_archived_priority_also_voronoi": float((resid_top & vor_top).sum() / max(resid_top.sum(), 1)),
        "voronoi_priority_n": int(vor_top.sum()),
        "voronoi_priority_people": census_people(vor, vor_top),
        "note": (
            "Squares are rasterized in sample order; a later square overwrites an earlier one, which is not a nearest-point rule. "
            "Voronoi assigns every positive 100 m pixel inside the administrative polygon to the nearest sample point in EPSG:32645. "
            "The two priority sets are the top fifth inside built probability >= 0.15 and population > 0, with components rescaled inside that set."
        ),
    }
    print(
        "  jaccard rebuilt", round(population_partition["jaccard_priority_rebuilt_universe"], 3),
        "reassigned people", round(ghsl_as["people_reassigned_to_a_different_point"]),
        "outside squares", round(ghsl_as["people_inside_admin_but_outside_every_square"]),
        flush=True,
    )

    print("residuals", flush=True)
    _, oof, _, _, y_all = cv_predict(work, feats, "spatial", 3000, "lgbm")
    resid = y_all - oof
    built_rows = work["built"].to_numpy(dtype=float) >= 0.15
    tr_block = spatial_block_ids(work["x"], work["y"], 3000).to_numpy()
    tr_block5 = spatial_block_ids(work["x"], work["y"], 5000).to_numpy()
    xy_tr = np.column_stack([work["x"].to_numpy(dtype=float), work["y"].to_numpy(dtype=float)])
    var_city = variogram(xy_tr, resid, n_origins=1500, seed=6)
    var_built = variogram(xy_tr[built_rows], resid[built_rows], n_origins=min(800, int(built_rows.sum())), seed=7, max_m=15000.0)
    print(
        "  range city", var_city["distance_m_at_95pct_sill"],
        "built", var_built["distance_m_at_95pct_sill"],
        "pairs at 5km", next((b["n_pairs"] for b in var_city["bins"] if b["distance_m"] == 4500 or abs(b["distance_m"] - 4500) < 1), None),
        flush=True,
    )

    def pools(block_ids, source_resid, source_blocks, fallback):
        bag = {}
        for b, r in zip(source_blocks, source_resid):
            bag.setdefault(str(b), []).append(float(r))
        out = []
        for b in block_ids:
            local = bag.get(str(b), [])
            out.append(np.asarray(local if len(local) >= 3 else fallback, dtype=float))
        return out

    def draw_fields(block_ids, pool_list, n, seed):
        rng = np.random.default_rng(seed)
        codes, _ = pd.factorize(np.asarray(block_ids))
        fields = []
        for _ in range(n):
            draw = np.array([rng.choice(p) for p in pool_list])
            fields.append(draw[codes])
        return fields

    g3 = spatial_block_ids(grid["x"], grid["y"], 3000).to_numpy()
    g5 = spatial_block_ids(grid["x"], grid["y"], 5000).to_numpy()
    built_resid = resid[built_rows]
    # Residential-conditional: only built-up out-of-fold residuals, 3 km and 5 km.
    fields_built3 = draw_fields(g3, pools(pd.factorize(g3)[1], built_resid, tr_block[built_rows], built_resid), 25, 21)
    fields_built5 = draw_fields(g5, pools(pd.factorize(g5)[1], built_resid, tr_block5[built_rows], built_resid), 25, 22)
    # The published design mixes whatever residuals fall in the block, then falls back to built or open.
    open_resid = resid[~built_rows]
    mixed_pools = []
    bag = {}
    for b, r in zip(tr_block, resid):
        bag.setdefault(str(b), []).append(float(r))
    codes3, uniq3 = pd.factorize(g3)
    has_built = np.zeros(len(uniq3), dtype=bool)
    for code, flag in zip(codes3, built):
        if flag:
            has_built[code] = True
    for b, is_b in zip(uniq3, has_built):
        local = bag.get(str(b), [])
        mixed_pools.append(np.asarray(local if len(local) >= 1 else (built_resid if is_b else open_resid), dtype=float))
    fields_mixed = draw_fields(g3, mixed_pools, 25, 11)

    weights = pd.read_csv(ROOT / "outputs" / "tables" / "uthi_weight_mc_draws.csv")
    W = weights[["w_TS", "w_EXP", "w_COOL", "w_SERV"]].to_numpy(dtype=float)
    W_joint = W[::50]

    def ensemble(fields, weight_rows, populations, ref, label):
        jacs, people, hits = [], [], np.zeros(len(grid), dtype=np.int32)
        n = 0
        for delta in fields:
            for population, kind in populations:
                for w in weight_rows:
                    mask = priority(grid, lst + delta, population, ref, p90, access, w)
                    hits += mask
                    jacs.append(jaccard(mask, resid_top))
                    if kind == "ghsl":
                        people.append(census_people(pop, mask))
                    else:
                        people.append(float(population[mask].sum() / max(float(np.nansum(population)), 1e-9) * CENSUS))
                    n += 1
        freq = hits / max(n, 1)
        stable = freq >= 0.90
        return {
            "label": label,
            "n": n,
            "jaccard_mean": float(np.mean(jacs)),
            "jaccard_p05": float(np.quantile(jacs, 0.05)),
            "stable90_n": int(stable.sum()),
            "stable90_people": census_people(pop, stable),
            "people_p50": float(np.median(people)),
        }

    temp_mixed = ensemble(fields_mixed, [w_equal], [(pop, "ghsl")], universe, "mixed block residuals, published design")
    temp_built3 = ensemble(fields_built3, [w_equal], [(pop, "ghsl")], universe, "built-up residuals, 3 km")
    temp_built5 = ensemble(fields_built5, [w_equal], [(pop, "ghsl")], universe, "built-up residuals, 5 km")
    joint_built3 = ensemble(
        fields_built3, W_joint[:8], [(pop, "ghsl"), (pop_w, "worldpop")], universe, "built residual joint"
    )
    print(
        "  temp mixed", round(temp_mixed["jaccard_mean"], 3),
        "built3", round(temp_built3["jaccard_mean"], 3),
        "built5", round(temp_built5["jaccard_mean"], 3),
        "joint built", round(joint_built3["jaccard_mean"], 3),
        flush=True,
    )

    print("thresholds", flush=True)
    threshold_rows = []
    specs = [
        ("GHSL > 0, built >= 0.15", built & (pop > 0)),
        ("GHSL >= 1, built >= 0.15", built & (pop >= 1)),
        ("GHSL >= 5, built >= 0.15", built & (pop >= 5)),
        ("GHSL >= 10, built >= 0.15", built & (pop >= 10)),
        ("GHSL > 0 and WorldPop > 0, built >= 0.15", built & (pop > 0) & (pop_w > 0)),
        ("GHSL > 0, built >= 0.30", (built_p >= 0.30) & (pop > 0)),
    ]
    for label, mask0 in specs:
        ref = mask0 & np.isfinite(lst) & np.isfinite(p90)
        top = priority(grid, lst, pop, ref, p90, access, w_equal)
        # Temperature-only stability with built-up residuals, so thresholds share one error model.
        hits = np.zeros(len(grid), dtype=np.int32)
        jacs = []
        for delta in fields_built3:
            m = priority(grid, lst + delta, pop, ref, p90, access, w_equal)
            hits += m
            jacs.append(jaccard(m, top))
        stable = (hits / len(fields_built3)) >= 0.90
        threshold_rows.append({
            "rule": label,
            "universe_n": int(ref.sum()),
            "priority_n": int(top.sum()),
            "priority_people": census_people(pop, top),
            "jaccard_vs_baseline_priority": jaccard(top, resid_top),
            "stable90_n": int(stable.sum()),
            "stable90_people": census_people(pop, stable),
            "temperature_jaccard_mean": float(np.mean(jacs)),
        })
        print(" ", label, "priority", int(top.sum()), "people", round(census_people(pop, top)), "stable", int(stable.sum()), flush=True)

    print("cross-year", flush=True)
    g5p = pd.read_parquet(
        ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2025.parquet",
        columns=["x", "y", "LST_C", "LST_P90"],
    )
    tree5 = cKDTree(np.column_stack([g5p["x"].to_numpy(dtype=float), g5p["y"].to_numpy(dtype=float)]))
    dist5, ix5 = tree5.query(xy, k=1)
    matched = universe & (dist5 <= 200) & np.isfinite(g5p["LST_C"].to_numpy()[ix5]) & np.isfinite(p90) & np.isfinite(lst)
    y25 = g5p["LST_C"].to_numpy(dtype=float)[ix5]
    p90_25 = g5p["LST_P90"].to_numpy(dtype=float)[ix5]
    sub = np.flatnonzero(matched)
    cross = {
        "n_matched_within_200m": int(matched.sum()),
        "mean_2024_predicted": float(lst[sub].mean()),
        "mean_2024_p90": float(p90[sub].mean()),
        "mean_2025_seasonal": float(y25[sub].mean()),
        "mean_2025_p90": float(p90_25[sub].mean()),
        "spearman_2024_pred_vs_2025_seasonal": float(spearmanr(lst[sub], y25[sub]).statistic),
        "spearman_2024_p90_vs_2025_seasonal": float(spearmanr(p90[sub], y25[sub]).statistic),
        "spearman_2024_pred_vs_2025_p90": float(spearmanr(lst[sub], p90_25[sub]).statistic),
        "spearman_2024_p90_vs_2025_p90": float(spearmanr(p90[sub], p90_25[sub]).statistic),
        "recall_2024_pred_for_2025_seasonal_hot_fifth": hot_recall(lst[sub], y25[sub]),
        "recall_2024_p90_for_2025_seasonal_hot_fifth": hot_recall(p90[sub], y25[sub]),
        "recall_2024_pred_for_2025_p90_hot_fifth": hot_recall(lst[sub], p90_25[sub]),
        "recall_2024_p90_for_2025_p90_hot_fifth": hot_recall(p90[sub], p90_25[sub]),
        "note": "2025 observations were not used to fit the 2024 model or the 2024 priority index. Ranks are within matched inhabited built-up points. Means differ between summers, so the comparison is a rank comparison.",
    }
    print(
        "  n", cross["n_matched_within_200m"],
        "recall pred", round(cross["recall_2024_pred_for_2025_seasonal_hot_fifth"], 3),
        "p90", round(cross["recall_2024_p90_for_2025_seasonal_hot_fifth"], 3),
        flush=True,
    )

    print("siting", flush=True)
    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    names = {n.lower() for n in RETROFIT}
    cand = fac[fac["fclass"].astype(str).str.lower().isin(names)].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])

    def demand_idx(mask, population, cap=450):
        score = np.where(mask, population, 0.0)
        idx = np.flatnonzero(score > 0)
        if len(idx) > cap:
            idx = idx[np.argsort(-score[idx])[:cap]]
        return idx

    def cover_of(idx, population, chosen):
        if len(idx) == 0:
            return float("nan"), None
        weights_d = population[idx]
        cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
        return covered_weight(weights_d, cov, chosen) / max(float(weights_d.sum()), 1e-9), cov

    idx0 = demand_idx(resid_top, pop)
    w0 = pop[idx0]
    cov0 = coverage_matrix(xy[idx0], cand_xy, 1125.0)
    exact0, _, status0 = solve_exact(w0, cov0, time_limit=15)
    exact_share0 = covered_weight(w0, cov0, exact0) / max(float(w0.sum()), 1e-9)
    all_idx = np.flatnonzero(resid_top & (pop > 0))
    all_share, _ = cover_of(all_idx, pop, exact0)
    # Voronoi priority siting, one exact solve.
    v_idx = demand_idx(vor_top, vor)
    v_cov = coverage_matrix(xy[v_idx], cand_xy, 1125.0)
    v_exact, _, v_status = solve_exact(vor[v_idx], v_cov, time_limit=15)
    vor_site_overlap = len(set(exact0) & set(v_exact))

    # Joint slice: 6 temperature fields x 2 weights x 2 populations.
    w_shift = W[int(np.argmax(np.abs(W - 0.25).sum(axis=1)))]
    joint_fields = fields_built3[::4][:6]
    joint_weights = [w_equal, w_shift]
    regrets, fixed_shares, re_shares, full_fixed, full_re = [], [], [], [], []
    statuses = []
    hit = np.zeros(len(cand), dtype=int)
    scenarios = []
    for delta in joint_fields:
        for population, kind in ((pop, "ghsl"), (pop_w, "worldpop")):
            for w in joint_weights:
                mask = priority(grid, lst + delta, population, universe, p90, access, w)
                idx = demand_idx(mask, population)
                weights_d = population[idx]
                cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
                chosen, _, status = solve_exact(weights_d, cov, time_limit=6)
                fixed = covered_weight(weights_d, cov, exact0) / max(float(weights_d.sum()), 1e-9)
                again = covered_weight(weights_d, cov, chosen) / max(float(weights_d.sum()), 1e-9)
                full_idx = np.flatnonzero(mask & (population > 0))
                full_f, _ = cover_of(full_idx, population, exact0)
                full_r, _ = cover_of(full_idx, population, chosen)
                regrets.append(again - fixed)
                fixed_shares.append(fixed)
                re_shares.append(again)
                full_fixed.append(full_f)
                full_re.append(full_r)
                statuses.append(status)
                hit[np.asarray(chosen, dtype=int)] += 1
                scenarios.append(kind)
    regrets = np.asarray(regrets, dtype=float)
    cand_ll = cand.to_crs(4326)
    siting = {
        "baseline_exact_status": status0,
        "baseline_exact_share_of_450": float(exact_share0),
        "baseline_exact_share_of_all_priority_points": float(all_share),
        "all_priority_n": int(len(all_idx)),
        "all_priority_pop_share_of_residential_priority": float(pop[all_idx].sum() / max(pop[resid_top].sum(), 1e-9)),
        "voronoi_exact_status": v_status,
        "voronoi_baseline_sites_in_common": int(vor_site_overlap),
        "joint_n": int(len(regrets)),
        "joint_design": "6 built-up residual fields x 2 weight vectors x GHSL and WorldPop. Ranked inside the archived inhabited built-up universe. Exact solver, 6 s limit, 1% gap.",
        "joint_status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "exact_regret_values": [float(v) for v in regrets],
        "exact_regret_p50": float(np.median(regrets)),
        "exact_regret_p05": float(np.quantile(regrets, 0.05)),
        "exact_regret_p95": float(np.quantile(regrets, 0.95)),
        "exact_regret_min": float(regrets.min()),
        "exact_regret_max": float(regrets.max()),
        "n_negative_exact_regret": int((regrets < -1e-8).sum()),
        "fixed_share_p50": float(np.median(fixed_shares)),
        "reopt_share_p50": float(np.median(re_shares)),
        "full_set_fixed_p50": float(np.nanmedian(full_fixed)),
        "full_set_reopt_p50": float(np.nanmedian(full_re)),
        "full_set_fixed_min": float(np.nanmin(full_fixed)),
        "always_selected": int(sum(hit[i] == len(regrets) for i in exact0)),
        "n_ever": int((hit > 0).sum()),
        "nonnegative_note": "A nonnegative exact regret is required by the definition of an optimum. It is not evidence that the plan is robust.",
        "sites": [
            {
                "name": None if pd.isna(cand.iloc[i]["name"]) else str(cand.iloc[i]["name"]),
                "fclass": str(cand.iloc[i]["fclass"]),
                "lon": float(cand_ll.geometry.iloc[i].x),
                "lat": float(cand_ll.geometry.iloc[i].y),
                "frequency": float(hit[i] / len(regrets)),
                "in_baseline": bool(i in set(exact0)),
                "scheme": "exact re-solve of the joint slice" if hit[i] else "",
            }
            for i in np.argsort(-hit) if hit[i] > 0 or i in set(exact0)
        ],
    }
    print(
        "  all-point cover", round(all_share, 3),
        "joint regret p50", round(siting["exact_regret_p50"], 4),
        "min", round(siting["exact_regret_min"], 4),
        "voronoi sites shared", vor_site_overlap,
        flush=True,
    )

    print("few-shot pairs", flush=True)
    bundle = joblib.load(ROOT / "outputs" / "models" / "lgbm_lst.joblib")
    yf = available_features(yining, bundle["features"])
    yuse = yining.dropna(subset=yf + ["LST_C", "x", "y"]).copy()
    pred = bundle["model"].predict(yuse[yf])
    obs = yuse["LST_C"].to_numpy(dtype=float)
    yb = spatial_block_ids(yuse["x"], yuse["y"], 3000).to_numpy()
    uniq = np.unique(yb)
    ylon = yuse.geometry.x.to_numpy(dtype=float)
    ylat = yuse.geometry.y.to_numpy(dtype=float)
    rng_c = np.random.default_rng(9)
    few = []
    for n_cal in (20, 50, 100, 200, 500):
        reps = []
        for rep in range(12):
            order_b = rng_c.permutation(uniq)
            cal_idx, chosen_blocks = [], []
            for b in order_b:
                if len(cal_idx) >= n_cal:
                    break
                members = np.flatnonzero(yb == b)
                need = n_cal - len(cal_idx)
                take = members if len(members) <= need else rng_c.choice(members, size=need, replace=False)
                cal_idx.extend(np.atleast_1d(take).tolist())
                chosen_blocks.append(b)
            cal_idx = np.asarray(cal_idx[:n_cal], dtype=int)
            test_idx = np.flatnonzero(~np.isin(yb, chosen_blocks))
            if len(test_idx) < 50:
                continue
            bias = float(np.mean(pred[cal_idx] - obs[cal_idx]))
            raw_m = extra(obs[test_idx], pred[test_idx])
            cal_m = extra(obs[test_idx], pred[test_idx] - bias)
            reps.append({
                "rep": rep,
                "raw_RMSE": raw_m["RMSE"],
                "cal_RMSE": cal_m["RMSE"],
                "delta_RMSE": raw_m["RMSE"] - cal_m["RMSE"],
                "raw_R2": raw_m["R2"],
                "cal_R2": cal_m["R2"],
                "n_test": int(len(test_idx)),
                "n_cal_blocks": int(len(chosen_blocks)),
                "cal_lon_span": float(ylon[cal_idx].max() - ylon[cal_idx].min()),
                "cal_lat_span": float(ylat[cal_idx].max() - ylat[cal_idx].min()),
                "block_fraction": float(len(chosen_blocks) / len(uniq)),
            })
        worst = min(reps, key=lambda r: r["cal_R2"])
        few.append({
            "n_cal": n_cal,
            "repeats": len(reps),
            "raw_RMSE_mean": float(np.mean([r["raw_RMSE"] for r in reps])),
            "raw_RMSE_sd": float(np.std([r["raw_RMSE"] for r in reps])),
            "cal_RMSE_mean": float(np.mean([r["cal_RMSE"] for r in reps])),
            "cal_RMSE_sd": float(np.std([r["cal_RMSE"] for r in reps])),
            "cal_R2_mean": float(np.mean([r["cal_R2"] for r in reps])),
            "cal_R2_sd": float(np.std([r["cal_R2"] for r in reps])),
            "delta_RMSE_mean": float(np.mean([r["delta_RMSE"] for r in reps])),
            "delta_RMSE_min": float(np.min([r["delta_RMSE"] for r in reps])),
            "delta_RMSE_max": float(np.max([r["delta_RMSE"] for r in reps])),
            "mean_cal_blocks": float(np.mean([r["n_cal_blocks"] for r in reps])),
            "mean_block_fraction": float(np.mean([r["block_fraction"] for r in reps])),
            "worst_repeat": worst,
            "repeats_detail": reps,
        })
        print("  n", n_cal, "delta", round(few[-1]["delta_RMSE_mean"], 2), "R2 sd", round(few[-1]["cal_R2_sd"], 2), "worst R2", round(worst["cal_R2"], 2), flush=True)

    # Held-out residuals for the existing map are already in revision2_maps.npz.
    payload = {
        "population_partition": {k: v for k, v in population_partition.items() if k not in {}},
        "conditional_error": {
            "published_mixed_block_rerun": temp_mixed,
            "built_residuals_3km": temp_built3,
            "built_residuals_5km": temp_built5,
            "joint_built_residuals_3km": joint_built3,
            "variogram_citywide_full_pairs": var_city,
            "variogram_built_full_pairs": var_built,
            "built_residual_sd": float(built_resid.std()),
            "citywide_residual_sd": float(resid.std()),
        },
        "thresholds": threshold_rows,
        "cross_year": cross,
        "siting": {k: v for k, v in siting.items() if k != "sites"},
        "siting_sites": siting["sites"],
        "few_shot": [{k: v for k, v in row.items() if k != "repeats_detail"} for row in few],
        "few_shot_deltas": {str(row["n_cal"]): [r["delta_RMSE"] for r in row["repeats_detail"]] for row in few},
        "few_shot_cal_r2": {str(row["n_cal"]): [r["cal_R2"] for r in row["repeats_detail"]] for row in few},
    }
    # overlap definition contains numpy types already cast. JSON check.
    OUT.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print("saved json", flush=True)
    make_figures(payload, lon, lat, grid, yuse)
    print("saved figures", flush=True)


def make_figures(payload, lon, lat, grid, yuse) -> None:
    import geopandas as gpd

    style()
    blue, verm, green, orange = "#0072B2", "#D55E00", "#009E73", "#E69F00"
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson")
    maps = np.load(RES / "revision2_maps.npz")
    prev = json.loads((RES / "revision_summary.json").read_text(encoding="utf-8"))
    d = prev["domains"]
    xlim, ylim = (87.30, 87.85), (43.65, 44.05)

    # Figure 1: add hot-fifth recall.
    labels = ["Citywide", "Built-up\nrows", "Four\ndistricts"]
    r2s = [d["citywide_model_citywide_test"]["R2"], d["citywide_model_built_test"]["R2"], d["citywide_model_core4_test"]["R2"]]
    refit = [np.nan, d["built_model_built_test"]["R2"], np.nan]
    nrmse = [d["citywide_model_citywide_test"]["NRMSE"], d["citywide_model_built_test"]["NRMSE"], d["citywide_model_core4_test"]["NRMSE"]]
    recall = [d["citywide_model_citywide_test"]["top20_recall"], d["citywide_model_built_test"]["top20_recall"], d["citywide_model_core4_test"]["top20_recall"]]
    recall_refit = [np.nan, d["built_model_built_test"]["top20_recall"], np.nan]
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.3))
    x = np.arange(3)
    axes[0].bar(x - 0.15, r2s, 0.3, color=blue, label="Citywide model")
    axes[0].bar(x + 0.15, refit, 0.3, color=orange, label="Refit in domain")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Out-of-fold R²")
    axes[0].set_title("A  Fit")
    axes[0].legend(frameon=False, fontsize=7)
    axes[1].bar(x, nrmse, color=blue)
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(labelsize=8)
    axes[0].tick_params(labelsize=8)
    axes[1].set_ylabel("RMSE / SD of observed LST")
    axes[1].set_title("B  Scale-free error")
    axes[2].bar(x - 0.15, recall, 0.3, color=blue, label="Citywide model")
    axes[2].bar(x + 0.15, recall_refit, 0.3, color=orange, label="Refit in domain")
    axes[2].set_xticks(x, labels)
    axes[2].set_ylim(0, 1)
    axes[2].set_ylabel("Hot-fifth recall")
    axes[2].set_title("C  Hottest fifth")
    axes[2].legend(frameon=False, fontsize=7)
    for ax in axes:
        ax.tick_params(labelsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_domain_accuracy.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig1_domain_accuracy.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 2 title clarifies that elevation is omitted from the display.
    sh = pd.DataFrame(prev["shap"]["rows"]).head(10).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(sh))
    axes[0].barh(y - 0.25, sh.citywide_model_all, 0.24, color=blue, label="Citywide model, all rows")
    axes[0].barh(y, sh.citywide_model_built_rows, 0.24, color=orange, label="Citywide model, built-up rows")
    axes[0].barh(y + 0.25, sh.built_model, 0.24, color=green, label="Refit on built-up rows")
    axes[0].set_yticks(y, sh.feature)
    axes[0].set_xlabel("Mean |TreeSHAP| (°C)")
    axes[0].set_title("A  Full scale")
    axes[0].legend(frameon=False, fontsize=6, loc="lower right")
    drop = sh[sh.feature != "elevation"]
    y2 = np.arange(len(drop))
    axes[1].barh(y2 - 0.25, drop.citywide_model_all, 0.24, color=blue)
    axes[1].barh(y2, drop.citywide_model_built_rows, 0.24, color=orange)
    axes[1].barh(y2 + 0.25, drop.built_model, 0.24, color=green)
    axes[1].set_yticks(y2, drop.feature)
    axes[1].set_xlabel("Mean |TreeSHAP| (°C)")
    axes[1].set_title("B  Elevation omitted from this panel")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_shap_domains.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig2_shap_domains.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Figure 3, one window for B–D.
    city_top = maps["city_top"].astype(bool)
    resid_top = maps["resid_top"].astype(bool)
    freq = maps["joint_freq"]
    inhabited = maps["pop_pos"].astype(bool)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.2))

    def frame(ax, zoom=False):
        boundary.boundary.plot(ax=ax, color="#666666", linewidth=0.4)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
        if zoom:
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)

    ax = axes[0, 0]
    frame(ax, zoom=False)
    ax.scatter(lon[city_top & ~inhabited], lat[city_top & ~inhabited], s=1, c=blue, marker="s", linewidths=0, rasterized=True, label="No residents")
    ax.scatter(lon[city_top & inhabited], lat[city_top & inhabited], s=3, c=verm, marker="s", linewidths=0, rasterized=True, label="Has residents")
    ax.legend(frameon=False, loc="lower left", markerscale=2, fontsize=6)
    ax.set_title("A  Citywide top 20%")
    ax.set_ylabel("Latitude")
    ax = axes[0, 1]
    frame(ax, zoom=True)
    ax.scatter(lon[resid_top], lat[resid_top], s=6, c=verm, marker="s", linewidths=0, rasterized=True)
    ax.set_title("B  Inhabited built-up top 20%")
    window = (lon >= xlim[0]) & (lon <= xlim[1]) & (lat >= ylim[0]) & (lat <= ylim[1])
    ax = axes[1, 0]
    frame(ax, zoom=True)
    show = window & (maps["universe"].astype(bool))
    sc = ax.scatter(lon[show], lat[show], c=freq[show], s=6, cmap="cividis", vmin=0, vmax=1, marker="s", linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Scenario frequency")
    ax.set_title("C  Residential scenario frequency")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax = axes[1, 1]
    frame(ax, zoom=True)
    only_city = city_top & ~resid_top & inhabited & window
    only_res = resid_top & ~city_top & window
    both = city_top & resid_top & window
    ax.scatter(lon[only_city], lat[only_city], s=5, c=blue, marker="s", linewidths=0, rasterized=True, label="Citywide only")
    ax.scatter(lon[only_res], lat[only_res], s=6, c=orange, marker="s", linewidths=0, rasterized=True, label="Residential only")
    ax.scatter(lon[both], lat[both], s=8, c=verm, marker="s", linewidths=0, rasterized=True, label="Both")
    ax.legend(frameon=False, loc="lower left", fontsize=6, markerscale=1.4)
    ax.set_title("D  Where the two sets meet")
    ax.set_xlabel("Longitude")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_priority_stability.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 4: exact regret, coverage map, frequency bars.
    sites = pd.DataFrame(payload["siting_sites"])
    base = sites[sites.in_baseline].copy()
    regret = np.asarray(payload["siting"]["exact_regret_values"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 3.6))
    axes[0].hist(regret, bins=10, color=verm, edgecolor="white")
    axes[0].axvline(float(np.median(regret)), color="black", lw=1, linestyle="--")
    axes[0].set_xlabel("Exact regret")
    axes[0].set_ylabel("Joint scenarios")
    axes[0].set_title("A  Exact regret")
    ax = axes[1]
    # 1,125 m in degrees at 43.8 N.
    width = 2 * 1125 / (111_320 * np.cos(np.deg2rad(43.8)))
    height = 2 * 1125 / 110_540
    for _, row in base.iterrows():
        ax.add_patch(Ellipse((row.lon, row.lat), width=width, height=height, facecolor=blue, edgecolor=blue, alpha=0.18, lw=0.4, zorder=1))
    ranked = base.sort_values("frequency", ascending=False).reset_index(drop=True)
    sc = ax.scatter(ranked.lon, ranked.lat, c=ranked.frequency, s=40, cmap="cividis", vmin=0, vmax=1, zorder=3)
    for i, row in ranked.iterrows():
        ax.annotate(str(i + 1), (row.lon, row.lat), textcoords="offset points", xytext=(4, 4), fontsize=7)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Exact re-solve frequency")
    ax.set_aspect("equal")
    ax.set_title("B  Sites")
    ax.set_xlabel("Longitude")
    order = ranked.sort_values("frequency", ascending=True)
    labels = []
    for _, row in order.iterrows():
        name = "" if not isinstance(row["name"], str) else row["name"].split("（")[0].split("(")[0]
        number = int(ranked.index[ranked.lon.eq(row.lon) & ranked.lat.eq(row.lat)][0]) + 1
        labels.append(f"{number}  {name if name else row['fclass']}")
    axes[2].barh(np.arange(len(order)), order.frequency, color=green)
    axes[2].set_yticks(np.arange(len(order)), labels, fontsize=6)
    axes[2].set_xlim(0, 1)
    axes[2].set_xlabel("Selection frequency")
    axes[2].set_title("C  How often the exact solver returns each site")
    fig.tight_layout()
    fig.savefig(FIG / "fig4_siting.png", bbox_inches="tight")
    plt.close(fig)

    # Figure 5: paired improvement, not two unpaired means.
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 6.4))
    raw = maps["yining_raw_resid"]
    cal = maps["yining_cal_resid"]
    ylon = maps["yining_lon"]
    ylat = maps["yining_lat"]
    rng = np.random.default_rng(1)
    take = rng.choice(len(raw), size=min(4000, len(raw)), replace=False)
    obs = yuse["LST_C"].to_numpy(dtype=float)
    # raw residual was predicted - observed in revision2.
    axes[0, 0].scatter(obs[take], obs[take] + raw[take], s=4, alpha=0.25, c=blue, linewidths=0, rasterized=True)
    lims = [float(np.nanmin(obs[take])), float(np.nanmax(obs[take]))]
    axes[0, 0].plot(lims, lims, color="black", lw=0.8)
    axes[0, 0].set_xlabel("Observed LST (°C)")
    axes[0, 0].set_ylabel("Predicted LST (°C)")
    axes[0, 0].set_title("A  Uncalibrated transfer")
    deltas = payload.get("few_shot_dispersed_deltas") or payload["few_shot_deltas"]
    positions = []
    data = []
    for i, n_cal in enumerate(("20", "50", "100", "200", "500")):
        positions.append(i)
        data.append(deltas[n_cal])
    axes[0, 1].boxplot(data, positions=positions, widths=0.55, patch_artist=True,
                       boxprops={"facecolor": "#F4A582", "edgecolor": "#444444"},
                       medianprops={"color": "black"})
    axes[0, 1].set_xticks(positions, ["20", "50", "100", "200", "500"])
    axes[0, 1].set_xlabel("Local points, one per spatial block")
    axes[0, 1].set_ylabel("RMSE reduction on the same blocks (°C)")
    axes[0, 1].set_title("B  Paired improvement, 12 repeats")
    vmax = float(np.nanquantile(np.abs(raw), 0.98))
    for ax, resid, title in (
        (axes[1, 0], raw, "C  Uncalibrated residual"),
        (axes[1, 1], cal, "D  After block intercept"),
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

    # Figure 6: full-pair variograms.
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, key, title in (
        (axes[0], "variogram_citywide_full_pairs", "A  Citywide out-of-fold residuals"),
        (axes[1], "variogram_built_full_pairs", "B  Built-up residuals only"),
    ):
        block = payload["conditional_error"][key]
        bins = block["bins"]
        xx = [b["distance_m"] / 1000 for b in bins]
        yy = [b["gamma"] for b in bins]
        nn = [b["n_pairs"] for b in bins]
        ax.plot(xx, yy, color=verm, marker="o", ms=3)
        ax.axhline(block["sill"], color="#444444", lw=0.8, linestyle="--")
        ax.set_xlabel("Distance (km)")
        ax.set_ylabel("Semivariance (°C²)")
        ax.set_title(title)
        ax.text(0.98, 0.05, f"pairs in the fullest bin: {max(nn):,}", transform=ax.transAxes, ha="right", fontsize=8, color="#444444")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_variogram.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig6_variogram.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
