"""Fifth-round audit: proven CBC gaps, unweighted coverage, recall intervals.

Writes only under analysis/. Does not modify archived grids or models.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis import FIG, RES, coverage_matrix, covered_weight, cv_predict  # noqa: E402
from revision2 import census_people, style  # noqa: E402
from revision3 import GHSL_TIF, WPOP_TIF, assign_population, hot_recall, priority  # noqa: E402
from revision4 import demand_idx  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import spatial_block_ids  # noqa: E402
from src.geoai.train_geoai import available_features  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

OUT = RES / "revision5_summary.json"
LOGS = RES / "cbc_logs"
CENSUS = 4_054_369
SHORT = (
    ("HongSham", "HongSham market"),
    ("十一师", "No. 11 Division school"),
    ("警史馆", "Police History Museum"),
    ("水利水电", "Water conservancy school"),
    ("农业大学", "Xinjiang Agricultural Univ."),
    ("师范大学", "Xinjiang Normal Univ."),
    ("花鸟鱼", "South Park market"),
    ("乌鲁木齐职业大学", "Urumqi Vocational Univ."),
    ("职业大学", "Xinjiang Vocational Univ."),
    ("共青团", "Youth League school"),
    ("兵团三中", "XPCC No. 3 school"),
)


def short_name(name: str) -> str:
    text = "" if not isinstance(name, str) else name
    for key, label in SHORT:
        if key in text:
            return label
    return "Candidate"


def parse_cbc_log(text: str) -> dict:
    result = ""
    objective = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("Result -"):
            result = stripped
        if "Objective value" in stripped:
            try:
                objective = float(stripped.split()[-1])
            except ValueError:
                pass
    proven = "Optimal solution found" in result
    return {
        "result_line": result,
        "log_objective": objective,
        "proven_optimal": proven,
        "stopped_on_gap": ("gap" in result.lower()) and not proven,
    }


def solve_proven(weights, cov, banned=None, time_limit=90, label="s"):
    import pulp
    from revision3 import _SOLVE

    _SOLVE["n"] += 1
    tag = f"g{_SOLVE['n']}_{label}"
    n_d, n_c = cov.shape
    prob = pulp.LpProblem(f"mclp_{tag}", pulp.LpMaximize)
    y = [pulp.LpVariable(f"{tag}_y{j}", cat="Binary") for j in range(n_c)]
    x = [pulp.LpVariable(f"{tag}_x{i}", cat="Binary") for i in range(n_d)]
    prob += pulp.lpDot(np.asarray(weights, dtype=float).tolist(), x)
    prob += pulp.lpSum(y) <= 10
    for i in range(n_d):
        js = np.flatnonzero(cov[i])
        if len(js) == 0:
            prob += x[i] == 0
        else:
            prob += x[i] <= pulp.lpSum(y[int(j)] for j in js)
    if banned:
        prob += pulp.lpSum(y[int(j)] for j in banned) <= max(len(banned) - 1, 0)
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}.log"
    t0 = time.perf_counter()
    status_code = prob.solve(
        pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0, gapAbs=0, logPath=str(log_path))
    )
    elapsed = time.perf_counter() - t0
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    parsed = parse_cbc_log(text)
    chosen = [j for j, v in enumerate(y) if v.value() is not None and v.value() > 0.5]
    obj = pulp.value(prob.objective)
    obj = None if obj is None else float(obj)
    bound = obj if parsed["proven_optimal"] and obj is not None else parsed["log_objective"]
    gap = None
    if obj is not None and bound is not None and abs(obj) > 0:
        gap = abs(bound - obj) / abs(obj)
    return {
        "sites": chosen,
        "objective": obj,
        "bound": bound,
        "relative_gap": gap,
        "status": pulp.LpStatus[status_code],
        "proven_optimal": bool(parsed["proven_optimal"]),
        "result_line": parsed["result_line"],
        "seconds": elapsed,
        "log": str(log_path.relative_to(ROOT)),
    }


def covered_population(pop_values, cov, sites) -> float:
    if len(sites) == 0:
        return 0.0
    hit = cov[:, np.asarray(sites, dtype=int)].any(axis=1)
    return float(np.asarray(pop_values, dtype=float)[hit].sum())


def quant(values):
    arr = np.asarray(values, dtype=float)
    return {
        "n": int(len(arr)),
        "p50": float(np.median(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def block_bootstrap(scores, truths, blocks, n_boot=400, seed=11):
    rng = np.random.default_rng(seed)
    uniq = np.unique(blocks)
    groups = [np.flatnonzero(blocks == b) for b in uniq]
    point = [hot_recall(s, t) for s, t in zip(scores, truths)]
    draws = []
    diffs = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        rows = np.concatenate([groups[i] for i in pick])
        recalls = [hot_recall(s[rows], t[rows]) for s, t in zip(scores, truths)]
        draws.append(recalls)
        diffs.append([recalls[1] - recalls[0], recalls[2] - recalls[1], recalls[2] - recalls[0]])
    draws = np.asarray(draws, dtype=float)
    diffs = np.asarray(diffs, dtype=float)
    names = ["predicted_seasonal_mean", "observed_seasonal_mean", "observed_p90"]
    diff_names = ["observed_mean_minus_predicted", "p90_minus_observed_mean", "p90_minus_predicted"]
    out = {"n_boot": n_boot, "seed": seed, "block_m": 3000, "n_blocks": int(len(uniq))}
    out["recall"] = {}
    for i, name in enumerate(names):
        out["recall"][name] = {
            "point": float(point[i]),
            "p05": float(np.quantile(draws[:, i], 0.05)),
            "p95": float(np.quantile(draws[:, i], 0.95)),
        }
    out["paired_difference"] = {}
    for i, name in enumerate(diff_names):
        out["paired_difference"][name] = {
            "point": float(point[[1, 2, 2][i]] - point[[0, 1, 0][i]]),
            "p05": float(np.quantile(diffs[:, i], 0.05)),
            "p95": float(np.quantile(diffs[:, i], 0.95)),
        }
    marginals = [out["recall"][n] for n in names]
    out["marginal_intervals_overlap"] = not (
        marginals[0]["p95"] < marginals[1]["p05"] or marginals[1]["p95"] < marginals[0]["p05"]
    ) or not (
        marginals[1]["p95"] < marginals[2]["p05"] or marginals[2]["p95"] < marginals[1]["p05"]
    )
    # The flag above is messy if only one pair overlaps. Store each pair.
    def overlaps(a, b):
        return not (a["p95"] < b["p05"] or b["p95"] < a["p05"])

    out["marginal_overlap"] = {
        "predicted_and_observed_mean": overlaps(marginals[0], marginals[1]),
        "observed_mean_and_p90": overlaps(marginals[1], marginals[2]),
        "predicted_and_p90": overlaps(marginals[0], marginals[2]),
    }
    return out


def summarize_rows(rows, key):
    return quant([row[key] for row in rows])


def main() -> None:
    import geopandas as gpd

    print("cross-year intervals", flush=True)
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    xy = np.column_stack([grid["x"].to_numpy(dtype=float), grid["y"].to_numpy(dtype=float)])
    built = grid["built"].to_numpy(dtype=float) >= 0.15
    lst = grid["LST_pred"].to_numpy(dtype=float)
    obs = grid["LST_C"].to_numpy(dtype=float)
    p90 = grid["LST_P90"].to_numpy(dtype=float)
    access = grid["access"].to_numpy(dtype=float)
    gap = np.clip(grid["Gap"].to_numpy(dtype=float), 0, 1)
    pop_square = grid["population"].to_numpy(dtype=float)
    universe = built & (pop_square > 0) & np.isfinite(lst) & np.isfinite(p90) & np.isfinite(obs)
    g5 = pd.read_parquet(
        ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2025.parquet",
        columns=["x", "y", "LST_C", "LST_P90"],
    )
    tree = cKDTree(np.column_stack([g5["x"].to_numpy(dtype=float), g5["y"].to_numpy(dtype=float)]))
    dist, ix = tree.query(xy, k=1)
    y25 = g5["LST_C"].to_numpy(dtype=float)[ix]
    sub = np.flatnonzero(universe & np.isfinite(y25) & (dist <= 200))
    blocks = spatial_block_ids(grid["x"].iloc[sub], grid["y"].iloc[sub], 3000).to_numpy()
    boot = block_bootstrap(
        [lst[sub], obs[sub], p90[sub]],
        [y25[sub], y25[sub], y25[sub]],
        blocks,
    )
    print("  overlap", boot["marginal_overlap"], flush=True)
    for name, row in boot["recall"].items():
        print(" ", name, round(row["point"], 3), round(row["p05"], 3), round(row["p95"], 3), flush=True)

    print("nearest-point fields", flush=True)
    ghsl_as = assign_population(GHSL_TIF, xy, boundary)
    wpop_as = assign_population(WPOP_TIF, xy, boundary)
    vor = ghsl_as["voronoi"]
    vor_w = wpop_as["voronoi"]
    _, oof, _, _, y_all = cv_predict(work, feats, "spatial", 3000, "lgbm")
    resid = y_all - oof
    built_rows = work["built"].to_numpy(dtype=float) >= 0.15
    tr_block = spatial_block_ids(work["x"], work["y"], 3000).to_numpy()
    g3 = spatial_block_ids(grid["x"], grid["y"], 3000).to_numpy()
    bag = {}
    for b, r in zip(tr_block[built_rows], resid[built_rows]):
        bag.setdefault(str(b), []).append(float(r))
    built_resid = resid[built_rows]
    codes, uniq = pd.factorize(g3)
    pools = [np.asarray(bag.get(str(b), []), dtype=float) for b in uniq]
    pools = [p if len(p) >= 3 else built_resid for p in pools]
    rng = np.random.default_rng(21)
    fields = [np.array([rng.choice(p) for p in pools])[codes] for _ in range(25)]
    ref = built & (vor > 0) & np.isfinite(lst) & np.isfinite(p90)
    w_equal = np.full(4, 0.25)
    base = priority(grid, lst, vor, ref, p90, access, w_equal)
    W = pd.read_csv(ROOT / "outputs" / "tables" / "uthi_weight_mc_draws.csv")
    W = W[["w_TS", "w_EXP", "w_COOL", "w_SERV"]].to_numpy(dtype=float)
    w_shift = W[int(np.argmax(np.abs(W - 0.25).sum(axis=1)))]

    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    cand = fac[fac["fclass"].astype(str).str.lower().isin({n.lower() for n in RETROFIT})].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])
    cand_ll = cand.to_crs(4326)

    def run_slice(field_ids, label):
        idx0 = demand_idx(base, vor)
        cov0 = coverage_matrix(xy[idx0], cand_xy, 1125.0)
        base_solve = solve_proven(vor[idx0], cov0, time_limit=120, label=f"{label}_base")
        exact0 = base_solve["sites"]
        rows = []
        hit = np.zeros(len(cand), dtype=int)
        for fi in field_ids:
            for kind, population in (("ghsl", vor), ("worldpop", vor_w)):
                for w_name, w in (("equal", w_equal), ("shifted", w_shift)):
                    mask = priority(grid, lst + fields[fi], population, ref, p90, access, w)
                    idx = demand_idx(mask, population)
                    pop_d = population[idx]
                    cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
                    solved = solve_proven(pop_d, cov, time_limit=60, label=f"{label}_{fi}_{kind}_{w_name}")
                    chosen = solved["sites"]
                    fixed_raw = covered_population(pop_d, cov, exact0)
                    opt_raw = covered_population(pop_d, cov, chosen)
                    weight_d = pop_d * (0.4 + 0.6 * gap[idx])
                    fixed_w = covered_population(weight_d, cov, exact0)
                    opt_w = covered_population(weight_d, cov, chosen)
                    total = float(np.nansum(population))
                    rows.append({
                        "field": int(fi),
                        "population": kind,
                        "weight": w_name,
                        "status": solved["status"],
                        "proven_optimal": solved["proven_optimal"],
                        "result_line": solved["result_line"],
                        "relative_gap": solved["relative_gap"],
                        "seconds": solved["seconds"],
                        "objective": solved["objective"],
                        "unweighted_fixed_raw": fixed_raw,
                        "unweighted_opt_raw": opt_raw,
                        "unweighted_delta_raw": opt_raw - fixed_raw,
                        "unweighted_fixed_census": fixed_raw / total * CENSUS,
                        "unweighted_opt_census": opt_raw / total * CENSUS,
                        "unweighted_delta_census": (opt_raw - fixed_raw) / total * CENSUS,
                        "objective_minus_unweighted_opt": None if solved["objective"] is None else solved["objective"] - opt_raw,
                        "weighted_fixed": fixed_w,
                        "weighted_opt": opt_w,
                        "weighted_delta": opt_w - fixed_w,
                        "normalized_regret": (opt_raw - fixed_raw) / max(float(pop_d.sum()), 1e-9),
                        "log": solved["log"],
                    })
                    hit[np.asarray(chosen, dtype=int)] += 1
                    print(" ", label, fi, kind, w_name, solved["proven_optimal"], round(solved["seconds"], 2), flush=True)
        return {"baseline": base_solve, "rows": rows, "hit": hit, "exact0": exact0, "idx0": idx0, "cov0": cov0}

    slice_a = run_slice([0, 4, 8, 12, 16, 20], "a")
    slice_b = run_slice([1, 5, 9, 13, 17, 21], "b")
    print("uniqueness", flush=True)
    second = solve_proven(
        vor[slice_a["idx0"]],
        slice_a["cov0"],
        banned=slice_a["exact0"],
        time_limit=120,
        label="a_second",
    )
    pop0 = vor[slice_a["idx0"]]
    first_people = covered_population(pop0, slice_a["cov0"], slice_a["exact0"])
    second_people = covered_population(pop0, slice_a["cov0"], second["sites"])
    total_vor = float(np.nansum(vor))
    only_first = sorted(set(slice_a["exact0"]) - set(second["sites"]))
    only_second = sorted(set(second["sites"]) - set(slice_a["exact0"]))

    def names_of(ids):
        out = []
        for i in ids:
            value = cand.iloc[i]["name"]
            out.append("" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))
        return out

    def pack(block):
        rows = block["rows"]
        proven = [r for r in rows if r["proven_optimal"]]
        by_pop = {}
        for kind in ("ghsl", "worldpop"):
            part = [r for r in rows if r["population"] == kind]
            by_pop[kind] = {
                "normalized_regret": summarize_rows(part, "normalized_regret"),
                "unweighted_delta_census": summarize_rows(part, "unweighted_delta_census"),
                "unweighted_fixed_census": summarize_rows(part, "unweighted_fixed_census"),
                "unweighted_opt_census": summarize_rows(part, "unweighted_opt_census"),
                "weighted_delta": summarize_rows(part, "weighted_delta"),
                "n_weighted_sign_differs": int(sum((r["weighted_delta"] > 0) != (r["unweighted_delta_raw"] > 0) and abs(r["weighted_delta"]) > 1e-6 and abs(r["unweighted_delta_raw"]) > 1e-6 for r in part)),
            }
        return {
            "n": len(rows),
            "n_proven_optimal": len(proven),
            "max_relative_gap": max((r["relative_gap"] or 0) for r in rows),
            "max_seconds": max(r["seconds"] for r in rows),
            "max_objective_minus_covered_population": max(abs(r["objective_minus_unweighted_opt"] or 0) for r in rows),
            "by_population": by_pop,
            "all_unweighted_delta_census": summarize_rows(rows, "unweighted_delta_census"),
            "all_normalized_regret": summarize_rows(rows, "normalized_regret"),
        }

    sites = []
    hit = slice_a["hit"]
    for i in slice_a["exact0"]:
        value = cand.iloc[i]["name"]
        text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
        sites.append({
            "name": text,
            "short_name": short_name(text),
            "lon": float(cand_ll.geometry.iloc[i].x),
            "lat": float(cand_ll.geometry.iloc[i].y),
            "frequency": float(hit[i] / len(slice_a["rows"])),
            "in_baseline": True,
        })
    sites = sorted(sites, key=lambda r: (-r["frequency"], r["short_name"]))
    payload = {
        "solver": "PuLP CBC, gapRel 0, gapAbs 0. Proven only when the CBC log says Optimal solution found.",
        "objective": "Sum of nearest-point population on the 450 largest population points in the scenario priority. Not population times the service-gap factor.",
        "recall_bootstrap": boot,
        "slice_a": pack(slice_a),
        "slice_b": pack(slice_b),
        "slice_a_rows": slice_a["rows"],
        "slice_b_rows": [{k: v for k, v in r.items() if k != "log"} for r in slice_b["rows"]],
        "baseline": {k: v for k, v in slice_a["baseline"].items() if k != "sites"},
        "uniqueness": {
            "first_proven": slice_a["baseline"]["proven_optimal"],
            "second_proven": second["proven_optimal"],
            "first_objective": slice_a["baseline"]["objective"],
            "second_objective": second["objective"],
            "first_result": slice_a["baseline"]["result_line"],
            "second_result": second["result_line"],
            "shared_sites": len(set(slice_a["exact0"]) & set(second["sites"])),
            "only_in_first": names_of(only_first),
            "only_in_second": names_of(only_second),
            "unweighted_delta_raw": first_people - second_people,
            "unweighted_delta_census": (first_people - second_people) / total_vor * CENSUS,
        },
        "siting_sites": sites,
        "slice_a_regrets": [r["normalized_regret"] for r in slice_a["rows"]],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("unique census", round(payload["uniqueness"]["unweighted_delta_census"], 1), payload["uniqueness"]["only_in_first"], payload["uniqueness"]["only_in_second"], flush=True)
    print("A ghsl", payload["slice_a"]["by_population"]["ghsl"]["unweighted_delta_census"], flush=True)
    print("saved", flush=True)
    draw(payload)
    print("figures", flush=True)


def scale_bar(ax, km, lat, corner="lower left"):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    dx = km / (111.32 * np.cos(np.deg2rad(lat)))
    if corner == "lower right":
        xb = x1 - 0.06 * (x1 - x0) - dx
        yb = y0 + 0.08 * (y1 - y0)
    else:
        xb = x0 + 0.08 * (x1 - x0)
        yb = y0 + 0.08 * (y1 - y0)
    ax.plot([xb, xb + dx], [yb, yb], color="black", lw=1.4, solid_capstyle="butt", zorder=5)
    ax.text(xb + dx / 2, yb + 0.012 * (y1 - y0), f"{km} km", ha="center", va="bottom", fontsize=7, zorder=5, color="black")


def draw(payload) -> None:
    style()
    sites = pd.DataFrame(payload["siting_sites"])
    rows = pd.DataFrame(payload["slice_a_rows"])
    ghsl = rows.loc[rows["population"] == "ghsl", "normalized_regret"].to_numpy(dtype=float)
    world = rows.loc[rows["population"] == "worldpop", "normalized_regret"].to_numpy(dtype=float)
    blue, verm, green = "#0072B2", "#D55E00", "#009E73"
    proven = payload["slice_a"]["n_proven_optimal"] == payload["slice_a"]["n"] and payload["baseline"]["proven_optimal"]
    regret_title = "B  Regret by population" if proven else "B  Solver-bounded regret"
    fig = plt.figure(figsize=(8.6, 5.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.45], height_ratios=[1, 1.25], wspace=0.92, hspace=0.45)
    ax = fig.add_subplot(gs[:, 0])
    width = 2 * 1125 / (111_320 * np.cos(np.deg2rad(43.8)))
    height = 2 * 1125 / 110_540
    ranked = sites.sort_values(["frequency", "short_name"], ascending=[False, True]).reset_index(drop=True)
    for _, row in ranked.iterrows():
        ax.add_patch(Ellipse((row.lon, row.lat), width=width, height=height, facecolor=blue, edgecolor=blue, alpha=0.16, lw=0.5, zorder=1))
    ax.scatter(ranked.lon, ranked.lat, c=ranked.frequency, s=60, cmap="cividis", vmin=0, vmax=1, zorder=3)
    for i, row in ranked.iterrows():
        ax.annotate(str(i + 1), (row.lon, row.lat), textcoords="offset points", xytext=(4, 3), fontsize=9)
    ax.set_xlim(87.50, 87.67)
    ax.set_ylim(43.75, 43.92)
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("A  Baseline sites, 1,125 m Euclidean")
    axr = fig.add_subplot(gs[0, 1])
    box = axr.boxplot(
        [ghsl, world],
        tick_labels=["GHSL", "WorldPop"],
        widths=0.55,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.2},
    )
    for patch, color in zip(box["boxes"], (verm, blue)):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)
    axr.set_ylabel("Exact regret")
    axr.set_title(regret_title)
    axr.tick_params(labelsize=8)
    axb = fig.add_subplot(gs[1, 1])
    order = ranked.sort_values("frequency", ascending=True)
    labels = []
    for _, row in order.iterrows():
        number = int(ranked.index[(ranked.lon == row.lon) & (ranked.lat == row.lat)][0]) + 1
        labels.append(f"{number}  {row.short_name}")
    axb.barh(np.arange(len(order)), order.frequency, color=green)
    axb.set_yticks(np.arange(len(order)), labels, fontsize=8)
    axb.tick_params(axis="y", pad=2)
    axb.set_xlim(0, 1.05)
    axb.set_xlabel("Selection frequency")
    axb.set_title("C  Return frequency")
    fig.savefig(FIG / "fig4_siting.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig4_siting.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    import geopandas as gpd

    maps = np.load(RES / "revision2_maps.npz")
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson").to_crs(4326)
    lon, lat = maps["lon"], maps["lat"]
    city_top = maps["city_top"].astype(bool)
    resid_top = maps["resid_top"].astype(bool)
    freq = maps["joint_freq"]
    inhabited = maps["pop_pos"].astype(bool)
    xlim, ylim = (87.30, 87.85), (43.65, 44.05)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 7.4))

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
    ax.legend(frameon=False, loc="upper left", markerscale=2, fontsize=7)
    ax.set_title("A  Citywide top 20%, Urumqi")
    ax.set_ylabel("Latitude")
    scale_bar(ax, 20, 43.8, "lower right")
    ax = axes[0, 1]
    frame(ax, zoom=True)
    ax.scatter(lon[resid_top], lat[resid_top], s=6, c=verm, marker="s", linewidths=0, rasterized=True)
    ax.set_title("B  Inhabited built-up top 20%")
    scale_bar(ax, 5, 43.8)
    window = (lon >= xlim[0]) & (lon <= xlim[1]) & (lat >= ylim[0]) & (lat <= ylim[1])
    ax = axes[1, 0]
    frame(ax, zoom=True)
    show = window & maps["universe"].astype(bool)
    sc = ax.scatter(lon[show], lat[show], c=freq[show], s=6, cmap="cividis", vmin=0, vmax=1, marker="s", linewidths=0, rasterized=True)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Scenario frequency")
    ax.set_title("C  Residential scenario frequency")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    scale_bar(ax, 5, 43.8, "lower right")
    ax = axes[1, 1]
    frame(ax, zoom=True)
    ax.scatter(lon[city_top & ~resid_top & inhabited & window], lat[city_top & ~resid_top & inhabited & window], s=8, c=blue, marker="s", linewidths=0, rasterized=True, label="Citywide only")
    ax.scatter(lon[resid_top & ~city_top & window], lat[resid_top & ~city_top & window], s=8, c="#E69F00", marker="s", linewidths=0, rasterized=True, label="Residential only")
    ax.scatter(lon[city_top & resid_top & window], lat[city_top & resid_top & window], s=10, c=verm, marker="s", linewidths=0, rasterized=True, label="Both")
    ax.legend(frameon=False, loc="upper left", fontsize=9, markerscale=1.8)
    ax.set_title("D  Where the two sets meet")
    ax.set_xlabel("Longitude")
    scale_bar(ax, 5, 43.8, "lower right")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_priority_stability.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG / "fig3_priority_stability.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
