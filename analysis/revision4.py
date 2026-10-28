"""Fourth-round checks. Writes only under analysis/."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from analysis import FIG, RES, coverage_matrix, covered_weight, cv_predict  # noqa: E402
from revision2 import census_people, jaccard  # noqa: E402
from revision3 import GHSL_TIF, WPOP_TIF, assign_population, hot_recall, priority, solve_exact, style  # noqa: E402
from src.config import get_settings  # noqa: E402
from src.geoai.spatial_cv import spatial_block_ids  # noqa: E402
from src.geoai.train_geoai import available_features  # noqa: E402
from src.gis.osm_network import RETROFIT  # noqa: E402

OUT = RES / "revision4_summary.json"
CENSUS = 4_054_369


def rank_block(score, seasonal, p90):
    return {
        "spearman_vs_seasonal": float(spearmanr(score, seasonal).statistic),
        "spearman_vs_p90": float(spearmanr(score, p90).statistic),
        "recall_seasonal_hot_fifth": hot_recall(score, seasonal),
        "recall_p90_hot_fifth": hot_recall(score, p90),
    }


def solve_alternates(weights, cov, k=10, time_limit=8, max_alts=4):
    import pulp
    from revision3 import _SOLVE

    first, obj, status = solve_exact(weights, cov, k=k, time_limit=time_limit)
    found = [{"sites": list(first), "objective": obj, "worse": False}]
    banned = [list(first)]
    for _ in range(max_alts):
        _SOLVE["n"] += 1
        tag = f"alt{_SOLVE['n']}"
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
        for prev in banned:
            if prev:
                prob += pulp.lpSum(y[int(j)] for j in prev) <= max(len(prev) - 1, 0)
        st = prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0.01))
        chosen = [j for j, v in enumerate(y) if v.value() is not None and v.value() > 0.5]
        obj2 = pulp.value(prob.objective)
        if obj2 is None or pulp.LpStatus[st] != "Optimal":
            break
        worse = obj is not None and obj2 < obj - 1e-4 * max(abs(obj), 1.0)
        found.append({"sites": chosen, "objective": float(obj2), "worse": worse})
        if worse:
            break
        banned.append(chosen)
    same = [f for f in found if not f["worse"]]
    return {
        "n_equal_optima_found": len(same),
        "first_objective": obj,
        "first_status": status,
        "later_objectives": [f["objective"] for f in found[1:]],
        "site_sets": [f["sites"] for f in same],
        "stopped_because_worse": any(f["worse"] for f in found),
    }


def demand_idx(mask, population, cap=450):
    score = np.where(mask, population, 0.0)
    idx = np.flatnonzero(score > 0)
    if len(idx) > cap:
        idx = idx[np.argsort(-score[idx])[:cap]]
    return idx


def main() -> None:
    import geopandas as gpd

    print("load", flush=True)
    grid = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2024.parquet")
    train = gpd.read_parquet(ROOT / "data" / "processed" / "heatcheck_train.parquet")
    boundary = gpd.read_file(ROOT / "data" / "raw" / "boundary_Wulumuqishi_union.geojson")
    settings = get_settings()
    feats = available_features(train, settings.feature_cols)
    work = train.dropna(subset=feats + ["LST_C", "x", "y", "built"]).copy()
    xy = np.column_stack([grid["x"].to_numpy(dtype=float), grid["y"].to_numpy(dtype=float)])
    pop = grid["population"].to_numpy(dtype=float)
    built_p = grid["built"].to_numpy(dtype=float)
    lst = grid["LST_pred"].to_numpy(dtype=float)
    obs = grid["LST_C"].to_numpy(dtype=float)
    p90 = grid["LST_P90"].to_numpy(dtype=float)
    access = grid["access"].to_numpy(dtype=float)
    built = built_p >= 0.15
    universe = built & (pop > 0) & np.isfinite(lst) & np.isfinite(p90) & np.isfinite(obs)
    g5 = pd.read_parquet(
        ROOT / "data" / "processed" / "heatcheck_grid_Wulumuqishi_2025.parquet",
        columns=["x", "y", "LST_C", "LST_P90"],
    )
    tree = cKDTree(np.column_stack([g5["x"].to_numpy(dtype=float), g5["y"].to_numpy(dtype=float)]))
    dist, ix = tree.query(xy, k=1)
    y25 = g5["LST_C"].to_numpy(dtype=float)[ix]
    p25 = g5["LST_P90"].to_numpy(dtype=float)[ix]
    ok = universe & np.isfinite(y25) & np.isfinite(p25)
    within = ok & (dist <= 200)
    sub = np.flatnonzero(within)
    targets = ix[sub]
    _, counts = np.unique(targets, return_counts=True)
    keep = {}
    for row, target, d in zip(sub, targets, dist[sub]):
        prev = keep.get(int(target))
        if prev is None or d < prev[1]:
            keep[int(target)] = (int(row), float(d))
    unique_rows = np.array([v[0] for v in keep.values()], dtype=int)

    def pack(rows):
        return {
            "n": int(len(rows)),
            "observed_2024_seasonal_mean": rank_block(obs[rows], y25[rows], p25[rows]),
            "predicted_2024_seasonal_mean": rank_block(lst[rows], y25[rows], p25[rows]),
            "observed_2024_p90": rank_block(p90[rows], y25[rows], p25[rows]),
        }

    match = {
        "universe_n": int(universe.sum()),
        "within_200m": int(within.sum()),
        "unmatched_or_farther_than_200m": int(universe.sum() - within.sum()),
        "distance_m_mean": float(dist[sub].mean()),
        "distance_m_median": float(np.median(dist[sub])),
        "distance_m_max": float(dist[sub].max()),
        "unique_2025_targets": int(len(counts)),
        "targets_used_more_than_once": int((counts > 1).sum()),
        "extra_2024_points_on_a_shared_target": int((counts[counts > 1] - 1).sum()) if (counts > 1).any() else 0,
        "max_2024_points_on_one_2025_point": int(counts.max()),
    }
    cross = {
        "all_matches_within_200m": pack(sub),
        "one_2024_point_per_2025_point": pack(unique_rows),
        "match": match,
    }
    print("match", match, flush=True)
    for name, block in cross["all_matches_within_200m"].items():
        if name == "n":
            continue
        print(" ", name, {k: round(v, 3) for k, v in block.items()}, flush=True)

    print("nearest-point metrics", flush=True)
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
    pools = []
    for b in uniq:
        local = bag.get(str(b), [])
        pools.append(np.asarray(local if len(local) >= 3 else built_resid, dtype=float))
    rng = np.random.default_rng(21)
    fields = []
    for _ in range(25):
        fields.append(np.array([rng.choice(p) for p in pools])[codes])
    ref = built & (vor > 0) & np.isfinite(lst) & np.isfinite(p90)
    w_equal = np.full(4, 0.25)
    base = priority(grid, lst, vor, ref, p90, access, w_equal)
    W = pd.read_csv(ROOT / "outputs" / "tables" / "uthi_weight_mc_draws.csv")
    W = W[["w_TS", "w_EXP", "w_COOL", "w_SERV"]].to_numpy(dtype=float)
    shift_row = int(np.argmax(np.abs(W - 0.25).sum(axis=1)))
    w_shift = W[shift_row]
    W_joint = W[::50][:8]

    def ensemble(field_list, weight_rows, populations):
        jacs = []
        hits = np.zeros(len(grid), dtype=np.int32)
        for delta in field_list:
            for population in populations:
                for w in weight_rows:
                    mask = priority(grid, lst + delta, population, ref, p90, access, w)
                    hits += mask
                    jacs.append(jaccard(mask, base))
        freq = hits / max(len(jacs), 1)
        stable = freq >= 0.90
        return {
            "n": len(jacs),
            "jaccard_mean": float(np.mean(jacs)),
            "stable90_n": int(stable.sum()),
            "stable90_people": census_people(vor, stable),
        }

    temp = ensemble(fields, [w_equal], [vor])
    joint = ensemble(fields, list(W_joint), [vor, vor_w])
    print("  temp", round(temp["jaccard_mean"], 3), "joint", round(joint["jaccard_mean"], 3), flush=True)

    fac = gpd.read_file(ROOT / "data" / "processed" / "facilities.geojson").to_crs(32645)
    fac["x"] = fac.geometry.centroid.x
    fac["y"] = fac.geometry.centroid.y
    cand = fac[fac["fclass"].astype(str).str.lower().isin({n.lower() for n in RETROFIT})].dropna(subset=["x", "y"]).head(180).copy()
    cand_xy = np.column_stack([cand["x"].to_numpy(dtype=float), cand["y"].to_numpy(dtype=float)])
    cand_ll = cand.to_crs(4326)

    def run_slice(field_ids, label):
        regrets, people_loss, full_fixed, statuses = [], [], [], []
        hit = np.zeros(len(cand), dtype=int)
        idx0 = demand_idx(base, vor)
        cov0 = coverage_matrix(xy[idx0], cand_xy, 1125.0)
        exact0, obj0, status0 = solve_exact(vor[idx0], cov0, time_limit=15)
        base_share = covered_weight(vor[idx0], cov0, exact0) / max(float(vor[idx0].sum()), 1e-9)
        all_idx = np.flatnonzero(base & (vor > 0))
        all_cov = coverage_matrix(xy[all_idx], cand_xy, 1125.0)
        all_share = covered_weight(vor[all_idx], all_cov, exact0) / max(float(vor[all_idx].sum()), 1e-9)
        for fi in field_ids:
            for population in (vor, vor_w):
                for w in (w_equal, w_shift):
                    mask = priority(grid, lst + fields[fi], population, ref, p90, access, w)
                    idx = demand_idx(mask, population)
                    ww = population[idx]
                    cov = coverage_matrix(xy[idx], cand_xy, 1125.0)
                    chosen, _, status = solve_exact(ww, cov, time_limit=6)
                    fixed = covered_weight(ww, cov, exact0) / max(float(ww.sum()), 1e-9)
                    again = covered_weight(ww, cov, chosen) / max(float(ww.sum()), 1e-9)
                    regrets.append(again - fixed)
                    people_loss.append((again - fixed) * float(ww.sum()) / max(float(np.nansum(population)), 1e-9) * CENSUS)
                    full_idx = np.flatnonzero(mask & (population > 0))
                    full_cov = coverage_matrix(xy[full_idx], cand_xy, 1125.0)
                    full_fixed.append(
                        covered_weight(population[full_idx], full_cov, exact0) / max(float(population[full_idx].sum()), 1e-9)
                    )
                    statuses.append(status)
                    hit[np.asarray(chosen, dtype=int)] += 1
        regrets = np.asarray(regrets, dtype=float)
        people_loss = np.asarray(people_loss, dtype=float)
        return {
            "label": label,
            "field_indices": [int(i) for i in field_ids],
            "n": int(len(regrets)),
            "baseline_status": status0,
            "baseline_objective": obj0,
            "baseline_share_of_450": float(base_share),
            "baseline_share_of_all_priority": float(all_share),
            "exact_regret_p50": float(np.median(regrets)),
            "exact_regret_p95": float(np.quantile(regrets, 0.95)),
            "exact_regret_min": float(regrets.min()),
            "exact_regret_max": float(regrets.max()),
            "people_loss_p50": float(np.median(people_loss)),
            "people_loss_p95": float(np.quantile(people_loss, 0.95)),
            "people_loss_max": float(people_loss.max()),
            "full_fixed_p50": float(np.median(full_fixed)),
            "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
            "always_selected": int(sum(hit[i] == len(regrets) for i in exact0)),
            "n_ever": int((hit > 0).sum()),
            "regret_values": [float(v) for v in regrets],
            "exact0": [int(i) for i in exact0],
            "hit": hit,
            "idx0": idx0,
            "cov0": cov0,
        }

    slice_a = run_slice(list(range(0, 25, 4))[:6], "fields 0,4,8,12,16,20")
    slice_b = run_slice(list(range(1, 25, 4))[:6], "fields 1,5,9,13,17,21")
    print("slice A", round(slice_a["exact_regret_p50"], 4), "people", round(slice_a["people_loss_p50"]), "B", round(slice_b["exact_regret_p50"], 4), flush=True)
    alt = solve_alternates(vor[slice_a["idx0"]], slice_a["cov0"], time_limit=12, max_alts=4)
    print("equal optima", alt["n_equal_optima_found"], alt["later_objectives"], flush=True)
    sites = []
    hit = slice_a["hit"]
    for i in np.argsort(-hit):
        if hit[i] <= 0 and i not in set(slice_a["exact0"]):
            continue
        if i not in set(slice_a["exact0"]):
            continue
        sites.append({
            "name": None if pd.isna(cand.iloc[i]["name"]) else str(cand.iloc[i]["name"]),
            "lon": float(cand_ll.geometry.iloc[i].x),
            "lat": float(cand_ll.geometry.iloc[i].y),
            "frequency": float(hit[i] / slice_a["n"]),
            "in_baseline": True,
        })

    def slim(block):
        return {k: v for k, v in block.items() if k not in {"regret_values", "hit", "idx0", "cov0", "exact0"}}

    payload = {
        "cross_year": cross,
        "nearest_point": {
            "universe_n": int(ref.sum()),
            "priority_n": int(base.sum()),
            "priority_people": census_people(vor, base),
            "temperature_only": temp,
            "joint_400": joint,
        },
        "scenario_design": {
            "temperature_draws": 25,
            "temperature_seed": 21,
            "residual_pool": "Built-up out-of-fold residuals. A 3 km block uses its own residuals when at least three exist; otherwise it draws from the built-up pool.",
            "slice_a_rule": "Indices 0, 4, 8, 12, 16 and 20 of that fixed draw. Not selected because they were extreme.",
            "slice_b_rule": "Indices 1, 5, 9, 13, 17 and 21. A shifted slice, used to see whether the first slice was unusual.",
            "weights": {
                "equal": [0.25, 0.25, 0.25, 0.25],
                "shifted_row_in_archived_draws": shift_row,
                "shifted_vector": [float(v) for v in w_shift],
                "why": "The archived weight vector with the largest L1 distance from equal weights.",
            },
            "populations": "Nearest-point GHSL and nearest-point WorldPop.",
            "solver": "PuLP CBC, 6 second limit, 1 percent relative gap. The baseline uses 15 seconds.",
        },
        "siting_slice_a": slim(slice_a),
        "siting_slice_b": slim(slice_b),
        "siting_slice_a_regrets": slice_a["regret_values"],
        "alternate_optima_baseline": {
            "n_equal_optima_found": alt["n_equal_optima_found"],
            "first_objective": alt["first_objective"],
            "first_status": alt["first_status"],
            "later_objectives": alt["later_objectives"],
            "stopped_because_worse": alt["stopped_because_worse"],
            "pairwise_shared_sites": [len(set(alt["site_sets"][0]) & set(other)) for other in alt["site_sets"][1:]],
        },
        "siting_sites": sites,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("saved", flush=True)
    draw(payload)
    print("figures", flush=True)


def draw(payload) -> None:
    style()
    prev = json.loads((RES / "revision3_summary.json").read_text(encoding="utf-8"))
    sites = pd.DataFrame(payload["siting_sites"])
    regret = np.asarray(payload["siting_slice_a_regrets"], dtype=float)
    blue, verm, green = "#0072B2", "#D55E00", "#009E73"
    fig = plt.figure(figsize=(7.8, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.15], height_ratios=[1, 1.35], wspace=0.72, hspace=0.48)
    ax = fig.add_subplot(gs[:, 0])
    width = 2 * 1125 / (111_320 * np.cos(np.deg2rad(43.8)))
    height = 2 * 1125 / 110_540
    ranked = sites.sort_values(["frequency", "name"], ascending=[False, True]).reset_index(drop=True)
    for _, row in ranked.iterrows():
        ax.add_patch(Ellipse((row.lon, row.lat), width=width, height=height, facecolor=blue, edgecolor=blue, alpha=0.16, lw=0.5, zorder=1))
    sc = ax.scatter(ranked.lon, ranked.lat, c=ranked.frequency, s=60, cmap="cividis", vmin=0, vmax=1, zorder=3)
    for i, row in ranked.iterrows():
        ax.annotate(str(i + 1), (row.lon, row.lat), textcoords="offset points", xytext=(5, 4), fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.02, label="Frequency")
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("A  Baseline sites, 1,125 m Euclidean")
    axr = fig.add_subplot(gs[0, 1])
    axr.hist(regret, bins=8, color=verm, edgecolor="white")
    axr.axvline(float(np.median(regret)), color="black", lw=1, linestyle="--")
    axr.set_xlabel("Exact regret")
    axr.set_ylabel("Scenarios")
    axr.set_title("B  Exact regret")
    axb = fig.add_subplot(gs[1, 1])
    order = ranked.sort_values("frequency", ascending=True)
    labels = []
    for _, row in order.iterrows():
        name = "" if not isinstance(row["name"], str) else row["name"].split("（")[0].split("(")[0]
        number = int(ranked.index[(ranked.lon == row.lon) & (ranked.lat == row.lat)][0]) + 1
        labels.append(f"{number}  {name if name else 'site'}")
    axb.barh(np.arange(len(order)), order.frequency, color=green)
    axb.set_yticks(np.arange(len(order)), labels, fontsize=6)
    axb.set_xlim(0, 1)
    axb.set_xlabel("Selection frequency")
    axb.set_title("C  Return frequency")
    fig.savefig(FIG / "fig4_siting.png", bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6), sharex="col")
    for col, key, title in (
        (0, "variogram_citywide_full_pairs", "Citywide residuals"),
        (1, "variogram_built_full_pairs", "Built-up residuals"),
    ):
        block = prev["conditional_error"][key]
        bins = block["bins"]
        xx = [b["distance_m"] / 1000 for b in bins]
        axes[0, col].plot(xx, [b["gamma"] for b in bins], color=verm, marker="o", ms=3)
        axes[0, col].axhline(block["sill"], color="#444444", lw=0.8, linestyle="--")
        axes[0, col].set_title(title)
        axes[0, col].set_ylabel("Semivariance (°C²)")
        axes[1, col].bar(xx, [b["n_pairs"] for b in bins], width=0.8, color=blue)
        axes[1, col].set_ylabel("Pairs in the bin")
        axes[1, col].set_xlabel("Distance (km)")
    fig.tight_layout()
    fig.savefig(FIG / "fig6_variogram.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

