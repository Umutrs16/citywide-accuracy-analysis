from __future__ import annotations

import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

from src.config import Settings
from src.utils import minmax, save_json, table_file

CLASS_A = {
    "library",
    "community_centre",
    "theatre",
    "arts_centre",
    "university",
    "school",
    "college",
    "marketplace",
    "museum",
}
CLASS_B = {"hospital", "clinic"}
CLASS_C = {"park", "garden", "playground", "recreation_ground", "park_proxy", "sports_centre", "stadium", "swimming_pool"}
RETROFIT = CLASS_A | {"sports_centre", "stadium", "swimming_pool"}
CLASS_WEIGHT = {"A": 1.0, "B": 0.8, "C": 0.5}


def classify_facility(fclass) -> str:
    name = str(fclass or "").lower()
    if name in CLASS_A:
        return "A"
    if name in CLASS_B:
        return "B"
    if name in CLASS_C:
        return "C"
    return "C"


def facility_weight(fclass) -> float:
    return CLASS_WEIGHT[classify_facility(fclass)]


def _polygon_from_gdf(gdf: gpd.GeoDataFrame):
    geom = unary_union(gdf.to_crs(4326).geometry)
    if geom.geom_type == "Polygon":
        return geom
    if geom.geom_type == "MultiPolygon":
        return max(geom.geoms, key=lambda g: g.area)
    return box(*gdf.to_crs(4326).total_bounds)


def builtup_polygon(grid: gpd.GeoDataFrame, built_thr: float = 0.15):
    """只取连续建成/高人口区，避免对 1.4 万 km² 全市拉路网。"""
    g = grid.to_crs(4326).copy()
    built = pd.to_numeric(g.get("built", 0), errors="coerce").fillna(0)
    pop = pd.to_numeric(g.get("pop_use", g.get("population", 0)), errors="coerce").fillna(0)
    mask = (built >= built_thr) | (pop >= max(float(pop.quantile(0.92)), 1.0))
    sub = g.loc[mask]
    if len(sub) < 80:
        sub = g
    geom = unary_union(sub.geometry.buffer(0.012))
    if geom.geom_type == "MultiPolygon":
        geom = max(geom.geoms, key=lambda x: x.area)
    return geom


def _green_proxy(grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    green = grid.copy()
    if "NDVI" in green.columns:
        green = green[green["NDVI"] >= green["NDVI"].quantile(0.92)]
    if green.empty:
        green = grid.sample(min(25, len(grid)), random_state=1)
    green = green.to_crs(4326)
    return gpd.GeoDataFrame(
        {
            "name": [f"green_proxy_{i}" for i in range(len(green))],
            "fclass": "park_proxy",
            "geometry": green.geometry.centroid.values,
        },
        crs=4326,
    )


def fetch_osm(settings: Settings, grid: gpd.GeoDataFrame, tag: str = "") -> tuple[gpd.GeoDataFrame, object | None]:
    facilities = gpd.GeoDataFrame(columns=["name", "fclass", "geometry"], geometry="geometry", crs=4326)
    graph = None
    try:
        from src.gis.osm_offline import CORE, downtown_polygon, fetch_downtown_osm

        g4326 = grid.to_crs(4326)
        core_poly = downtown_polygon(settings)
        if g4326.intersects(core_poly).any():
            fac_off, graph_off, meta = fetch_downtown_osm(settings)
            if len(fac_off) >= 20:
                outside = g4326[~g4326.intersects(core_poly)]
                proxy = _green_proxy(outside if len(outside) > 50 else g4326)
                facilities = gpd.GeoDataFrame(
                    pd.concat([fac_off[["name", "fclass", "geometry"]], proxy], ignore_index=True),
                    geometry="geometry",
                    crs=4326,
                )
                graph = graph_off
                save_json({**meta, "osm_network": "downtown_xml_plus_proxy", "n_proxy": int(len(proxy))}, table_file(settings, "osm_status", tag))
    except Exception as exc:
        save_json({"osm_offline_error": str(exc)}, table_file(settings, "osm_status", tag))
    if facilities.empty:
        try:
            poly = builtup_polygon(grid)
        except Exception:
            poly = _polygon_from_gdf(grid)
        tags = settings.raw["osm"]["facility_tags"]
        try:
            import osmnx as ox

            ox.settings.use_cache = True
            ox.settings.timeout = 240
            last_err = None
            pois = gpd.GeoDataFrame()
            for endpoint in (
                "https://overpass.kumi.systems/api",
                "https://overpass.openstreetmap.fr/api",
                "https://maps.mail.ru/osm/tools/overpass/api",
                "https://overpass-api.de/api",
            ):
                try:
                    ox.settings.overpass_endpoint = endpoint
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        graph = ox.graph_from_polygon(poly, network_type=settings.raw["osm"]["network_type"], simplify=True)
                        pois = ox.features_from_polygon(poly, tags)
                    last_err = None
                    break
                except Exception as exc:
                    last_err = exc
                    graph = None
                    pois = gpd.GeoDataFrame()
            if last_err is not None and graph is None:
                raise last_err
            if not pois.empty:
                pois = pois.reset_index()
                keep = [c for c in ["name", "amenity", "leisure", "landuse", "tourism", "geometry"] if c in pois.columns]
                pois = pois[keep].copy()
                pois = pois.to_crs(3857)
                pois["geometry"] = pois.geometry.centroid
                pois = pois.to_crs(4326)
                pois["fclass"] = (
                    pois.get("amenity", pd.Series(index=pois.index))
                    .fillna(pois.get("leisure", pd.Series(index=pois.index)))
                    .fillna(pois.get("tourism", pd.Series(index=pois.index)))
                    .fillna("public")
                )
                facilities = gpd.GeoDataFrame(pois, geometry="geometry", crs=4326)
            if graph is not None:
                save_json({"osm_network": "builtup", "n_nodes": int(len(graph.nodes))}, table_file(settings, "osm_status", tag))
        except Exception as exc:
            save_json({"osm_error": str(exc)}, table_file(settings, "osm_status", tag))
    if facilities.empty:
        facilities = _green_proxy(grid)
    facilities = facilities.copy()
    facilities["service_class"] = facilities.get("fclass", "public").map(classify_facility)
    facilities["service_weight"] = facilities["service_class"].map(CLASS_WEIGHT)
    return facilities, graph


def accessibility(
    settings: Settings,
    grid: gpd.GeoDataFrame,
    facilities: gpd.GeoDataFrame,
    graph=None,
    tag: str = "",
) -> gpd.GeoDataFrame:
    out = grid.copy()
    metric = settings.crs_metric
    g = out.to_crs(metric)
    f = facilities.to_crs(metric)
    speed = float(settings.raw["access"]["walk_speed_kmh"])
    minutes = float(settings.raw["access"]["minutes"])
    radius = speed * 1000.0 * minutes / 60.0

    g_cent = g.geometry.centroid
    f_cent = f.geometry.centroid
    if "service_weight" not in f.columns:
        f["service_weight"] = f.get("fclass", "public").map(facility_weight)
        f["service_class"] = f.get("fclass", "public").map(classify_facility)
    coords_g = np.column_stack([g_cent.x, g_cent.y])
    coords_f = np.column_stack([f_cent.x, f_cent.y])
    weights_f = pd.to_numeric(f["service_weight"], errors="coerce").fillna(0.5).to_numpy()
    dmin = np.full(len(coords_g), np.inf)
    service = np.zeros(len(coords_g), dtype=float)
    indoor_dist = np.full(len(coords_g), np.inf)
    chunk = 400
    indoor = f["service_class"].isin(["A", "B"]).to_numpy() if "service_class" in f.columns else np.ones(len(f), dtype=bool)
    for i in range(0, len(coords_g), chunk):
        sl = coords_g[i : i + chunk]
        d = np.sqrt(((sl[:, None, :] - coords_f[None, :, :]) ** 2).sum(axis=2))
        dmin[i : i + chunk] = d.min(axis=1)
        contrib = weights_f[None, :] * np.clip(1.0 - d / radius, 0, 1)
        service[i : i + chunk] = contrib.max(axis=1)
        if indoor.any():
            indoor_dist[i : i + chunk] = d[:, indoor].min(axis=1)
    eu_dist = dmin
    eu_access = service

    net_access = None
    if graph is not None:
        try:
            import osmnx as ox

            G = ox.project_graph(graph, to_crs=metric)
            nodes, _ = ox.graph_to_gdfs(G)
            real = ~f.get("fclass", pd.Series("", index=f.index)).astype(str).str.contains("proxy", case=False)
            poi_xy = np.column_stack([f_cent.x.to_numpy()[real.to_numpy()], f_cent.y.to_numpy()[real.to_numpy()]])
            if len(poi_xy) == 0:
                poi_xy = np.column_stack([f_cent.x, f_cent.y])
            poi_nodes = []
            for x, y in poi_xy:
                nid = ox.nearest_nodes(G, x, y)
                poi_nodes.append(nid)
            poi_nodes = list(set(poi_nodes))
            dist_map = nx.multi_source_dijkstra_path_length(G, poi_nodes, weight="length")
            node_x = nodes.geometry.x.to_numpy()
            node_y = nodes.geometry.y.to_numpy()
            node_ids = nodes.index.to_numpy()
            gx = g_cent.x.to_numpy()
            gy = g_cent.y.to_numpy()
            prefilter = 1500.0
            acc = np.full(len(gx), np.nan)
            snap = np.full(len(gx), np.inf)
            for i, (x, y) in enumerate(zip(gx, gy)):
                if eu_dist[i] > prefilter:
                    continue
                j = int(np.argmin((node_x - x) ** 2 + (node_y - y) ** 2))
                snap_i = float(np.hypot(node_x[j] - x, node_y[j] - y))
                snap[i] = snap_i
                if snap_i > prefilter:
                    continue
                d = dist_map.get(node_ids[j], np.inf)
                if not np.isfinite(d):
                    continue
                d_net = d + snap_i
                acc[i] = 0.0 if d_net >= radius else max(0.0, 1.0 - d_net / radius)
            net_access = acc
        except Exception:
            net_access = None

    out["dist_facility_m"] = eu_dist
    out["access_euclid"] = eu_access
    if net_access is not None:
        out["access"] = np.where(np.isfinite(net_access), net_access, eu_access)
        out["access_mode"] = np.where(np.isfinite(net_access), "network", "euclid")
    else:
        out["access"] = eu_access
        out["access_mode"] = "euclid"
    out["covered_15min"] = (out["access"] > 0).astype(int)
    out["covered_15min_indoor"] = (indoor_dist <= radius).astype(int)

    if "pop_use" not in out.columns:
        out["pop_use"] = pd.to_numeric(out.get("population", 0), errors="coerce").fillna(0)
    if "UTHI" not in out.columns:
        out["UTHI"] = 0.5
    out["Gap"] = minmax(out["UTHI"]) * minmax(out["pop_use"]) * (1.0 - out["access"].clip(0, 1))
    save_json(
        {
            "walk_speed_kmh": speed,
            "minutes": minutes,
            "radius_m": radius,
            "n_facilities": int(len(facilities)),
            "facility_class_counts": facilities.get("service_class", pd.Series(dtype=str)).value_counts().to_dict()
            if hasattr(facilities, "get")
            else {},
            "network_used": bool(net_access is not None),
            "access_definition": "欧氏≤1.5 km预筛后，主城用路网最短路径判定 d_net≤1125 m；外围无路网处退回欧氏。室内A/B欧氏覆盖只作理论上界。",
            "covered_pop_share": float(
                (out["pop_use"] * out["covered_15min"]).sum() / max(out["pop_use"].sum(), 1e-6)
            ),
            "covered_pop_share_indoor_euclid_upper": float(
                (out["pop_use"] * out["covered_15min_indoor"]).sum() / max(out["pop_use"].sum(), 1e-6)
            ),
            "n_network_cells": int((out.get("access_mode") == "network").sum()) if "access_mode" in out.columns else 0,
        },
        table_file(settings, "access_summary", tag),
    )
    return out
