#!/usr/bin/env python
# coding: utf-8

import os
import io
import zipfile
import tempfile
import shutil
import json
from typing import Tuple, List

import requests
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium.plugins import Fullscreen
from streamlit_folium import folium_static
import simplekml


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="GADM Boundary Exporter",
    page_icon="🗺️",
    layout="wide",
)


# -----------------------------
# Project folders (portable)
# -----------------------------
BASE_DATA_DIR = os.path.join("data", "gadm_shapefiles")
EXPORTS_DIR = "exports"
ASSETS_DIR = "assets"
LOGO_PATH = os.path.join(ASSETS_DIR, "logo.png")

os.makedirs(BASE_DATA_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)


# -----------------------------
# GADM helpers
# -----------------------------
def gadm_zip_url(country_code: str) -> str:
    code = country_code.upper().strip()
    return f"https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{code}_shp.zip"


def country_dir(country_code: str) -> str:
    code = country_code.upper().strip()
    path = os.path.join(BASE_DATA_DIR, code)
    os.makedirs(path, exist_ok=True)
    return path


def level_shp_path(country_code: str, level: int) -> str:
    code = country_code.upper().strip()
    return os.path.join(country_dir(code), f"gadm41_{code}_{level}.shp")


def download_and_extract_gadm(country_code: str) -> None:
    code = country_code.upper().strip()
    out_dir = country_dir(code)
    url = gadm_zip_url(code)
    zip_path = os.path.join(out_dir, f"gadm41_{code}_shp.zip")

    try:
        r = requests.get(url, timeout=90)
        if r.status_code != 200:
            raise RuntimeError(f"Download failed: HTTP {r.status_code}")

        with open(zip_path, "wb") as f:
            f.write(r.content)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(out_dir)

    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except OSError:
                pass


def available_levels(country_code: str, max_level: int = 4) -> List[int]:
    code = country_code.upper().strip()
    levels = []
    for lvl in range(0, max_level + 1):
        if os.path.exists(level_shp_path(code, lvl)):
            levels.append(lvl)
    return levels


@st.cache_data(show_spinner=False)
def load_country_gdf(country_code: str) -> gpd.GeoDataFrame:
    code = country_code.upper().strip()
    gdfs = []
    for lvl in range(0, 5):
        p = level_shp_path(code, lvl)
        if os.path.exists(p):
            g = gpd.read_file(p)
            g["level"] = lvl
            gdfs.append(g)

    if not gdfs:
        return gpd.GeoDataFrame()

    return gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)


def find_name_columns(gdf: gpd.GeoDataFrame) -> List[str]:
    return [c for c in gdf.columns if c.startswith("NAME_")]


def match_query(gdf: gpd.GeoDataFrame, query: str, name_cols: List[str]) -> gpd.GeoDataFrame:
    q = query.strip().lower()
    if not q or gdf.empty or not name_cols:
        return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)

    mask = gdf[name_cols].apply(
        lambda row: any(q in str(v).lower() for v in row.values if pd.notnull(v)),
        axis=1,
    )
    return gdf[mask].copy()


def build_label(row: pd.Series, name_cols: List[str]) -> str:
    parts = [str(row[c]) for c in name_cols if c in row and pd.notnull(row[c])]
    return " > ".join(parts) if parts else "Unnamed"


# -----------------------------
# Geodesy helpers
# -----------------------------
def utm_epsg_from_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 + zone) if lat >= 0 else (32700 + zone)


def centroid_and_area_km2(gdf_wgs84: gpd.GeoDataFrame) -> Tuple[Tuple[float, float], float, int]:
    if gdf_wgs84.empty:
        return (0.0, 0.0), 0.0, 0

    g = gdf_wgs84.to_crs(epsg=4326)
    approx_centroid = g.geometry.unary_union.centroid
    lon, lat = float(approx_centroid.x), float(approx_centroid.y)

    utm_epsg = utm_epsg_from_lonlat(lon, lat)
    projected = g.to_crs(epsg=utm_epsg)

    area_m2 = float(projected.geometry.area.sum())
    area_km2 = area_m2 / 1e6

    proj_centroid = projected.geometry.unary_union.centroid
    centroid_wgs = gpd.GeoSeries([proj_centroid], crs=f"EPSG:{utm_epsg}").to_crs(epsg=4326).iloc[0]
    return (float(centroid_wgs.y), float(centroid_wgs.x)), area_km2, utm_epsg


def _dynamic_simplify_tolerance_m(gdf_3857: gpd.GeoDataFrame) -> float:
    minx, miny, maxx, maxy = gdf_3857.total_bounds
    scale = max(maxx - minx, maxy - miny)

    if scale > 2_000_000:
        return 800.0
    if scale > 800_000:
        return 400.0
    if scale > 200_000:
        return 200.0
    if scale > 50_000:
        return 80.0
    return 20.0


def make_folium_map(gdf_wgs84: gpd.GeoDataFrame) -> folium.Map:
    g = gdf_wgs84.to_crs(epsg=4326).copy()
    g = g[g.geometry.notnull()].copy()
    if g.empty:
        return folium.Map(location=[0, 0], zoom_start=2, tiles="OpenStreetMap")

    g3857 = g.to_crs(epsg=3857).copy()
    g3857["geometry"] = g3857.geometry.buffer(0)

    tol = _dynamic_simplify_tolerance_m(g3857)
    g3857["geometry"] = g3857.geometry.simplify(tol, preserve_topology=True)
    g = g3857.to_crs(epsg=4326)

    (lat, lon), _, _ = centroid_and_area_km2(g)

    m = folium.Map(
        location=[lat, lon],
        zoom_start=6,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True,
    )

    Fullscreen(position="topleft").add_to(m)

    fg = folium.FeatureGroup(name="Boundary", show=True)
    folium.GeoJson(
        g,
        name="Boundary",
        style_function=lambda _: {
            "color": "#ff0000",
            "weight": 4,
            "opacity": 1.0,
            "fillColor": "#ff0000",
            "fillOpacity": 0.12,
        },
        highlight_function=lambda _: {"weight": 6, "fillOpacity": 0.20},
    ).add_to(fg)
    fg.add_to(m)

    try:
        bounds = g.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    except Exception:
        pass

    return m


# -----------------------------
# Export helpers
# -----------------------------
def export_to_kml_bytes(gdf_wgs84: gpd.GeoDataFrame) -> bytes:
    g = gdf_wgs84.to_crs(epsg=4326)
    kml = simplekml.Kml()

    for _, row in g.iterrows():
        geom = row.geometry
        if geom is None:
            continue

        if geom.geom_type == "Polygon":
            outer = [(x, y) for x, y in geom.exterior.coords]
            pol = kml.newpolygon(name="Area", outerboundaryis=outer)
            if geom.interiors:
                pol.innerboundaryis = [[(x, y) for x, y in ring.coords] for ring in geom.interiors]

        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                outer = [(x, y) for x, y in part.exterior.coords]
                pol = kml.newpolygon(name="Area", outerboundaryis=outer)
                if part.interiors:
                    pol.innerboundaryis = [[(x, y) for x, y in ring.coords] for ring in part.interiors]

    with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        kml.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def export_to_geojson_bytes(gdf_wgs84: gpd.GeoDataFrame) -> bytes:
    g = gdf_wgs84.to_crs(epsg=4326)
    return g.to_json().encode("utf-8")


def export_to_shapefile_zip_bytes(gdf_wgs84: gpd.GeoDataFrame, stem: str) -> bytes:
    g = gdf_wgs84.to_crs(epsg=4326)

    safe_stem = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in stem)[:50] or "export"
    with tempfile.TemporaryDirectory() as tmpdir:
        shp_path = os.path.join(tmpdir, f"{safe_stem}.shp")
        g.to_file(shp_path, driver="ESRI Shapefile")

        mem = io.BytesIO()
        with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for fn in os.listdir(tmpdir):
                if fn.startswith(safe_stem + "."):
                    z.write(os.path.join(tmpdir, fn), arcname=fn)

        mem.seek(0)
        return mem.read()


def build_export_filename(query: str, suffix: str, ext: str) -> str:
    base = query.strip().replace(" ", "_")
    base = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in base)
    base = base[:80] or "export"
    return f"{base}_{suffix}.{ext}"


# -----------------------------
# Persist selection for Map Preview tab
# -----------------------------
def gdf_from_geojson_str(geojson_str: str) -> gpd.GeoDataFrame:
    obj = json.loads(geojson_str)
    feats = obj.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(feats, crs="EPSG:4326")


# -----------------------------
# Cache clear logic
# -----------------------------
def clear_country_cache(country_code: str) -> None:
    code = country_code.upper().strip()
    if not code:
        return
    path = os.path.join(BASE_DATA_DIR, code)
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

    try:
        load_country_gdf.clear()
    except Exception:
        pass

    try:
        st.cache_data.clear()
    except Exception:
        pass


# -----------------------------
# Session state
# -----------------------------
if "recent_searches" not in st.session_state:
    st.session_state["recent_searches"] = []

if "last_export" not in st.session_state:
    st.session_state["last_export"] = None

if "selected_geojson" not in st.session_state:
    st.session_state["selected_geojson"] = None


# -----------------------------
# Sidebar UI (logo + controls + cache clear)
# -----------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, use_container_width=True)

    st.header("Controls")

    country_code = st.text_input("ISO3 Country Code", value="DEU").strip().upper()
    query = st.text_input("Search name", placeholder="Berlin, Punjab, Maharashtra...")

    fmt = st.selectbox("Export format", ["geojson", "shp", "kml"])
    combine = st.checkbox("Combine matches", value=False)

    admin_filter = st.multiselect(
        "Filter admin levels (optional)",
        options=[0, 1, 2, 3, 4],
        default=[0, 1, 2, 3, 4],
        help="Restrict search results to certain administrative levels.",
    )

    st.divider()

    col_a, col_b = st.columns(2)
    with col_a:
        run = st.button("Search & Export", use_container_width=True)
    with col_b:
        clear_btn = st.button("Clear cache", use_container_width=True)

    if clear_btn:
        if not country_code:
            st.warning("Enter a country code first.")
        else:
            clear_country_cache(country_code)
            st.session_state["last_export"] = None
            st.session_state["selected_geojson"] = None
            st.success(f"Cache cleared for {country_code}. Download will happen again next run.")

    st.caption("Tiles: OpenStreetMap only")


# -----------------------------
# Main UI
# -----------------------------
st.title("GADM Boundary Search and Export")
st.caption("Data source: GADM 4.1 shapefiles (UC Davis mirror). Downloaded on-demand and stored locally per country.")

# Ensure data exists for the country
if country_code:
    lvl0 = level_shp_path(country_code, 0)
    if not os.path.exists(lvl0):
        with st.spinner(f"Downloading GADM shapefiles for {country_code}..."):
            try:
                download_and_extract_gadm(country_code)
                load_country_gdf.clear()
            except Exception as e:
                st.error(f"Could not download GADM for {country_code}. Error: {e}")

full_gdf = load_country_gdf(country_code) if country_code else gpd.GeoDataFrame()

# KPI row
k1, k2, k3, k4 = st.columns(4)
levels = available_levels(country_code) if country_code else []
k1.metric("Country", country_code if country_code else "-")
k2.metric("Loaded Levels", ", ".join(map(str, levels)) if levels else "None")
k3.metric("Cache Folder", os.path.join(BASE_DATA_DIR, country_code) if country_code else "-")
k4.metric("Exports Folder", EXPORTS_DIR)

tabs = st.tabs(["Search Results", "Map Preview", "Export", "Data Summary"])

matches = gpd.GeoDataFrame()
selected_wgs84 = gpd.GeoDataFrame()

if run:
    if not country_code:
        st.error("Please enter a country code.")
    elif full_gdf.empty:
        st.error("No data loaded for this country code.")
    elif not query.strip():
        st.error("Please enter a search term.")
    else:
        name_cols = find_name_columns(full_gdf)
        if not name_cols:
            st.error("No NAME_* columns found in the dataset.")
        else:
            filtered = full_gdf[full_gdf["level"].isin(admin_filter)].copy()
            matches = match_query(filtered, query, name_cols)

            rs = st.session_state["recent_searches"]
            rs.insert(0, f"{country_code}: {query}")
            st.session_state["recent_searches"] = rs[:5]

            if matches.empty:
                st.warning("No match found.")
            else:
                st.success(f"Found {len(matches)} match(es).")

                if combine:
                    selected = matches.dissolve().reset_index(drop=True)
                    suffix = "combined"
                else:
                    labels = [build_label(row, name_cols) for _, row in matches.iterrows()]
                    idx = st.selectbox(
                        "Select one match",
                        options=list(range(len(labels))),
                        format_func=lambda i: labels[i],
                    )
                    selected = matches.iloc[[idx]].copy()
                    suffix = "selected"

                selected_wgs84 = selected.to_crs(epsg=4326)

                # Persist boundary so Map Preview tab always shows it
                st.session_state["selected_geojson"] = selected_wgs84.to_json()

                (lat, lon), area_km2, utm_epsg = centroid_and_area_km2(selected_wgs84)

                if fmt == "geojson":
                    data_bytes = export_to_geojson_bytes(selected_wgs84)
                    file_name = build_export_filename(query, suffix, "geojson")
                    mime = "application/geo+json"
                elif fmt == "kml":
                    data_bytes = export_to_kml_bytes(selected_wgs84)
                    file_name = build_export_filename(query, suffix, "kml")
                    mime = "application/vnd.google-earth.kml+xml"
                else:
                    data_bytes = export_to_shapefile_zip_bytes(selected_wgs84, f"{query}_{suffix}")
                    file_name = build_export_filename(query, suffix, "zip")
                    mime = "application/zip"

                local_path = os.path.join(EXPORTS_DIR, file_name)
                try:
                    with open(local_path, "wb") as f:
                        f.write(data_bytes)
                except OSError:
                    pass

                st.session_state["last_export"] = {
                    "file_name": file_name,
                    "path": local_path,
                    "format": fmt,
                    "matches": len(matches),
                    "area_km2": area_km2,
                    "centroid": (lat, lon),
                    "utm_epsg": utm_epsg,
                    "suffix": suffix,
                    "mime": mime,
                    "data_bytes": data_bytes,
                }

with tabs[0]:
    if st.session_state["recent_searches"]:
        st.caption("Recent searches: " + " | ".join(st.session_state["recent_searches"]))

    if run and not full_gdf.empty and query.strip():
        if matches.empty:
            st.info("No results to show.")
        else:
            show_cols = ["level"] + [c for c in matches.columns if c.startswith("NAME_")]
            st.dataframe(matches[show_cols].head(300), use_container_width=True)
    else:
        st.info("Run a search to see results.")

with tabs[1]:
    geojson_str = st.session_state.get("selected_geojson")
    if geojson_str:
        selected_for_map = gdf_from_geojson_str(geojson_str)
        m = make_folium_map(selected_for_map)
        folium_static(m, width=1100, height=650)
    else:
        st.info("Run a search to preview the selected boundary on the map.")

with tabs[2]:
    last = st.session_state.get("last_export")
    if last:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Matches", str(last["matches"]))
        c2.metric("Area (km²)", f"{last['area_km2']:.2f}")
        c3.metric("Centroid", f"{last['centroid'][0]:.5f}, {last['centroid'][1]:.5f}")
        c4.metric("UTM EPSG", str(last["utm_epsg"]))

        st.download_button(
            label=f"⬇️ Download {last['file_name']}",
            data=last["data_bytes"],
            file_name=last["file_name"],
            mime=last["mime"],
            use_container_width=True,
        )
        st.caption(f"Saved locally at: {last['path']}")
    else:
        st.info("No export yet. Run a search and export to enable download here.")

with tabs[3]:
    if not full_gdf.empty:
        st.write("Row count:", len(full_gdf))
        st.write("Columns:")
        st.code(", ".join(list(full_gdf.columns)))
        if "level" in full_gdf.columns:
            st.write("Rows per admin level:")
            st.dataframe(full_gdf["level"].value_counts().sort_index().rename("rows"), use_container_width=True)
    else:
        st.info("No data loaded yet.")
