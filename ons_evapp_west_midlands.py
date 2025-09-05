import dash
from dash import dcc, html, Input, Output, State, dash_table
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import Draw, MarkerCluster, BeautifyIcon
from shapely.geometry import shape, Point, Polygon, box, mapping
from shapely.ops import unary_union
try:
    from shapely import wkt as shapely_wkt  # Shapely 2.x
except Exception:
    import shapely.wkt as shapely_wkt       # Shapely 1.x fallback
from shapely.prepared import prep
import json, io, base64, time, requests, re
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import plotly.express as px
from xml.etree import ElementTree as ET
from pandas.api.types import is_string_dtype

# =========================
# Robust Google Sheets loader (CSV export)
# =========================
SHEETS_EXPORT_BASE = "https://docs.google.com/spreadsheets/d/{sid}/export?format=csv"

def _extract_sheet_id_and_gid(url_or_id: str):
    """Return (sheet_id, gid_or_None) from a Sheets share URL or raw id."""
    if url_or_id.startswith("http"):
        p = urlparse(url_or_id)
        m = re.search(r"/d/([^/]+)/", p.path)
        sid = m.group(1) if m else None
        gid = parse_qs(p.query).get("gid", [None])[0]
        if not sid:
            raise ValueError("Could not extract spreadsheet id from the provided URL.")
        return sid, gid
    return url_or_id, None

def load_csv_from_gsheet(url_or_id: str, max_retries: int = 5, timeout: int = 30) -> pd.DataFrame:
    sid, gid = _extract_sheet_id_and_gid(url_or_id)
    url = SHEETS_EXPORT_BASE.format(sid=sid) + (f"&gid={gid}" if gid else "")
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    backoff = 1
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            # If Google returns HTML (auth page) instead of CSV, fail fast
            ct = r.headers.get("Content-Type", "")
            if "text/html" in ct and b"," not in r.content[:1024]:
                raise requests.HTTPError("Received HTML instead of CSV (is the sheet public?)")
            return pd.read_csv(io.BytesIO(r.content), low_memory=False)
        except Exception as e:
            last_err = e
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)
    raise last_err or RuntimeError("Unknown error reading Google Sheet")

# =========================
# Load base charger data (from your provided Sheet)
# =========================
SHEET_URL = "https://docs.google.com/spreadsheets/d/1xjD-NH6rX7_ueOU89jxZsKXahURfRAyA/edit?usp=sharing&ouid=118120094376416558501&rtpof=true&sd=true"
df = load_csv_from_gsheet(SHEET_URL)

# Try to be forgiving about column names
def _col(df, *candidates):
    for c in candidates:
        if c in df.columns: return c
    # case-insensitive fallback
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower_map: return lower_map[c.lower()]
    return None

lat_col = _col(df, "latitude", "Latitude", "lat", "Lat", "LAT")
lon_col = _col(df, "longitude", "Longitude", "lon", "lng", "Lon", "Lng", "LONGITUDE")
town_col = _col(df, "town", "Town", "city", "City", "settlement", "Settlement")

if not all([lat_col, lon_col, town_col]):
    missing = [n for n, c in {"latitude":lat_col, "longitude":lon_col, "town":town_col}.items() if not c]
    raise ValueError(f"Sheet is missing required columns (case-insensitive): {missing}")

# Optional / nice-to-have columns
date_col   = _col(df, "dateCreated", "DateCreated", "installedOn", "InstalledOn")
op_col     = _col(df, "deviceControllerName", "Operator", "operator")
status_col = _col(df, "chargeDeviceStatus", "Status", "status")
pay_col    = _col(df, "paymentRequired", "PaymentRequired", "payment", "Payment")

target_towns = [
    "Birmingham","Coventry","Wolverhampton",
    "Dudley","Walsall","Solihull","West Bromwich","Sutton Coldfield",
    "Stourbridge","Halesowen","Smethwick","Oldbury","Tipton","Wednesbury",
    "Rowley Regis","Brierley Hill","Kingswinford","Bilston","Darlaston",
    "Willenhall","Aldridge","Brownhills","Sedgley","Wednesfield","Quarry Bank",
    "Wordsley","Cradley Heath","Bearwood","Castle Bromwich","Meriden"
]

df['Latitude']  = pd.to_numeric(df[lat_col], errors='coerce')
df['Longitude'] = pd.to_numeric(df[lon_col], errors='coerce')
df['Town']      = df[town_col].astype(str).str.strip().str.title()

# =========================
# Robust date parsing (explicit formats first; silent mixed fallback)
# =========================
KNOWN_DT_FORMATS = [
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
    "%d/%m/%Y",            # UK day-first
    "%d-%m-%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d-%m-%Y %H:%M:%S",
    "%d %b %Y",
    "%d %B %Y",
    "%Y-%m-%dT%H:%M:%S",   # ISO without Z
    "%Y-%m-%dT%H:%M:%S.%f"
]

def parse_dates_series(s: pd.Series) -> pd.Series:
    """Try known exact formats first, then a mixed-format, day-first fallback without warnings."""
    s_clean = s.astype(str).str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    mask_remaining = s_clean.ne("") & ~s_clean.isna()

    for fmt in KNOWN_DT_FORMATS:
        try:
            parsed = pd.to_datetime(s_clean.where(mask_remaining), format=fmt, errors="coerce", utc=False)
        except Exception:
            continue
        hits = parsed.notna()
        out = out.where(~hits, parsed)
        mask_remaining &= ~hits
        if not mask_remaining.any():
            break

    if mask_remaining.any():
        # pandas >= 2.0 supports format="mixed"; TypeError on older versions
        try:
            parsed_fallback = pd.to_datetime(
                s_clean.where(mask_remaining),
                format="mixed",
                dayfirst=True,
                errors="coerce",
                utc=False
            )
        except TypeError:
            from dateutil import parser
            def _du(x):
                try:
                    return parser.parse(x, dayfirst=True)
                except Exception:
                    return pd.NaT
            parsed_fallback = s_clean.where(mask_remaining).apply(_du).astype("datetime64[ns]")
        hits = parsed_fallback.notna()
        out = out.where(~hits, parsed_fallback)

    return out

if date_col:
    df['DateCreated'] = parse_dates_series(df[date_col])
else:
    df['DateCreated'] = pd.NaT

df['Operator'] = (df[op_col] if op_col else pd.Series(index=df.index)).fillna("Unknown")
df['Status']   = (df[status_col] if status_col else pd.Series(index=df.index)).fillna("Unknown")

if pay_col:
    # accept bools or strings like "Yes/No", "Y/N", "True/False"
    def norm_pay(v):
        if pd.isna(v): return 'Unknown'
        if isinstance(v, bool): return 'Yes' if v else 'No'
        s = str(v).strip().lower()
        if s in {'yes','y','true','t','1'}: return 'Yes'
        if s in {'no','n','false','f','0'}: return 'No'
        return 'Unknown'
    df['PaymentRequired'] = df[pay_col].map(norm_pay)
else:
    df['PaymentRequired'] = 'Unknown'

df = df.dropna(subset=['Latitude', 'Longitude', 'Town'])
df = df[df['Town'].isin([t.title() for t in target_towns])]

df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
west_midlands_gdf = gdf.copy()

# =========================
# Study extent (West Midlands bbox)
# =========================
WM_BBOX = box(-3.0, 52.25, -1.3, 52.75)

# =========================
# EA Flood-Monitoring API (live) – OGL v3
# =========================
EA_FLOODS_URL = "https://environment.data.gov.uk/flood-monitoring/id/floods"
EA_AREAS_URL  = "https://environment.data.gov.uk/flood-monitoring/id/floodAreas"

SEVERITY_LABEL = {
    1: "Severe flood warning",
    2: "Flood warning",
    3: "Flood alert",
    4: "Warning no longer in force"
}

# RAG colours
RED, AMBER, GREEN = "#D32F2F", "#FFC107", "#2E7D32"

def _sev_to_rag(sev: int) -> str:
    if sev in (1, 2): return RED
    if sev == 3: return AMBER
    return GREEN

def fetch_ea_floods():
    try:
        r = requests.get(EA_FLOODS_URL, timeout=25)
        r.raise_for_status()
        return r.json().get('items', [])
    except Exception:
        return []

def _paged_get_all_items(url: str, timeout: int = 30):
    items, next_url, tries = [], url, 0
    while next_url and tries < 10:
        tries += 1
        try:
            resp = requests.get(next_url, timeout=timeout)
            resp.raise_for_status()
            js = resp.json()
            items.extend(js.get('items', []))
            next_url = js.get('next')
        except Exception:
            break
    return items

def fetch_ea_flood_areas():
    out = {}
    try:
        items = _paged_get_all_items(EA_AREAS_URL, timeout=40)
        for it in items:
            poly_wkt = it.get('polygon')
            if not poly_wkt:
                continue
            try:
                geom = shapely_wkt.loads(poly_wkt)
            except Exception:
                continue
            key = it.get('notation') or it.get('code') or it.get('id')
            out[key] = {
                'geometry': geom,
                'eaAreaName': it.get('label') or it.get('eaAreaName') or "",
                'riverOrSea': it.get('riverOrSea') or ""
            }
    except Exception:
        pass
    return out

EA_AREAS_CACHE = fetch_ea_flood_areas()

# =========================
# OSM water overlay (Overpass API)
# =========================
OVERPASS_URL = "https://overpass-api.de/api/interpreter"

def fetch_osm_water_polys(bbox_polygon):
    w, s, e, n = bbox_polygon.bounds
    query = f"""
    [out:json][timeout:25];
    (
      way["natural"="water"]({s},{w},{n},{e});
      relation["natural"="water"]({s},{w},{n},{e});
      way["waterway"="riverbank"]({s},{w},{n},{e});
      relation["waterway"="riverbank"]({s},{w},{n},{e});
    );
    out body geom;
    """
    try:
        r = requests.post(OVERPASS_URL, data=query, timeout=60)
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []

    geoms = []
    for el in data.get("elements", []):
        if el.get("type") == "way" and "geometry" in el:
            coords = [(pt["lon"], pt["lat"]) for pt in el["geometry"]]
            if len(coords) >= 4 and coords[0] == coords[-1]:
                try:
                    poly = Polygon(coords)
                    if not poly.is_empty:
                        geoms.append(poly)
                except Exception:
                    continue
        if el.get("type") == "relation" and el.get("members"):
            rings = []
            for m in el["members"]:
                if m.get("role") in ("outer", "inner") and "geometry" in m:
                    coords = [(pt["lon"], pt["lat"]) for pt in m["geometry"]]
                    if len(coords) >= 4 and coords[0] == coords[-1]:
                        try:
                            rings.append((m.get("role"), Polygon(coords)))
                        except Exception:
                            pass
            if rings:
                outers = [p for role, p in rings if role == "outer"]
                inners = [p for role, p in rings if role == "inner"]
                if outers:
                    try:
                        mp = unary_union(outers)
                        if inners:
                            mp = Polygon(mp.exterior.coords, holes=[i.exterior.coords for i in inners if i.is_valid])
                        if not mp.is_empty:
                            geoms.append(mp)
                    except Exception:
                        pass

    clipped = []
    for g in geoms:
        try:
            gg = g.intersection(WM_BBOX)
            if not gg.is_empty:
                clipped.append(gg)
        except Exception:
            pass
    return clipped

WATER_CACHE = None

# =========================
# EA WMS overlay helpers (discover + add)
# =========================
WMS_FZ2 = "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-2/wms"
WMS_FZ3 = "https://environment.data.gov.uk/spatialdata/flood-map-for-planning-rivers-and-sea-flood-zone-3/wms"
WMS_RoFRS = "https://environment.data.gov.uk/spatialdata/risk-of-flooding-from-rivers-and-sea/wms"
WMS_FloodWarnings = "https://environment.data.gov.uk/spatialdata/flood-warning-areas/wms"

def _wms_find_layers(base_url, keywords):
    """Return list of (layer_name, layer_title) whose Title/Name contains all keywords (case-insensitive)."""
    try:
        params = {"service": "WMS", "request": "GetCapabilities", "version": "1.3.0"}
        r = requests.get(base_url, params=params, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []

    layers = []
    for lyr in root.findall(".//{*}Layer"):
        name_el = lyr.find("{*}Name")
        title_el = lyr.find("{*}Title")
        if name_el is None or title_el is None:
            continue
        name = (name_el.text or "").strip()
        title = (title_el.text or "").strip()
        hay = f"{name} {title}".lower()
        if all(kw.lower() in hay for kw in keywords):
            layers.append((name, title))
    return layers

def add_ea_wms(m: folium.Map, base_url: str, *, keywords, layer_label: str, show: bool = False, opacity: float = 0.55):
    """Discover a suitable EA WMS layer by keywords and add as a transparent overlay to folium map 'm'."""
    matches = _wms_find_layers(base_url, keywords)
    if not matches:
        print(f"[WMS] No match for {layer_label} using {keywords}.")
        return
    layer_name, _layer_title = matches[0]
    folium.raster_layers.WmsTileLayer(
        url=base_url,
        name=layer_label,
        layers=layer_name,
        fmt="image/png",
        transparent=True,
        version="1.3.0",
        attr="© Environment Agency",
        overlay=True,
        control=True,
        show=show,
        opacity=opacity,
    ).add_to(m)

# =========================
# EA WFS (Flood Zone polygons) – for station colouring by FZ2/FZ3
# =========================
WFS_FZ2 = WMS_FZ2.replace("/wms", "/wfs")
WFS_FZ3 = WMS_FZ3.replace("/wms", "/wfs")

FZ2_UNION_PREP = None
FZ3_UNION_PREP = None

def _wfs_find_layers(base_url, keywords):
    """Find candidate FeatureType names in WFS GetCapabilities that contain all keywords."""
    try:
        r = requests.get(base_url, params={"service":"WFS","request":"GetCapabilities","version":"2.0.0"}, timeout=20)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for ft in root.findall(".//{*}FeatureType"):
        name_el = ft.find("{*}Name")
        title_el = ft.find("{*}Title")
        if name_el is None:
            continue
        name = (name_el.text or "").strip()
        title = (title_el.text or "").strip()
        hay = f"{name} {title}".lower()
        if all(kw.lower() in hay for kw in keywords):
            out.append(name)
    return out

def _wfs_get_geojson(base_url, layer_name, bbox):
    """Get GeoJSON features for a layer clipped by bbox=(w,s,e,n). Tries WFS 2.0 then 1.1."""
    w,s,e,n = bbox
    params20 = {
        "service":"WFS","request":"GetFeature","version":"2.0.0",
        "typeNames":layer_name,"outputFormat":"application/json",
        "srsName":"EPSG:4326","bbox":f"{w},{s},{e},{n},EPSG:4326"
    }
    try:
        r = requests.get(base_url, params=params20, timeout=40)
        r.raise_for_status()
        return r.json()
    except Exception:
        params11 = {
            "service":"WFS","request":"GetFeature","version":"1.1.0",
            "typeName":layer_name,"outputFormat":"application/json",
            "srsName":"EPSG:4326","bbox":f"{w},{s},{e},{n},EPSG:4326"
        }
        r = requests.get(base_url, params=params11, timeout=40)
        r.raise_for_status()
        return r.json()

def _wfs_fetch_polys_union_prepared(base_url, keywords, clip_poly):
    """Find a WFS layer by keywords, fetch polygons in bbox, union them, return prepared geometry."""
    layers = _wfs_find_layers(base_url, keywords)
    if not layers:
        return None
    geojson = _wfs_get_geojson(base_url, layers[0], clip_poly.bounds)
    geoms = []
    for feat in geojson.get("features", []):
        try:
            g = shape(feat.get("geometry"))
            if not g.is_empty:
                g = g.intersection(clip_poly)
                if not g.is_empty:
                    geoms.append(g)
        except Exception:
            continue
    if not geoms:
        return None
    return prep(unary_union(geoms))

def _ensure_fz_unions():
    """Compute and cache prepared unions for Flood Zone 2 and 3 within WM_BBOX."""
    global FZ2_UNION_PREP, FZ3_UNION_PREP
    if FZ2_UNION_PREP is None:
        FZ2_UNION_PREP = _wfs_fetch_polys_union_prepared(WFS_FZ2, ["zone","flood","2"], WM_BBOX)
    if FZ3_UNION_PREP is None:
        FZ3_UNION_PREP = _wfs_fetch_polys_union_prepared(WFS_FZ3, ["zone","flood","3"], WM_BBOX)

# =========================
# Map builder (OSM water overlay + EA polygons + WMS + FZ-based station colouring)
# =========================
def build_map(selected_severities=None, floods=None):
    m = folium.Map(location=[52.4862, -1.8904], zoom_start=10, tiles=None)
    BASEMAPS = {
        "OSM Standard": ("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png","© OpenStreetMap contributors"),
        "CARTO Positron": ("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
        "CARTO Dark Matter": ("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png","© OpenStreetMap, © CARTO"),
        "Esri World Imagery (satellite)": ("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}","Tiles © Esri"),
        "Esri World Gray Canvas": ("https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}","Tiles © Esri"),
    }
    for name, (tiles, attr) in BASEMAPS.items():
        folium.TileLayer(tiles=tiles, attr=attr, name=name, control=True,
                         show=(name == "Esri World Imagery (satellite)")).add_to(m)

    # --- OSM Water overlay (optional visual context) ---
    global WATER_CACHE
    if WATER_CACHE is None:
        WATER_CACHE = fetch_osm_water_polys(WM_BBOX)
    if WATER_CACHE:
        water_fg = folium.FeatureGroup(name="OSM Water (lakes/riverbanks)", show=True)
        WATER_STYLE = dict(fillColor="#1565C0", color="#0D47A1", weight=1.0, fillOpacity=0.55)
        for geom in WATER_CACHE:
            try:
                folium.GeoJson(data=mapping(geom),
                               style_function=lambda _f, s=WATER_STYLE: s).add_to(water_fg)
            except Exception:
                continue
        water_fg.add_to(m)

    # === EA WMS overlays: clearly mark flood areas ===
    add_ea_wms(m, WMS_FZ2,           keywords=['zone','flood','2'],       layer_label="Flood Zone 2 (undefended)", show=False)
    add_ea_wms(m, WMS_FZ3,           keywords=['zone','flood','3'],       layer_label="Flood Zone 3 (undefended)", show=False)
    add_ea_wms(m, WMS_RoFRS,         keywords=['risk','river','sea'],     layer_label="Risk of Flooding (Rivers & Sea)", show=True, opacity=0.5)
    add_ea_wms(m, WMS_FloodWarnings, keywords=['flood','warning','area'], layer_label="Flood Warning Areas", show=False)

    # === Live EA Flood Risk Areas overlay (from flood-monitoring feed) ===
    raw_floods = floods if floods is not None else fetch_ea_floods()
    if selected_severities:
        floods_for_polys = [f for f in raw_floods if str(f.get('severityLevel','')).isdigit()
                            and int(f['severityLevel']) in selected_severities]
    else:
        floods_for_polys = raw_floods

    areas_fg = folium.FeatureGroup(name="EA Flood Risk Areas (live)", show=True)
    for it in floods_for_polys:
        try:
            sev = int(it.get('severityLevel', 0) or 0)
        except Exception:
            continue
        if sev not in SEVERITY_LABEL:
            continue

        fa = it.get('floodArea') or {}
        geom = None

        poly_wkt = fa.get('polygon')
        if poly_wkt:
            try:
                geom = shapely_wkt.loads(poly_wkt)
            except Exception:
                geom = None
        if geom is None:
            key = fa.get('notation') or fa.get('id') or fa.get('code')
            area = EA_AREAS_CACHE.get(key)
            if area:
                geom = area['geometry']
        if geom is None:
            continue

        try:
            geom = geom.intersection(WM_BBOX)
            if geom.is_empty:
                continue
        except Exception:
            continue

        rag_colour = _sev_to_rag(sev)
        feature = {
            "type": "Feature",
            "properties": {
                "severity": SEVERITY_LABEL[sev],
                "label": fa.get('label') or fa.get('eaAreaName') or "",
                "river": fa.get('riverOrSea') or "",
                "rag": rag_colour
            },
            "geometry": mapping(geom)
        }

        folium.GeoJson(
            data=feature,
            name=f"EA {SEVERITY_LABEL[sev]}",
            style_function=lambda f: {
                "fillColor": f["properties"]["rag"],
                "color": f["properties"]["rag"],
                "weight": 0.8,
                "fillOpacity": 0.35,
            },
            highlight_function=lambda f: {
                "weight": 2.0,
                "fillOpacity": 0.55,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["label", "severity", "river"],
                aliases=["Area", "Severity", "River/Sea"],
                sticky=True,
            ),
        ).add_to(areas_fg)
    areas_fg.add_to(m)

    # === Flood Zone-based classification for chargers (FZ3 -> FZ2 -> Outside) ===
    _ensure_fz_unions()

    def fz_class(row):
        pt = row.geometry
        if FZ3_UNION_PREP and FZ3_UNION_PREP.contains(pt):
            return ("Flood Zone 3 (undefended)", RED)
        if FZ2_UNION_PREP and FZ2_UNION_PREP.contains(pt):
            return ("Flood Zone 2 (undefended)", AMBER)
        return ("Outside FZ2/FZ3", GREEN)

    classes = west_midlands_gdf.apply(fz_class, axis=1, result_type='expand')
    west_midlands_gdf['RiskLabel'] = classes[0]
    west_midlands_gdf['RiskColor'] = classes[1]

    # === Charger markers clustered by Flood Zone category ===
    red_group   = folium.FeatureGroup(name="Chargers: Flood Zone 3 (red)", show=True)
    amber_group = folium.FeatureGroup(name="Chargers: Flood Zone 2 (amber)", show=True)
    safe_group  = folium.FeatureGroup(name="Chargers: Outside FZ2/FZ3 (green)", show=True)

    red_cluster   = MarkerCluster(name="Cluster: FZ3").add_to(red_group)
    amber_cluster = MarkerCluster(name="Cluster: FZ2").add_to(amber_group)
    safe_cluster  = MarkerCluster(name="Cluster: Outside").add_to(safe_group)

    def make_icon(color_hex):
        border = {RED: "#B71C1C", AMBER: "#FF8F00", GREEN: "#1B5E20"}.get(color_hex, "#1B5E20")
        return BeautifyIcon(icon="bolt", icon_shape="marker",
                            background_color=color_hex, border_color=border, border_width=3,
                            text_color="white", inner_icon_style="font-size:22px;padding-top:2px;")

    for _, row in west_midlands_gdf.iterrows():
        label = row['RiskLabel']; color = row['RiskColor']
        popup_html = (f"<b>Town/City:</b> {row['Town']}<br>"
                      f"<b>Status:</b> {row['Status']}<br>"
                      f"<b>Operator:</b> {row['Operator']}<br>"
                      f"<b>Zone:</b> {label}")
        marker = folium.Marker(location=[row['Latitude'], row['Longitude']],
                               icon=make_icon(color), popup=folium.Popup(popup_html, max_width=420))
        if color == RED: red_cluster.add_child(marker)
        elif color == AMBER: amber_cluster.add_child(marker)
        else: safe_cluster.add_child(marker)

    red_group.add_to(m); amber_group.add_to(m); safe_group.add_to(m)
    Draw(export=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # === Top-left alert blocks (live feed counts – optional, kept) ===
    count_sev1 = sum(1 for f in raw_floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == 1)
    count_sev2 = sum(1 for f in raw_floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == 2)
    count_sev3 = sum(1 for f in raw_floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == 3)
    count_sev4 = sum(1 for f in raw_floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == 4)
    count_red   = count_sev1 + count_sev2

    alert_panel = f"""
    <div style="position: fixed; top: 12px; left: 12px; z-index: 9999; display:flex; gap:8px; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
      <div style="background:{RED}; color:white; padding:6px 10px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,.25);">
        <div style="font-size:12px;opacity:.9;">Warning/Severe</div>
        <div style="font-size:18px;font-weight:700;line-height:1;">{count_red}</div>
      </div>
      <div style="background:{AMBER}; color:black; padding:6px 10px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,.25);">
        <div style="font-size:12px;opacity:.9;">Alerts</div>
        <div style="font-size:18px;font-weight:700;line-height:1;">{count_sev3}</div>
      </div>
      <div style="background:{GREEN}; color:white; padding:6px 10px; border-radius:8px; box-shadow:0 1px 3px rgba(0,0,0,.25);">
        <div style="font-size:12px;opacity:.9;">No longer in force</div>
        <div style="font-size:18px;font-weight:700;line-height:1;">{count_sev4}</div>
      </div>
    </div>"""
    m.get_root().html.add_child(folium.Element(alert_panel))

    # === Legends ===
    rag_legend = f"""
    <div style="position: fixed; bottom: 120px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>EA Flood Areas (RAG)</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:{RED};margin-right:6px;border:1px solid #555;"></span> Warning/Severe (1–2)<br>
      <span style="display:inline-block;width:12px;height:12px;background:{AMBER};margin-right:6px;border:1px solid #555;"></span> Alert (3)<br>
      <span style="display:inline-block;width:12px;height:12px;background:{GREEN};margin-right:6px;border:1px solid #555;"></span> No longer in force (4)
    </div>"""
    m.get_root().html.add_child(folium.Element(rag_legend))

    chargers_legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>Chargers by Flood Zone (undefended)</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:{RED};margin-right:6px;border:1px solid #555;"></span> Flood Zone 3<br>
      <span style="display:inline-block;width:12px;height:12px;background:{AMBER};margin-right:6px;border:1px solid #555;"></span> Flood Zone 2<br>
      <span style="display:inline-block;width:12px;height:12px;background:{GREEN};margin-right:6px;border:1px solid #555;"></span> Outside FZ2/FZ3
    </div>"""
    m.get_root().html.add_child(folium.Element(chargers_legend))

    return m

# =========================
# Live feed + table helpers
# =========================
def parse_iso(ts: str):
    if not ts: return None
    try: return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except Exception: return None

def format_feed_items(floods, tz="Europe/London", limit=50):
    def best_time(it):
        return (
            parse_iso(it.get('timeSeverityChanged')) or
            parse_iso(it.get('timeMessageChanged')) or
            parse_iso(it.get('timeRaised')) or
            parse_iso(it.get('timeUpdated')) or
            datetime.now(timezone.utc)
        )
    floods_sorted = sorted(floods, key=lambda it: (int(it.get('severityLevel', 9) or 9), best_time(it)))
    out = []
    for it in floods_sorted[:limit]:
        sev = int(it.get('severityLevel', 0) or 0)
        sev_txt = SEVERITY_LABEL.get(sev, f"Unknown ({sev})")
        chip = _sev_to_rag(sev)
        fa = it.get('floodArea', {}) or {}
        area = fa.get('label') or fa.get('eaAreaName') or fa.get('description') or fa.get('riverOrSea') or "Area"
        msg = it.get('message') or it.get('description') or it.get('severity') or ""
        when = (parse_iso(it.get('timeSeverityChanged')) or parse_iso(it.get('timeMessageChanged')) or
                parse_iso(it.get('timeRaised')) or parse_iso(it.get('timeUpdated')))
        when_local = when.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M") if when else "—"
        link = it.get('@id') or it.get('id') or EA_FLOODS_URL
        badge = html.Span(sev_txt, style={'background': chip, 'color': ('black' if chip==AMBER else 'white'),
                                          'padding':'2px 6px','borderRadius':'6px','fontSize':'12px','marginRight':'8px'})
        row = html.Div([
            badge, html.Span(area, style={'fontWeight':'600'}),
            html.Span(f" — {msg}", style={'opacity':0.85, 'marginLeft':'6px'}),
            html.Span(f" · {when_local}", style={'float':'right','fontSize':'12px','opacity':0.7}),
            html.Br(), html.A("details", href=link, target="_blank", style={'fontSize':'12px'})
        ], style={'padding':'6px 0','borderBottom':'1px dashed #ddd'})
        out.append(row)
    return out

def flatten_item(it):
    fa = it.get('floodArea') or {}
    return {
        "severityLevel": it.get("severityLevel"),
        "severity": it.get("severity"),
        "area_label": fa.get("label") or fa.get("eaAreaName") or "",
        "area_notation": fa.get("notation") or "",
        "area_riverOrSea": fa.get("riverOrSea") or "",
        "message": (it.get("message") or it.get("description") or "")[:280],
        "timeSeverityChanged": it.get("timeSeverityChanged"),
        "timeUpdated": it.get("timeUpdated"),
        "timeRaised": it.get("timeRaised"),
        "id": it.get("@id") or it.get("id") or ""
    }

def kpi_badge(title, value, color):
    return html.Div([
        html.Div(title, style={'fontSize':'12px','opacity':0.7}),
        html.Div(str(value), style={'fontSize':'24px','fontWeight':'700'})
    ], style={'flex':'1', 'background':color, 'color':('black' if color==AMBER else 'white'),
              'padding':'12px 14px','borderRadius':'8px','textAlign':'center'})

# =========================
# Initial map HTML + Dash app
# =========================
MAP_HTML = "ons_ev_map_west_midlands.html"
build_map().save(MAP_HTML)

app = dash.Dash(__name__)
with open(MAP_HTML, "r", encoding="utf-8") as f:
    map_html = f.read()

app.layout = html.Div([
    html.H1("EV Chargers – West Midlands (Flood Zones, EA Live Areas, & Risk-Clustering)", style={'textAlign': 'center'}),
    html.Div(style={'height':'6px'}),

    # Controls
    html.Div([
        html.Label([
            html.Span(
                "Filter EA Flood Severity (live polygons):ℹ️",
                title=("Severe flood warning (1) · Flood warning (2) · Flood alert (3) · Warning no longer in force (4)\n"
                       "Colours use RAG: 1–2 Red, 3 Amber, 4 Green."),
                style={'textDecoration': 'underline', 'cursor': 'help'}
            )
        ]),
        dcc.Checklist(
            id='flood-severity',
            options=[{'label': f"{v} ({k})", 'value': k} for k, v in SEVERITY_LABEL.items()],
            value=[1, 2, 3], inline=True
        ),
        dcc.Interval(id='ea-poll', interval=60_000, n_intervals=0)
    ], style={'marginBottom': '10px'}),

    # KPIs for live EA feed (outside the map)
    html.Div([
        html.Div(id='kpi-sev1'), html.Div(id='kpi-sev2'),
        html.Div(id='kpi-sev3'), html.Div(id='kpi-sev4'),
        html.Div(id='kpi-total')
    ], style={'display':'flex','gap':'10px','margin':'8px 0'}),

    # Map
    html.Iframe(id='map', srcDoc=map_html, width='100%', height='650'),

    # Live warning feed + table
    html.Div([
        html.Div([
            html.H2("Live EA Warnings"),
            html.Div(id='last-update', style={'fontSize': '12px', 'opacity': 0.7, 'marginBottom': '6px'}),
            html.Div(id='live-feed',
                     style={'maxHeight': '280px', 'overflowY': 'auto', 'border': '1px solid #ccc',
                            'padding': '8px','borderRadius': '6px', 'background': '#fafafa'}),
        ], style={'flex':'1'}),

        html.Div([
            html.H2("Live EA Flood Items (Table)"),
            dash_table.DataTable(
                id='ea-table',
                columns=[
                    {"name": "severityLevel", "id": "severityLevel"},
                    {"name": "severity", "id": "severity"},
                    {"name": "area_label", "id": "area_label"},
                    {"name": "area_notation", "id": "area_notation"},
                    {"name": "area_riverOrSea", "id": "area_riverOrSea"},
                    {"name": "message", "id": "message"},
                    {"name": "timeSeverityChanged", "id": "timeSeverityChanged"},
                    {"name": "timeUpdated", "id": "timeUpdated"},
                    {"name": "timeRaised", "id": "timeRaised"},
                    {"name": "id", "id": "id"},
                ],
                data=[],
                page_size=10,
                sort_action="native",
                filter_action="native",
                style_table={'maxHeight':'320px','overflowY':'auto'},
                style_cell={'fontSize':'12px','whiteSpace':'normal','height':'auto'},
            )
        ], style={'flex':'1', 'marginLeft': '20px'})
    ], style={'display':'flex','gap':'20px','marginTop':'16px'}),

    html.H2("Charger Installations Over Time (Stacked by Town)"),
    html.Div([
        html.Div([
            html.Label("Status:"),
            dcc.Dropdown(
                id='status-filter',
                options=[{'label': s, 'value': s} for s in sorted(west_midlands_gdf['Status'].dropna().unique())],
                multi=True
            )
        ], style={'width': '40%','display':'inline-block'}),
        html.Div([
            html.Label("Payment Required:"),
            dcc.Dropdown(
                id='payment-filter',
                options=[{'label': p, 'value': p} for p in sorted(west_midlands_gdf['PaymentRequired'].dropna().unique())],
                multi=True
            )
        ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'})
    ]),
    dcc.Graph(id='time-series'),

    html.H2("Upload Polygon to Filter Chargers"),
    dcc.Upload(id='upload-polygon', children=html.Button("Upload GeoJSON / GeoJSON FeatureCollection"), multiple=False),
    html.Div(id='output-count')
])

# =========================
# Live refresh callback (map + KPIs + feed + table)
# =========================
@app.callback(
    Output('map', 'srcDoc'),
    Output('live-feed', 'children'),
    Output('last-update', 'children'),
    Output('kpi-sev1', 'children'),
    Output('kpi-sev2', 'children'),
    Output('kpi-sev3', 'children'),
    Output('kpi-sev4', 'children'),
    Output('kpi-total', 'children'),
    Output('ea-table', 'data'),
    Input('flood-severity', 'value'),
    Input('ea-poll', 'n_intervals')
)
def refresh_live(severities, _n):
    floods = fetch_ea_floods()
    selected_severities = severities or []
    floods_filtered = [f for f in floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) in selected_severities] if selected_severities else floods

    # Map: pass full floods for overlays; station colours are by Flood Zone via WFS unions
    m = build_map(selected_severities=selected_severities, floods=floods)
    map_html_str = m.get_root().render()

    # Feed
    feed_children = format_feed_items(floods_filtered)
    ts = datetime.now(ZoneInfo("Europe/London")).strftime("Last updated: %Y-%m-%d %H:%M %Z")

    # KPIs
    def count_lvl(k): return sum(1 for f in floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == k)
    k1 = kpi_badge("Severe (1)", count_lvl(1), RED)
    k2 = kpi_badge("Warning (2)", count_lvl(2), RED)
    k3 = kpi_badge("Alert (3)", count_lvl(3), AMBER)
    k4 = kpi_badge("No longer in force (4)", count_lvl(4), GREEN)
    kt = kpi_badge("Total active", len(floods), "#283593")

    # Table (flatten)
    table_rows = [flatten_item(it) for it in floods_filtered]

    return map_html_str, feed_children, ts, k1, k2, k3, k4, kt, table_rows

# =========================
# Polygon filter callback
# =========================
def _extract_polygon(geojson_obj):
    if not isinstance(geojson_obj, dict):
        raise ValueError('Invalid GeoJSON content.')
    if geojson_obj.get('type') == 'FeatureCollection':
        geoms = [shape(f['geometry']) for f in geojson_obj.get('features', [])
                 if f.get('geometry') and f['geometry'].get('type') in {'Polygon','MultiPolygon'}]
        if not geoms:
            raise ValueError('No Polygon/MultiPolygon found in FeatureCollection.')
        return unary_union(geoms)
    if geojson_obj.get('type') == 'Feature':
        geom = geojson_obj.get('geometry')
        if not geom or geom.get('type') not in {'Polygon','MultiPolygon'}:
            raise ValueError('Feature geometry is not a Polygon/MultiPolygon.')
        return shape(geom)
    if geojson_obj.get('type') in {'Polygon','MultiPolygon'}:
        return shape(geojson_obj)
    raise ValueError('Unsupported GeoJSON type for polygon extraction.')

@app.callback(
    Output('output-count', 'children'),
    Input('upload-polygon', 'contents'),
    State('upload-polygon', 'filename')
)
def filter_polygon(contents, filename):
    if contents:
        try:
            content_string = contents.split(',')[1]
            decoded = base64.b64decode(content_string)
            geojson_obj = json.load(io.StringIO(decoded.decode('utf-8')))
            poly_shape = _extract_polygon(geojson_obj)
            mask = west_midlands_gdf.geometry.within(poly_shape)
            selected = west_midlands_gdf[mask]
            preview = selected[['Town','Latitude','Longitude']].head(10).to_string(index=False)
            return html.Div([html.P(f"{len(selected)} charging points in selected polygon."), html.Pre(preview)])
        except Exception as e:
            return html.Div([html.P("Error reading GeoJSON. Ensure it is a Polygon/MultiPolygon, Feature, or FeatureCollection."),
                             html.Pre(str(e))])
    return "Upload a valid polygon GeoJSON (.geojson or .json)."

# =========================
# Stacked bar chart callback
# =========================
@app.callback(
    Output('time-series', 'figure'),
    Input('status-filter', 'value'),
    Input('payment-filter', 'value')
)
def update_time_series(statuses, payments):
    dff = west_midlands_gdf.copy()
    if statuses:
        dff = dff[dff['Status'].isin(statuses)]
    if payments:
        dff = dff[dff['PaymentRequired'].isin(payments)]
    dff = dff.dropna(subset=['DateCreated'])
    if dff.empty:
        return px.bar(title='New Chargers Over Time (Stacked)')
    dff['Month'] = dff['DateCreated'].dt.to_period('M').astype(str)
    time_counts = dff.groupby(['Month', 'Town']).size().reset_index(name='Count')
    fig = px.bar(time_counts, x='Month', y='Count', color='Town',
                 title='New Chargers Over Time (Stacked by Town)',
                 barmode='stack', color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(xaxis_title='Month', yaxis_title='Charger Count', height=420, legend_title_text='Town')
    return fig

if __name__ == "__main__":
    app.run(debug=True)
