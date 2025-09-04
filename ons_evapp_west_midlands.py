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
import json, io, base64, time, requests
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import plotly.express as px

# =========================
# Robust Google Drive loader
# =========================
DRIVE_EXPORT = "https://drive.google.com/uc?export=download&id={file_id}"

def load_csv_from_drive(file_id: str, max_retries: int = 5, timeout: int = 30) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    url = DRIVE_EXPORT.format(file_id=file_id)
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            r = session.get(url, timeout=timeout, allow_redirects=True)
            token = next((v for k, v in r.cookies.items() if k.startswith("download_warning")), None)
            if token:
                r = session.get(f"{url}&confirm={token}", timeout=timeout, allow_redirects=True)
            if r.status_code in {429, 500, 502, 503, 504} or not r.content:
                raise requests.HTTPError(f"HTTP {r.status_code}")
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), low_memory=False)
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 16)

# =========================
# Load base charger data
# =========================
file_id = "16xhVfgn4T4MEET_8ziEdBhs3nhpc_0PL"
df = load_csv_from_drive(file_id)

target_towns = [
    "Birmingham","Coventry","Wolverhampton",
    "Dudley","Walsall","Solihull","West Bromwich","Sutton Coldfield",
    "Stourbridge","Halesowen","Smethwick","Oldbury","Tipton","Wednesbury",
    "Rowley Regis","Brierley Hill","Kingswinford","Bilston","Darlaston",
    "Willenhall","Aldridge","Brownhills","Sedgley","Wednesfield","Quarry Bank",
    "Wordsley","Cradley Heath","Bearwood","Castle Bromwich","Meriden"
]

required_cols = {'latitude', 'longitude', 'town'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df['Latitude']  = pd.to_numeric(df['latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['DateCreated'] = pd.to_datetime(df.get('dateCreated'), errors='coerce')
df['Town'] = df['town'].astype(str).str.strip().str.title()
df['Operator'] = df.get('deviceControllerName', pd.Series(index=df.index)).fillna("Unknown")
df['Status'] = df.get('chargeDeviceStatus', pd.Series(index=df.index)).fillna("Unknown")
df['PaymentRequired'] = df.get('paymentRequired').map({True: 'Yes', False: 'No'}).fillna('Unknown')

df = df.dropna(subset=['Latitude', 'Longitude', 'Town'])
df = df[df['Town'].isin([t.title() for t in target_towns])]

df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
west_midlands_gdf = gdf.copy()

# =========================
# EA Flood-Monitoring API (live)
# =========================
EA_FLOODS_URL = "https://environment.data.gov.uk/flood-monitoring/id/floods"
EA_AREAS_URL  = "https://environment.data.gov.uk/flood-monitoring/id/floodAreas"

SEVERITY_LABEL = {
    1: "Severe flood warning",
    2: "Flood warning",
    3: "Flood alert",
    4: "Warning no longer in force"
}
SEVERITY_COLOR = {1: "darkred", 2: "red", 3: "orange", 4: "green"}

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
WM_BBOX = box(-3.0, 52.25, -1.3, 52.75)

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
WATER_UNION_27700 = None
DIST_READY = False

RED, AMBER, GREEN = "#D32F2F", "#FFC107", "#2E7D32"

def ensure_water_and_distances():
    global WATER_CACHE, WATER_UNION_27700, DIST_READY, west_midlands_gdf
    if WATER_CACHE is None:
        WATER_CACHE = fetch_osm_water_polys(WM_BBOX)
    if not WATER_CACHE:
        west_midlands_gdf['DistWater_m'] = pd.NA
        DIST_READY = True
        return
    water_gdf = gpd.GeoDataFrame(geometry=[unary_union(WATER_CACHE)], crs="EPSG:4326")
    water_27700 = water_gdf.to_crs(epsg=27700).geometry.iloc[0]
    WATER_UNION_27700 = prep(water_27700)
    chargers_27700 = west_midlands_gdf.to_crs(epsg=27700)
    dists = chargers_27700.geometry.apply(lambda p: 0.0 if WATER_UNION_27700.contains(p) else p.distance(water_27700))
    west_midlands_gdf['DistWater_m'] = dists.values
    DIST_READY = True

def classify_distance(dist_m):
    if pd.isna(dist_m):
        return (0, "Safe (>5 km)", GREEN)
    if dist_m <= 300:
        return (2, "High: ≤ 300 m", RED)
    if dist_m <= 5000:
        return (1, "Amber: 301 m–5 km", AMBER)
    return (0, "Safe (>5 km)", GREEN)

def ea_risk_score_for_points(floods):
    sev12_polys, sev3_polys = [], []
    for it in floods:
        sev = int(it.get('severityLevel', 0)) if str(it.get('severityLevel', '')).isdigit() else 0
        if sev not in (1, 2, 3):
            continue
        fa = it.get('floodArea', {})
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
        geom = geom.intersection(WM_BBOX)
        if geom.is_empty:
            continue
        if sev in (1, 2):
            sev12_polys.append(geom)
        elif sev == 3:
            sev3_polys.append(geom)
    sev12_union = prep(unary_union(sev12_polys)) if sev12_polys else None
    sev3_union  = prep(unary_union(sev3_polys))  if sev3_polys  else None
    scores = []
    for pt in west_midlands_gdf.geometry:
        score = 0
        if sev3_union and sev3_union.contains(pt):
            score = max(score, 1)
        if sev12_union and sev12_union.contains(pt):
            score = max(score, 2)
        scores.append(score)
    return pd.Series(scores, index=west_midlands_gdf.index)

# =========================
# Map builder
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

    global WATER_CACHE
    if WATER_CACHE is None:
        WATER_CACHE = fetch_osm_water_polys(WM_BBOX)
    if WATER_CACHE:
        water_fg = folium.FeatureGroup(name="OSM Water (lakes/riverbanks)", show=True)
        WATER_STYLE = dict(fillColor="#1565C0", color="#0D47A1", weight=2.0, fillOpacity=0.60)
        names = []
        for geom in WATER_CACHE:
            try:
                gj = folium.GeoJson(data=mapping(geom), style_function=lambda _f, s=WATER_STYLE: s)
                gj.add_to(water_fg)
                names.append(gj.get_name())
            except Exception:
                continue
        water_fg.add_to(m)
        pulse_js = f"""
        <script>(function(){{
          var layers=[{",".join([f"window.{n}" for n in names])}].filter(Boolean);var hi=true;
          function pulse(){{layers.forEach(function(l){{try{{l.setStyle({{fillOpacity:(hi?0.75:0.5),weight:(hi?2.8:1.6)}})}}catch(e){{}}));hi=!hi;}} setInterval(pulse,1200);}})();</script>"""
        m.get_root().html.add_child(folium.Element(pulse_js))

    if not DIST_READY:
        ensure_water_and_distances()

    if floods is None:
        floods = fetch_ea_floods()
    if selected_severities:
        floods = [f for f in floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) in selected_severities]

    by_sev = {}
    for f in floods:
        try:
            sev = int(f.get('severityLevel', 0))
        except Exception:
            continue
        if sev not in SEVERITY_LABEL:
            continue
        by_sev.setdefault(sev, []).append(f)

    for sev, items in by_sev.items():
        fg = folium.FeatureGroup(name=f"EA: {SEVERITY_LABEL[sev]}", show=True)
        for it in items:
            fa = it.get('floodArea', {})
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
            style = dict(fillColor=SEVERITY_COLOR[sev], color=SEVERITY_COLOR[sev], weight=1, fillOpacity=0.25)
            folium.GeoJson(data=mapping(geom), style_function=lambda _f, s=style: s).add_to(fg)
        fg.add_to(m)

    ea_scores = ea_risk_score_for_points(floods) if floods else pd.Series(0, index=west_midlands_gdf.index)

    def combined_class(row):
        dist_score, _dl, _dc = classify_distance(row['DistWater_m'])
        ea_score = ea_scores.loc[row.name]
        score = max(dist_score, ea_score)
        if score == 2: return ("High (≤300 m or EA warning)", RED)
        if score == 1: return ("Amber (301 m–5 km or EA alert)", AMBER)
        return ("Safe (>5 km)", GREEN)

    classes = west_midlands_gdf.apply(combined_class, axis=1, result_type='expand')
    west_midlands_gdf['RiskLabel'] = classes[0]
    west_midlands_gdf['RiskColor'] = classes[1]

    red_group   = folium.FeatureGroup(name="Chargers: High risk (red)", show=True)
    amber_group = folium.FeatureGroup(name="Chargers: Amber risk", show=True)
    safe_group  = folium.FeatureGroup(name="Chargers: Safe (green)", show=True)

    red_cluster   = MarkerCluster(name="Cluster: High").add_to(red_group)
    amber_cluster = MarkerCluster(name="Cluster: Amber").add_to(amber_group)
    safe_cluster  = MarkerCluster(name="Cluster: Safe").add_to(safe_group)

    def make_icon(color_hex):
        border = {"#D32F2F": "#B71C1C", "#FFC107": "#FF8F00", "#2E7D32": "#1B5E20"}.get(color_hex, "#1B5E20")
        return BeautifyIcon(icon="bolt", icon_shape="marker",
                            background_color=color_hex, border_color=border, border_width=3,
                            text_color="white", inner_icon_style="font-size:22px;padding-top:2px;")

    for _, row in west_midlands_gdf.iterrows():
        label = row['RiskLabel']; color = row['RiskColor']
        dist_txt = "N/A" if pd.isna(row['DistWater_m']) else f"{row['DistWater_m']:.0f} m"
        popup_html = (f"<b>Town/City:</b> {row['Town']}<br>"
                      f"<b>Status:</b> {row['Status']}<br>"
                      f"<b>Operator:</b> {row['Operator']}<br>"
                      f"<b>Distance to water:</b> {dist_txt}<br>"
                      f"<b>Risk:</b> {label}")
        marker = folium.Marker(location=[row['Latitude'], row['Longitude']],
                               icon=make_icon(color), popup=folium.Popup(popup_html, max_width=420))
        if color == RED: red_cluster.add_child(marker)
        elif color == AMBER: amber_cluster.add_child(marker)
        else: safe_cluster.add_child(marker)

    red_group.add_to(m); amber_group.add_to(m); safe_group.add_to(m)
    Draw(export=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    risk_legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>Charger Risk (distance & EA)</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:{RED};margin-right:6px;border:1px solid #555;"></span> High: ≤ 300 m or EA Warning<br>
      <span style="display:inline-block;width:12px;height:12px;background:{AMBER};margin-right:6px;border:1px solid #555;"></span> Amber: 301 m–5 km or EA Alert<br>
      <span style="display:inline-block;width:12px;height:12px;background:{GREEN};margin-right:6px;border:1px solid #555;"></span> Safe: > 5 km
    </div>"""
    m.get_root().html.add_child(folium.Element(risk_legend))

    ea_legend = """
    <div style="position: fixed; bottom: 120px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>EA Flood Severity</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#8B0000;margin-right:6px;border:1px solid #555;"></span> Severe (1)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#FF0000;margin-right:6px;border:1px solid #555;"></span> Warning (2)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#FFA500;margin-right:6px;border:1px solid #555;"></span> Alert (3)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#008000;margin-right:6px;border:1px solid #555;"></span> No longer in force (4)
    </div>"""
    m.get_root().html.add_child(folium.Element(ea_legend))
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
        color = SEVERITY_COLOR.get(sev, "gray")
        fa = it.get('floodArea', {}) or {}
        area = fa.get('label') or fa.get('eaAreaName') or fa.get('description') or fa.get('riverOrSea') or "Area"
        msg = it.get('message') or it.get('description') or it.get('severity') or ""
        when = (parse_iso(it.get('timeSeverityChanged')) or parse_iso(it.get('timeMessageChanged')) or
                parse_iso(it.get('timeRaised')) or parse_iso(it.get('timeUpdated')))
        when_local = when.astimezone(ZoneInfo(tz)).strftime("%Y-%m-%d %H:%M") if when else "—"
        link = it.get('@id') or it.get('id') or EA_FLOODS_URL
        badge = html.Span(sev_txt, style={'background': color, 'color': 'white','padding':'2px 6px',
                                          'borderRadius':'6px','fontSize':'12px','marginRight':'8px'})
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
        "message": (it.get("message") or it.get("description") or "")[:280],
        "timeRaised": it.get("timeRaised"),
        "timeUpdated": it.get("timeUpdated"),
        "timeSeverityChanged": it.get("timeSeverityChanged"),
        "station": (it.get("eaAreaName") or ""),
        "area_label": fa.get("label") or fa.get("eaAreaName") or "",
        "area_notation": fa.get("notation") or "",
        "area_riverOrSea": fa.get("riverOrSea") or "",
        "id": it.get("@id") or it.get("id") or ""
    }

def kpi_badge(title, value, color):
    return html.Div([
        html.Div(title, style={'fontSize':'12px','opacity':0.7}),
        html.Div(str(value), style={'fontSize':'24px','fontWeight':'700'})
    ], style={'flex':'1', 'background':color, 'color':'white', 'padding':'12px 14px',
              'borderRadius':'8px', 'textAlign':'center'})

# =========================
# Initial map HTML
# =========================
MAP_HTML = "ons_ev_map_west_midlands.html"
build_map().save(MAP_HTML)

# =========================
# Dash app
# =========================
app = dash.Dash(__name__)
with open(MAP_HTML, "r", encoding="utf-8") as f:
    map_html = f.read()

app.layout = html.Div([
    html.H1("EV Chargers – West Midlands (Live EA Flood Risk)", style={'textAlign': 'center'}),
    html.Div(style={'height':'6px'}),

    # Controls
    html.Div([
        html.Label([
            html.Span(
                "Filter EA Flood Severity (polygon overlay): ℹ️",
                title=("Severe flood warning (1) · Flood warning (2) · Flood alert (3) · Warning no longer in force (4)"),
                style={'textDecoration': 'underline', 'cursor': 'help'}
            )
        ]),
        dcc.Checklist(
            id='flood-severity',
            options=[{'label': f"{v} ({k})", 'value': k} for k, v in SEVERITY_LABEL.items()],
            value=[1, 2, 3], inline=True
        ),
        dcc.Interval(id='ea-poll', interval=60_000, n_intervals=0)  # poll every 60s
    ], style={'marginBottom': '10px'}),

    # KPIs for live EA feed
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
        ], style={'flex':'1', 'marginLeft':'20px'})
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
        ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'}),
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

    # Map (always built from full floods so markers reflect EA risk comprehensively)
    m = build_map(selected_severities=selected_severities, floods=floods)
    map_html_str = m.get_root().render()

    # Feed
    feed_children = format_feed_items(floods_filtered)
    ts = datetime.now(ZoneInfo("Europe/London")).strftime("Last updated: %Y-%m-%d %H:%M %Z")

    # KPIs
    def count_lvl(k): return sum(1 for f in floods if str(f.get('severityLevel','')).isdigit() and int(f['severityLevel']) == k)
    k1 = kpi_badge("Severe (1)", count_lvl(1), "#8B0000")
    k2 = kpi_badge("Warning (2)", count_lvl(2), "#C62828")
    k3 = kpi_badge("Alert (3)", count_lvl(3), "#EF6C00")
    k4 = kpi_badge("No longer in force (4)", count_lvl(4), "#2E7D32")
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

# This code surfaces the **exact contents** of `https://environment.data.gov.uk/flood-monitoring/id/floods` live in three places: the **map polygons**, a **live feed**, and a **live table**, all refreshed automatically every minute.
