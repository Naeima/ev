import dash
from dash import dcc, html, Input, Output, State
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
# Load data
# =========================
file_id = "16xhVfgn4T4MEET_8ziEdBhs3nhpc_0PL"
df = load_csv_from_drive(file_id)

# =========================
# Define relevant towns (West Midlands county)
# =========================
target_towns = [
    "Birmingham","Coventry","Wolverhampton",
    "Dudley","Walsall","Solihull","West Bromwich","Sutton Coldfield",
    "Stourbridge","Halesowen","Smethwick","Oldbury","Tipton","Wednesbury",
    "Rowley Regis","Brierley Hill","Kingswinford","Bilston","Darlaston",
    "Willenhall","Aldridge","Brownhills","Sedgley","Wednesfield","Quarry Bank",
    "Wordsley","Cradley Heath","Bearwood","Castle Bromwich","Meriden"
]

# =========================
# Clean and filter
# =========================
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

# Geometry
df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
west_midlands_gdf = gdf.copy()

# =========================
# EA Flood-Monitoring API (live overlays)
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
        r = requests.get(EA_FLOODS_URL, timeout=20)
        r.raise_for_status()
        return r.json().get('items', [])
    except Exception:
        return []

def fetch_ea_flood_areas():
    out = {}
    try:
        r = requests.get(EA_AREAS_URL, timeout=30)
        r.raise_for_status()
        items = r.json().get('items', [])
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
WM_BBOX = box(-3.0, 52.25, -1.3, 52.75)  # rough West Midlands bbox

def fetch_osm_water_polys(bbox_polygon):
    w, s, e, n = bbox_polygon.bounds  # lon_min, lat_min, lon_max, lat_max
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

# Globals
WATER_CACHE = None
WATER_UNION_27700 = None
DIST_READY = False

# =========================
# Distance + risk helpers
# =========================
RED, AMBER, GREEN = "#D32F2F", "#FFC107", "#2E7D32"

def ensure_water_and_distances():
    """Fetch water polygons if needed and compute per-charger distance to water (meters)."""
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
    """Return (score, label, color) from distance threshold."""
    if pd.isna(dist_m):
        return (0, "Safe (>5 km)", GREEN)
    if dist_m <= 300:
        return (2, "High: ≤ 300 m", RED)
    if dist_m <= 5000:
        return (1, "Amber: 301 m–5 km", AMBER)
    return (0, "Safe (>5 km)", GREEN)

def ea_risk_score_for_points(floods):
    """
    Returns a pandas Series of EA scores aligned to west_midlands_gdf index:
    2 if inside any severity 1/2 polygon, 1 if inside severity 3 polygon, else 0.
    """
    sev12_polys, sev3_polys = [], []
    for it in floods:
        sev = int(it.get('severityLevel', 0)) if str(it.get('severityLevel', '')).isdigit() else 0
        if sev not in (1,2,3):
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
        if sev in (1,2):
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
# Build Folium map
# =========================
def build_map(selected_severities=None):
    # ---- Basemaps (nicer defaults) ----
    m = folium.Map(location=[52.4862, -1.8904], zoom_start=10, tiles=None)
    BASEMAPS = {
        "CARTO Voyager (default)": (
            "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap contributors, © CARTO"
        ),
        "CARTO Positron (light)": (
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap contributors, © CARTO"
        ),
        "CARTO Dark Matter": (
            "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap contributors, © CARTO"
        ),
        "OpenStreetMap HOT": (
            "https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
            "© OpenStreetMap contributors, Tiles courtesy of Humanitarian OpenStreetMap Team"
        ),
        "OpenTopoMap": (
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "© OpenStreetMap contributors, SRTM | Map style: © OpenTopoMap (CC-BY-SA)"
        ),
        "Esri World Imagery (satellite)": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
            "Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, GIS User Community"
        ),
        "Esri World Gray Canvas": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "Tiles © Esri — Esri, DeLorme, NAVTEQ"
        ),
    }
    for name, (tiles, attr) in BASEMAPS.items():
        folium.TileLayer(
            tiles=tiles, attr=attr, name=name, control=True,
            show=(name == "CARTO Voyager (default)")
        ).add_to(m)

    # --- Water overlay (animated, dark blue) ---
    global WATER_CACHE
    if WATER_CACHE is None:
        WATER_CACHE = fetch_osm_water_polys(WM_BBOX)

    if WATER_CACHE:
        water_fg = folium.FeatureGroup(name="OSM Water (lakes/riverbanks)", show=True)
        WATER_STYLE = dict(fillColor="#1565C0", color="#0D47A1", weight=2.0, fillOpacity=0.60)
        water_layer_names = []
        for geom in WATER_CACHE:
            try:
                gj = folium.GeoJson(data=mapping(geom), style_function=lambda _f, s=WATER_STYLE: s)
                gj.add_to(water_fg)
                water_layer_names.append(gj.get_name())
            except Exception:
                continue
        water_fg.add_to(m)
        # Pulse animation
        OPACITY_A, OPACITY_B = 0.75, 0.50
        WEIGHT_A,  WEIGHT_B  = 2.8, 1.6
        INTERVAL_MS = 1200
        js_layers = ",".join([f"window.{name}" for name in water_layer_names])
        pulse_js = f"""
        <script>
          (function() {{
            var layers = [{js_layers}].filter(Boolean);
            var hi = true;
            function pulse() {{
              layers.forEach(function(layer) {{
                try {{
                  layer.setStyle({{fillOpacity: (hi ? {OPACITY_A} : {OPACITY_B}),
                                  weight: (hi ? {WEIGHT_A}  : {WEIGHT_B} )}});
                }} catch(e) {{}}
              }});
              hi = !hi;
            }}
            setInterval(pulse, {INTERVAL_MS});
          }})();
        </script>
        """
        m.get_root().html.add_child(folium.Element(pulse_js))

    # --- Ensure distances computed ---
    if not DIST_READY:
        ensure_water_and_distances()

    # --- EA polygons & scores ---
    floods = fetch_ea_floods()
    if selected_severities:
        floods = [f for f in floods if str(f.get('severityLevel', '')).isdigit()
                  and int(f['severityLevel']) in selected_severities]

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

    # --- Combined risk classification ---
    def combined_class(row):
        dist_score, dist_label, dist_color = classify_distance(row['DistWater_m'])
        ea_score = ea_scores.loc[row.name]
        score = max(dist_score, ea_score)
        if score == 2:
            return ("High (≤300 m or EA warning)", RED)
        if score == 1:
            return ("Amber (301 m–5 km or EA alert)", AMBER)
        return ("Safe (>5 km)", GREEN)

    classes = west_midlands_gdf.apply(combined_class, axis=1, result_type='expand')
    west_midlands_gdf['RiskLabel'] = classes[0]
    west_midlands_gdf['RiskColor'] = classes[1]

    # --- Clustered layers with BIG icon markers ---
    red_group   = folium.FeatureGroup(name="Chargers: High risk (red)", show=True)
    amber_group = folium.FeatureGroup(name="Chargers: Amber risk", show=True)
    safe_group  = folium.FeatureGroup(name="Chargers: Safe (green)", show=True)

    red_cluster   = MarkerCluster(name="Cluster: High").add_to(red_group)
    amber_cluster = MarkerCluster(name="Cluster: Amber").add_to(amber_group)
    safe_cluster  = MarkerCluster(name="Cluster: Safe").add_to(safe_group)

    def make_icon(color_hex):
        border = {"#D32F2F": "#B71C1C", "#FFC107": "#FF8F00", "#2E7D32": "#1B5E20"}.get(color_hex, "#1B5E20")
        return BeautifyIcon(
            icon="bolt", icon_shape="marker",
            background_color=color_hex, border_color=border, border_width=3,
            text_color="white",
            inner_icon_style="font-size:22px;padding-top:2px;"
        )

    for _, row in west_midlands_gdf.iterrows():
        label = row['RiskLabel']
        color = row['RiskColor']
        dist_txt = "N/A" if pd.isna(row['DistWater_m']) else f"{row['DistWater_m']:.0f} m"
        popup_html = (
            f"<b>Town/City:</b> {row['Town']}<br>"
            f"<b>Status:</b> {row['Status']}<br>"
            f"<b>Operator:</b> {row['Operator']}<br>"
            f"<b>Distance to water:</b> {dist_txt}<br>"
            f"<b>Risk:</b> {label}"
        )
        marker = folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            icon=make_icon(color),
            popup=folium.Popup(popup_html, max_width=420)
        )
        if color == RED:
            red_cluster.add_child(marker)
        elif color == AMBER:
            amber_cluster.add_child(marker)
        else:
            safe_cluster.add_child(marker)

    red_group.add_to(m)
    amber_group.add_to(m)
    safe_group.add_to(m)

    Draw(export=True).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # Legends
    risk_legend = f"""
    <div style="position: fixed; bottom: 20px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>Charger Risk (distance & EA)</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:{RED};margin-right:6px;border:1px solid #555;"></span> High: ≤ 300 m or EA Warning<br>
      <span style="display:inline-block;width:12px;height:12px;background:{AMBER};margin-right:6px;border:1px solid #555;"></span> Amber: 301 m–5 km or EA Alert<br>
      <span style="display:inline-block;width:12px;height:12px;background:{GREEN};margin-right:6px;border:1px solid #555;"></span> Safe: > 5 km
    </div>
    """
    m.get_root().html.add_child(folium.Element(risk_legend))

    ea_legend = """
    <div style="position: fixed; bottom: 120px; left: 20px; z-index: 9999; background: white; padding: 8px 10px; border: 1px solid #ccc; font-size: 13px;">
      <b>EA Flood Severity</b><br>
      <span style="display:inline-block;width:12px;height:12px;background:#8B0000;margin-right:6px;border:1px solid #555;"></span> Severe (1)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#FF0000;margin-right:6px;border:1px solid #555;"></span> Warning (2)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#FFA500;margin-right:6px;border:1px solid #555;"></span> Alert (3)<br>
      <span style="display:inline-block;width:12px;height:12px;background:#008000;margin-right:6px;border:1px solid #555;"></span> No longer in force (4)
    </div>
    """
    m.get_root().html.add_child(folium.Element(ea_legend))

    return m

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
    html.H1("EV Chargers – West Midlands (Oct-24)", style={'textAlign': 'center'}),
    html.H2("Map of EV Charging Stations + Live Flood Risk (Environment Agency)"),

    html.Div([
        html.Label("Filter EA Flood Severity (polygon overlay):"),
        dcc.Checklist(
            id='flood-severity',
            options=[{'label': f"{v} ({k})", 'value': k} for k, v in SEVERITY_LABEL.items()],
            value=[1, 2, 3], inline=True
        )
    ], style={'marginBottom': '10px'}),

    html.Iframe(id='map', srcDoc=map_html, width='100%', height='650'),

    html.Div([
        html.Label("Filter by Status:"),
        dcc.Dropdown(
            id='status-filter',
            options=[{'label': s, 'value': s} for s in sorted(west_midlands_gdf['Status'].dropna().unique())],
            multi=True
        )
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Filter by Payment Required:"),
        dcc.Dropdown(
            id='payment-filter',
            options=[{'label': p, 'value': p} for p in sorted(west_midlands_gdf['PaymentRequired'].dropna().unique())],
            multi=True
        )
    ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'}),

    html.H2("Time Series of Charger Installations"),
    dcc.Graph(id='time-series'),

    html.H2("Upload Polygon to Filter"),
    dcc.Upload(id='upload-polygon', children=html.Button("Upload GeoJSON / GeoJSON FeatureCollection"), multiple=False),
    html.Div(id='output-count')
])

# =========================
# Callbacks
# =========================
@app.callback(
    Output('map', 'srcDoc'),
    Input('flood-severity', 'value')
)
def refresh_map(severities):
    m = build_map(selected_severities=severities or [])
    html_path = MAP_HTML
    m.save(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()

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
            return html.Div([
                html.P(f"{len(selected)} charging points in selected polygon."),
                html.Pre(preview)
            ])
        except Exception as e:
            return html.Div([
                html.P("Error reading GeoJSON. Ensure it is a Polygon/MultiPolygon, Feature, or FeatureCollection."),
                html.Pre(str(e))
            ])
    return "Upload a valid polygon GeoJSON (.geojson or .json)."

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
        return px.line(title='New Chargers Over Time')
    dff['Month'] = dff['DateCreated'].dt.to_period('M').astype(str)
    time_counts = dff.groupby(['Month','Town']).size().reset_index(name='Count')
    fig = px.line(
        time_counts, x='Month', y='Count', color='Town',
        title='New Chargers Over Time', markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title='Month', yaxis_title='Charger Count', height=400)
    return fig

if __name__ == "__main__":
    app = dash.Dash(__name__)
    app.run_server(debug=True)
