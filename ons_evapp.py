# dashboard_ev_merged_rcsp_anim.py
# Adds moving-car animation + "Extreme weather" mode to the graph/DP dashboard.

import io, os, time, json, tempfile, http.client, requests, math, heapq
from io import StringIO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import dash
from dash import dcc, html, Input, Output, State
from flask import Response
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
from shapely.ops import split as shp_split

import folium
from folium.plugins import MarkerCluster, Draw
from folium.raster_layers import WmsTileLayer

# Optional graph libs
try:
    import osmnx as ox
    import networkx as nx
    HAS_OSMNX = True
except Exception:
    HAS_OSMNX = False

# =========================
# Config
# =========================
EV_GDRIVE_FILE_ID = "16xhVfgn4T4MEET_8ziEdBhs3nhpc_0PL"
LOCAL_EV_CSV = "south_wales_ev.csv"
DEFAULT_NEAR_M = 300
CACHE_DIR = ".cache_wfs"; os.makedirs(CACHE_DIR, exist_ok=True)

OWS_BASE = "https://datamap.gov.wales/geoserver/ows"

FRAW_WMS = {
    "FRAW – Rivers":  "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW – Sea":     "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW – Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WMS = {
    "FMfP – Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP – Surface/Small Watercourses": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
}
LIVE_WMS = {
    "Live – Warning Areas": "inspire-nrw:NRW_FLOOD_WARNING",
    "Live – Alert Areas":   "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}
CONTEXT_WMS = {"Historic Flood Extents": "inspire-nrw:NRW_HISTORIC_FLOODMAP"}

FRAW_WFS = {
    "Rivers":  "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "Sea":     "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
LIVE_WFS = {
    "Warnings": "inspire-nrw:NRW_FLOOD_WARNING",
    "Alerts":   "inspire-nrw:NRW_FLOOD_WATCH_AREAS",
}

SIM_DEFAULTS = dict(
    start_lat=51.4816, start_lon=-3.1791,  # Cardiff
    end_lat=51.6214,   end_lon=-3.9436,    # Swansea
    battery_kwh=64.0, init_soc=90.0, reserve_soc=10.0, target_soc=80.0,
    kwh_per_km=0.18, max_charger_offset_km=1.5, min_leg_km=20.0,
    route_buffer_m=30, wfs_pad_m=800
)

# RCSP knobs
SOC_STEP = 0.05
CHARGE_STEP = 0.10
DEFAULT_POWER_KW = 50.0
BASE_RISK_PENALTY_PER_KM = 60.0   # sec/km
EXTREME_RISK_PENALTY_PER_KM = 240.0
EXTREME_BUFFER_M = 60.0
MAX_GRAPH_BBOX_DEG = 0.25

# =========================
# Utilities
# =========================
def _gd_url(x): return x if "drive.google.com" in x else f"https://drive.google.com/uc?export=download&id={x}"

def read_csv_resilient_gdrive(file_id_or_url: str, **kw):
    url = _gd_url(file_id_or_url)
    sess = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET","HEAD"]))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    headers = {"User-Agent":"Mozilla/5.0 (EV-Dashboard)"}
    try:
        r = sess.get(url, headers=headers, timeout=30, stream=True); r.raise_for_status()
        token = next((v for k,v in r.cookies.items() if k.startswith("download_warning")), None)
        if token:
            r = sess.get(url, headers=headers, params={"confirm":token}, timeout=30, stream=True); r.raise_for_status()
        return pd.read_csv(io.BytesIO(r.content), **({"low_memory":False}|kw))
    except Exception as e:
        try:
            import gdown
            with tempfile.TemporaryDirectory() as td:
                out = os.path.join(td,"data.csv")
                gdown.download(url, out, quiet=True, fuzzy=True)
                return pd.read_csv(out, **({"low_memory":False}|kw))
        except Exception:
            pass
        if "export=download" not in url:
            r = requests.get(url.replace("/uc?","/uc?export=download&"), headers=headers, timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), **({"low_memory":False}|kw))
        raise RuntimeError(f"Google Drive fetch failed: {e}")

def haversine_km(a,b):
    R=6371.0088
    lat1,lon1=math.radians(a[0]), math.radians(a[1])
    lat2,lon2=math.radians(b[0]), math.radians(b[1])
    dlat=lat2-lat1; dlon=lon2-lon1
    h=math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return 2*R*math.asin(math.sqrt(h))

def bbox_expand(bounds, pad_m):
    minx,miny,maxx,maxy = bounds
    pad_deg = max(0.002, pad_m/111_320.0)
    return (minx-pad_deg, miny-pad_deg, maxx+pad_deg, maxy+pad_deg)

def _bbox_for(df_like, pad_m=DEFAULT_NEAR_M*2):
    if isinstance(df_like, gpd.GeoDataFrame):
        minx,miny,maxx,maxy = df_like.total_bounds
    else:
        minx,miny,maxx,maxy = gdf_ev.total_bounds
    return bbox_expand((minx,miny,maxx,maxy), pad_m)

def _cache_path(layer,bbox):
    safe = f"{layer}_{bbox[0]:.4f}_{bbox[1]:.4f}_{bbox[2]:.4f}_{bbox[3]:.4f}".replace(":","_").replace(",","_")
    return os.path.join(CACHE_DIR, f"{safe}.geojson")

def fetch_wfs_layer_cached(layer, bbox, ttl_h=48):
    p = _cache_path(layer, bbox)
    if os.path.exists(p) and time.time()-os.path.getmtime(p) < ttl_h*3600:
        try:
            gj = json.load(open(p,"r",encoding="utf-8"))
            return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
        except Exception:
            pass
    from urllib.parse import urlencode
    params = {
        "service":"WFS","request":"GetFeature","version":"2.0.0",
        "typenames":layer,"outputFormat":"application/json","srsName":"EPSG:4326",
        "bbox": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]},EPSG:4326"
    }
    url = f"{OWS_BASE}?{urlencode(params)}"
    try:
        gj = requests.get(url, timeout=30).json()
        with open(p,"w",encoding="utf-8") as f: json.dump(gj,f)
        return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
    except Exception:
        if os.path.exists(p):
            try:
                gj = json.load(open(p,"r",encoding="utf-8"))
                return gpd.GeoDataFrame.from_features(gj.get("features",[]), crs="EPSG:4326")
            except Exception:
                pass
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs="EPSG:4326")

# =========================
# Load EV data
# =========================
if os.path.exists(LOCAL_EV_CSV):
    df = pd.read_csv(LOCAL_EV_CSV, low_memory=False)
else:
    df = read_csv_resilient_gdrive(EV_GDRIVE_FILE_ID)

TARGET_AREAS = [
    "Blaenau Gwent","Bridgend","Caerphilly","Cardiff","Carmarthenshire","Merthyr Tydfil",
    "Monmouthshire","Neath Port Talbot","Newport","Pembrokeshire","Rhondda Cynon Taf",
    "Swansea","The Vale Of Glamorgan","Torfaen"
]
area_col = 'adminArea' if 'adminArea' in df.columns else 'town'
df[area_col] = df[area_col].astype(str).str.strip().str.title()
df = df[df[area_col].isin([t.title() for t in TARGET_AREAS])].copy()

df['Latitude']  = pd.to_numeric(df['latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df = df.dropna(subset=['Latitude','Longitude'])
df['Town'] = df[area_col]

def classify_availability(s):
    s = str(s).lower().strip()
    if any(k in s for k in ["available","in service","operational","working","ok","service"]): return True
    if any(k in s for k in ["not operational","fault","out of service","offline","unavailable","down"]): return False
    return None

df['Available'] = df.get('chargeDeviceStatus', pd.Series(index=df.index)).apply(classify_availability)
df['AvailabilityLabel'] = df['Available'].map({True:"Operational", False:"Not operational"}).fillna("Unknown")
df['Operator'] = df.get('deviceControllerName', df.get('Operator','Unknown'))
df['Postcode'] = df.get('postcode', df.get('Postcode','N/A'))
df['dateCreated'] = pd.to_datetime(df.get('dateCreated', df.get('DateCreated')), errors='coerce', dayfirst=True)

df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_ev = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf_ev['ROW_ID'] = gdf_ev.index.astype(int)
TOWN_OPTIONS = sorted([t for t in gdf_ev['Town'].dropna().astype(str).unique() if t])

# =========================
# Risk for chargers (lazy)
# =========================
def build_fraw_water_gdf(ev_gdf):
    bbox = _bbox_for(ev_gdf)
    chunks=[]
    for lyr in FRAW_WFS.values():
        g = fetch_wfs_layer_cached(lyr, bbox)
        if not g.empty: chunks.append(g[['geometry']])
    if not chunks:
        return gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs='EPSG:4326')
    G = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), geometry='geometry', crs='EPSG:4326')
    try: G = G.explode(index_parts=False).reset_index(drop=True)
    except Exception: pass
    return G

def compute_at_risk(ev_gdf, max_m):
    water = build_fraw_water_gdf(ev_gdf)
    out = ev_gdf[['ROW_ID']].copy()
    if water.empty:
        out['DistWater_m']=float('nan'); out['AtRisk']=False; return out
    try:
        ev_m = ev_gdf.to_crs('EPSG:27700'); water_m = water.to_crs('EPSG:27700')
    except Exception:
        ev_m = ev_gdf.to_crs('EPSG:3857');  water_m = water.to_crs('EPSG:3857')
    try:
        water_m['geometry']=water_m['geometry'].buffer(0).simplify(3, preserve_topology=True)
    except Exception:
        pass
    joined = gpd.sjoin_nearest(ev_m, water_m, how='left', max_distance=max_m, distance_col='DistWater_m') \
                .drop(columns=[c for c in ['index_right'] if c in ['index_right']]).to_crs('EPSG:4326')
    out = joined[['ROW_ID','DistWater_m']].copy()
    out['AtRisk'] = out['DistWater_m'].le(max_m)
    return out

def risk_cache_path(threshold_m: int) -> str:
    return f"cache_atrisk_{int(threshold_m)}.parquet"

def preload_risk_json(threshold_m: int) -> str:
    p = risk_cache_path(threshold_m)
    if os.path.exists(p):
        try:
            cache = pd.read_parquet(p)
            need = {"ROW_ID","DistWater_m","AtRisk"}
            if need.issubset(cache.columns):
                return cache[list(need)].to_json(orient="records")
        except Exception:
            pass
    return "[]"

# =========================
# Routing helpers
# =========================
def osrm_route(sl, so, el, eo):
    url = f"https://router.project-osrm.org/route/v1/driving/{so},{sl};{eo},{el}"
    params = {"overview":"full","geometries":"geojson","alternatives":"false","steps":"false"}
    try:
        r = requests.get(url, params=params, timeout=15).json()
        if r.get("routes"):
            coords = r["routes"][0]["geometry"]["coordinates"]
            ln = LineString([(c[0],c[1]) for c in coords])
            return ln, float(r["routes"][0]["distance"]), float(r["routes"][0]["duration"]), "OSRM"
    except Exception:
        pass
    ln = LineString([(so,sl),(eo,el)])
    d_km = haversine_km((sl,so),(el,eo))
    return ln, d_km*1000.0, d_km/70.0*3600.0, "Great-circle (approx)"

def get_flood_union(bounds, include_live=True, include_fraw=True, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    bbox = bbox_expand(bounds, pad_m); chunks=[]
    if include_fraw:
        for lyr in FRAW_WFS.values():
            g = fetch_wfs_layer_cached(lyr,bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if include_live:
        for lyr in LIVE_WFS.values():
            g = fetch_wfs_layer_cached(lyr,bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if not chunks: return None
    G = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), geometry='geometry', crs='EPSG:4326')
    try: G = G.explode(index_parts=False).reset_index(drop=True)
    except Exception: pass
    try: return G.to_crs('EPSG:27700').union_all()
    except Exception: return G.to_crs('EPSG:27700').unary_union

def segment_route_by_risk(line_wgs84, risk_union_metric, buffer_m=30):
    if risk_union_metric is None: return [line_wgs84], []
    try: line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:27700').iloc[0]
    except Exception: line_m = gpd.GeoSeries([line_wgs84], crs='EPSG:4326').to_crs('EPSG:3857').iloc[0]
    hit = risk_union_metric.buffer(buffer_m)
    try: pieces = list(shp_split(line_m, hit.boundary))
    except Exception: pieces = [line_m]
    safe_m, risk_m = [], []
    for seg in pieces: (risk_m if seg.intersects(hit) else safe_m).append(seg)
    safe = gpd.GeoSeries(safe_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if safe_m else []
    risk = gpd.GeoSeries(risk_m, crs='EPSG:27700').to_crs('EPSG:4326').tolist() if risk_m else []
    return safe, risk

# =========================
# RCSP optimiser (graph/DP)
# =========================
def rcsp_optimize(start_lat, start_lon, end_lat, end_lon,
                  battery_kwh, init_soc, reserve_soc, target_soc,
                  kwh_per_km, chargers_df, flood_union_m,
                  extreme=False):
    if not HAS_OSMNX:
        raise RuntimeError("OSMnx not installed")

    # Bbox
    minlat, maxlat = sorted([start_lat, end_lat])
    minlon, maxlon = sorted([start_lon, end_lon])
    pad_deg = 0.05
    south, north = minlat-pad_deg, maxlat+pad_deg
    west,  east  = minlon-pad_deg, maxlon+pad_deg
    if (east-west) > MAX_GRAPH_BBOX_DEG or (north-south) > MAX_GRAPH_BBOX_DEG:
        raise RuntimeError("Area too large for local graph; zoom in or use OSRM fallback")

    # Graph
    G = ox.graph_from_bbox(north, south, east, west, network_type="drive", simplify=True)
    G = ox.add_edge_speeds(G); G = ox.add_edge_travel_times(G)
    Gm = ox.project_graph(G, to_crs="EPSG:27700")
    nodes, edges = ox.graph_to_gdfs(G, nodes=True, edges=True, node_geometry=True, fill_edge_geometry=True)
    edges_m = edges.to_crs("EPSG:27700")

    # Risk mark
    if flood_union_m is not None:
        try:
            edges_m["risk"] = edges_m.geometry.buffer(0).intersects(flood_union_m.buffer(EXTREME_BUFFER_M if extreme else 0.0))
        except Exception:
            edges_m["risk"] = False
    else:
        edges_m["risk"] = False

    if "length" not in edges.columns:
        edges["length"] = edges.geometry.length*0.0

    edges_lookup = {}
    joined = edges.join(edges_m[["risk"]])
    for (u,v,k), row in joined.iterrows():
        L = float(row.get("length", row.geometry.length*111000))
        T = float(row.get("travel_time", L/13.9))
        R = bool(row["risk"])
        edges_lookup[(u,v,k)] = (L,T,R)

    u = ox.nearest_nodes(G, start_lon, start_lat)
    v = ox.nearest_nodes(G, end_lon, end_lat)

    # Chargers mapped to nodes
    chargers = {}
    if not chargers_df.empty:
        for _, r in chargers_df.iterrows():
            try:
                nid = ox.nearest_nodes(G, r["Longitude"], r["Latitude"])
                chargers[nid] = dict(power_kW=DEFAULT_POWER_KW,
                                     operational=(r.get("AvailabilityLabel","")=="Operational"))
            except Exception:
                continue

    # RCSP
    step = SOC_STEP
    Q = [round(i*step,2) for i in range(0, int(1/step)+1)]
    def q_to_idx(q): return max(0, min(len(Q)-1, int(round(q/step))))
    reserve_q = reserve_soc/100.0; tgt_q = target_soc/100.0; init_q = init_soc/100.0

    INF = 10**18
    best = {}; pred = {}
    hq = []
    start_idx = (u, q_to_idx(init_q))
    best[start_idx] = 0.0
    heapq.heappush(hq, (0.0, u, q_to_idx(init_q)))

    # Collapse multiedges
    adj = {}
    for (uu,vv,kk), (L,T,R) in edges_lookup.items():
        cur = adj.get((uu,vv))
        if (cur is None) or (T < cur[1]):
            adj[(uu,vv)] = (L,T,R)
    out = {}
    for (uu,vv), (L,T,R) in adj.items():
        out.setdefault(uu, []).append((vv,L,T,R))

    risk_penalty = EXTREME_RISK_PENALTY_PER_KM if extreme else BASE_RISK_PENALTY_PER_KM

    while hq:
        cost, node, qi = heapq.heappop(hq)
        if best.get((node,qi), INF) < cost - 1e-9: continue
        if node == v and Q[qi] >= reserve_q: break

        # Drive
        for (vv,L,T,R) in out.get(node, []):
            E_kWh = (L/1000.0) * kwh_per_km
            dq = E_kWh / battery_kwh
            if Q[qi] - dq < reserve_q - 1e-9: continue
            qj = q_to_idx(max(reserve_q, Q[qi] - dq))
            new_cost = cost + T + (risk_penalty*(L/1000.0) if R else 0.0)
            key = (vv, qj)
            if new_cost + 1e-9 < best.get(key, INF):
                best[key] = new_cost
                pred[key] = (node, qi, "drive", dict(L=L,T=T,R=R))
                heapq.heappush(hq, (new_cost, vv, qj))

        # Charge (if operational charger)
        ch = chargers.get(node)
        if ch and ch.get("operational", True):
            p_kw = float(ch.get("power_kW", DEFAULT_POWER_KW))
            max_target = max(tgt_q, Q[qi])
            for dq_step in [CHARGE_STEP, 2*CHARGE_STEP, 3*CHARGE_STEP]:
                q_next = min(1.0, Q[qi] + dq_step)
                if q_next <= Q[qi] or q_next < max_target - 1e-9: continue
                added_kWh = battery_kwh*(q_next - Q[qi])
                charge_time_s = 3600.0 * (added_kWh / max(1e-6, p_kw))
                key = (node, q_to_idx(q_next))
                new_cost = cost + charge_time_s
                if new_cost + 1e-9 < best.get(key, INF):
                    best[key] = new_cost
                    pred[key] = (node, qi, "charge", dict(p_kW=p_kw, added_kWh=added_kWh, dt=charge_time_s))
                    heapq.heappush(hq, (new_cost, node, q_to_idx(q_next)))

    # Reconstruct
    goal = None; goal_cost = INF
    for qi in range(len(Q)-1, -1, -1):
        k = (v, qi)
        if k in best and best[k] < goal_cost:
            goal, goal_cost = k, best[k]
    if goal is None:
        raise RuntimeError("No feasible RCSP solution in bbox.")

    path_nodes = []; charges = []
    k = goal
    while k in pred:
        prev, pqi, act, info = pred[k]
        if act == "charge":
            charges.append((k[0], Q[pqi], Q[k[1]], info))
        path_nodes.append(k[0])
        k = (prev, pqi)
    path_nodes.append(u); path_nodes.reverse(); charges.reverse()

    lats = [G.nodes[n]['y'] for n in path_nodes]; lons = [G.nodes[n]['x'] for n in path_nodes]
    line = LineString([(lon,lat) for lon,lat in zip(lons,lats)])
    safe_lines, risk_lines = segment_route_by_risk(line, flood_union_m,
                                                   buffer_m=(EXTREME_BUFFER_M if extreme else SIM_DEFAULTS["route_buffer_m"]))
    planned_stops = []
    if not chargers_df.empty:
        for (nid, q_before, q_after, info) in charges:
            lat, lon = G.nodes[nid]['y'], G.nodes[nid]['x']
            try:
                idx = ((chargers_df["Latitude"]-lat)**2 + (chargers_df["Longitude"]-lon)**2).idxmin()
                row = chargers_df.loc[idx]
                planned_stops.append(dict(
                    ROW_ID=int(row["ROW_ID"]), Operational=(row.get("AvailabilityLabel","")=="Operational"),
                    AtRisk=bool(row.get("AtRisk", False)), proj_km=0.0, offset_km=0.0
                ))
            except Exception:
                pass
    return line, safe_lines, risk_lines, planned_stops, goal_cost

# =========================
# Folium rendering (with animation)
# =========================
def add_wms_group(fmap, title_to_layer: dict, visible=True, opacity=0.55):
    for title, layer in title_to_layer.items():
        try:
            WmsTileLayer(url=OWS_BASE, layers=layer, name=f"{title} (WMS)",
                         fmt="image/png", transparent=True, opacity=opacity, version="1.3.0", show=visible).add_to(fmap)
        except Exception: pass

def add_live_wfs_popups(fmap, df_like):
    bbox = _bbox_for(df_like if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty else gdf_ev)
    for title, layer in LIVE_WFS.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty: continue
        folium.GeoJson(g.to_json(), name=f"{title} (WFS)",
                       style_function=lambda _: {'fillOpacity':0.15,'weight':2}).add_to(fmap)

def js_car_animation(coords_latlng, speed_kmh=50):
    """
    Returns a <script> block that animates a marker along coords_latlng [[lat,lon],...].
    speed_kmh: constant speed along polyline. Simple linear interpolation.
    """
    # Light JS: distance calc and tweening
    return folium.Element(f"""
<script>
(function() {{
  if (!window._ev_anim) window._ev_anim = {{active:false}};
  const coords = {json.dumps(coords_latlng)};
  if (!coords || coords.length < 2) return;

  // kill previous animation if any
  if (window._ev_anim.active && window._ev_anim.stop) {{ try {{ window._ev_anim.stop(); }} catch(e){{}} }}

  // find map object
  var map = null;
  for (var k in window) {{ if (k.startsWith('map_')) {{ map = window[k]; break; }} }}
  if (!map) return;

  // icon
  var carIcon = L.divIcon({{
    html: '<div style="font-size:20px;transform:translate(-8px,-8px);">🚗</div>',
    iconSize: [16,16], className: 'car-icon'
  }});

  // polyline (for reference)
  var poly = L.polyline(coords, {{color:'#3388ff', weight:4, opacity:0.0}}).addTo(map); // invisible, used for bounds
  try {{ map.fitBounds(poly.getBounds(), {{padding:[20,20]}}); }} catch(e){{}}

  var marker = L.marker(coords[0], {{icon: carIcon}}).addTo(map);

  // distances
  function toRad(d) {{ return d * Math.PI/180; }}
  function hav(a,b) {{
    var R=6371008.8;
    var dLat = toRad(b[0]-a[0]), dLon = toRad(b[1]-a[1]);
    var la1 = toRad(a[0]), la2 = toRad(b[0]);
    var h = Math.sin(dLat/2)**2 + Math.cos(la1)*Math.cos(la2)*Math.sin(dLon/2)**2;
    return 2*R*Math.asin(Math.sqrt(h));
  }}

  var seg = [], total=0;
  for (var i=0;i<coords.length-1;i++) {{
    var d = hav(coords[i], coords[i+1]);
    seg.push(d); total += d;
  }}
  if (total <= 0) return;

  var speed = Math.max(1, {float(speed_kmh)}) * 1000.0/3600.0; // m/s
  var t_total = total / speed; // seconds
  var t0 = performance.now();
  var idx = 0, acc = 0;

  function interp(a,b,t) {{ return [a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t]; }}

  var handle = null;
  function step(ts) {{
    var dt = (ts - t0)/1000.0; // s
    var dist = Math.min(total, dt*speed);
    // find segment
    var run = 0; idx = 0;
    while (idx < seg.length && run + seg[idx] < dist) {{ run += seg[idx]; idx++; }}
    var p = 0;
    if (idx < seg.length && seg[idx] > 0) {{
      p = (dist - run)/seg[idx];
    }} else {{
      idx = seg.length-1; p = 1;
    }}
    var a = coords[idx], b = coords[Math.min(idx+1, coords.length-1)];
    var ll = interp(a,b,p);
    marker.setLatLng(ll);
    if (dist >= total - 1) {{ cancelAnimationFrame(handle); window._ev_anim.active=false; return; }}
    handle = requestAnimationFrame(step);
    window._ev_anim.handle = handle;
  }}
  window._ev_anim.active = true;
  window._ev_anim.stop = function() {{ try {{ cancelAnimationFrame(handle); }} catch(e){{}} window._ev_anim.active=false; }};
  handle = requestAnimationFrame(step);
}})();
</script>
""")

def render_map_html_route(full_line, route_safe, route_risk, start, end, chargers,
                          animate=False, speed_kmh=50, show_live_backdrops=False):
    m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=11,
                   tiles="OpenStreetMap", control_scale=True)
    cluster = MarkerCluster(name="Planned chargers").add_to(m)

    def add_lines(lines, color, name):
        if not lines: return
        fg = folium.FeatureGroup(name=name).add_to(m)
        for ln in lines:
            coords = [(lat,lon) for lon,lat in ln.coords]
            folium.PolyLine(coords, color=color, weight=6, opacity=0.9).add_to(fg)

    add_lines(route_safe, "#2b8cbe", "Route – safe")
    add_lines(route_risk, "#e31a1c", "Route – flood risk")
    folium.Marker(start, tooltip="Start", icon=folium.Icon(color="green")).add_to(m)
    folium.Marker(end,   tooltip="End",   icon=folium.Icon(color="blue")).add_to(m)

    for st in chargers:
        row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st.get("ROW_ID",-1))].iloc[0] if "ROW_ID" in st else None
        if row is None: continue
        color = "green" if st.get("Operational", False) and not st.get("AtRisk", False) else ("red" if st.get("AtRisk", False) else "gray")
        popup = folium.Popup(
            f"<b>{row.get('Operator','')}</b><br>{row.get('Town','')} {row.get('Postcode','')}<br>"
            f"Operational: {st.get('Operational')}<br>Flood-risk: {st.get('AtRisk')}", max_width=320
        )
        folium.Marker([row["Latitude"], row["Longitude"]],
                      tooltip=f"{row.get('Operator','')} ({row.get('Town','')})",
                      popup=popup,
                      icon=folium.Icon(color=color, icon="bolt", prefix="fa")).add_to(cluster)

    # Optional live backdrop when extreme is on
    if show_live_backdrops:
        add_wms_group(m, LIVE_WMS, visible=True, opacity=0.65)

    folium.LayerControl(collapsed=True).add_to(m)

    # Animation along the full route polyline
    if animate and isinstance(full_line, LineString) and len(full_line.coords) >= 2:
        coords_latlng = [(lat, lon) for lon, lat in list(full_line.coords)]
        m.get_root().html.add_child(js_car_animation(coords_latlng, speed_kmh))

    return m.get_root().render()

def render_map_html_ev(df_map, show_fraw, show_fmfp, show_live, show_ctx, near_threshold):
    m = folium.Map(location=[51.6,-3.2], zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    cluster = MarkerCluster(name="EV Chargers").add_to(m)
    Draw(export=False, position='topleft',
         draw_options={'polygon': {'allowIntersection': False, 'showArea': True},
                       'rectangle': True, 'polyline': False, 'circle': False,
                       'circlemarker': False, 'marker': False},
         edit_options={'edit': True}).add_to(m)
    for _, row in df_map.iterrows():
        status = row.get("AvailabilityLabel","Unknown")
        near = bool(row.get("AtRisk")) if 'AtRisk' in df_map.columns else False
        dist_m = row.get("DistWater_m")
        parts = [f"<b>Operator:</b> {row.get('Operator','EV')}",
                 f"<b>Status:</b> {status}"]
        if pd.notna(dist_m): parts.append(f"<b>Dist. to water:</b> {dist_m:.0f} m")
        if pd.notna(row.get('dateCreated', pd.NaT)): parts.append(f"<b>Installed:</b> {row['dateCreated']}")
        color = 'red' if near else ('green' if status=='Operational' else ('red' if status=='Not operational' else 'gray'))
        folium.Marker([row['Latitude'], row['Longitude']],
                      tooltip=f"{row.get('Operator','')} ({row.get('Town','')})",
                      popup=folium.Popup("<br>".join(parts), max_width=320),
                      icon=folium.Icon(color=color, icon='bolt', prefix='fa')).add_to(cluster)
    if show_fraw: add_wms_group(m, FRAW_WMS, True, 0.50)
    if show_fmfp: add_wms_group(m, FMFP_WMS, True, 0.55)
    if show_ctx:  add_wms_group(m, CONTEXT_WMS, False, 0.45)
    if show_live:
        add_wms_group(m, LIVE_WMS, True, 0.65)
        add_live_wfs_popups(m, df_map)
    folium.LayerControl(collapsed=True).add_to(m)
    m.get_root().html.add_child(folium.Element(
        f"""<div style="position: fixed; bottom:20px; left:20px; z-index:9999; background:white; padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;">
            <b>Legend</b>
            <div style="margin-top:6px"><span style="display:inline-block;width:12px;height:12px;background:#ff0000;margin-right:6px;"></span> EV near water (≤ {near_threshold} m)</div>
        </div>"""
    ))
    return m.get_root().render()

# =========================
# Dash app
# =========================
app = dash.Dash(__name__)
server = app.server
LATEST_HTML = "<html><body style='font-family:sans-serif;padding:10px'>Use the controls below.</body></html>"

@server.route("/__map")
def _serve_map():
    return Response(LATEST_HTML, mimetype="text/html")

app.layout = html.Div([
    html.H1("South Wales EV – Chargers, Flood Risk, Graph/DP & Animation"),

    html.H2("A) Chargers & Flood Overlays"),
    html.Div([
        html.Div([html.Label("Town(s)"),
            dcc.Dropdown(id="f-town",
                options=[{"label":t, "value":t} for t in TOWN_OPTIONS],
                value=[], multi=True, placeholder="All towns")], style={"minWidth":"260px"}),
        html.Div([html.Label("Town contains"),
            dcc.Input(id="f-town-like", type="text", placeholder="substring", debounce=True)], style={"minWidth":"220px"}),
        html.Div([html.Label("Operational"),
            dcc.Checklist(id="f-op",
                options=[{"label":"Operational","value":"op"},
                         {"label":"Not operational","value":"down"},
                         {"label":"Unknown","value":"unk"}],
                value=["op","down","unk"], inputStyle={"marginRight":"6px"})], style={"minWidth":"320px"}),
        html.Div([html.Label("Risk"),
            dcc.Checklist(id="f-risk",
                options=[{"label":"At risk","value":"at"},
                         {"label":"Not at risk","value":"no"},
                         {"label":"Unknown","value":"unk"}],
                value=["at","no","unk"], inputStyle={"marginRight":"6px"})], style={"minWidth":"320px"}),
    ], style={"display":"flex","gap":"12px","alignItems":"end","flexWrap":"wrap","margin":"6px 0"}),

    html.Div([
        html.Div([html.Label("Near-water threshold (m)"),
            dcc.Input(id="near-m", type="number", min=50, max=2000, step=50, value=DEFAULT_NEAR_M, style={"width":"140px"})]),
        html.Button("Compute/Update near-water", id="btn-risk", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),

        html.Div(style={"width":"16px"}),

        html.Div([html.Label("Show overlays"),
            dcc.Checklist(id="layers",
                options=[{"label":"FRAW","value":"fraw"},
                         {"label":"FMfP","value":"fmfp"},
                         {"label":"Live warnings","value":"live"},
                         {"label":"Context","value":"ctx"}],
                value=["fraw","fmfp","live"], inputStyle={"marginRight":"6px"})], style={"minWidth":"360px"}),

        html.Button("Refresh layers", id="btn-refresh", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),
    ], style={"display":"flex","gap":"10px","alignItems":"end","flexWrap":"wrap","margin":"6px 0 12px"}),

    html.H2("B) Journey Simulator"),
    html.Div([
        html.Div([html.Label("Start (lat, lon)"),
            dcc.Input(id="start-lat", type="number", value=SIM_DEFAULTS["start_lat"], step=0.0001, style={"width":"120px"}),
            dcc.Input(id="start-lon", type="number", value=SIM_DEFAULTS["start_lon"], step=0.0001, style={"width":"120px"})]),
        html.Div([html.Label("End (lat, lon)"),
            dcc.Input(id="end-lat", type="number", value=SIM_DEFAULTS["end_lat"], step=0.0001, style={"width":"120px"}),
            dcc.Input(id="end-lon", type="number", value=SIM_DEFAULTS["end_lon"], step=0.0001, style={"width":"120px"})]),
        html.Div([html.Label("Battery / SOC start,res,target"),
            dcc.Input(id="batt-kwh", type="number", value=SIM_DEFAULTS["battery_kwh"], step=1, style={"width":"90px"}),
            dcc.Input(id="soc-init", type="number", value=SIM_DEFAULTS["init_soc"], step=1, style={"width":"70px"}),
            dcc.Input(id="soc-res",  type="number", value=SIM_DEFAULTS["reserve_soc"], step=1, style={"width":"70px"}),
            dcc.Input(id="soc-tgt",  type="number", value=SIM_DEFAULTS["target_soc"], step=1, style={"width":"70px"})]),
        html.Div([html.Label("kWh/km, max offset (km), min leg (km)"),
            dcc.Input(id="cons-kwhkm", type="number", value=SIM_DEFAULTS["kwh_per_km"], step=0.01, style={"width":"90px"}),
            dcc.Input(id="max-offset", type="number", value=SIM_DEFAULTS["max_charger_offset_km"], step=0.1, style={"width":"110px"}),
            dcc.Input(id="min-leg",   type="number", value=SIM_DEFAULTS["min_leg_km"], step=1, style={"width":"100px"})]),
        html.Div([html.Label("Exact optimiser (graph/DP)"),
            dcc.Checklist(id="use-rcsp", options=[{"label":"Enable","value":"on"}], value=[])],
            style={"minWidth":"180px"}),
        html.Div([html.Label("Extreme weather"),
            dcc.Checklist(id="extreme", options=[{"label":"On","value":"on"}], value=[])],
            style={"minWidth":"150px"}),
        html.Div([html.Label("Animate car / Speed (km/h)"),
            dcc.Checklist(id="animate", options=[{"label":"Animate","value":"on"}], value=["on"]),
            dcc.Input(id="speed-kmh", type="number", value=45, step=5, style={"width":"90px"})],
            style={"minWidth":"260px"}),
        html.Button("Simulate", id="btn-sim", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),
    ], style={"display":"flex","gap":"12px","alignItems":"end","flexWrap":"wrap","marginBottom":"10px"}),

    dcc.Loading(html.Iframe(id="map", src="/__map",
                            style={'width':'100%','height':'620px','border':'1px solid #ddd','borderRadius':'8px'})),
    html.Div(id="itinerary", style={"marginTop":"10px"}),

    dcc.Store(id="store-risk", data=preload_risk_json(DEFAULT_NEAR_M)),
    dcc.Store(id="overlay-refresh-token"),
    dcc.Interval(id="init", interval=250, n_intervals=0),
])

# -------------------------
# Callbacks
# -------------------------
@app.callback(
    Output("overlay-refresh-token", "data"),
    Input("btn-refresh", "n_clicks"),
    State("overlay-refresh-token", "data"),
    prevent_initial_call=True
)
def _bump_refresh(_n, tok):
    return (tok or 0) + 1

@app.callback(
    Output("store-risk", "data"),
    Input("btn-risk", "n_clicks"),
    State("near-m", "value"),
    prevent_initial_call=True
)
def _compute_risk(_n, near_m):
    thr = int(near_m or DEFAULT_NEAR_M)
    out = compute_at_risk(gdf_ev, thr)
    try: out.to_parquet(risk_cache_path(thr), index=False)
    except Exception: pass
    return out.to_json(orient="records")

@app.callback(
    Output("map", "src"),
    Output("itinerary", "children"),
    # EV filters + overlays
    Input("f-town", "value"),
    Input("f-town-like", "value"),
    Input("f-op", "value"),
    Input("f-risk", "value"),
    Input("near-m", "value"),
    Input("layers", "value"),
    Input("store-risk", "data"),
    Input("overlay-refresh-token", "data"),
    # Simulation trigger + params
    Input("btn-sim", "n_clicks"),
    State("start-lat","value"), State("start-lon","value"),
    State("end-lat","value"),   State("end-lon","value"),
    State("batt-kwh","value"),  State("soc-init","value"),
    State("soc-res","value"),   State("soc-tgt","value"),
    State("cons-kwhkm","value"),
    State("max-offset","value"), State("min-leg","value"),
    State("use-rcsp","value"),
    State("extreme","value"),
    State("animate","value"),
    State("speed-kmh","value"),
    State("init","n_intervals"),
)
def _update_map(towns, town_like, op_vals, risk_vals, near_m, layers_vals, risk_json, _tok,
                sim_clicks, sla, slo, ela, elo, batt, si, sres, stgt, kwhkm, maxoff, minleg,
                use_rcsp, extreme_vals, animate_vals, speed_kmh, _init_n):
    global LATEST_HTML

    # Base dataset + risk merge
    d = gdf_ev.copy()
    if risk_json and risk_json != "[]":
        try:
            extra = pd.read_json(StringIO(risk_json))
            if {"ROW_ID","DistWater_m","AtRisk"}.issubset(extra.columns) and not extra.empty:
                d = d.merge(extra[["ROW_ID","DistWater_m","AtRisk"]], on="ROW_ID", how="left")
        except Exception:
            pass

    # Filters
    if towns: d = d[d['Town'].isin(towns)]
    if town_like:
        s = str(town_like).strip().lower()
        if s: d = d[d['Town'].str.lower().str.contains(s, na=False)]
    op_vals = set(op_vals or [])
    if op_vals and len(op_vals) < 3:
        mask = pd.Series(False, index=d.index)
        if "op" in op_vals:   mask |= d['AvailabilityLabel'].eq("Operational")
        if "down" in op_vals: mask |= d['AvailabilityLabel'].eq("Not operational")
        if "unk" in op_vals:  mask |= d['AvailabilityLabel'].eq("Unknown") | d['AvailabilityLabel'].isna()
        d = d[mask]
    risk_vals = set(risk_vals or [])
    if risk_vals and len(risk_vals) < 3:
        if 'AtRisk' in d.columns:
            mask = pd.Series(False, index=d.index)
            if "at" in risk_vals: mask |= d['AtRisk'].fillna(False)
            if "no" in risk_vals: mask |= (~d['AtRisk'].fillna(False))
            if "unk" in risk_vals: mask |= d['AtRisk'].isna()
            d = d[mask]
        else:
            d = d.iloc[0:0]

    # Overlays
    layers_vals = set(layers_vals or [])
    show_fraw = "fraw" in layers_vals
    show_fmfp = "fmfp" in layers_vals
    show_live = "live" in layers_vals
    show_ctx  = "ctx"  in layers_vals

    itinerary_children = html.Div()

    # Route mode?
    if sim_clicks:
        extreme = "on" in (extreme_vals or [])
        animate = "on" in (animate_vals or [])
        speed = float(speed_kmh or 45)

        # Build flood union (live+FRAW; extreme widens later)
        bounds = (min(slo,elo), min(sla,ela), max(slo,elo), max(sla,ela))
        flood_union_m = get_flood_union(bounds, include_live=True, include_fraw=True, pad_m=SIM_DEFAULTS["wfs_pad_m"])

        # RCSP if requested
        if ("on" in (use_rcsp or [])) and HAS_OSMNX:
            try:
                line, safe_lines, risk_lines, stops, total_cost = rcsp_optimize(
                    float(sla), float(slo), float(ela), float(elo),
                    float(batt), float(si), float(sres), float(stgt),
                    float(kwhkm), d if not d.empty else gdf_ev, flood_union_m,
                    extreme=extreme
                )
                LATEST_HTML = render_map_html_route(
                    full_line=line, route_safe=safe_lines, route_risk=risk_lines,
                    start=(float(sla),float(slo)), end=(float(ela),float(elo)),
                    chargers=stops, animate=animate, speed_kmh=speed,
                    show_live_backdrops=extreme
                )
                rows = [f"**Exact (graph/DP)** — generalised cost ≈ {total_cost/60:.1f} min "
                        f"({'extreme weather' if extreme else 'normal'})"]
                if stops:
                    rows.append("---")
                    for i, st in enumerate(stops, 1):
                        row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st["ROW_ID"])].iloc[0]
                        rows.append(f"**Stop {i}** — {row.get('Operator','')} ({row.get('Town','')}) {row.get('Postcode','')}  "
                                    f"Operational: {st.get('Operational')}  Flood-risk: {st.get('AtRisk')}")
                itinerary_children = dcc.Markdown("\n\n".join(rows))
                return f"/__map?_ts={int(time.time())}", itinerary_children
            except Exception as e:
                itinerary_children = dcc.Markdown(f"**Exact optimiser fallback:** {e}")

        # Fallback: OSRM + animate line
        line, dist_m, dur_s, src = osrm_route(float(sla), float(slo), float(ela), float(elo))
        safe_lines, risk_lines = segment_route_by_risk(line, flood_union_m,
                                                       buffer_m=(EXTREME_BUFFER_M if extreme else SIM_DEFAULTS["route_buffer_m"]))
        LATEST_HTML = render_map_html_route(
            full_line=line, route_safe=safe_lines, route_risk=risk_lines,
            start=(float(sla),float(slo)), end=(float(ela),float(elo)),
            chargers=[], animate=animate, speed_kmh=speed,
            show_live_backdrops=extreme
        )
        itinerary_children = dcc.Markdown(f"**Fallback route:** {src} • ≈ {dist_m/1000.0:.1f} km • ≈ {dur_s/3600.0:.2f} h "
                                          f"({'extreme weather' if extreme else 'normal'})")
        return f"/__map?_ts={int(time.time())}", itinerary_children

    # Else: EV overview map
    LATEST_HTML = render_map_html_ev(d, show_fraw, show_fmfp, show_live, show_ctx, int(near_m or DEFAULT_NEAR_M))
    return f"/__map?_ts={int(time.time())}", itinerary_children

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
