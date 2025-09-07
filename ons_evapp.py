# dashboard_ev_merged_rcsp_zones_weather.py
# Chargers coloured by FLOOD MODEL ZONE + Routing risk from ALL flood layers + Weather
# Light/Incremental mode: fast startup with no WFS/WMS feature fetch until you opt in.

import io, os, time, json, tempfile, requests, math, heapq
from io import StringIO, BytesIO
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

# Optional graph libs for exact optimiser
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
CACHE_DIR = ".cache_wfs"; os.makedirs(CACHE_DIR, exist_ok=True)

# Logo from Google Drive (provided by you)
LOGO_GDRIVE_FILE_OR_URL = "https://drive.google.com/file/d/173EoNipH7ifFAMBbuvXfycO9nRmyQHVm/view?usp=sharing"
LOGO_CACHE_PATH = os.path.join(CACHE_DIR, "cleets_logo.png")

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
    "FRAW Rivers":  "inspire-nrw:NRW_FLOOD_RISK_FROM_RIVERS",
    "FRAW Sea":     "inspire-nrw:NRW_FLOOD_RISK_FROM_SEA",
    "FRAW Surface": "inspire-nrw:NRW_FLOOD_RISK_FROM_SURFACE_WATER_SMALL_WATERCOURSES",
}
FMFP_WFS = {
    "FMfP Rivers & Sea": "inspire-nrw:NRW_FLOODZONE_RIVERS_SEAS_MERGED",
    "FMfP Surface/Small": "inspire-nrw:NRW_FLOODZONE_SURFACE_WATER_AND_SMALL_WATERCOURSES",
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
def _gd_url(x):
    if "drive.google.com" in x:
        if "/file/d/" in x:
            fid = x.split("/file/d/")[1].split("/")[0]
            return f"https://drive.google.com/uc?export=download&id={fid}"
        return x
    return f"https://drive.google.com/uc?export=download&id={x}"

def _requests_session():
    sess = requests.Session()
    retry = Retry(total=3, connect=3, read=3, backoff_factor=0.5,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET","HEAD"]))
    sess.mount("https://", HTTPAdapter(max_retries=retry))
    sess.headers.update({"User-Agent":"Mozilla/5.0 (EV-Dashboard)"})
    return sess

def read_csv_resilient_gdrive(file_id_or_url: str, **kw):
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    try:
        r = sess.get(url, timeout=30, stream=True); r.raise_for_status()
        token = next((v for k,v in r.cookies.items() if k.startswith("download_warning")), None)
        if token:
            r = sess.get(url, params={"confirm":token}, timeout=30, stream=True); r.raise_for_status()
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
            r = requests.get(url.replace("/uc?","/uc?export=download&"), timeout=30)
            r.raise_for_status()
            return pd.read_csv(io.BytesIO(r.content), **({"low_memory":False}|kw))
        raise RuntimeError(f"Google Drive fetch failed: {e}")

def read_bytes_resilient_gdrive(file_id_or_url: str) -> bytes:
    url = _gd_url(file_id_or_url)
    sess = _requests_session()
    r = sess.get(url, timeout=30, stream=True); r.raise_for_status()
    token = next((v for k,v in r.cookies.items() if k.startswith("download_warning")), None)
    if token:
        r = sess.get(url, params={"confirm":token}, timeout=30, stream=True); r.raise_for_status()
    return r.content

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

def _bbox_for(df_like, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty:
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
# Flood model zones → label/colour
# =========================
ZONE_COLORS = {
    "Zone 3": "#D32F2F", "High": "#D32F2F",
    "Zone 2": "#FFC107", "Medium": "#FFC107",
    "Zone 1": "#2E7D32", "Low": "#2E7D32", "Very Low": "#2E7D32",
    "Outside": "#2E7D32", "Unknown": "#2E7D32"
}
ZONE_PRIORITY = ["Zone 3", "High", "Zone 2", "Medium", "Zone 1", "Low", "Very Low", "Outside", "Unknown"]
_PRI = {z:i for i,z in enumerate(ZONE_PRIORITY)}

def zone_to_icon(z: str) -> str:
    z = (z or "").strip()
    if z in ("Zone 3", "High"):   return "red"
    if z in ("Zone 2", "Medium"): return "orange"
    return "green"

def _norm_zone(props: dict, layer_name: str) -> str:
    txt = " ".join([str(v) for v in props.values() if v is not None]).lower()
    if "zone 3" in txt: return "Zone 3"
    if "zone 2" in txt: return "Zone 2"
    if "zone 1" in txt: return "Zone 1"
    if "very low" in txt: return "Very Low"
    if "high" in txt:     return "High"
    if "medium" in txt:   return "Medium"
    if "low" in txt:      return "Low"
    return "Unknown"

def fetch_model_zones_gdf(ev_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    bbox = _bbox_for(ev_gdf, pad_m=SIM_DEFAULTS.get("wfs_pad_m", 800))
    chunks = []
    for title, layer in {**FMFP_WFS, **FRAW_WFS}.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty: continue
        props_df = g.drop(columns=["geometry"], errors="ignore")
        zlabs = [_norm_zone(r.to_dict(), title) for _, r in props_df.iterrows()]
        g = g.assign(zone=zlabs, color=[ZONE_COLORS.get(z,"#2E7D32") for z in zlabs], model=title)
        try: g["geometry"] = g["geometry"].buffer(0)
        except Exception: pass
        try: g = g.explode(index_parts=False).reset_index(drop=True)
        except Exception: pass
        chunks.append(g[["zone","color","model","geometry"]])
    if not chunks:
        return gpd.GeoDataFrame(columns=["zone","color","model","geometry"], geometry="geometry", crs="EPSG:4326")
    G = pd.concat(chunks, ignore_index=True)
    return gpd.GeoDataFrame(G, geometry="geometry", crs="EPSG:4326")

def compute_model_zones_for_points(ev_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    zones = fetch_model_zones_gdf(ev_gdf)
    out = ev_gdf[["ROW_ID"]].copy()
    out["ZoneLabel"] = "Outside"; out["ZoneColor"] = ZONE_COLORS["Outside"]
    if zones.empty or ev_gdf.empty:
        return out[["ROW_ID","ZoneLabel","ZoneColor"]]
    try:
        ev_m = ev_gdf.to_crs("EPSG:27700"); zn_m = zones.to_crs("EPSG:27700")
    except Exception:
        ev_m = ev_gdf.to_crs("EPSG:3857");  zn_m = zones.to_crs("EPSG:3857")
    try:
        joined = gpd.sjoin(ev_m[["ROW_ID","geometry"]], zn_m, how="left", predicate="within")
    except Exception:
        joined = gpd.sjoin(ev_m[["ROW_ID","geometry"]], zn_m, how="left", predicate="intersects")
    if joined.empty:
        return out[["ROW_ID","ZoneLabel","ZoneColor"]]
    joined["pri"] = joined["zone"].map(_PRI).fillna(_PRI["Unknown"])
    idx = joined.sort_values(["ROW_ID","pri"]).groupby("ROW_ID", as_index=False).first()
    lut = idx.set_index("ROW_ID")
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneLabel"] = lut["zone"]
    out.loc[out["ROW_ID"].isin(lut.index), "ZoneColor"] = lut["zone"].map(ZONE_COLORS).fillna("#2E7D32")
    return out[["ROW_ID","ZoneLabel","ZoneColor"]]

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

df['Latitude']  = pd.to_numeric(df.get('latitude'), errors='coerce')
df['Longitude'] = pd.to_numeric(df.get('longitude'), errors='coerce')
df = df.dropna(subset=['Latitude','Longitude'])
df['Town'] = df[area_col]

def classify_availability(s):
    s = str(s).lower().strip()
    if any(k in s for k in ["available","in service","operational","working","ok","service"]): return True
    if any(k in s for k in ["not operational","fault","out of service","offline","unavailable","down"]): return False
    return None

_df_status = df.get('chargeDeviceStatus', pd.Series(index=df.index))
df['Available'] = _df_status.apply(classify_availability)
df['AvailabilityLabel'] = df['Available'].map({True:"Operational", False:"Not operational"}).fillna("Unknown")
df['Operator'] = df.get('deviceControllerName', df.get('Operator','Unknown'))
df['Postcode'] = df.get('postcode', df.get('Postcode','N/A'))
df['dateCreated'] = pd.to_datetime(df.get('dateCreated', df.get('DateCreated')), errors='coerce', dayfirst=True)

df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf_ev = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
gdf_ev['ROW_ID'] = gdf_ev.index.astype(int)
TOWN_OPTIONS = sorted([t for t in gdf_ev['Town'].dropna().astype(str).unique() if t])

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

def get_flood_union(bounds, include_live=True, include_fraw=True, include_fmfp=True, pad_m=SIM_DEFAULTS["wfs_pad_m"]):
    bbox = bbox_expand(bounds, pad_m); chunks=[]
    if include_fmfp:
        for lyr in FMFP_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if include_fraw:
        for lyr in FRAW_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if include_live:
        for lyr in LIVE_WFS.values():
            g = fetch_wfs_layer_cached(lyr, bbox)
            if not g.empty: chunks.append(g[['geometry']])
    if not chunks: return None
    G = gpd.GeoDataFrame(pd.concat(chunks, ignore_index=True), geometry='geometry', crs='EPSG:4326')
    try: G["geometry"] = G["geometry"].buffer(0)
    except Exception: pass
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
    _ = ox.project_graph(G, to_crs="EPSG:27700")
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

    # Chargers to nodes
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

        # Charge
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
        path_nodes.append(k[0]); k = (prev, pqi)
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
                    ROW_ID=int(row["ROW_ID"]), Operational=(row.get("AvailabilityLabel","")=="Operational")
                ))
            except Exception:
                pass
    return line, safe_lines, risk_lines, planned_stops, goal_cost

# =========================
# Folium helpers — tiles & overlays
# =========================
def add_wms_group(fmap, title_to_layer: dict, visible=True, opacity=0.55):
    for title, layer in title_to_layer.items():
        try:
            WmsTileLayer(url=OWS_BASE, layers=layer, name=f"{title} (WMS)",
                         fmt="image/png", transparent=True, opacity=opacity, version="1.3.0", show=visible).add_to(fmap)
        except Exception:
            pass

def add_live_wfs_popups(fmap, df_like):
    bbox = _bbox_for(df_like if isinstance(df_like, gpd.GeoDataFrame) and not df_like.empty else gdf_ev)
    for title, layer in LIVE_WFS.items():
        g = fetch_wfs_layer_cached(layer, bbox)
        if g.empty: continue
        folium.GeoJson(g.to_json(), name=f"{title} (WFS)",
                       style_function=lambda _: {'fillOpacity':0.15,'weight':2}).add_to(fmap)

def add_base_tiles(m):
    # OpenStreetMap (explicit URL + attribution)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        name="OpenStreetMap",
        attr="© OpenStreetMap contributors",
        control=True,
        overlay=False, max_zoom=19
    ).add_to(m)

    # CartoDB Positron / Dark Matter
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Positron",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)
    folium.TileLayer(
        tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
        name="CartoDB Dark Matter",
        attr="© OpenStreetMap contributors, © CARTO",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)

    # OSM Humanitarian (use this instead of Stamen Terrain/Toner)
    folium.TileLayer(
        tiles="https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png",
        name="OSM Humanitarian",
        attr="© OpenStreetMap contributors, Tiles courtesy of HOT",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)

    # Esri World Imagery
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/"
              "World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Esri WorldImagery",
        attr="Tiles © Esri & contributors",
        control=True, overlay=False, max_zoom=19
    ).add_to(m)

    # OpenTopoMap
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="OpenTopoMap",
        attr="© OpenStreetMap contributors, SRTM; style © OpenTopoMap (CC-BY-SA)",
        control=True, overlay=False, max_zoom=17
    ).add_to(m)

# def add_base_tiles(m):
#     folium.TileLayer("OpenStreetMap", name="OpenStreetMap", control=True).add_to(m)
#     folium.TileLayer("CartoDB positron", name="CartoDB Positron", control=True).add_to(m)
#     folium.TileLayer("CartoDB dark_matter", name="CartoDB Dark Matter", control=True).add_to(m)
#     folium.TileLayer("Stamen Terrain", name="Stamen Terrain", control=True).add_to(m)
#     folium.TileLayer("Stamen Toner", name="Stamen Toner", control=True).add_to(m)
#     folium.TileLayer(
#         tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
#         attr="Esri WorldImagery", name="Esri WorldImagery", control=True,
#         overlay=False, max_zoom=19).add_to(m)
#     folium.TileLayer(
#         tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
#         attr="&copy; OpenTopoMap contributors", name="OpenTopoMap", control=True,
#         overlay=False, max_zoom=17).add_to(m)

# =========================
# Rendering (icons by zone)
# =========================
def render_map_html_route(full_line, route_safe, route_risk, start, end, chargers,
                          animate=False, speed_kmh=50, show_live_backdrops=False):
    m = folium.Map(location=[(start[0]+end[0])/2, (start[1]+end[1])/2], zoom_start=11,
                   tiles=None, control_scale=True)
    add_base_tiles(m)
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
        zlabel = row.get("ZoneLabel","Outside")
        icon_col = zone_to_icon(zlabel)
        popup = folium.Popup(
            f"<b>{row.get('Operator','')}</b><br>{row.get('Town','')} {row.get('Postcode','')}"
            f"<br><b>Flood model zone:</b> {zlabel}"
            f"<br><b>Operational:</b> {st.get('Operational')}",
            max_width=320
        )
        folium.Marker([row["Latitude"], row["Longitude"]],
                      tooltip=f"{row.get('Operator','')} ({row.get('Town','')})",
                      popup=popup,
                      icon=folium.Icon(color=icon_col, icon="bolt", prefix="fa")).add_to(cluster)

    if show_live_backdrops:
        add_wms_group(m, LIVE_WMS, visible=True, opacity=0.65)

    folium.LayerControl(collapsed=True).add_to(m)

    legend_html = """
    <div style="position: fixed; bottom:20px; left:20px; z-index:9999; background:white; padding:10px 12px; border:1px solid #ccc; border-radius:6px; font-size:13px;">
      <b>Charger icon colour — Flood model zone</b>
      <div style="margin-top:6px"><span style="display:inline-block;width:12px;height:12px;background:#D32F2F;margin-right:6px;"></span> Red: Zone 3 / High</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#FFC107;margin-right:6px;"></span> Amber: Zone 2 / Medium</div>
      <div><span style="display:inline-block;width:12px;height:12px;background:#2E7D32;margin-right:6px;"></span> Green: Zone 1 / Low–Very Low / Outside</div>
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    return m.get_root().render()

def render_map_html_ev(df_map, show_fraw, show_fmfp, show_live, show_ctx, light=False):
    m = folium.Map(location=[51.6,-3.2], zoom_start=9, tiles=None, control_scale=True)
    add_base_tiles(m)
    cluster = MarkerCluster(name="EV Chargers").add_to(m)
    Draw(export=False, position='topleft',
         draw_options={'polygon': {'allowIntersection': False, 'showArea': True},
                       'rectangle': True, 'polyline': False, 'circle': False,
                       'circlemarker': False, 'marker': False},
         edit_options={'edit': True}).add_to(m)

    for _, row in df_map.iterrows():
        zlabel = row.get("ZoneLabel", "Outside") or "Outside"
        icon_col = zone_to_icon(zlabel)
        parts = [
            f"<b>Operator:</b> {row.get('Operator','EV')}",
            f"<b>Status:</b> {row.get('AvailabilityLabel','Unknown')}",
            f"<b>Flood model zone:</b> {zlabel}",
        ]
        if pd.notna(row.get('dateCreated', pd.NaT)): parts.append(f"<b>Installed:</b> {row['dateCreated']}")
        folium.Marker([row['Latitude'], row['Longitude']],
                      tooltip=f"{row.get('Operator','')} ({row.get('Town','')})",
                      popup=folium.Popup("<br>".join(parts), max_width=320),
                      icon=folium.Icon(color=icon_col, icon='bolt', prefix='fa')).add_to(cluster)

    # Tile overlays are client-side; allow WMS tiles even in light mode.
    if show_fraw: add_wms_group(m, FRAW_WMS, True, 0.50)
    if show_fmfp: add_wms_group(m, FMFP_WMS, True, 0.55)
    if show_ctx:  add_wms_group(m, CONTEXT_WMS, False, 0.45)

    # WFS feature overlays (server fetch) — skip in light mode
    if show_live and not light:
        add_wms_group(m, LIVE_WMS, True, 0.65)
        add_live_wfs_popups(m, df_map)
    elif show_live:
        add_wms_group(m, LIVE_WMS, True, 0.65)

    folium.LayerControl(collapsed=True).add_to(m)

    if light:
        banner = """
        <div style="position: fixed; top:18px; right:18px; z-index:9999; background:#fff3cd; color:#6c4f00;
             padding:8px 10px; border:1px solid #ffe69c; border-radius:6px; font: 13px/1.3 system-ui, sans-serif;">
          Light mode: zones & live WFS skipped. Click “Compute/Update zones” to enable.
        </div>"""
        m.get_root().html.add_child(folium.Element(banner))

    return m.get_root().render()

# =========================
# Weather (Open-Meteo default; Met Office optional)
# =========================
from functools import lru_cache

@lru_cache(maxsize=128)
def cached_get(url, headers_tuple=(), params_tuple=()):
    headers = dict(headers_tuple) if headers_tuple else {}
    params = dict(params_tuple) if params_tuple else {}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()

def _ts():
    return str(int(time.time()//300))

METOFFICE_KEY = os.environ.get("METOFFICE_KEY", "").strip()
METOFFICE_SITE_API = os.environ.get("METOFFICE_SITE_API", "").strip()  # optional

def get_weather(lat=51.48, lon=-3.18):
    try:
        if METOFFICE_KEY and METOFFICE_SITE_API:
            headers = (("apikey", METOFFICE_KEY),)
            params = (("latitude", str(lat)), ("longitude", str(lon)),)
            data = cached_get(METOFFICE_SITE_API, headers, params)
            return {"provider":"Met Office", "raw": data}
        else:
            url = "https://api.open-meteo.com/v1/forecast"
            params = (
                ("latitude", str(lat)), ("longitude", str(lon)),
                ("current", "temperature_2m,precipitation,wind_speed_10m"),
                ("hourly", "temperature_2m,precipitation_probability,wind_speed_10m"),
                ("timezone", "Europe/London"),
                ("_ts", _ts()),
            )
            data = cached_get(url, (), params)
            return {"provider":"Open-Meteo", "raw": data}
    except Exception as e:
        return {"provider":"error", "error": str(e)}

def _parse_metoffice_timeseries(raw):
    try:
        feats = raw.get("features") or []
        if feats and isinstance(feats, list):
            ts = feats[0].get("properties", {}).get("timeSeries") or []
            times = [r.get("time") for r in ts if "time" in r][:24]
            temps = [r.get("screenTemperature") for r in ts][:24]
            pops  = [r.get("precipitationProbability") or r.get("precipProb") for r in ts][:24]
            if times and temps:
                return {"time": times, "temp": temps, "pop": pops or [None]*len(times)}
    except Exception:
        pass
    return {}

# =========================
# Dash app + header logo + KML download
# =========================
app = dash.Dash(__name__)
server = app.server

# Serve logo from Google Drive (cached to disk)
@server.route("/__logo")
def _serve_logo():
    try:
        if not os.path.exists(LOGO_CACHE_PATH) or (time.time()-os.path.getmtime(LOGO_CACHE_PATH)) > 7*24*3600:
            content = read_bytes_resilient_gdrive(LOGO_GDRIVE_FILE_OR_URL)
            with open(LOGO_CACHE_PATH, "wb") as f: f.write(content)
        with open(LOGO_CACHE_PATH, "rb") as f:
            return Response(f.read(), mimetype="image/png")
    except Exception:
        # tiny transparent placeholder
        return Response(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06'
                        b'\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\x0bIDATx\x9cc``\x00\x00\x00\x02\x00\x01'
                        b'\xe2!\xbc3\x00\x00\x00\x00IEND\xaeB`\x82', mimetype="image/png")

WALES_LOCS = {
    "Cardiff": (51.4816, -3.1791),
    "Swansea": (51.6214, -3.9436),
    "Newport": (51.5842, -2.9977),
    "Aberystwyth": (52.4153, -4.0829),
    "Bangor": (53.2290, -4.1294)
}

def preload_zones_json() -> str:
    p = "cache_model_zones.parquet"
    if os.path.exists(p):
        try:
            cache = pd.read_parquet(p)
            if {"ROW_ID","ZoneLabel","ZoneColor"}.issubset(cache.columns) and not cache.empty:
                return cache[["ROW_ID","ZoneLabel","ZoneColor"]].to_json(orient="records")
        except Exception:
            pass
    return "[]"

# Build KML from route+stops
def _kml_escape(s: str) -> str:
    return (s or "").replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def build_kml(route_data: dict) -> str:
    name = "EV Journey Simulator"
    linestring = ""
    coords = route_data.get("route") or []
    if coords:
        coord_str = " ".join([f"{p['lon']:.6f},{p['lat']:.6f},0" for p in coords])
        linestring = f"""
        <Placemark>
          <name>Planned route</name>
          <Style><LineStyle><color>ff8a2be2</color><width>4</width></LineStyle></Style>
          <LineString><tessellate>1</tessellate><coordinates>{coord_str}</coordinates></LineString>
        </Placemark>"""
    def mk_pt(title, lat, lon, color_hex=None):
        kml_color = "ff2e7d32" if color_hex is None else "ff" + "".join(reversed([color_hex[i:i+2] for i in (1,3,5)]))
        return f"""
        <Placemark>
          <name>{_kml_escape(title)}</name>
          <Style><IconStyle><color>{kml_color}</color></IconStyle></Style>
          <Point><coordinates>{lon:.6f},{lat:.6f},0</coordinates></Point>
        </Placemark>"""
    pts = []
    s = route_data.get("start"); e = route_data.get("end")
    if s: pts.append(mk_pt("Start", s["lat"], s["lon"], "#2E7D32"))
    if e: pts.append(mk_pt("End",   e["lat"], e["lon"], "#1f78b4"))
    for i, st in enumerate(route_data.get("stops") or [], 1):
        title = st.get("name") or f"Stop {i}"
        pts.append(mk_pt(title, st["lat"], st["lon"], "#FFC107"))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
  <name>{_kml_escape(name)}</name>
  {linestring}
  {''.join(pts)}
</Document>
</kml>"""

# ---------- Layout (iframe uses srcDoc now) ----------
app.layout = html.Div([
    html.Div([
        html.Img(src="/__logo", style={"height":"240px"}),  # bigger logo
        html.H1("South Wales EV – Chargers coloured by Flood Model Zone (FMfP/FRAW) + Routing & Weather",
                style={"margin":"0"})
    ], style={"display":"flex","alignItems":"center","gap":"12px","marginBottom":"10px"}),

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
        html.Div([html.Label("Show overlays"),
            dcc.Checklist(id="layers",
                options=[{"label":"FRAW","value":"fraw"},
                         {"label":"FMfP","value":"fmfp"},
                         {"label":"Live warnings","value":"live"},
                         {"label":"Context","value":"ctx"}],
                value=["fraw","fmfp"], inputStyle={"marginRight":"6px"})], style={"minWidth":"360px"}),
        html.Div([html.Label("Start-up mode"),
            dcc.Checklist(id="light", options=[{"label":"Light mode (fast start)","value":"on"}],
                          value=["on"])], style={"minWidth":"260px"}),
        html.Button("Compute/Update zones", id="btn-zones", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),
        html.Button("Refresh overlays", id="btn-refresh", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),
    ], style={"display":"flex","gap":"12px","alignItems":"end","flexWrap":"wrap","margin":"6px 0 12px"}),

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
        html.Button("Download route for Google Maps (KML)", id="btn-kml", n_clicks=0, style={"height":"38px","marginLeft":"8px"}),
    ], style={"display":"flex","gap":"12px","alignItems":"end","flexWrap":"wrap","marginBottom":"10px"}),

    # Iframe uses srcDoc; updated directly from callbacks
    dcc.Loading(html.Iframe(
        id="map",
        srcDoc="<html><body style='font-family:sans-serif;padding:10px'>Loading…</body></html>",
        style={'width':'100%','height':'620px','border':'1px solid #ddd','borderRadius':'8px'}
    )),
    html.Div(id="itinerary", style={"marginTop":"10px"}),

    html.H2("C) Weather for Wales"),
    html.Div([
        html.Div([html.Label("Location"),
                  dcc.Dropdown(id='wales-location', value='Cardiff',
                               options=[{"label":k, "value":k} for k in WALES_LOCS.keys()],
                               clearable=False, style={"width":"220px"})], style={"marginRight":"16px"}),
        dcc.Interval(id='wx-refresh', interval=5*60*1000, n_intervals=0)
    ], style={"display":"flex","alignItems":"center","gap":"12px"}),
    html.Div(id='weather-split', style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"12px","alignItems":"stretch"}),

    dcc.Store(id="store-zones", data=preload_zones_json()),
    dcc.Store(id="overlay-refresh-token"),
    dcc.Store(id="store-route"),
    dcc.Download(id="dl-kml"),
    # Fire once on load so the map renders immediately
    dcc.Interval(id="init", interval=250, n_intervals=0, max_intervals=1),
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
    Output("store-zones", "data"),
    Input("btn-zones", "n_clicks"),
    prevent_initial_call=True
)
def _compute_zones(_n):
    zones = compute_model_zones_for_points(gdf_ev)
    try: zones.to_parquet("cache_model_zones.parquet", index=False)
    except Exception: pass
    return zones.to_json(orient="records")

@app.callback(
    Output("map", "srcDoc"),
    Output("itinerary", "children"),
    Output("store-route", "data"),
    # EV filters + overlays
    Input("f-town", "value"),
    Input("f-town-like", "value"),
    Input("f-op", "value"),
    Input("layers", "value"),
    Input("light", "value"),              # <-- Light/Incremental toggle
    Input("store-zones", "data"),
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
    # Trigger once on load
    Input("init","n_intervals"),
)
def _update_map(towns, town_like, op_vals, layers_vals, light_vals, zones_json, _tok,
                sim_clicks, sla, slo, ela, elo, batt, si, sres, stgt, kwhkm, maxoff, minleg,
                use_rcsp, extreme_vals, animate_vals, speed_kmh, _init_n):

    light = "on" in (light_vals or [])

    # Base dataset + zone join (skip heavy WFS in light mode unless user pressed Compute)
    d = gdf_ev.copy()
    zones_df = None
    if zones_json and zones_json != "[]":
        try:
            zextra = pd.read_json(StringIO(zones_json))
            if {"ROW_ID","ZoneLabel","ZoneColor"}.issubset(zextra.columns) and not zextra.empty:
                zones_df = zextra[["ROW_ID","ZoneLabel","ZoneColor"]]
        except Exception:
            pass
    if zones_df is None:
        if light:
            zones_df = pd.DataFrame({"ROW_ID": gdf_ev["ROW_ID"],
                                     "ZoneLabel": "Outside",
                                     "ZoneColor": ZONE_COLORS["Outside"]})
        else:
            zones_df = compute_model_zones_for_points(gdf_ev)

    d = d.merge(zones_df, on="ROW_ID", how="left")
    d["ZoneLabel"] = d["ZoneLabel"].fillna("Outside")
    d["ZoneColor"] = d["ZoneColor"].fillna(ZONE_COLORS["Outside"])

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

    # Overlays
    layers_vals = set(layers_vals or [])
    show_fraw = "fraw" in layers_vals
    show_fmfp = "fmfp" in layers_vals
    show_live = "live" in layers_vals and not light  # live WFS popups skipped in light mode
    show_ctx  = "ctx"  in layers_vals

    itinerary_children = html.Div()
    route_store = {}

    # Route mode
    if sim_clicks:
        try:
            extreme = "on" in (extreme_vals or [])
            animate = "on" in (animate_vals or [])
            speed = float(speed_kmh or 45)

            # Risk union can be very heavy — skip in light mode
            flood_union_m = None
            if not light:
                bounds = (min(slo,elo), min(sla,ela), max(slo,elo), max(sla,ela))
                flood_union_m = get_flood_union(bounds, include_live=True, include_fraw=True, include_fmfp=True,
                                                pad_m=SIM_DEFAULTS["wfs_pad_m"])

            if ("on" in (use_rcsp or [])) and HAS_OSMNX:
                line, safe_lines, risk_lines, stops, total_cost = rcsp_optimize(
                    float(sla), float(slo), float(ela), float(elo),
                    float(batt), float(si), float(sres), float(stgt),
                    float(kwhkm), d if not d.empty else gdf_ev, flood_union_m,
                    extreme=(extreme and not light)
                )
                # enrich stops with coords + names for KML
                stop_features = []
                for st in stops:
                    row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st["ROW_ID"])].iloc[0]
                    stop_features.append(dict(
                        lat=float(row["Latitude"]), lon=float(row["Longitude"]),
                        name=f"{row.get('Operator','')} ({row.get('Town','')})",
                        desc=f"Postcode: {row.get('Postcode','')}; Zone: {row.get('ZoneLabel','Outside')}"
                    ))

                html_str = render_map_html_route(
                    full_line=line, route_safe=safe_lines, route_risk=risk_lines,
                    start=(float(sla),float(slo)), end=(float(ela),float(elo)),
                    chargers=stops, animate=animate, speed_kmh=speed,
                    show_live_backdrops=(extreme and not light)
                )
                rows = [f"**Exact (graph/DP)** — generalised cost ≈ {total_cost/60:.1f} min "
                        f"({'light mode' if light else ('extreme weather' if extreme else 'normal')})"]
                if stops:
                    rows.append("---")
                    for i, st in enumerate(stops, 1):
                        row = gdf_ev.loc[gdf_ev["ROW_ID"].eq(st["ROW_ID"])].iloc[0]
                        rows.append(f"**Stop {i}** — {row.get('Operator','')} ({row.get('Town','')}) {row.get('Postcode','')} — Zone: {row.get('ZoneLabel','Outside')}")
                itinerary_children = dcc.Markdown("\n\n".join(rows))

                coords_latlng = [{"lat":lat, "lon":lon} for lon,lat in list(line.coords)]
                route_store = dict(
                    start={"lat":float(sla), "lon":float(slo)},
                    end={"lat":float(ela), "lon":float(elo)},
                    route=coords_latlng,
                    stops=stop_features,
                    created_ts=time.time()
                )
                return html_str, itinerary_children, route_store

            # Fallback to OSRM / straight line
            line, dist_m, dur_s, src = osrm_route(float(sla), float(slo), float(ela), float(elo))
            safe_lines, risk_lines = segment_route_by_risk(line, flood_union_m,
                                                           buffer_m=(EXTREME_BUFFER_M if (extreme and not light) else SIM_DEFAULTS["route_buffer_m"]))
            html_str = render_map_html_route(
                full_line=line, route_safe=safe_lines, route_risk=risk_lines,
                start=(float(sla),float(slo)), end=(float(ela),float(elo)),
                chargers=[], animate=animate, speed_kmh=speed,
                show_live_backdrops=(extreme and not light)
            )
            itinerary_children = dcc.Markdown(
                f"**Fallback route:** {src} • ≈ {dist_m/1000.0:.1f} km • ≈ {dur_s/3600.0:.2f} h "
                f"({'light mode' if light else ('extreme weather' if extreme else 'normal')})"
            )
            coords_latlng = [{"lat":lat, "lon":lon} for lon,lat in list(line.coords)]
            route_store = dict(
                start={"lat":float(sla), "lon":float(slo)},
                end={"lat":float(ela), "lon":float(elo)},
                route=coords_latlng,
                stops=[],
                created_ts=time.time()
            )
            return html_str, itinerary_children, route_store

        except Exception as e:
            try:
                line = LineString([(float(slo),float(sla)), (float(elo),float(ela))])
            except Exception:
                line = LineString([(-3.2,51.5), (-3.9,51.6)])
            html_str = render_map_html_route(
                full_line=line, route_safe=[line], route_risk=[],
                start=(float(sla or 51.5), float(slo or -3.2)),
                end=(float(ela or 51.6), float(elo or -3.9)),
                chargers=[], animate=False, speed_kmh=45, show_live_backdrops=False
            )
            itinerary_children = dcc.Markdown(f"**Simulation error:** {e}")
            return html_str, itinerary_children, {}

    # EV overview
    html_str = render_map_html_ev(d, show_fraw, show_fmfp, show_live, show_ctx, light=light)
    return html_str, itinerary_children, {}

# Weather split
@app.callback(
    Output("weather-split", "children"),
    Input("wales-location", "value"),
    Input("wx-refresh", "n_intervals")
)
def _wx_split(loc, _n):
    lat, lon = WALES_LOCS.get(loc, (51.60, -3.20))
    data = get_weather(lat, lon)
    prov = data.get("provider","?")
    raw = data.get("raw") or {}

    left_children = [html.H3(f"{loc} – Current ({prov})")]
    if prov == "Open-Meteo":
        cur = raw.get("current", {})
        if cur:
            left_children += [
                html.Div(f"Temperature: {cur.get('temperature_2m','?')} °C"),
                html.Div(f"Precipitation: {cur.get('precipitation','?')} mm"),
                html.Div(f"Wind: {cur.get('wind_speed_10m','?')} m/s"),
            ]
    elif prov == "Met Office":
        left_children += [html.Pre(json.dumps(raw, indent=2)[:1200])]
    elif prov == "error":
        left_children += [html.Div("Weather error: " + data.get("error",""))]

    left = html.Div(style={"border":"1px solid #eee","borderRadius":"10px","padding":"10px"}, children=left_children)

    try:
        import plotly.graph_objects as go
        if prov == "Open-Meteo":
            hrs = raw.get("hourly", {})
            times = hrs.get("time") or []
            temps = hrs.get("temperature_2m") or []
            pops  = hrs.get("precipitation_probability") or []
            times = times[:24]; temps = temps[:24]; pops = pops[:24]
            df2 = pd.DataFrame({"time": times, "temp": temps, "pop": pops})
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df2["time"], y=df2["pop"], name="Precip %", yaxis="y2", opacity=0.5))
            fig.add_trace(go.Scatter(x=df2["time"], y=df2["temp"], name="Temp °C"))
            fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320,
                              xaxis_title="", yaxis_title="Temp (°C)",
                              yaxis2=dict(title="Precip (%)", overlaying="y", side="right"))
        elif prov == "Met Office":
            ts = _parse_metoffice_timeseries(raw)
            fig = go.Figure()
            if ts:
                df2 = pd.DataFrame({"time": ts.get("time",[]), "temp": ts.get("temp",[]), "pop": ts.get("pop",[])})
                if any(df2.get("pop", [])):
                    fig.add_trace(go.Bar(x=df2["time"], y=df2["pop"], name="Precip %", yaxis="y2", opacity=0.5))
                fig.add_trace(go.Scatter(x=df2["time"], y=df2["temp"], name="Temp °C"))
                fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320,
                                  xaxis_title="", yaxis_title="Temp (°C)",
                                  yaxis2=dict(title="Precip (%)", overlaying="y", side="right"))
            else:
                fig.update_layout(title="Met Office: timeseries not found", height=320)
        else:
            import plotly.graph_objects as go
            fig = go.Figure(); fig.update_layout(margin=dict(l=10,r=10,t=30,b=10), height=320)
    except Exception as e:
        import plotly.graph_objects as go
        fig = go.Figure(); fig.update_layout(title=f"Weather chart error: {e}", height=320)

    right = html.Div(style={"border":"1px solid #eee","borderRadius":"10px","padding":"10px"},
                     children=[html.H3("Next 24h forecast"), dcc.Graph(figure=fig, config={"displayModeBar": False})])

    return [left, right]

# ============ KML Download ============
@app.callback(
    Output("dl-kml", "data"),
    Input("btn-kml", "n_clicks"),
    State("store-route", "data"),
    prevent_initial_call=True
)
def _download_kml(_n, route_data):
    if not route_data or not (route_data.get("route") and route_data.get("start") and route_data.get("end")):
        return dash.no_update
    kml = build_kml(route_data)
    return dict(content=kml, filename="ev_journey.kml", type="application/vnd.google-earth.kml+xml")

# -------------------------
# Run
# -------------------------
if __name__ == "__main__":
    app.run_server(debug=True)
