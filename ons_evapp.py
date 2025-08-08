import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import Draw
from shapely.geometry import shape, Point
from shapely.ops import unary_union
import json
import io
import base64
import plotly.express as px

# --- Load data ---
file_id = "16xhVfgn4T4MEET_8ziEdBhs3nhpc_0PL"
url = f"https://drive.google.com/uc?id={file_id}"
df = pd.read_csv(url, low_memory=False)

# --- Define relevant towns ---
target_towns = [
    'Cardiff', 'Swansea', 'Newport', 'Bridgend', 'Merthyr Tydfil',
    'Llanelli', 'Carmarthen', 'Abergavenny', 'Pontypridd',
    'Neath', 'Port Talbot', 'Barry', 'Tenby'
]

# --- Clean and filter ---
# Ensure required columns exist
required_cols = {'latitude', 'longitude', 'town'}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Coerce numeric first, then drop rows with invalid coordinates
df['Latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# Basic cleaning
if 'dateCreated' in df.columns:
    df['DateCreated'] = pd.to_datetime(df['dateCreated'], errors='coerce')
else:
    df['DateCreated'] = pd.NaT

df['Town'] = df['town'].astype(str).str.strip().str.title()
df['Operator'] = df.get('deviceControllerName', pd.Series(index=df.index)).fillna("Unknown")
df['Status'] = df.get('chargeDeviceStatus', pd.Series(index=df.index)).fillna("Unknown")

# Map boolean to Yes/No/Unknown robustly
if 'paymentRequired' in df.columns:
    df['PaymentRequired'] = df['paymentRequired'].map({True: 'Yes', False: 'No'})
else:
    df['PaymentRequired'] = 'Unknown'
df['PaymentRequired'] = df['PaymentRequired'].fillna('Unknown')

# Drop invalid coords and towns
df = df.dropna(subset=['Latitude', 'Longitude', 'Town'])
df = df[df['Town'].isin(target_towns)]

# Geometry
df['geometry'] = [Point(xy) for xy in zip(df['Longitude'], df['Latitude'])]
gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
south_wales_gdf = gdf.copy()

# --- Folium map creation ---
m = folium.Map(location=[51.6, -3.2], zoom_start=9)
for _, row in south_wales_gdf.iterrows():
    popup_html = (
        f"<b>Town:</b> {row['Town']}<br>"
        f"<b>Status:</b> {row['Status']}<br>"
        f"<b>Operator:</b> {row['Operator']}<br>"
        f"<b>Postcode:</b> {row.get('postcode', 'N/A')}"
    )
    status = str(row['Status']).lower()
    icon_color = 'gray'
    if ('available' in status) or ('in service' in status):
        icon_color = 'green'
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_html,
        icon=folium.Icon(color=icon_color, icon='bolt', prefix='fa')
    ).add_to(m)

Draw(export=True).add_to(m)
folium.LayerControl().add_to(m)

MAP_HTML = "ons_ev_map.html"
m.save(MAP_HTML)

# --- Dash app ---
app = dash.Dash(__name__)

# Read the saved folium map HTML
with open(MAP_HTML, "r", encoding="utf-8") as f:
    map_html = f.read()

app.layout = html.Div([
    html.H1("ONS EV Chargers – South Wales (Selected Towns)", style={'textAlign': 'center'}),
    html.H2("Map of EV Charging Stations"),
    html.Iframe(id='map', srcDoc=map_html, width='100%', height='600'),

    html.Div([
        html.Label("Filter by Status:"),
        dcc.Dropdown(
            id='status-filter',
            options=[{'label': s, 'value': s} for s in sorted(south_wales_gdf['Status'].dropna().unique())],
            multi=True
        )
    ], style={'width': '40%', 'display': 'inline-block'}),

    html.Div([
        html.Label("Filter by Payment Required:"),
        dcc.Dropdown(
            id='payment-filter',
            options=[{'label': p, 'value': p} for p in sorted(south_wales_gdf['PaymentRequired'].dropna().unique())],
            multi=True
        )
    ], style={'width': '40%', 'display': 'inline-block', 'marginLeft': '5%'}),

    html.H2("Charger Count by Town"),
    dcc.Graph(id='bar-chart'),

    html.H2("Time Series of Charger Installations"),
    dcc.Graph(id='time-series'),

    html.H2("Upload Polygon to Filter"),
    dcc.Upload(id='upload-polygon', children=html.Button("Upload GeoJSON / GeoJSON FeatureCollection"), multiple=False),
    html.Div(id='output-count')
])

# --- Callbacks ---
@app.callback(
    Output('bar-chart', 'figure'),
    Input('status-filter', 'value'),
    Input('payment-filter', 'value')
)
def update_bar(statuses, payments):
    dff = south_wales_gdf.copy()
    if statuses:
        dff = dff[dff['Status'].isin(statuses)]
    if payments:
        dff = dff[dff['PaymentRequired'].isin(payments)]
    bar = dff['Town'].value_counts().reset_index()
    bar.columns = ['Town', 'Charger Count']
    return px.bar(
        bar, x='Town', y='Charger Count', title='Chargers by Town', color='Town',
        color_discrete_sequence=px.colors.qualitative.Pastel
    )


def _extract_polygon(geojson_obj):
    """Return a shapely polygon/multipolygon from various GeoJSON shapes.
    Accepts Feature, FeatureCollection, or Geometry object.
    If multiple polygons are present, returns their unary union.
    """
    if not isinstance(geojson_obj, dict):
        raise ValueError('Invalid GeoJSON content.')

    # If FeatureCollection
    if geojson_obj.get('type') == 'FeatureCollection':
        geoms = []
        for feat in geojson_obj.get('features', []):
            geom = feat.get('geometry')
            if geom and geom.get('type') in {'Polygon', 'MultiPolygon'}:
                geoms.append(shape(geom))
        if not geoms:
            raise ValueError('No Polygon/MultiPolygon found in FeatureCollection.')
        return unary_union(geoms)

    # If single Feature
    if geojson_obj.get('type') == 'Feature':
        geom = geojson_obj.get('geometry')
        if not geom:
            raise ValueError('Feature has no geometry.')
        if geom.get('type') not in {'Polygon', 'MultiPolygon'}:
            raise ValueError('Feature geometry is not a Polygon/MultiPolygon.')
        return shape(geom)

    # If bare geometry
    if geojson_obj.get('type') in {'Polygon', 'MultiPolygon'}:
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
            # points strictly inside polygon (change to .intersects for inclusive)
            mask = south_wales_gdf.geometry.within(poly_shape)
            selected = south_wales_gdf[mask]
            preview = selected[['Town', 'Latitude', 'Longitude']].head(10).to_string(index=False)
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
    dff = south_wales_gdf.copy()
    if statuses:
        dff = dff[dff['Status'].isin(statuses)]
    if payments:
        dff = dff[dff['PaymentRequired'].isin(payments)]
    dff = dff.dropna(subset=['DateCreated'])
    if dff.empty:
        return px.line(title='New Chargers Over Time')
    dff['Month'] = dff['DateCreated'].dt.to_period('M').astype(str)
    time_counts = dff.groupby(['Month', 'Town']).size().reset_index(name='Count')
    fig = px.line(
        time_counts,
        x='Month', y='Count', color='Town', title='New Chargers Over Time', markers=True,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_layout(xaxis_title='Month', yaxis_title='Charger Count', height=400)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
