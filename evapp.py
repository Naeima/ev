import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import geopandas as gpd
import folium
from folium.plugins import Draw
from shapely.geometry import shape, Point
import json
import io
import requests
import base64
from dash import dash_table

# Download and load dataset from Google Drive
file_id = "1jqzdEnrhvgbDqw4dbQPrLjVddGm5FkSy"
url = f"https://drive.google.com/uc?id={file_id}"
ev_df = pd.read_csv(url)
ev_df = ev_df.dropna(subset=['Latitude', 'Longitude', 'Town'])

# Prepare filtered subset
# Plot all stations without filtering by operational status
map_data = ev_df.copy()
map_data = map_data[['Latitude', 'Longitude', 'Town', 'PowerKW', 'PowerLevel', 'CurrentType',
                     'IsOperational', 'Level.IsFastChargeCapable', 'Operator', 'DateLastStatusUpdate']].dropna(subset=['Latitude', 'Longitude', 'Town'])
gdf = gpd.GeoDataFrame(map_data, geometry=gpd.points_from_xy(map_data.Longitude, map_data.Latitude), crs="EPSG:4326")

# Approximate coastline filter: bounding box within ~2km of coastline (crude heuristic)
def is_near_sea(lat, lon):
    return (51.3 <= lat <= 51.7) and (lon <= -3.5)

gdf['NearSea'] = gdf.apply(lambda row: is_near_sea(row['Latitude'], row['Longitude']), axis=1)

# Create Folium map
m = folium.Map(location=[51.48, -3.18], zoom_start=10, control_scale=True)
for _, row in gdf.iterrows():
    popup_html = "<br>".join([f"<b>{col}</b>: {row[col]}" for col in ['Town', 'PowerKW', 'PowerLevel', 'Operator', 'Latitude', 'Longitude'] if pd.notna(row[col])])
    if row['IsOperational'] == 1:
        color = 'green' if not row['NearSea'] else 'blue'
    else:
        color = 'gray'
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=popup_html,
        icon=folium.Icon(color=color, icon='bolt', prefix='fa')
    ).add_to(m)

draw = Draw(export=True, filename='polygon.json', draw_options={
    'polygon': True, 'rectangle': False, 'circle': False, 'polyline': False, 'marker': False, 'circlemarker': False
}, position='topleft')
draw.add_to(m)
folium.LayerControl().add_to(m)


# Add EV journey from Newport to Swansea via Cardiff with moving icon and estimated travel details
route_coords = {
    "Newport": [51.5842, -2.9977],
    "Cardiff": [51.4816, -3.1791],
    "Swansea": [51.6214, -3.9436]
}

# Draw route polyline and moving marker animation
from folium.plugins import TimestampedGeoJson

# Create journey segments
journey_points = [
    {"coordinates": coord, "city": city, "timestamp": f"2024-01-01T0{i}:00:00"} 
    for i, (city, coord) in enumerate(route_coords.items())
]

features = [{
    "type": "Feature",
    "geometry": {"type": "Point", "coordinates": [p['coordinates'][1], p['coordinates'][0]]},
    "properties": {
        "time": p['timestamp'],
        "popup": p['city'],
        "icon": "car"
    }
} for p in journey_points]

TimestampedGeoJson({
    "type": "FeatureCollection",
    "features": features
},
    period="PT1H",
    add_last_point=True,
    auto_play=True,
    loop=False,
    max_speed=1,
    loop_button=True,
    date_options="YYYY-MM-DD HH:mm",
    time_slider_drag_update=True
).add_to(m)

# Draw polyline for static journey
folium.PolyLine(
    locations=[route_coords[city] for city in ["Newport", "Cardiff", "Swansea"]],
    color="red",
    weight=2,
    opacity=0.8,
    tooltip="Planned EV Route",
    dash_array='5, 5'
).add_to(m)

# Add route markers
for city, coord in route_coords.items():
    folium.Marker(
        location=coord,
        popup=f"{city} (Stop)",
        icon=folium.Icon(color='red', icon='car', prefix='fa')
    ).add_to(m)

# Estimate energy usage and simulate state of charge (SOC) dynamics
# Assume 6 km/kWh efficiency and 100 km between each city
# Parameters
battery_capacity_kwh = 60
start_soc_pct = 80  # 80% of full battery
km_per_kwh = 6

# Route distances (km)
distances_km = {"Newport-Cardiff": 25, "Cardiff-Swansea": 66}

# Calculate consumption
soc = start_soc_pct / 100 * battery_capacity_kwh
segment_energy = {seg: dist / km_per_kwh for seg, dist in distances_km.items()}
soc_after_cardiff = soc - segment_energy["Newport-Cardiff"]
soc_after_swansea = soc_after_cardiff - segment_energy["Cardiff-Swansea"]

journey_case_text = f"""<b>Journey Case Study</b><br>
Starting SOC: {start_soc_pct}% ({soc:.1f} kWh)<br>
Newport → Cardiff (~25 km): -{segment_energy['Newport-Cardiff']:.1f} kWh → {soc_after_cardiff:.1f} kWh<br>
Cardiff → Swansea (~66 km): -{segment_energy['Cardiff-Swansea']:.1f} kWh → {soc_after_swansea:.1f} kWh<br>
<b>Recommendation:</b> Stop in Cardiff or Bridgend if SOC &lt; 40%."""
m.get_root().html.add_child(folium.Element(f"<div style='position: fixed; bottom: 10px; right: 10px; background: white; padding: 10px; z-index: 9999; font-size: 14px;'>{journey_case_text}</div>"))

# Highlight charging options within 2km of each stop
for stop, coord in route_coords.items():
    buffer = Point(coord[::-1]).buffer(0.02)  # approx 2km buffer
    nearby_stations = gdf[gdf.geometry.within(buffer)]
    for _, row in nearby_stations.iterrows():
        folium.CircleMarker(
            location=[row['Latitude'], row['Longitude']],
            radius=6,
            color='purple',
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Town']}: {row['PowerKW']}kW"
        ).add_to(m)

m.save("ev_charger_map.html")

# Launch Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    # Date range filter: allows user to explore temporal trends in charger installation and usage
    html.Div([
        html.Label("Select Date Range:"),
        dcc.DatePickerRange(
            id='date-range-picker',
            start_date=gdf['DateLastStatusUpdate'].min(),
            end_date=gdf['DateLastStatusUpdate'].max(),
            display_format='YYYY-MM-DD'
        )
    ], style={'marginBottom': 30}),
    # Interactive map with markers for chargers and tools to draw a polygon for spatial filtering
    html.H1("South Wales EV Chargers Map with Polygon Tool"),
    html.Iframe(id='map', srcDoc=open('ev_charger_map.html', 'r').read(), width='100%', height='600'),
    html.P("Legend: Green = Operational (Inland), Blue = Operational (Near Sea), Gray = Non-Operational", style={'fontSize': '18px', 'marginTop': '5px'}),
    html.P("Draw a polygon on the map, export it, and upload the GeoJSON below to filter chargers."),
    dcc.Upload(id='upload-polygon', children=html.Button('Upload Polygon JSON'), multiple=False),
    html.Div(id='output-count'),
    # List and CSV download of stations within ~2km of coastline
    html.H3("Chargers Near the Sea (within ~2km of coast):"),
    html.P("This section lists EV charging stations that are geographically close to the coast based on a spatial filter."),
    html.Div(id='near-sea-output'),
    html.Br(),
    html.A("Download CSV of Coastal Chargers", id='download-link', download="coastal_chargers.csv", href="", target="_blank"),
    # Bar chart summarizing charger counts by town, filtered by selected date range
    html.H3("Charger Distribution by Town"),
    html.P("Bar chart showing the number of charging stations per town within the selected date range."),
    dcc.Graph(id='bar-plot'),
    
    # Battery simulator: calculates and visualizes EV charge levels for a trip across major cities
    html.H3("EV Journey Battery Simulator"),
    html.P("Estimate and visualize the battery usage of an EV traveling from Newport to Swansea via Cardiff based on user-defined parameters."),
    html.Div([
        html.Label("Initial SOC (%)"),
        dcc.Slider(id='soc-slider', min=10, max=100, step=5, value=80, marks={i: str(i) for i in range(10, 101, 10)}),
        html.Label("Battery Capacity (kWh)"),
        dcc.Input(id='battery-capacity', type='number', min=10, max=120, step=5, value=60),
        html.Label("Efficiency (km/kWh)"),
        dcc.Input(id='efficiency', type='number', min=2, max=10, step=0.5, value=6)
    ], style={'marginBottom': 20}),
    dcc.Graph(id='soc-graph'),
    # Correlation explorer: analyze relationships between charger features with filtering options
    html.H3("Advanced Correlation Explorer"),
    html.P("This interactive tool lets you correlate features such as charger power, operational status, and sea proximity with custom filters."),
    html.Div([
        html.Label("Filter: Operational Status"),
        dcc.Checklist(
            id='operational-filter',
            options=[
                {'label': 'Only Operational', 'value': 1},
                {'label': 'Only Non-Operational', 'value': 0}
            ],
            value=[]
        ),
        html.Label("Filter: Minimum Power kW"),
        dcc.Slider(
            id='power-filter',
            min=0,
            max=150,
            step=5,
            value=0,
            marks={i: str(i) for i in range(0, 151, 25)}
        ),
        html.Label("Filter: Fast Charge Capability"),
        dcc.Checklist(
            id='fast-charge-filter',
            options=[
                {'label': 'Fast Charge Only', 'value': 1}
            ],
            value=[]
        )
    ], style={'marginBottom': 20}),
    html.Div([
        html.Label("Select X variable:"),
        dcc.Dropdown(id='x-dropdown', options=[
            {'label': col, 'value': col} for col in ['PowerKW', 'IsOperational', 'Level.IsFastChargeCapable']
        ], value='PowerKW')
    ], style={'width': '45%', 'display': 'inline-block'}),
    html.Div([
        html.Label("Select Y variable:"),
        dcc.Dropdown(id='y-dropdown', options=[
            {'label': col, 'value': col} for col in ['IsOperational', 'Level.IsFastChargeCapable', 'NearSea']
        ], value='IsOperational')
    ], style={'width': '45%', 'display': 'inline-block', 'marginLeft': '5%'}),
    dcc.Graph(id='correlation-graph')
])

@app.callback(
    Output('output-count', 'children'),
    Input('upload-polygon', 'contents'),
    State('upload-polygon', 'filename')
)
def filter_by_polygon(contents, filename):
    if contents and filename.endswith('.json'):
        content_string = contents.split(',')[1]
        decoded = base64.b64decode(content_string)
        polygon_geojson = json.load(io.StringIO(decoded.decode('utf-8')))
        poly_shape = shape(polygon_geojson['geometry'])
        selected = gdf[gdf.geometry.within(poly_shape)]
        return html.Div([
            html.P(f"{len(selected)} charging points found within the polygon."),
            dcc.Markdown('### Preview of filtered chargers:'),
            html.Pre(selected[['Town', 'PowerKW', 'PowerLevel', 'Operator']].head(10).to_string(index=False))
        ])
    return "Upload a valid polygon GeoJSON file to filter chargers."

@app.callback(
    Output('near-sea-output', 'children'),
    Output('download-link', 'href'),
    Input('upload-polygon', 'contents')
)
def update_near_sea(_):
    near_sea_chargers = gdf[gdf['NearSea'] == True]
    preview = near_sea_chargers[['Town', 'Latitude', 'Longitude', 'PowerKW', 'PowerLevel', 'Operator']].head(10)
    csv_string = near_sea_chargers.to_csv(index=False, encoding='utf-8')
    csv_encoded = "data:text/csv;charset=utf-8," + requests.utils.quote(csv_string)
    return html.Pre(preview.to_string(index=False)), csv_encoded

@app.callback(
    Output('bar-plot', 'figure'),
    Input('upload-polygon', 'contents'),
    Input('date-range-picker', 'start_date'),
    Input('date-range-picker', 'end_date')
)
def update_bar_plot(_, start_date, end_date):
    filtered = gdf.copy()
    filtered['Date'] = pd.to_datetime(filtered['DateLastStatusUpdate'], errors='coerce')
    if start_date and end_date:
        filtered = filtered[(filtered['Date'] >= pd.to_datetime(start_date)) & (filtered['Date'] <= pd.to_datetime(end_date))]
    bar_data = filtered.groupby('Town').size().reset_index(name='ChargerCount')
    fig = px.bar(bar_data, x='Town', y='ChargerCount', title='Number of Chargers by Town')
    return fig

@app.callback(
    Output('soc-graph', 'figure'),
    Input('soc-slider', 'value'),
    Input('battery-capacity', 'value'),
    Input('efficiency', 'value')
)
def update_soc_chart(soc_pct, capacity, efficiency):
    route_segments = ["Start", "Newport", "Cardiff", "Swansea"]
    distances = [0, 0, 25, 91]  # cumulative km
    soc_kwh = soc_pct / 100 * capacity
    remaining_kwh = [soc_kwh]
    for i in range(1, len(distances)):
        consumed = (distances[i] - distances[i-1]) / efficiency
        soc_kwh -= consumed
        remaining_kwh.append(max(soc_kwh, 0))
    df = pd.DataFrame({"Stop": route_segments, "SOC_kWh": remaining_kwh})
    return px.line(df, x="Stop", y="SOC_kWh", markers=True, title="Estimated State of Charge Along Journey")

@app.callback(
    Output('correlation-graph', 'figure'),
    Input('x-dropdown', 'value'),
    Input('y-dropdown', 'value'),
    Input('operational-filter', 'value'),
    Input('power-filter', 'value'),
    Input('fast-charge-filter', 'value')
)
def update_correlation(x_col, y_col, op_filter, min_kw, fast_filter):
    dff = gdf.copy()
    if op_filter:
        dff = dff[dff['IsOperational'].isin(op_filter)]
    if min_kw > 0:
        dff = dff[dff['PowerKW'] >= min_kw]
    if fast_filter:
        dff = dff[dff['Level.IsFastChargeCapable'] == 1]
    fig = px.scatter(dff, x=x_col, y=y_col, color='Town', hover_data=['Operator'], title=f"Correlation: {x_col} vs {y_col}")
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
