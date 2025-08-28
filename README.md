# EV Dashboards South Wales and West Midlands (UK)

A local Dash web app for exploring EV charging infrastructure and coastal spatial data in South Wales.

## 🚀 How to Run

1. Ensure Python 3.9+ is installed.
2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Open Data Map app (South Wales):
```bash
python evapp.py
```
3. Run the ONS (oct-2024) app (South Wales):
```bash
python ons_evapp.py
```
3. Run the ONS (oct-2024) app (West Midlands):
```bash
python ons_evapp_west_midlands.py
```

Then open your browser to: [http://127.0.0.1:8050](http://127.0.0.1:8050)

## 📁 Contents

- `app.py`: Main Dash app
- `ev_charger_map.html`: Pre-rendered interactive Folium map
- `requirements.txt`: Python dependencies
