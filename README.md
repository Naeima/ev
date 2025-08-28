# EV Dashboards South Wales and West Midlands (UK)

A local Dash web app for exploring EV charging infrastructure and coastal spatial data in South Wales and another one for West Midlands.

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

docs: 
If you encounter an error when running the app locally, replace 
`app.run_server(debug=True)` with `app.run(debug=True)`.  
`app.run_server()` is Dash-specific, while `app.run()` is Flask’s method.  
Some setups require the Flask call for compatibility.

## 📁 Contents

### Files Included  

- **`evapp.py`** – South Wales Open Map Data application  
- **`ons_evapp.py`** – Office for National Statistics (ONS, Oct 2024) Dash application for South Wales  
- **`ons_evapp_west_midlands.py`** – Office for National Statistics (ONS, Oct 2024) Dash application for the West Midlands  
- **`requirements.txt`** – Python dependencies  

