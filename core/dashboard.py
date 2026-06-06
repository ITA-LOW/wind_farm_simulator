import os
import sys
import yaml
import json
import pyproj
import numpy as np

from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, PointDrawTool, HoverTool, Div, TapTool
from bokeh.plotting import figure
import xyzservices.providers as xyz
from shapely.geometry import shape

# Configure path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.boundary import SiteBoundary
from core.aep import calcAEP, getWindRoseYAML, getTurbAtrbtYAML
from core.cabling_v3 import analisar_layout_completo

# ---------------------------------------------------------
# 1. INITIAL SETUP AND LOADING
# ---------------------------------------------------------

output_dir = sys.argv[1]
config_path = sys.argv[2]

with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Load Turbine
turb_yaml = os.path.join(ROOT, config.get("turbine_yaml", "config/turbines/iea37-335mw.yaml"))
turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(turb_yaml)

# Load Geographical Boundaries
geojson_path = os.path.join(ROOT, config["boundary_geojson"])
boundary = SiteBoundary.from_geojson(geojson_path)
lon0, lat0 = boundary.lon0, boundary.lat0
proj_local = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
transformer = pyproj.Transformer.from_crs(proj_local, "EPSG:3857", always_xy=True)
inv_transformer = pyproj.Transformer.from_crs("EPSG:3857", proj_local, always_xy=True)

# Process Boundaries for Bokeh
site_mp_xs, site_mp_ys = [], []
polys = boundary._poly.geoms if boundary._poly.geom_type == "MultiPolygon" else [boundary._poly]
for poly in polys:
    ext_x, ext_y = poly.exterior.coords.xy
    ext_x, ext_y = transformer.transform(ext_x, ext_y)
    poly_xs = [list(ext_x)]
    poly_ys = [list(ext_y)]
    for interior in poly.interiors:
        in_x, in_y = interior.coords.xy
        in_x, in_y = transformer.transform(in_x, in_y)
        poly_xs.append(list(in_x))
        poly_ys.append(list(in_y))
    site_mp_xs.append(poly_xs)
    site_mp_ys.append(poly_ys)

bound_xs, bound_ys = [site_mp_xs], [site_mp_ys]

# Load Wind Data
wind_npz = os.path.join(output_dir, "wind_data.npz")
if os.path.exists(wind_npz):
    print("Loading wind data locally from NPZ (skipping API fetch)...")
    npz_data = np.load(wind_npz)
    wind_dir = npz_data['wind_dir']
    wind_freq = npz_data['wind_freq']
    wind_speed = npz_data['wind_speed']
else:
    wind_config = config.get("windrose_yaml", "config/windrose/iea37-windrose.yaml")
    if wind_config == "auto":
        from core.wind_rose import get_automatic_wind_rose
        try:
            with open(turb_yaml, "r") as fy:
                tdata = yaml.safe_load(fy)
            hub_height = float(tdata['definitions']['rotor']['properties']['hub_height']['default'])
        except:
            hub_height = 110.0
            
        print("Fetching ERA5 wind data...")
        wind_dir, wind_freq, wind_speed, _ = get_automatic_wind_rose(
            lat0, lon0, hub_height, turb_ci, turb_co, rated_ws
        )
    else:
        wind_yaml = os.path.join(ROOT, wind_config)
        wind_dir, wind_freq, wind_speed = getWindRoseYAML(wind_yaml)

# Load All Pareto Layouts
pareto_file = os.path.join(output_dir, "pareto_solutions.json")

if not os.path.exists(pareto_file):
    print(f"Could not find {pareto_file}")
    sys.exit(1)

with open(pareto_file, "r") as f:
    pareto_data = json.load(f)

# Extract data from the entire Pareto front
aeps = []
capexs = []
ranks = []
turb_coords_list = []
sub_pos_list = []
groups_list = []

knee_idx = 0
for idx, sol in enumerate(pareto_data):
    if sol.get("is_knee_point", False):
        knee_idx = idx
    aeps.append(sol["net_aep_gwh"])
    capexs.append(sol["cable_capex_usd"] / 1000.0) # kUSD
    ranks.append(sol["rank_by_aep"])
    turb_coords_list.append(np.array(sol["turbine_coordinates"]))
    sub_pos_list.append(np.array(sol["substation_position"]))
    groups_list.append(sol["cable_groups"])

# ---------------------------------------------------------
# 2. BOKEH SOURCES AND PHYSICS FUNCTIONS
# ---------------------------------------------------------
source_pareto = ColumnDataSource(data=dict(
    x=capexs,
    y=aeps,
    rank=ranks,
    idx=list(range(len(pareto_data)))
))

turb_source = ColumnDataSource(data=dict(x=[], y=[]))
sub_source = ColumnDataSource(data=dict(x=[], y=[]))
cable_source = ColumnDataSource(data=dict(x=[], y=[], length=[], cost=[]))

# HTML Information Panel
info_div = Div(text="", sizing_mode="stretch_width")

# Global to hold the number of cable groups for the clicked solution
current_n_groups = groups_list[knee_idx]

def load_layout(idx):
    """Loads the coordinates of the Pareto point onto the map."""
    global current_n_groups
    coords = turb_coords_list[idx]
    sub = sub_pos_list[idx]
    current_n_groups = groups_list[idx]
    
    wm_x, wm_y = transformer.transform(coords[:, 0], coords[:, 1])
    sub_wm_x, sub_wm_y = transformer.transform([sub[0]], [sub[1]])
    
    # Update sub_source FIRST, as turb_source.data triggers on_change instantaneously
    sub_source.data = dict(x=list(sub_wm_x), y=list(sub_wm_y))
    turb_source.data = dict(x=list(wm_x), y=list(wm_y))
    
    recalculate_physics()

def recalculate_physics():
    """Reads the turbine points on the screen, recalculates AEP and Cabling, and updates the lines."""
    screen_x = turb_source.data['x']
    screen_y = turb_source.data['y']
    
    sub_wm_x = sub_source.data['x']
    sub_wm_y = sub_source.data['y']
    
    if len(screen_x) == 0 or len(sub_wm_x) == 0:
        return
        
    # 2. Convert back to Local Meters
    loc_x, loc_y = inv_transformer.transform(screen_x, screen_y)
    current_turb_coords = np.column_stack((loc_x, loc_y))
    sub_loc_x, sub_loc_y = inv_transformer.transform(sub_wm_x, sub_wm_y)
    sub_pos_local = np.array([sub_loc_x[0], sub_loc_y[0]])
    
    # 3. Calculate Physical AEP (Wake Effect)
    gross_aep = np.sum(calcAEP(current_turb_coords, wind_freq, wind_speed, wind_dir,
                               turb_diam, turb_ci, turb_co, rated_ws, rated_pwr))
    
    # 4. Calculate Smart Cabling (SAP)
    combined_coords = np.vstack([current_turb_coords, sub_pos_local.reshape((1, 2))])
    sub_index = len(current_turb_coords)
    
    try:
        planta, res = analisar_layout_completo(combined_coords, sub=sub_index, n_grupos=current_n_groups)
        paths = planta.paths
        capex = res["custo_total_usd"]
        losses = res["perda_anual_mwh"]
        secao = res["secao_cabo_mm2"]
    except Exception as e:
        print("Cabling error:", e)
        planta = None
        paths = []
        capex = 0
        losses = 0
        secao = 0

    net_aep = gross_aep - losses
    
    # 5. Prepare new cable data for the screen (Web Mercator)
    sol_lines_x = []
    sol_lines_y = []
    sol_len = []
    sol_cost = []
    
    if planta:
        for p_idx, path in enumerate(paths):
            cable_path = planta.Cb[p_idx]
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                cable = cable_path[i]
                
                # Coordinates of Point A
                px_a = sub_wm_x[0] if a == sub_index else screen_x[a]
                py_a = sub_wm_y[0] if a == sub_index else screen_y[a]
                    
                # Coordinates of Point B
                px_b = sub_wm_x[0] if b == sub_index else screen_x[b]
                py_b = sub_wm_y[0] if b == sub_index else screen_y[b]
                    
                sol_lines_x.append([px_a, px_b])
                sol_lines_y.append([py_a, py_b])
                sol_len.append(cable.lc)
                sol_cost.append(cable.Ctot)
                
    cable_source.data = dict(x=sol_lines_x, y=sol_lines_y, length=sol_len, cost=sol_cost)
    
    # 6. Update top HTML panel
    info_div.text = f"""
    <div style="padding: 12px 20px; background-color: #f8fafc; border-left: 5px solid #10b981; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); font-family: sans-serif;">
        <h4 style="margin: 0 0 10px 0; color: #0f172a; font-size: 14px; text-transform: uppercase;">Pareto Front Solutions</h4>
        <div style="display: flex; justify-content: flex-start; gap: 40px; align-items: center; font-size: 16px;">
            <div><strong>Net AEP:</strong> <span style="color: #22c55e;">{net_aep/1e3:.3f} GWh</span></div>
            <div><strong>Cabling CAPEX:</strong> <span style="color: #ef4444;">${capex/1e3:,.2f} kUSD</span></div>
            <div><strong>Cross-Section:</strong> {secao} mm²</div>
            <div><strong>Number of Strings:</strong> {current_n_groups}</div>
        </div>
    </div>
    """

# Connect the "Drag" turbines event
def on_turb_change(attr, old, new):
    recalculate_physics()
turb_source.on_change('data', on_turb_change)

# Connect the click on the Pareto Chart
def on_pareto_select(attr, old, new):
    if new:
        idx = new[0] # Get the clicked index
        load_layout(idx)
source_pareto.selected.on_change('indices', on_pareto_select)

# ---------------------------------------------------------
# 3. BOKEH INTERFACE (PARETO CHART + MAP)
# ---------------------------------------------------------

# --- Pareto Chart ---
p_pareto = figure(
    title="Pareto Front (Click a point to load layout)",
    x_axis_label="Cabling CAPEX (kUSD)",
    y_axis_label="Net AEP (GWh)",
    sizing_mode="stretch_both",
    tools="pan,wheel_zoom,box_zoom,reset,tap"
)
pareto_scatter = p_pareto.scatter(
    'x', 'y', source=source_pareto, size=10,
    color="#0284c7", alpha=0.7,
    nonselection_alpha=0.2, selection_color="#ef4444"
)

# Highlight Knee Point
p_pareto.scatter(
    x=[capexs[knee_idx]], y=[aeps[knee_idx]],
    size=20, marker="star", color="gold", line_color="#d97706", line_width=1.5,
    legend_label="Knee Point"
)

p_pareto.add_tools(HoverTool(renderers=[pareto_scatter], tooltips=[
    ("Rank", "@rank"), ("AEP", "@y{0.00} GWh"), ("CAPEX", "$@x{0,0} kUSD")
]))

p_pareto.legend.location = "bottom_right"

# --- Mapa Interativo ---
p_map = figure(
    sizing_mode="stretch_both",
    x_axis_type="mercator", y_axis_type="mercator",
    title="Drag and Drop Physics Editor",
    tools="pan,wheel_zoom,save,reset"
)
p_map.add_tile(xyz.CartoDB.Positron)
p_map.axis.visible = False

if bound_xs and bound_ys:
    p_map.multi_polygons(xs=bound_xs, ys=bound_ys, fill_color="#38bdf8", fill_alpha=0.08, line_color="#0284c7", line_width=1.5, line_dash="dashed")

cable_renderer = p_map.multi_line('x', 'y', line_color="#475569", line_width=2, line_alpha=0.8, source=cable_source)
p_map.add_tools(HoverTool(renderers=[cable_renderer], tooltips=[
    ("Length", "@length{0,0} m"),
    ("Segment Cost", "$@cost{0,0} USD")
]))

p_map.scatter('x', 'y', size=15, marker="square", color="#f97316", line_color="#c2410c", source=sub_source, legend_label="Substation")

turb_renderer = p_map.scatter('x', 'y', size=10, marker="circle", color="#1e293b", line_color="white", source=turb_source, legend_label="Wind Turbine")
draw_tool = PointDrawTool(renderers=[turb_renderer], add=False)
p_map.add_tools(draw_tool)
p_map.toolbar.active_drag = draw_tool

# Start the dashboard loading the Knee Point
source_pareto.selected.indices = [knee_idx]
load_layout(knee_idx)

# Finalize Layout
layout = column(info_div, row(p_pareto, p_map, sizing_mode="stretch_both"), sizing_mode="stretch_both")
curdoc().add_root(layout)
curdoc().title = "Wind Farm Simulator"
