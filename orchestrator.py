#!/usr/bin/env python3
"""
orchestrator.py — Main Entry Point for Wind Farm Optimization
===========================================================
A clean and modular orchestrator that manages the two-phase optimization:
1. Phase 1: Layout Optimization (AEP maximization) via DEAP eaSimple.
2. Phase 2: Genome Expansion & Co-design (Layout + Cables) via DEAP NSGA-II.
3. Plotting: Renders metrics and visualizes the evolution as GIFs.
"""

import os
import sys
import time
import subprocess
import yaml
import json
import argparse
import random
import numpy as np

# Reproducibility
random.seed(42)
np.random.seed(42)

# Set up paths
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.phase_1 import run_phase_1
from core.phase_2 import run_phase_2
from core.plot import generate_evolution_gifs, plot_pareto_front
from core.boundary import SiteBoundary
from core.aep import getTurbAtrbtYAML, getWindRoseYAML

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def banner(text: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)

def save_logs(output_dir, h_p1_aep, h_p2_net, h_p2_capex, hof_p2, config):
    # 1. Save convergence history
    history = {
        "phase1_gross_aep_gwh": [v / 1e3 for v in h_p1_aep],
        "phase2_net_aep_gwh": [v / 1e3 for v in h_p2_net],
        "phase2_capex_kusd": [v / 1e3 for v in h_p2_capex] if h_p2_capex else []
    }
    with open(os.path.join(output_dir, "convergence_history.json"), "w") as f:
        json.dump(history, f, indent=4)
        
    # 2. Save Pareto solutions
    if not hof_p2:
        return
        
    valid_hof = [ind for ind in hof_p2 if ind.fitness.values[0] > 0]
    valid_hof.sort(key=lambda ind: ind.fitness.values[0], reverse=True) # Sort by AEP descending
    
    n_turb = config.get("n_turbines", 16)
    n_coords = n_turb * 2
    min_g = config.get("cable_groups", {}).get("min_groups", 2)
    max_g = config.get("cable_groups", {}).get("max_groups", 16)
    sub_mode = "optimize"
    if isinstance(config.get("substation"), dict):
        sub_mode = config["substation"].get("mode", "optimize")
        
    # Find knee point
    knee_idx = -1
    if len(valid_hof) > 1:
        gen_aeps = np.array([ind.fitness.values[0] for ind in valid_hof])
        gen_capexs = np.array([ind.fitness.values[1] for ind in valid_hof])
        aep_norm = (gen_aeps - np.min(gen_aeps)) / (np.max(gen_aeps) - np.min(gen_aeps) + 1e-6)
        capex_norm = (np.max(gen_capexs) - gen_capexs) / (np.max(gen_capexs) - np.min(gen_capexs) + 1e-6)
        dists = np.sqrt((1.0 - aep_norm)**2 + (1.0 - capex_norm)**2)
        knee_idx = np.argmin(dists)
    elif len(valid_hof) == 1:
        knee_idx = 0
        
    pareto_data = []
    for rank, ind in enumerate(valid_hof):
        g_norm = ind[n_coords]
        groups = int(np.round(min_g + g_norm * (max_g - min_g)))
        groups = max(min_g, min(n_turb, groups))
        
        if sub_mode == "optimize":
            sub_pos = [ind[n_coords+1], ind[n_coords+2]]
        else:
            fixed = config.get("substation", {}).get("fixed_pos", [-1350.0, 0.0])
            sub_pos = [fixed[0], fixed[1]]
            
        coords = np.array(ind[:n_coords]).reshape((n_turb, 2)).tolist()
        
        pareto_data.append({
            "rank_by_aep": int(rank + 1),
            "net_aep_gwh": float(ind.fitness.values[0] / 1e3),
            "cable_capex_usd": float(ind.fitness.values[1]),
            "cable_groups": int(groups),
            "substation_position": [float(sub_pos[0]), float(sub_pos[1])],
            "turbine_coordinates": [[float(c[0]), float(c[1])] for c in coords],
            "is_knee_point": bool(rank == knee_idx)
        })
        
    with open(os.path.join(output_dir, "pareto_solutions.json"), "w") as f:
        json.dump(pareto_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instantiate custom wind farm simulation with cabling co-design.")
    parser.add_argument("--case", type=str, default="cases/case_example.yaml", help="Path to YAML configuration case file")
    parser.add_argument("--output", type=str, default="results/user_run", help="Output directory to save results")
    parser.add_argument("--max-gens", type=int, default=1000, help="Maximum generations per phase (default: 1000)")
    args = parser.parse_args()

    t0 = time.time()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load Configuration
    config = load_config(args.case)
    if config is None:
        config = {}
    
    banner(f"Starting Orchestrator: {config.get('name', 'Wind Farm Simulation')}")
    print(f"Case Config : {args.case}")
    print(f"Output Dir  : {args.output}")
    print(f"Max Gens    : {args.max_gens} per phase")
    
    # ── 0. Pre-load Global Data (Wind & Turbines) ───────────────────────────────
    turb_yaml = os.path.join(ROOT, config.get("turbine_yaml", "config/iea37-335mw.yaml"))
    geojson_path = os.path.join(ROOT, config["boundary_geojson"])
    
    turb_atrbt_data = getTurbAtrbtYAML(turb_yaml)
    boundary = SiteBoundary.from_geojson(geojson_path)
    
    # Smart Substation Auto-detect
    if boundary.substation_pos is not None:
        config["substation"] = {
            "mode": "fixed",
            "fixed_pos": boundary.substation_pos
        }
    else:
        if "substation" not in config or config.get("substation") in ["optimize", "from_geojson"]:
            config["substation"] = "optimize"
    
    wind_config = config.get("windrose_yaml")
    if not wind_config:
        print("\n[ERROR] Wind configuration not found! Please define 'windrose_yaml' in your case configuration file (e.g., 'auto' or a path to a YAML file).\n")
        sys.exit(1)
        
    if wind_config == "auto":
        from core.wind_rose import get_automatic_wind_rose
        from core.plot import plot_wind_rose
        from shapely.geometry import shape
        from core.boundary import _extract_geometries
        
        banner("Generating Auto-Wind Rose (ERA5 API)")
        # Get lat/lon from raw GeoJSON (WGS84)
        with open(geojson_path, "r") as f:
            gj = json.load(f)
        geoms = _extract_geometries(gj)
        # Use largest polygon for centroid
        shapes = [shape(g) for g in geoms if g is not None]
        polys = [s for s in shapes if s.geom_type in ["Polygon", "MultiPolygon"]]
        polys.sort(key=lambda p: p.area, reverse=True)
        lon, lat = polys[0].centroid.x, polys[0].centroid.y
        
        # Get hub height
        with open(turb_yaml, "r") as f:
            tdata = yaml.safe_load(f)
        try:
            hub_height = float(tdata['definitions']['rotor']['properties']['hub_height']['default'])
        except KeyError:
            hub_height = 110.0
            
        turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
        wind_dir, wind_freq, wind_speed, diagnostic = get_automatic_wind_rose(
            lat, lon, hub_height, turb_ci, turb_co, rated_ws
        )
        wind_rose_data = (wind_dir, wind_freq, wind_speed)
        
        # Save diagnostic json
        with open(os.path.join(args.output, "wind_diagnostic.json"), "w") as f:
            json.dump(diagnostic, f, indent=4)
        
        # Save wind data as NPZ for the interactive editor to load instantly
        np.savez(os.path.join(args.output, "wind_data.npz"), wind_dir=wind_dir, wind_freq=wind_freq, wind_speed=wind_speed)

        # Save plot
        plot_path = os.path.join(args.output, "auto_wind_rose.png")
        plot_wind_rose(wind_dir, wind_freq, wind_speed, plot_path)
    else:
        from core.plot import plot_wind_rose
        wind_yaml = os.path.join(ROOT, wind_config)
        wind_rose_data = getWindRoseYAML(wind_yaml)
        wind_dir, wind_freq, wind_speed = wind_rose_data
        
        # Save wind data as NPZ for the interactive editor to load instantly
        np.savez(os.path.join(args.output, "wind_data.npz"), wind_dir=wind_dir, wind_freq=wind_freq, wind_speed=wind_speed)

        # Save plot
        plot_path = os.path.join(args.output, "auto_wind_rose.png")
        plot_wind_rose(wind_dir, wind_freq, wind_speed, plot_path)
    
    # ── 1. Run Phase 1 ──────────────────────────────────────────────────────────
    banner("Executing PHASE 1: Layout Optimization")
    best_p1_coords, h_p1_aep, pop_p1, log_p1, p1_frames = run_phase_1(config, max_gens=args.max_gens, wind_rose_data=wind_rose_data)
    
    # ── 2. Run Phase 2 ──────────────────────────────────────────────────────────
    banner("Executing PHASE 2: Co-design & Cabling")
    hof_p2, p2_frames, h_p2_net, h_p2_capex = run_phase_2(config, best_p1_coords, pop_p1, max_gens=args.max_gens, wind_rose_data=wind_rose_data)
    
    # ── 3. Render Outputs ───────────────────────────────────────────────────────
    banner("Rendering Outputs and GIFs")


    generate_evolution_gifs(
        p1_frames=p1_frames,
        p2_frames=p2_frames,
        h_p1_aep=h_p1_aep,
        h_p2_net=h_p2_net,
        h_p2_capex=h_p2_capex,
        config=config,
        turb_atrbt_data=turb_atrbt_data,
        wind_rose_data=wind_rose_data,
        boundary=boundary,
        output_dir=args.output
    )
    
    save_logs(args.output, h_p1_aep, h_p2_net, h_p2_capex, hof_p2, config)
    
    # 4. Save Pareto Front Plot
    pareto_data_plot = [{"aep": ind.fitness.values[0], "cost": ind.fitness.values[1]} for ind in hof_p2 if ind.fitness.values[0] > 0]
    if pareto_data_plot:
        plot_pareto_front(pareto_data_plot, config.get("name", "Simulation"), args.output)

    execution_time = (time.time() - t0) / 60.0
    
    # Launch Interactive Dashboard
    subprocess.run(["bokeh", "serve", "--show", "core/dashboard.py", "--args", args.output, args.case])
        
    print("\n" + "═"*70)
    print(f"  Optimization Finished in {execution_time:.2f} minutes!")
    print(f"Check the {args.output} folder for the generated GIFs and logs.")
