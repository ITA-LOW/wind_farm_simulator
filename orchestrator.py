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
from config.iea37_aepcalc import getTurbAtrbtYAML, getWindRoseYAML

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
    
    # ── 1. Run Phase 1 ──────────────────────────────────────────────────────────
    banner("Executing PHASE 1: Layout Optimization")
    best_p1_coords, h_p1_aep, pop_p1, log_p1, p1_frames = run_phase_1(config, max_gens=args.max_gens)
    
    # ── 2. Run Phase 2 ──────────────────────────────────────────────────────────
    banner("Executing PHASE 2: Co-design & Cabling")
    hof_p2, p2_frames, h_p2_net, h_p2_capex = run_phase_2(config, best_p1_coords, pop_p1, max_gens=args.max_gens)
    
    # ── 3. Render Outputs ───────────────────────────────────────────────────────
    banner("Rendering Outputs and GIFs")
    
    # Load required data for plotting
    turb_yaml = os.path.join(os.path.dirname(ROOT), config.get("turbine_yaml", "config/iea37-335mw.yaml"))
    wind_yaml = os.path.join(os.path.dirname(ROOT), config.get("windrose_yaml", "config/iea37-windrose.yaml"))
    
    turb_atrbt_data = getTurbAtrbtYAML(turb_yaml)
    wind_rose_data = getWindRoseYAML(wind_yaml)
    
    generate_evolution_gifs(
        p1_frames=p1_frames,
        p2_frames=p2_frames,
        h_p1_aep=h_p1_aep,
        h_p2_net=h_p2_net,
        h_p2_capex=h_p2_capex,
        config=config,
        turb_atrbt_data=turb_atrbt_data,
        wind_rose_data=wind_rose_data,
        output_dir=args.output
    )
    
    save_logs(args.output, h_p1_aep, h_p2_net, h_p2_capex, hof_p2, config)
    
    # 4. Save Pareto Front Plot
    pareto_data_plot = [{"aep": ind.fitness.values[0], "cost": ind.fitness.values[1]} for ind in hof_p2 if ind.fitness.values[0] > 0]
    if pareto_data_plot:
        plot_pareto_front(pareto_data_plot, config.get("name", "Simulation"), args.output)

    t1 = time.time()
    banner(f"Optimization Finished in {(t1 - t0) / 60.0:.2f} minutes!")
    print(f"Check the {args.output} folder for the generated GIFs and logs.")
