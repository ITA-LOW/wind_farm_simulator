import matplotlib.animation as animation
from matplotlib.patches import Polygon as mplPolygon, Circle
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import io
from PIL import Image

import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.aep import calcAEP
from core.cabling_v3 import analisar_layout_completo
from core.wfwe import WindFarm


# Publication style configuration (Modern and Clean)
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'image.cmap': 'viridis'
})

estilo = 'seaborn-v0_8-whitegrid'

def plot_solution_circle(x, y, radius, output_dir='.'):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, 'bo', markersize=6)
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--', linewidth=2)
    ax.add_artist(circle)
    ax.set_xlim(-radius-100, radius+100)
    ax.set_ylim(-radius-100, radius+100)
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('X Coordinate (m)', fontsize=28)
    ax.set_ylabel('Y Coordinate (m)', fontsize=28)
    ax.set_title('Optimized turbines', fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'wind_farm_solution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_logbook_to_csv(logbook, filename, output_dir='.'):
    filepath = os.path.join(output_dir, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Generation', 'MaxFitness'])
        for entry in logbook:
            writer.writerow([entry['gen'], entry['max']])

def plot_fitness(x, y, output_dir='.'):
    plt.style.use(estilo)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, label='Max Fitness', color='b', linewidth=1)
    ax.set_title('Max Fitness x Generations', fontsize=16, fontweight='bold')
    ax.set_xlabel('Generations', fontsize=14)
    ax.set_ylabel('Max Fitness', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join(output_dir, 'max_fitness_vs_generations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_benchmark_results(gen_data, fit_data, coords, radius, case_name, output_dir='.'):
    plt.style.use(estilo)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f'Optimization Results - {case_name}', fontsize=20)
    
    ax1.plot(gen_data, fit_data, 'b-', label='Max Fitness')
    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Max Fitness (AEP)', fontsize=12)
    ax1.set_title('Fitness Evolution', fontsize=14)
    ax1.grid(True, linestyle='--')
    ax1.legend()
    ax1.ticklabel_format(style='plain', axis='y')
    
    ax2.plot(coords[:, 0], coords[:, 1], 'bo', markersize=8)
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='--', linewidth=1.5)
    ax2.add_artist(circle)
    ax2.set_xlim(-radius - 100, radius + 100)
    ax2.set_ylim(-radius - 100, radius + 100)
    ax2.set_aspect('equal', adjustable='box')
    ax2.set_title(f'Final Layout', fontsize=14)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True, linestyle='--')
    
    filename = f'result_{case_name}.png'
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Results for {case_name} saved to '{filename}'")

def plot_pareto_front(study_or_data, case, output_dir):
    """Plot Pareto front matching simulate.py design."""
    import json
    json_path = os.path.join(output_dir, "pareto_solutions.json")
    if not os.path.exists(json_path):
        return

    with open(json_path, "r") as f:
        data = json.load(f)
        
    if not data:
        return

    aeps = np.array([d["net_aep_gwh"] for d in data])
    capexs = np.array([d["cable_capex_usd"] for d in data])
    
    knee_data = next((d for d in data if d.get("is_knee_point", False)), data[0])
    knee_aep = knee_data["net_aep_gwh"]
    knee_capex = knee_data["cable_capex_usd"]
    knee_grp = knee_data.get("cable_groups", "N/A")

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0a1628")
    ax.set_facecolor("#0f2038")
    
    sort_indices = np.argsort(capexs)
    ax.plot(capexs[sort_indices] / 1e3, aeps[sort_indices], color="#38bdf8", alpha=0.5, lw=2, linestyle="-")
    ax.scatter(capexs / 1e3, aeps, color="#38bdf8", edgecolors="#0f2038", s=60, zorder=3, label="Pareto Solutions")
    
    # Highlight Knee Point
    ax.scatter(knee_capex / 1e3, knee_aep, color="#f59e0b", edgecolors="white", marker="*", s=250, zorder=5, label="Knee Point (Ideal Compromise)")
    ax.axhline(knee_aep, color="#f59e0b", linestyle=":", alpha=0.6, lw=1.2)
    ax.axvline(knee_capex / 1e3, color="#f59e0b", linestyle=":", alpha=0.6, lw=1.2)
    
    # Knee Info Box
    info_text = (f"★ Knee Point (Best Compromise) ★\n"
                 f"Net AEP: {knee_aep:.3f} GWh\n"
                 f"Cable CAPEX: ${knee_capex:,.2f} USD\n"
                 f"Groups: {knee_grp}")
    ax.text(0.05, 0.05, info_text, transform=ax.transAxes, ha="left", va="bottom",
            color="white", fontsize=9, fontweight="bold",
            bbox=dict(facecolor="#0f2038", edgecolor="#f59e0b", boxstyle="round,pad=0.6", alpha=0.9))
    
    ax.set_title("Pareto Front — Net AEP vs Cable CAPEX", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Cable CAPEX [kUSD]", color="white", fontsize=10)
    ax.set_ylabel("Net AEP [GWh]", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color="#1e3050", lw=0.5, linestyle=":")
    
    leg = ax.legend(loc="upper right", facecolor="#0a1628", edgecolor="#1e3050")
    if leg:
        for text in leg.get_texts():
            text.set_color("white")
            
    for s in ax.spines.values():
        s.set_edgecolor("#1e3050")
        
    leg = ax.legend(facecolor="#0a1628", edgecolor="#1e3050", fontsize=10)
    for text in leg.get_texts():
        text.set_color("white")
        
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "pareto_front.png"), dpi=300, bbox_inches="tight", facecolor="#0a1628")
    plt.close()

def plot_cabling_layout(coords, sub_idx, planta, case, output_dir, title=""):
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Park boundary
    radius = 1300 if "16" in case else (2000 if "36" in case else 3000)
    circle = Circle((0, 0), radius, fill=False, linestyle='--', color='black', alpha=0.5)
    ax.add_patch(circle)
    
    # Plot paths
    cmap = plt.cm.tab10
    for i, path in enumerate(planta.paths):
        path_coords = coords[path]
        plt.plot(path_coords[:, 0], path_coords[:, 1], '-o', ms=8, lw=2, 
                 color=cmap(i % 10), label=f"String {i+1}")
        
    # Highlight substation
    plt.scatter(coords[sub_idx, 0], coords[sub_idx, 1], marker='*', s=400, 
                color='gold', edgecolors='black', label="Substation", zorder=10)
    
    plt.title(f"Cabling Layout - {case}\n{title}", fontsize=18, fontweight='bold')
    plt.xlabel("X (m)", fontsize=14)
    plt.ylabel("Y (m)", fontsize=14)
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend(ncol=2, loc='upper right')
    
    plt.tight_layout()
    result_path = os.path.join(output_dir, f"cabling_layout_{case}.png")
    plt.savefig(result_path, dpi=300)
    plt.close()
    return result_path

def plot_wake_fields(farm, title, filename, output_dir='.'):
    farm.plot_layout_with_wake_field(
        title=title,
        save_path=os.path.join(output_dir, filename)
    )

def generate_evolution_gifs(p1_frames, p2_frames, h_p1_aep, h_p2_net, h_p2_capex,
                            config, turb_atrbt_data, wind_rose_data, boundary, output_dir):
    print("\n" + "═"*50)
    print("  Rendering Animation GIFs")
    print("═"*50)
    
    n_turb = config.get("n_turbines", 16)

    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = turb_atrbt_data
    wind_dir, wind_freq, wind_speed = wind_rose_data

    # Determine dominant wind direction for accurate plotting
    dominant_dir_idx = np.argmax(wind_freq)
    dominant_wind_dir = float(wind_dir[dominant_dir_idx])

    xmin, ymin, xmax, ymax = boundary.bbox
    pad = max(xmax - xmin, ymax - ymin) * 0.12
    ax_xlim = (xmin - pad, xmax + pad)
    ax_ylim = (ymin - pad, ymax + pad)
    
    total_p1 = len(p1_frames)
    total_p2 = len(p2_frames)
    total_frames = total_p1 + total_p2
    
    skip_step = max(1, total_frames // 150)
    
    # 1. Physical Evolution GIF
    print("Rendering evolution_layout.gif...")
    frames_layout = []
    for idx in range(0, total_frames, skip_step):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(9.5, 7), facecolor="#0a1628")
        ax.set_facecolor("#0a1628")
        
        boundary_patch = boundary.to_patch()
        ax.add_patch(boundary_patch)

        ax.set_xlim(*ax_xlim)
        ax.set_ylim(*ax_ylim)
        ax.set_aspect("equal")
        ax.grid(True, color="#1e3050", lw=0.4)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlabel("X [m]", color="white", fontsize=9)
        ax.set_ylabel("Y [m]", color="white", fontsize=9)
        for s in ax.spines.values():
            s.set_edgecolor("#1e3050")
            
        if idx < total_p1:
            coords = p1_frames[idx]
            phase = 1
        else:
            coords, sub_pos, n_groups = p2_frames[idx - total_p1]
            phase = 2
            
        farm = WindFarm(coords,
                        wind_direction=dominant_wind_dir,
                        wind_speed_free_stream=float(wind_speed[0]),
                        turbine_diameter=turb_diam)
        x_lo, x_hi = ax_xlim
        y_lo, y_hi = ax_ylim
        X, Y, V = farm.get_velocity_field(resolution=100, x_bounds=(x_lo, x_hi), y_bounds=(y_lo, y_hi))
        ax.contourf(X, Y, V, levels=30, cmap="coolwarm_r", alpha=0.85, zorder=1)
        
        if phase == 2:
            combined_coords = np.vstack([coords, sub_pos.reshape((1, 2))])
            sub_index = n_turb
            try:
                planta, res = analisar_layout_completo(combined_coords, sub=sub_index, n_grupos=n_groups)
                capex = res["custo_total_usd"]
                losses = res["perda_anual_mwh"]
                
                cmap = plt.cm.tab10
                for string_idx, path in enumerate(planta.paths):
                    xs = [combined_coords[k, 0] for k in path]
                    ys = [combined_coords[k, 1] for k in path]
                    ax.plot(xs, ys, "-", color=cmap(string_idx % 10), lw=2.0, zorder=3)
            except Exception:
                capex = 1e12
                losses = 0.0
        
        ax.scatter(coords[:, 0], coords[:, 1], s=40, color="white", edgecolors="#0a1628", linewidths=0.8, zorder=5)
        
        if phase == 2:
            ax.scatter(sub_pos[0], sub_pos[1], s=180, color="#f97316", marker="s", edgecolors="white", label="Substation", zorder=10)
            
        if phase == 1:
            aep = np.sum(calcAEP(coords, wind_freq, wind_speed, wind_dir,
                                 turb_diam, turb_ci, turb_co, rated_ws, rated_pwr))
            ax.set_title("Phase 1 — Layout Optimization", color="white", fontsize=11, fontweight="bold")
            ax.text(1.05, 0.95, f"★ Phase 1 Stats ★\nGross AEP: {aep/1e3:.2f} GWh", transform=ax.transAxes, ha="left", va="top",
                    color="#22c55e", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
        else:
            gross_aep = np.sum(calcAEP(coords, wind_freq, wind_speed, wind_dir,
                                       turb_diam, turb_ci, turb_co, rated_ws, rated_pwr))
            net_aep = gross_aep - losses
            ax.set_title("Phase 2 — Co-design Optimization", color="white", fontsize=11, fontweight="bold")
            
            p1_final_aep = h_p1_aep[-1] if h_p1_aep else gross_aep
            ax.text(1.05, 0.95, f"★ Phase 1 Seed ★\nGross AEP: {p1_final_aep/1e3:.2f} GWh", transform=ax.transAxes, ha="left", va="top",
                    color="#22c55e", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
            
            ax.text(1.05, 0.72, f"★ Phase 2 Stats ★\nNet AEP: {net_aep/1e3:.2f} GWh\nCAPEX: ${capex/1e3:.1f}k\nGroups: {n_groups}", transform=ax.transAxes, ha="left", va="top",
                    color="#f59e0b", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
            
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a1628")
        plt.close(fig)
        buf.seek(0)
        frames_layout.append(Image.open(buf).copy())
        buf.close()
        
    gif_path_layout = os.path.join(output_dir, "evolution_layout.gif")
    frames_layout[0].save(gif_path_layout, save_all=True, append_images=frames_layout[1:], duration=150, loop=0, optimize=False, disposal=2)
    print(f"Saved: {gif_path_layout}")
    
    png_path_layout = os.path.join(output_dir, "knee_layout.png")
    frames_layout[-1].save(png_path_layout)
    print(f"Saved: {png_path_layout}")
    
    # 2. AEP & CAPEX Evolution Graph GIF
    print("Rendering aep_evolution.gif...")
    frames_aep = []
    x1 = list(range(1, len(h_p1_aep) + 1))
    y1 = [max(v/1e3, 0.0) for v in h_p1_aep]
    
    x2 = list(range(len(h_p1_aep), len(h_p1_aep) + len(h_p2_net) + 1))
    
    # Ensure the graph connects cleanly
    y2_start = y1[-1] if y1 else 0
    y2 = [y2_start] + [max(v/1e3, 0.0) for v in h_p2_net]
    
    all_y = y1 + y2
    valid_y = [y for y in all_y if y > 0]
    y_min_val = min(valid_y) if valid_y else 0
    y_max_val = max(valid_y) if valid_y else 1
    
    y_range = y_max_val - y_min_val if y_max_val != y_min_val else 1.0
    y_ylim = (y_min_val - 0.10 * y_range, y_max_val + 0.10 * y_range)
    
    if h_p2_capex is not None and len(h_p2_capex) > 0:
        y_cost = [c / 1e3 for c in h_p2_capex]
        y_cost = [y_cost[0]] + y_cost
        cost_min, cost_max = min(y_cost), max(y_cost)
        cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
        cost_ylim = (cost_min - 0.05 * cost_range, cost_max + 0.05 * cost_range)
    
    for idx in range(0, total_frames, skip_step):
        plt.close('all')
        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#0a1628")
        ax.set_facecolor("#0f2038")
        
        ax.set_xlim(0, total_frames + 5)
        ax.set_ylim(y_ylim)
        
        ax_cost = None
        if h_p2_capex is not None and len(h_p2_capex) > 0:
            ax_cost = ax.twinx()
            ax_cost.set_ylabel("Cabling CAPEX [kUSD]", color="#ef4444", fontsize=10)
            ax_cost.tick_params(colors="#ef4444", labelsize=8)
            ax_cost.set_ylim(cost_ylim)
            ax_cost.grid(False)
        
        if idx < total_p1:
            curr_x1 = x1[:idx+1]
            curr_y1 = y1[:idx+1]
            ax.plot(curr_x1, curr_y1, color="#3b82f6", lw=2.5, label="Phase 1 — Gross AEP")
        else:
            ax.plot(x1, y1, color="#3b82f6", lw=2.5, label="Phase 1 — Gross AEP")
            if len(h_p1_aep) > 0:
                ax.axvline(len(h_p1_aep), color="#f59e0b", lw=1.5, linestyle="--", alpha=0.8)
            
            idx_p2 = idx - total_p1
            curr_x2 = x2[:idx_p2 + 1]
            curr_y2 = y2[:idx_p2 + 1]
            ax.plot(curr_x2, curr_y2, color="#22c55e", lw=2.5, label="Phase 2 — Net AEP (with cable losses)")
            
            if h_p2_capex is not None and ax_cost is not None:
                curr_y_cost = y_cost[:idx_p2 + 1]
                ax_cost.plot(curr_x2, curr_y_cost, color="#ef4444", lw=2.0, linestyle="-", label="Phase 2 — Cabling CAPEX")
            
        ax.set_title("Two-Phase Optimization — AEP & CAPEX Evolution", color="white", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Generation", color="white", fontsize=10)
        ax.set_ylabel("AEP [GWh]", color="white", fontsize=10)
        ax.tick_params(colors="white", labelsize=8)
        ax.grid(True, color="#1e3050", lw=0.5, linestyle=":")
        
        lines, labels = ax.get_legend_handles_labels()
        if ax_cost is not None:
            lines2, labels2 = ax_cost.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        
        if lines:
            leg = ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, facecolor="#0a1628", edgecolor="#1e3050", fontsize=9)
            for text in leg.get_texts():
                text.set_color("white")
                
        for s in ax.spines.values():
            s.set_edgecolor("#1e3050")
            
        fig.subplots_adjust(bottom=0.28, left=0.1, right=0.9, top=0.9)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a1628")
        plt.close(fig)
        buf.seek(0)
        frames_aep.append(Image.open(buf).copy())
        buf.close()
        
    gif_path_aep = os.path.join(output_dir, "aep_evolution.gif")
    frames_aep[0].save(gif_path_aep, save_all=True, append_images=frames_aep[1:], duration=150, loop=0, optimize=True)
    print(f"Saved: {gif_path_aep}")
    
    png_path_aep = os.path.join(output_dir, "aep_evolution.png")
    frames_aep[-1].save(png_path_aep)
    print(f"Saved: {png_path_aep}")


def plot_wind_rose(wind_dir, wind_freq, wind_speed, save_path):
    """
    Generate a polar plot of the wind rose and save it.
    Matches the dark aesthetic of the simulation outputs.
    """
    # Create figure with dark background
    fig = plt.figure(figsize=(8, 8), facecolor='#0f172a')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('#1e293b')
    
    # Set polar parameters for meteorological convention
    # North at top (0 deg), angles increase clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Data conversion
    theta = np.radians(wind_dir)
    width = np.radians(360.0 / len(wind_dir))
    
    # Plot bars (frequency dictates length)
    bars = ax.bar(theta, wind_freq, width=width, bottom=0.0, 
                  color='#3b82f6', edgecolor='#60a5fa', alpha=0.8, zorder=3)
                  
    # Styling grids and ticks
    ax.tick_params(colors='#94a3b8')
    ax.grid(color='#334155', linestyle='--', linewidth=0.5, zorder=0)
    ax.spines['polar'].set_color('#334155')
    
    # Labels
    ax.set_title("Site-Specific Wind Rose (ERA5 API)", color='white', pad=20, fontsize=14, fontweight='bold')
    
    # Custom radial ticks (percentage)
    max_freq = np.max(wind_freq)
    if max_freq == 0:
        max_freq = 1.0 # fallback
    rticks = np.linspace(0, max_freq, 5)
    ax.set_rticks(rticks)
    ax.set_yticklabels([f"{(t*100):.1f}%" for t in rticks], color='#94a3b8', fontsize=9)
    
    # Compass labels
    ax.set_xticks(np.radians([0, 45, 90, 135, 180, 225, 270, 315]))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    plt.close()
    print(f"Wind rose plot saved to {save_path}")