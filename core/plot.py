import matplotlib.animation as animation
from matplotlib.patches import Polygon as mplPolygon, Circle
import csv
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd


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
    """Plot Pareto front from an Optuna study or a list of dicts with 'aep'/'cost' keys."""
    try:
        trials = study_or_data.best_trials
        if len(trials) == 0:
            print(f"No best trials found for study {study_or_data.study_name}")
            return
        aeps  = [t.values[0] / 1e3 for t in trials]
        costs = [t.values[1] / 1e3 for t in trials]
        knee_label = lambda idx: f'Knee Point (Trial {trials[idx].number})'
    except AttributeError:
        data  = study_or_data
        aeps  = [d["aep"] / 1e3 for d in data]
        costs = [d["cost"] / 1e3 for d in data]
        knee_label = lambda idx: 'Knee Point'

    plt.figure(figsize=(10, 7))
    plt.scatter(costs, aeps, color='navy', marker='o', alpha=0.7, label='Pareto Optimal')

    if len(aeps) > 1:
        c_arr  = np.array(costs)
        a_arr  = np.array(aeps)
        c_norm = (c_arr - c_arr.min()) / (c_arr.max() - c_arr.min() + 1e-6)
        a_norm = (a_arr - a_arr.min()) / (a_arr.max() - a_arr.min() + 1e-6)
        dists  = np.sqrt(c_norm**2 + (1 - a_norm)**2)
        ki     = np.argmin(dists)
        plt.scatter(costs[ki], aeps[ki], color='gold', marker='*', s=300,
                    edgecolors='black', label=knee_label(ki), zorder=5)

    plt.title(f"Pareto Front - {case}\n(AEP vs Cabling Cost)", fontsize=16, fontweight='bold')
    plt.xlabel("Total Cabling Cost (kUSD)", fontsize=12)
    plt.ylabel("Annual Energy Production (GWh)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"pareto_front_{case}.png"), dpi=300)
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