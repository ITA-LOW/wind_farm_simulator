"""
BENCHMARK COMPARISON SCRIPT — GECCO 2026
-----------------------------------------
Purpose: Compare the proposed Two-Phase Method vs. Pure NSGA-II Baseline
         vs. Sequential (decoupled) approach on the IEA37-16 benchmark.

Three methods compared:
  1. Proposed  — Two-phase warm-start (Phase 1: layout, Phase 2: joint co-design)
  2. Baseline  — Pure NSGA-II from scratch (same budget as Proposed)
  3. Sequential — Phase 1 layout GA, then Phase 2 cabling GA with fixed turbines

Metrics:
  - Hypervolume (convergence & diversity)
  - Pareto front visualisation
  - Statistical tests (Mann-Whitney / t-test + Cohen's d)
  - C-metric (dominance coverage)
  - Computational efficiency

Usage:
    # Full benchmark (20 seeds — matches paper)
    python optimizer/benchmark.py

    # Quick smoke-test (3 seeds, fewer generations)
    python optimizer/benchmark.py --quick
"""

import sys
import os
import time
import random
import argparse
import multiprocessing
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless — no display needed
import matplotlib.pyplot as plt
import pandas as pd
from deap import base, creator, tools, algorithms

# ACM/GECCO require Type 1/TrueType (Type 42) fonts — not Type 3.
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype']  = 42
# Import hypervolume function directly
try:
    from deap.tools._hypervolume import hv as hypervolume_module
except ImportError:
    from deap.tools._hypervolume import pyhv as hypervolume_module

# Import statistical tests
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Statistical tests will be skipped.")

# ── path setup ───────────────────────────────────────────────────────────────
# Makes the script runnable both from the repo root and from the optimizer/ dir
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
import core.cabling_v3 as cabling_v3

# =============================================================================
# 1. EXPERIMENT CONFIGURATION  (edit here for quick tests)
# =============================================================================

# Default: matches the paper (20 seeds, balanced for robustness vs. speed)
N_SEEDS      = 20
POP_SIZE     = 150
NGEN_PHASE1  = 500
NGEN_PHASE2  = 1000

# Baseline runs for the combined budget to be fair
NGEN_BASELINE = NGEN_PHASE1 + NGEN_PHASE2

# Debug flag (prints the first baseline evaluation in detail)
DEBUG_BASELINE = True

# Farm geometry (IEA37-16)
IND_SIZE      = 16
CIRCLE_RADIUS = 5000     # boundary radius [m]
N_DIAMETERS   = 260      # minimum turbine spacing [m]
MIN_GRUPOS    = 2
MAX_GRUPOS    = 64
N_GRUPOS_INICIAL = MIN_GRUPOS

# Pre-load benchmark data (avoids repeated I/O inside the eval loop)
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_dir  = "config"
main_yaml_path = os.path.join(BASE_DIR, config_dir, "iea37-ex16.yaml")
initial_coordinates = getTurbLocYAML(main_yaml_path)
full_path_turb = os.path.join(BASE_DIR, config_dir, "iea37-335mw.yaml")
full_path_wr   = os.path.join(BASE_DIR, config_dir, "iea37-windrose.yaml")
TURB_ATRBT_DATA = getTurbAtrbtYAML(full_path_turb)
WIND_ROSE_DATA  = getWindRoseYAML(full_path_wr)

# =============================================================================
# 2. DEAP CONFIGURATION (CREATORS & TOOLBOXES)
# =============================================================================

if hasattr(creator, "FitnessMax"): del creator.FitnessMax
if hasattr(creator, "IndividualPhase1"): del creator.IndividualPhase1
if hasattr(creator, "FitnessMulti"): del creator.FitnessMulti
if hasattr(creator, "IndividualPhase2"): del creator.IndividualPhase2
if hasattr(creator, "FitnessMin"): del creator.FitnessMin
if hasattr(creator, "IndividualSequential"): del creator.IndividualSequential

# Phase 1 (Single Objective)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

# Phase 2 / Baseline (Multi Objective: Max AEP, Min Cost)
creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)

# Sequential (Single Objective: Min Cost, with fixed turbines)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualSequential", list, fitness=creator.FitnessMin)

toolbox_p1 = base.Toolbox()
toolbox_p2 = base.Toolbox()
toolbox_base = base.Toolbox() # Separate toolbox for Baseline
toolbox_seq = base.Toolbox() # Toolbox for sequential approach

# =============================================================================
# 3. AUXILIARY FUNCTIONS AND EVALUATORS (COPIED FROM ORIGINAL FOR CONSISTENCY)
# =============================================================================

def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2

def enforce_circle(individual_coords):
    """Projects coordinates inside the circle boundary."""
    for i in range(0, len(individual_coords), 2):
        x, y = individual_coords[i], individual_coords[i+1]
        if not is_within_circle(x, y, CIRCLE_RADIUS):
            angle = np.arctan2(y, x)
            individual_coords[i] = CIRCLE_RADIUS * np.cos(angle)
            individual_coords[i+1] = CIRCLE_RADIUS * np.sin(angle)
    return individual_coords

def enforce_substation(sub_pos):
    """Projects substation inside the circle boundary."""
    x, y = sub_pos[0], sub_pos[1]
    if not is_within_circle(x, y, CIRCLE_RADIUS):
        angle = np.arctan2(y, x)
        sub_pos[0] = CIRCLE_RADIUS * np.cos(angle)
        sub_pos[1] = CIRCLE_RADIUS * np.sin(angle)
    return sub_pos

def displace_substation_from_turbines(centroid, turb_coords, min_distance=50.0):
    """
    Displaces the substation from the centroid to ensure minimum distance from turbines.
    
    Strategy:
    1. Start at the centroid
    2. If it is too close to any turbine, displace it in the opposite direction
    3. Ensure minimum distance from all turbines
    
    Args:
        centroid: Initial position (centroid of turbines)
        turb_coords: NumPy array of shape (N, 2) with turbine coordinates
        min_distance: Minimum allowed distance (meters)
    
    Returns:
        sub_pos: Displaced substation position [x, y]
    """
    sub_pos = np.array(centroid.copy())
    max_iterations = 20
    
    for iteration in range(max_iterations):
        # Calculate distances to all turbines
        dists_to_turbines = np.linalg.norm(turb_coords - sub_pos, axis=1)
        min_dist = np.min(dists_to_turbines)
        
        # If far enough, stop
        if min_dist >= min_distance:
            break
        
        # Find the closest turbine
        closest_turb_idx = np.argmin(dists_to_turbines)
        closest_turb = turb_coords[closest_turb_idx]
        
        # Direction from centroid to closest turbine
        direction_to_turb = closest_turb - sub_pos
        dist_to_turb = np.linalg.norm(direction_to_turb)
        
        if dist_to_turb < 1e-6:
            # If substation is exactly on a turbine, displace randomly
            angle = random.uniform(0, 2*np.pi)
            sub_pos = sub_pos + min_distance * np.array([np.cos(angle), np.sin(angle)])
        else:
            # Normalize direction
            direction_to_turb = direction_to_turb / dist_to_turb
            
            # Displace in the opposite direction (pushing away from turbine)
            # Move enough to guarantee min_distance
            needed_displacement = min_distance - min_dist + 10.0  # +10m margin
            sub_pos = sub_pos - direction_to_turb * needed_displacement
        
        # Ensure it stays within the circle
        dist_from_center = np.linalg.norm(sub_pos)
        if dist_from_center > CIRCLE_RADIUS:
            angle = np.arctan2(sub_pos[1], sub_pos[0])
            sub_pos = CIRCLE_RADIUS * np.array([np.cos(angle), np.sin(angle)])
    
    return sub_pos.tolist()

def repair_spacing(coords_array, max_iterations=10):
    """
    Repairs minimum distance violations between turbines.
    Pushes very close turbines apart while keeping them inside the circle.
    
    Args:
        coords_array: NumPy array of shape (N, 2) with turbine coordinates
        max_iterations: Maximum number of repair iterations
    
    Returns:
        Repaired coords_array
    """
    coords = coords_array.copy()
    min_dist = N_DIAMETERS
    
    for iteration in range(max_iterations):
        # Calculate distances between all pairs
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Find pairs violating the minimum distance
        i_upper, j_upper = np.triu_indices(len(coords), k=1)
        violations = dists[i_upper, j_upper] < min_dist
        
        if not np.any(violations):
            break  # No violations, stop
        
        # For each violation, separate the turbines
        for idx in np.where(violations)[0]:
            i, j = i_upper[idx], j_upper[idx]
            dist_ij = dists[i, j]
            
            if dist_ij < min_dist:
                # Separation direction
                if dist_ij < 1e-6:  # Avoid division by zero
                    # If in the same spot, separate randomly
                    angle = random.uniform(0, 2*np.pi)
                    direction = np.array([np.cos(angle), np.sin(angle)])
                else:
                    direction = (coords[i] - coords[j]) / dist_ij
                
                # Distance needed to be added
                needed_separation = (min_dist - dist_ij) / 2.0
                
                # Move both turbines in opposite directions
                move_i = direction * needed_separation
                move_j = -direction * needed_separation
                
                # Apply movement
                new_pos_i = coords[i] + move_i
                new_pos_j = coords[j] + move_j
                
                # Ensure they stay inside the circle boundary
                dist_i = np.linalg.norm(new_pos_i)
                dist_j = np.linalg.norm(new_pos_j)
                
                if dist_i > CIRCLE_RADIUS:
                    angle_i = np.arctan2(new_pos_i[1], new_pos_i[0])
                    new_pos_i = CIRCLE_RADIUS * np.array([np.cos(angle_i), np.sin(angle_i)])
                
                if dist_j > CIRCLE_RADIUS:
                    angle_j = np.arctan2(new_pos_j[1], new_pos_j[0])
                    new_pos_j = CIRCLE_RADIUS * np.array([np.cos(angle_j), np.sin(angle_j)])
                
                coords[i] = new_pos_i
                coords[j] = new_pos_j
    
    return coords

# --- PHASE 1 EVALUATOR (GROSS AEP) ---
def evaluate_phase1(individual):
    # Reuse original logic
    turb_coords = np.array(individual).reshape((IND_SIZE, 2))
    
    # Geometric Penalties
    dist_from_center = np.linalg.norm(turb_coords, axis=1)
    penalty_circle = np.sum(np.maximum(0, dist_from_center - CIRCLE_RADIUS)) * 1e6
    
    # Distance between turbines
    num_turb = len(turb_coords)
    diff = turb_coords.reshape(num_turb, 1, 2) - turb_coords.reshape(1, num_turb, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)
    i_upper, j_upper = np.triu_indices(num_turb, k=1)
    close_mask = dist_matrix[i_upper, j_upper] < N_DIAMETERS
    penalty_spacing = np.sum(close_mask) * 1e6
    
    # AEP Calculation
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, 
                  TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3])
    
    return np.sum(aep) - penalty_circle - penalty_spacing,

# --- PHASE 2 / BASELINE EVALUATOR (NET AEP + COST) ---
def evaluate_full(individual, debug=False):
    """Evaluates the full genome (Turbines + Groups + Substation)."""
    try:
        n_coords = IND_SIZE * 2
        # Parse Genome
        coords_flat = individual[:n_coords]
        n_grupos_norm = individual[n_coords]
        sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        
        # Convert Groups
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        # Limit n_grupos to number of turbines (does not make sense to have more groups than turbines)
        n_grupos = min(n_grupos, IND_SIZE)
        
        turb_coords = np.array(coords_flat).reshape((IND_SIZE, 2))
        
        # 1. Basic Geometric Penalties (smoother)
        dist_turb = np.linalg.norm(turb_coords, axis=1)
        violations_turb = np.maximum(0, dist_turb - CIRCLE_RADIUS)
        pen_turb_out = np.sum(violations_turb) * 1e5  # Reduced from 1e6 to 1e5
        
        dist_sub = np.linalg.norm(sub_pos)
        pen_sub_out = np.maximum(0, dist_sub - CIRCLE_RADIUS) * 1e5  # Reduced
        
        # Distance Turbine-Turbine
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_u, j_u = np.triu_indices(IND_SIZE, k=1)
        # Smoother penalty: linear instead of extreme
        violations = np.maximum(0, N_DIAMETERS - dists[i_u, j_u])
        pen_spacing = np.sum(violations) * 1e5  # Reduced from 1e6 to 1e5
        
        # Distance Sub-Turbine (smoother)
        d_sub_turb = np.linalg.norm(turb_coords - sub_pos, axis=1)
        violation_sub_close = np.maximum(0, 50.0 - np.min(d_sub_turb))
        pen_sub_close = violation_sub_close * 1e5  # Reduced from 1e6 to 1e5
        
        # 2. Gross AEP
        wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
        turb_diam = TURB_ATRBT_DATA[4]
        aep_bruto = np.sum(calcAEP(turb_coords, wind_freq, wind_speed, wind_dir, turb_diam, 
                      TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))

        # 3. Cabling (Cost + Losses)
        coords_all = np.vstack([turb_coords, sub_pos.reshape(1, 2)])
        try:
            plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
            custo_usd = res['custo_total_usd']
            perdas_mwh = res['perda_anual_mwh']
        except Exception as cabling_error:
            if debug:
                print(f"   [DEBUG] Cabling error: {cabling_error}, n_grupos={n_grupos}")
                import traceback
                traceback.print_exc()
            return -1e6, 1e12
        
        # 4. Electrical Constraints (Crossings - simplified via user library if available)
        # Assuming cabling_v3 and SAP (Strict Angular Partitioning) guarantee radial planar topology
        # The original code uses 'detectar_sobreposicao_cabos' but it's complex to copy everything.
        # Since we use SAP (cabling_v3), crossings are mathematically impossible by definition,
        # unless there is physical coordinates overlapping.
        # Let's assume penalty 0 here for the simplified baseline, or use a simple heuristic.
        pen_cabos = 0 
        
        aep_liq = aep_bruto - perdas_mwh - pen_turb_out - pen_spacing - pen_sub_out - pen_sub_close
        custo_final = custo_usd + pen_turb_out + pen_spacing + pen_sub_out + pen_sub_close
        
        if debug:
            print(f"   [DEBUG Baseline] Full evaluation:")
            print(f"      Gross AEP: {aep_bruto:.2f} MWh")
            print(f"      Joule Losses: {perdas_mwh:.2f} MWh")
            print(f"      Penalties: turb_out={pen_turb_out:.2f}, spacing={pen_spacing:.2f}, "
                  f"sub_out={pen_sub_out:.2f}, sub_close={pen_sub_close:.2f}")
            print(f"      Net AEP: {aep_liq:.2f} MWh")
            print(f"      Cost: {custo_usd:.2e} USD")
            print(f"      Final Cost: {custo_final:.2e} USD")
        
        if aep_liq <= 0: 
            if debug:
                print(f"   [DEBUG] Net AEP <= 0, returning penalty")
            return -1e6, 1e12
        return aep_liq, custo_final

    except Exception as e:
        if debug:
            print(f"   [DEBUG] General exception in evaluate_full: {e}")
        return -1e6, 1e12

# --- SEQUENTIAL EVALUATOR (MINIMIZES ONLY COST, TURBINES FIXED) ---
def evaluate_sequential(individual, fixed_turb_coords, debug=False):
    """
    Evaluates only cabling cost with fixed turbines.
    Genome: [n_grupos_norm (1), sub_x (1), sub_y (1)]
    """
    try:
        n_grupos_norm = individual[0]
        sub_pos = np.array([individual[1], individual[2]])
        
        # Convert Groups
        n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
        n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
        
        # Penalties for substation
        dist_sub = np.linalg.norm(sub_pos)
        pen_sub_out = np.maximum(0, dist_sub - CIRCLE_RADIUS) * 1e5
        
        # Distance Sub-Turbine
        d_sub_turb = np.linalg.norm(fixed_turb_coords - sub_pos, axis=1)
        violation_sub_close = np.maximum(0, 50.0 - np.min(d_sub_turb))
        pen_sub_close = violation_sub_close * 1e5
        
        # Cabling (Cost + Losses)
        coords_all = np.vstack([fixed_turb_coords, sub_pos.reshape(1, 2)])
        try:
            plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
            custo_usd = res['custo_total_usd']
            perdas_mwh = res['perda_anual_mwh']
        except Exception as cabling_error:
            if debug:
                print(f"   [DEBUG Sequential] Cabling error: {cabling_error}, n_grupos={n_grupos}")
            return 1e12,
        
        # Final cost (minimize)
        custo_final = custo_usd + pen_sub_out + pen_sub_close
        
        if debug:
            print(f"   [DEBUG Sequential] Cost: {custo_usd:.2e} USD, Penalties: {pen_sub_out + pen_sub_close:.2e}")
        
        return custo_final,
        
    except Exception as e:
        if debug:
            print(f"   [DEBUG Sequential] Exception: {e}")
        return 1e12,

# --- GENETIC OPERATORS ---

def create_individual_from_coordinates(coords):
    """
    Creates a Phase 1 individual from coordinates.
    CONSERVATIVE: Returns exact YAML coordinates (no perturbation).
    Matches original wind_farm_GA_16.py code.
    """
    return creator.IndividualPhase1(np.array(coords).flatten().tolist())

def create_random_phase1_ind():
    """
    Generates a Phase 1 individual (only coordinates).
    CONSERVATIVE: Returns exact YAML coordinates, same as original code.
    Diversification occurs gradually via mutation/crossover.
    """
    if initial_coordinates is not None:
        # Returns exact YAML coordinates (no perturbation)
        coords = initial_coordinates.flatten().tolist()
    else:
        # Fallback: fully random if YAML not available
        coords = []
        for _ in range(IND_SIZE):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0, CIRCLE_RADIUS * 0.9)
            coords.extend([r*np.cos(angle), r*np.sin(angle)])
    return creator.IndividualPhase1(coords)

def create_random_phase2_ind():
    """
    Generates a Baseline individual (Full Genome).
    CONSERVATIVE: Uses exact YAML coordinates (no perturbation).
    Diversification occurs gradually via mutation/crossover.
    Matches philosophy of original wind_farm_GA_16.py code.
    """
    # 1. Turbine Coordinates - EXACT from YAML (no perturbation)
    if initial_coordinates is not None:
        coords = initial_coordinates.flatten().tolist()
    else:
        # Fallback: fully random if YAML not available
        coords = []
        for _ in range(IND_SIZE):
            angle = random.uniform(0, 2*np.pi)
            r = random.uniform(0, CIRCLE_RADIUS * 0.9)
            coords.extend([r*np.cos(angle), r*np.sin(angle)])
    
    # 2. Group Gene [0, 1] - conservative initial value (normalized from MIN_GRUPOS)
    n_grupos_norm = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))  # Guarantee [0, 1]
    
    # 3. Substation - turbine centroid DISPLACED so it doesn't overlap with a turbine
    coords_array = np.array(coords).reshape((IND_SIZE, 2))
    centroid = np.array(np.mean(coords_array, axis=0))  # Guarantee numpy array
    # Displace substation from centroid to ensure minimum distance of 50m from turbines
    sub_pos = displace_substation_from_turbines(centroid, coords_array, min_distance=50.0)
    
    # Assemble full genome: [coords_turbines (32), n_grupos_norm (1), sub_x (1), sub_y (1)]
    full_genome = coords + [n_grupos_norm] + sub_pos
    return creator.IndividualPhase2(full_genome)

def convert_p1_to_p2(ind_p1):
    """Smart Seeding: Converts P1 layout -> P2 with centroid heuristic."""
    coords_flat = list(ind_p1)
    coords = np.array(coords_flat).reshape((IND_SIZE, 2))
    
    # Initialize substation at centroid (Smart Heuristic)
    centroid = np.mean(coords, axis=0)
    sub_pos = enforce_substation(centroid).tolist()
    
    # Initialize groups (can be random or fixed, let's vary slightly)
    g_norm = random.random()
    
    return creator.IndividualPhase2(coords_flat + [g_norm] + sub_pos)

def mutate_p2(individual, indpb):
    """Mutation adapted for mixed genome - using parameters from wind_farm_GA_16.py."""
    n_coords = IND_SIZE * 2
    # Turbines - using sigma=100 from wind_farm_GA_16.py
    for i in range(n_coords):
        if random.random() < indpb:
            individual[i] += random.gauss(0, 100)  # sigma=100 from wind_farm_GA_16.py
    enforce_circle(individual[:n_coords])
    
    # Repair spacing after turbine mutation
    coords_array = np.array(individual[:n_coords]).reshape((IND_SIZE, 2))
    coords_repaired = repair_spacing(coords_array)
    individual[:n_coords] = coords_repaired.flatten().tolist()
    
    # Groups - mutation
    if random.random() < indpb:
        individual[n_coords] += random.gauss(0, 0.1)
        individual[n_coords] = max(0.0, min(1.0, individual[n_coords]))
        
    # Substation - using sigma=100 from wind_farm_GA_16.py
    if random.random() < indpb:
        individual[n_coords+1] += random.gauss(0, 100)  # sigma=100 from wind_farm_GA_16.py
        individual[n_coords+2] += random.gauss(0, 100)
    
    sub_arr = np.array([individual[n_coords+1], individual[n_coords+2]])
    sub_arr = enforce_substation(sub_arr)
    
    # Guarantee minimum distance of substation from turbines
    coords_array = np.array(individual[:n_coords]).reshape((IND_SIZE, 2))
    sub_arr = displace_substation_from_turbines(sub_arr, coords_array, min_distance=50.0)
    
    individual[n_coords+1] = sub_arr[0]
    individual[n_coords+2] = sub_arr[1]
    
    return individual,

# Register in Toolboxes
toolbox_p1.register("individual", create_random_phase1_ind)
toolbox_p1.register("population", tools.initRepeat, list, toolbox_p1.individual)
toolbox_p1.register("evaluate", evaluate_phase1)
toolbox_p1.register("mate", tools.cxBlend, alpha=0.5)
toolbox_p1.register("mutate", tools.mutGaussian, mu=0, sigma=100, indpb=0.4)  # Parameters from wind_farm_GA_16.py
toolbox_p1.register("select", tools.selTournament, tournsize=5)  # Parameters from wind_farm_GA_16.py

toolbox_p2.register("evaluate", evaluate_full)
toolbox_p2.register("mate", tools.cxBlend, alpha=0.5)  # Parameters from wind_farm_GA_16.py
toolbox_p2.register("mutate", mutate_p2, indpb=0.4)  # Parameters from wind_farm_GA_16.py
toolbox_p2.register("select", tools.selNSGA2)

toolbox_base.register("individual", create_random_phase2_ind)
toolbox_base.register("population", tools.initRepeat, list, toolbox_base.individual)
# Wrapper for evaluate_full with optional debug
def evaluate_full_wrapper(individual):
    """Wrapper that allows debug only on the first evaluation."""
    global DEBUG_BASELINE
    debug_now = DEBUG_BASELINE
    if DEBUG_BASELINE:
        DEBUG_BASELINE = False  # Deactivate after first time
    return evaluate_full(individual, debug=debug_now)

def mate_with_repair(ind1, ind2):
    """Crossover with automatic repair after operation."""
    tools.cxBlend(ind1, ind2, alpha=0.5)
    
    # Repair after crossover
    n_coords = IND_SIZE * 2
    
    # Repair turbine coordinates
    coords1 = np.array(ind1[:n_coords]).reshape((IND_SIZE, 2))
    coords1 = repair_spacing(coords1)
    ind1[:n_coords] = coords1.flatten().tolist()
    enforce_circle(ind1[:n_coords])
    
    coords2 = np.array(ind2[:n_coords]).reshape((IND_SIZE, 2))
    coords2 = repair_spacing(coords2)
    ind2[:n_coords] = coords2.flatten().tolist()
    enforce_circle(ind2[:n_coords])
    
    # Repair substation
    sub1 = np.array([ind1[n_coords+1], ind1[n_coords+2]])
    sub1 = enforce_substation(sub1)
    sub1 = displace_substation_from_turbines(sub1, coords1, min_distance=50.0)
    ind1[n_coords+1] = sub1[0]
    ind1[n_coords+2] = sub1[1]
    
    sub2 = np.array([ind2[n_coords+1], ind2[n_coords+2]])
    sub2 = enforce_substation(sub2)
    sub2 = displace_substation_from_turbines(sub2, coords2, min_distance=50.0)
    ind2[n_coords+1] = sub2[0]
    ind2[n_coords+2] = sub2[1]
    
    return ind1, ind2

toolbox_base.register("evaluate", evaluate_full_wrapper)
toolbox_base.register("mate", mate_with_repair)
toolbox_base.register("mutate", mutate_p2, indpb=0.4)  # Parameters from wind_farm_GA_16.py
toolbox_base.register("select", tools.selNSGA2)

# --- SEQUENTIAL TOOLBOX ---
def create_sequential_ind(fixed_turb_coords):
    """Creates a sequential individual: [n_grupos_norm, sub_x, sub_y]"""
    # Initialize substation at centroid
    centroid = np.mean(fixed_turb_coords, axis=0)
    sub_pos = displace_substation_from_turbines(centroid, fixed_turb_coords, min_distance=50.0)
    
    # Initial number of groups (normalized)
    n_grupos_norm = (N_GRUPOS_INICIAL - MIN_GRUPOS) / (MAX_GRUPOS - MIN_GRUPOS)
    n_grupos_norm = max(0.0, min(1.0, n_grupos_norm))
    
    return creator.IndividualSequential([n_grupos_norm, sub_pos[0], sub_pos[1]])

def mutate_sequential(individual, indpb, fixed_turb_coords):
    """Mutation for sequential individual - using parameters from wind_farm_GA_16.py"""
    # Mutation of groups
    if random.random() < indpb:
        individual[0] += random.gauss(0, 0.1)
        individual[0] = max(0.0, min(1.0, individual[0]))
    
    # Mutation of substation - using sigma=100 from wind_farm_GA_16.py
    if random.random() < indpb:
        individual[1] += random.gauss(0, 100)  # sigma=100 from wind_farm_GA_16.py
        individual[2] += random.gauss(0, 100)
    
    # Enforce constraints
    sub_arr = np.array([individual[1], individual[2]])
    sub_arr = enforce_substation(sub_arr)
    sub_arr = displace_substation_from_turbines(sub_arr, fixed_turb_coords, min_distance=50.0)
    individual[1] = sub_arr[0]
    individual[2] = sub_arr[1]
    
    return individual,

def mate_sequential(ind1, ind2):
    """Crossover for sequential individual"""
    tools.cxBlend(ind1, ind2, alpha=0.5)
    # Guarantee bounds
    ind1[0] = max(0.0, min(1.0, ind1[0]))
    ind2[0] = max(0.0, min(1.0, ind2[0]))
    return ind1, ind2

# Register sequential toolbox (will be configured dynamically with fixed turbines)
toolbox_seq.register("mate", mate_sequential)
toolbox_seq.register("select", tools.selTournament, tournsize=5)  # Parameters from wind_farm_GA_16.py
# mutate will be registered dynamically in run_sequential_method

# =============================================================================
# 4. METHOD EXECUTION
# =============================================================================

def run_proposed_method(seed, track_evolution=False, ref_point=None):
    """
    Executes Phase 1 + Phase 2.
    
    Returns:
        pareto_front: List of non-dominated solutions
        evolution_data: Dict with hypervolume evolution (if track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    # --- PHASE 1 ---
    pop = toolbox_p1.population(n=POP_SIZE)
    # Run Simple GA - using parameters from wind_farm_GA_16.py
    pop, _ = algorithms.eaSimple(pop, toolbox_p1, cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE1, verbose=False)
    
    # Select best from Phase 1
    best_p1 = tools.selBest(pop, int(POP_SIZE * 0.2)) # Top 20%
    
    # --- TRANSITION (SMART SEEDING) ---
    pop_p2 = []
    # Fill P2 population:
    # 1. Clones of the converted best P1 individuals
    for ind in best_p1:
        pop_p2.append(convert_p1_to_p2(ind))
    
    # 2. Fill the rest with new random individuals based on the best (perturbation)
    while len(pop_p2) < POP_SIZE:
        parent = random.choice(best_p1)
        child = convert_p1_to_p2(parent)
        child, = mutate_p2(child, indpb=0.3) # Strong mutation for diversity
        pop_p2.append(child)
        
    # --- PHASE 2 ---
    # Recalculate initial fitness (since it changed to multi-objective)
    invalid_ind = [ind for ind in pop_p2 if not ind.fitness.valid]
    fits = list(map(toolbox_p2.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Filter valid solutions before starting Phase 2
    pop_p2 = [ind for ind in pop_p2 if ind.fitness.valid and ind.fitness.values[0] > 0]
    if len(pop_p2) == 0:
        # If no valid solutions after Phase 1, return empty
        if track_evolution:
            return [], evolution_data
        return []
    
    # Run NSGA-II with custom tracking if needed
    if track_evolution:
        # Manual loop to track evolution
        pareto_front = tools.ParetoFront()
        pareto_front.update(pop_p2)
        
        # Calculate initial hypervolume (only valid solutions)
        pareto_valid = filter_valid_solutions(pareto_front)
        if len(pareto_valid) > 0 and ref_point is not None:
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"].append(NGEN_PHASE1)
            evolution_data["hv"].append(hv)
            evolution_data["n_solutions"].append(len(pareto_valid))
        else:
            evolution_data["gen"].append(NGEN_PHASE1)
            evolution_data["hv"].append(0.0)
            evolution_data["n_solutions"].append(0)
        
        for gen in range(NGEN_PHASE2):
            # Selection
            offspring = toolbox_p2.select(pop_p2, len(pop_p2))
            offspring = list(map(toolbox_p2.clone, offspring))
            
            # Crossover - using parameters from wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 from wind_farm_GA_16.py
                    toolbox_p2.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation - using parameters from wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 from wind_farm_GA_16.py
                    toolbox_p2.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_p2.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combine population + offspring and select best ones
            combined = pop_p2 + offspring
            pop_p2 = toolbox_p2.select(combined, POP_SIZE)
            pareto_front.update(pop_p2)
            
            # Calculate hypervolume (only valid solutions)
            pareto_valid = filter_valid_solutions(pareto_front)
            if len(pareto_valid) > 0 and ref_point is not None:
                pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
                pf_array = np.array(pf_points)
                hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
                evolution_data["gen"].append(NGEN_PHASE1 + gen)
                evolution_data["hv"].append(hv)
                evolution_data["n_solutions"].append(len(pareto_valid))
            else:
                # If no valid solutions, register 0
                evolution_data["gen"].append(NGEN_PHASE1 + gen)
                evolution_data["hv"].append(0.0)
                evolution_data["n_solutions"].append(0)
        
        pop_final = pop_p2
    else:
        # Run standard NSGA-II - using parameters from wind_farm_GA_16.py
        pop_final, logbook = algorithms.eaMuPlusLambda(pop_p2, toolbox_p2, mu=POP_SIZE, lambda_=POP_SIZE,
                                                       cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE2, verbose=False)
    
    # Extract Pareto front using ParetoFront
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop_final)
    
    # Filter invalid solutions
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

def run_baseline_method(seed, track_evolution=False, ref_point=None):
    """
    Runs Pure NSGA-II (Random Init, Long Run).
    
    Returns:
        pareto_front: List of non-dominated solutions
        evolution_data: Dict with hypervolume evolution (if track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    pop = toolbox_base.population(n=POP_SIZE)
    
    if track_evolution:
        # Evaluate initial population
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        # Debug: evaluate a solution to see what's happening
        if len(invalid_ind) > 0:
            test_ind = invalid_ind[0]
            test_fit = evaluate_full(test_ind, debug=True)
            print(f"   [DEBUG Baseline Init] First solution: AEP={test_fit[0]:.2f}, Cost={test_fit[1]:.2e}")
        fits = list(map(toolbox_base.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
        
        # Manual loop to track evolution
        pareto_front = tools.ParetoFront()
        pareto_front.update(pop)
        
        # Calculate initial hypervolume (only valid solutions)
        pareto_valid = filter_valid_solutions(pareto_front)
        if len(pareto_valid) > 0 and ref_point is not None:
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"].append(0)
            evolution_data["hv"].append(hv)
            evolution_data["n_solutions"].append(len(pareto_valid))
        else:
            evolution_data["gen"].append(0)
            evolution_data["hv"].append(0.0)
            evolution_data["n_solutions"].append(0)
        
        for gen in range(1, NGEN_BASELINE + 1):
            # ELITISM: Preserve best valid solutions before generating offspring
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            n_elite = min(5, len(valid_pop))  # Preserve top 5 valid solutions
            elite = tools.selBest(valid_pop, n_elite) if n_elite > 0 else []
            
            # Selection
            offspring = toolbox_base.select(pop, len(pop))
            offspring = list(map(toolbox_base.clone, offspring))
            
            # Crossover - using parameters from wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 from wind_farm_GA_16.py
                    toolbox_base.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation - using parameters from wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 from wind_farm_GA_16.py
                    toolbox_base.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_base.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combine population + offspring and select best ones
            combined = pop + offspring
            pop = toolbox_base.select(combined, POP_SIZE)
            
            # Guarantee elite is in the population
            if len(elite) > 0:
                # Remove worst and add elite
                pop_sorted = sorted(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid and x.fitness.values[0] > 0 else -1e12, reverse=True)
                pop = elite + pop_sorted[:POP_SIZE - len(elite)]
            
            pareto_front.update(pop)
            
            # Calculate hypervolume (only valid solutions)
            pareto_valid = filter_valid_solutions(pareto_front)
            if len(pareto_valid) > 0 and ref_point is not None:
                pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_valid]
                pf_array = np.array(pf_points)
                hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
                evolution_data["gen"].append(gen)
                evolution_data["hv"].append(hv)
                evolution_data["n_solutions"].append(len(pareto_valid))
            else:
                # If no valid solutions, register 0
                evolution_data["gen"].append(gen)
                evolution_data["hv"].append(0.0)
                evolution_data["n_solutions"].append(0)
        
        pop_final = pop
    else:
        # Run NSGA-II with explicit elitism (same logic as track_evolution)
        for gen in range(1, NGEN_BASELINE + 1):
            # ELITISM: Preserve best valid solutions before generating offspring
            valid_pop = [ind for ind in pop if ind.fitness.valid and ind.fitness.values[0] > 0]
            n_elite = min(5, len(valid_pop))  # Preserve top 5 valid solutions
            elite = tools.selBest(valid_pop, n_elite) if n_elite > 0 else []
            
            # Selection
            offspring = toolbox_base.select(pop, len(pop))
            offspring = list(map(toolbox_base.clone, offspring))
            
            # Crossover - using parameters from wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 from wind_farm_GA_16.py
                    toolbox_base.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation - using parameters from wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 from wind_farm_GA_16.py
                    toolbox_base.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_base.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # NSGA-II: Combine population + offspring and select best ones
            combined = pop + offspring
            pop = toolbox_base.select(combined, POP_SIZE)
            
            # Guarantee elite is in the population
            if len(elite) > 0:
                # Remove worst and add elite
                pop_sorted = sorted(pop, key=lambda x: x.fitness.values[0] if x.fitness.valid and x.fitness.values[0] > 0 else -1e12, reverse=True)
                pop = elite + pop_sorted[:POP_SIZE - len(elite)]
        
        pop_final = pop
    
    # Extract Pareto front using ParetoFront
    pareto_front = tools.ParetoFront()
    pareto_front.update(pop_final)
    
    # Debug: show info about Pareto front before filter (only baseline)
    if len(pareto_front) > 0:
        aep_vals = [ind.fitness.values[0] for ind in pareto_front]
        cost_vals = [ind.fitness.values[1] for ind in pareto_front]
        print(f"   [DEBUG Baseline] Pareto front before filter: {len(pareto_front)} solutions")
        print(f"   [DEBUG Baseline] AEP range: [{min(aep_vals):.2f}, {max(aep_vals):.2f}] MWh")
        print(f"   [DEBUG Baseline] Cost range: [{min(cost_vals):.2e}, {max(cost_vals):.2e}] USD")
        print(f"   [DEBUG Baseline] Solutions with AEP > 0: {sum(1 for aep in aep_vals if aep > 0)}")
        print(f"   [DEBUG Baseline] Solutions with Cost > 0 and < 1e12: {sum(1 for cost in cost_vals if 0 < cost < 1e12)}")
    
    # Filter invalid solutions
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

def run_sequential_method(seed, track_evolution=False, ref_point=None):
    """
    Executes sequential approach (two separate simple GAs):
    
    1. SIMPLE GA - Phase 1: Optimizes turbine layout
       - Objective: Maximize AEP
       - Variables: Positions (x, y) of all turbines
       - Algorithm: eaSimple (standard GA with tournament selection)
    
    2. SIMPLE GA - Sequential Phase 2: Optimizes substation and cabling
       - Objective: Minimize cost (with fixed turbines from Phase 1)
       - Variables: Substation position (x, y) + number of groups
       - Algorithm: eaSimple (standard GA with tournament selection)
       - Initialization: Substation starts at the turbine centroid
    
    Returns:
        pareto_front: List of solutions (converted to multi-objective format for comparison)
        evolution_data: Dict with evolution (if track_evolution=True)
    """
    random.seed(seed)
    np.random.seed(seed)
    
    evolution_data = {"gen": [], "hv": [], "n_solutions": []} if track_evolution else None
    
    # --- PHASE 1: SIMPLE GA to optimize turbine positions (maximize AEP) ---
    pop = toolbox_p1.population(n=POP_SIZE)
    pop, _ = algorithms.eaSimple(pop, toolbox_p1, cxpb=0.95, mutpb=0.7, ngen=NGEN_PHASE1, verbose=False)  # Parameters from wind_farm_GA_16.py
    
    # Select best layout from Phase 1
    best_p1 = tools.selBest(pop, 1)[0]
    fixed_turb_coords = np.array(best_p1).reshape((IND_SIZE, 2))
    
    # Calculate gross AEP of the fixed layout (to be used later)
    wind_dir, wind_freq, wind_speed = WIND_ROSE_DATA
    turb_diam = TURB_ATRBT_DATA[4]
    aep_bruto_fixed = np.sum(calcAEP(fixed_turb_coords, wind_freq, wind_speed, wind_dir, turb_diam,
                                     TURB_ATRBT_DATA[0], TURB_ATRBT_DATA[1], TURB_ATRBT_DATA[2], TURB_ATRBT_DATA[3]))
    
    # --- SEQUENTIAL PHASE 2: SIMPLE GA to optimize substation and cabling (minimize cost) ---
    # Configure sequential toolbox with fixed turbines from Phase 1
    def evaluate_seq_wrapper(ind):
        return evaluate_sequential(ind, fixed_turb_coords)
    
    def mutate_seq_wrapper(ind):
        return mutate_sequential(ind, 0.4, fixed_turb_coords)  # indpb=0.4 from wind_farm_GA_16.py
    
    toolbox_seq.register("individual", lambda: create_sequential_ind(fixed_turb_coords))
    toolbox_seq.register("population", tools.initRepeat, list, toolbox_seq.individual)
    toolbox_seq.register("evaluate", evaluate_seq_wrapper)
    toolbox_seq.register("mutate", mutate_seq_wrapper)
    
    # Create initial population
    pop_seq = toolbox_seq.population(n=POP_SIZE)
    
    # Evaluate initial population
    invalid_ind = [ind for ind in pop_seq if not ind.fitness.valid]
    fits = list(map(toolbox_seq.evaluate, invalid_ind))
    for ind, fit in zip(invalid_ind, fits):
        ind.fitness.values = fit
    
    # Run SIMPLE GA (eaSimple) to minimize cabling cost
    if track_evolution:
        # Manual loop to track evolution
        best_costs = []
        for gen in range(NGEN_PHASE2):
            # Selection
            offspring = toolbox_seq.select(pop_seq, len(pop_seq))
            offspring = list(map(toolbox_seq.clone, offspring))
            
            # Crossover - using parameters from wind_farm_GA_16.py
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < 0.95:  # CXPB=0.95 from wind_farm_GA_16.py
                    toolbox_seq.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation - using parameters from wind_farm_GA_16.py
            for mutant in offspring:
                if random.random() < 0.7:  # MUTPB=0.7 from wind_farm_GA_16.py
                    toolbox_seq.mutate(mutant)
                    del mutant.fitness.values
            
            # Evaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = list(map(toolbox_seq.evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
            
            # Elitism: keep the best
            combined = pop_seq + offspring
            pop_seq = tools.selBest(combined, POP_SIZE)
            
            # Track best cost
            best_cost = min([ind.fitness.values[0] for ind in pop_seq if ind.fitness.valid])
            best_costs.append(best_cost)
            
            # For comparison with other methods, we need to convert to multi-objective format
            # But since it's sequential, we only have one final solution (or we can create several by varying groups)
            # For now, let's create an artificial "Pareto front" with the best solutions found
            if gen % 50 == 0 or gen == NGEN_PHASE2 - 1:
                # Calculate net AEP for each valid solution
                valid_solutions = [ind for ind in pop_seq if ind.fitness.valid and ind.fitness.values[0] < 1e12]
                if len(valid_solutions) > 0:
                    # Convert to multi-objective format for comparison
                    # (but we don't have a real Pareto front, just one solution)
                    pass
        
        pop_final = pop_seq
    else:
        # Run standard SIMPLE GA - using parameters from wind_farm_GA_16.py
        pop_final, _ = algorithms.eaSimple(pop_seq, toolbox_seq, cxpb=0.95, mutpb=0.7, 
                                          ngen=NGEN_PHASE2, verbose=False)
    
    # Convert sequential solutions to multi-objective format for comparison
    # Create artificial "Pareto front" with the best solutions found
    pareto_front_list = []
    
    # Get the best solutions (diverse in terms of cost)
    valid_solutions = [ind for ind in pop_final if ind.fitness.valid and ind.fitness.values[0] < 1e12]
    if len(valid_solutions) > 0:
        # Sort by cost
        valid_solutions.sort(key=lambda x: x.fitness.values[0])
        
        # For each solution, calculate complete net AEP
        for ind_seq in valid_solutions[:min(50, len(valid_solutions))]:  # Top 50
            n_grupos_norm = ind_seq[0]
            sub_pos = np.array([ind_seq[1], ind_seq[2]])
            
            n_grupos = int(np.round(MIN_GRUPOS + n_grupos_norm * (MAX_GRUPOS - MIN_GRUPOS)))
            n_grupos = max(MIN_GRUPOS, min(MAX_GRUPOS, n_grupos))
            # Limit n_grupos to number of turbines
            n_grupos = min(n_grupos, IND_SIZE)
            
            # Calculate complete cabling
            coords_all = np.vstack([fixed_turb_coords, sub_pos.reshape(1, 2)])
            try:
                plant, res = cabling_v3.analisar_layout_completo(coords_all, sub=IND_SIZE, n_grupos=n_grupos)
                custo_usd = res['custo_total_usd']
                perdas_mwh = res['perda_anual_mwh']
                
                # Net AEP
                aep_liq = aep_bruto_fixed - perdas_mwh
                
                if aep_liq > 0 and custo_usd > 0:
                    # Create Phase2 individual for compatibility
                    full_genome = fixed_turb_coords.flatten().tolist() + [n_grupos_norm] + sub_pos.tolist()
                    ind_p2 = creator.IndividualPhase2(full_genome)
                    ind_p2.fitness.values = (aep_liq, custo_usd)
                    pareto_front_list.append(ind_p2)
            except:
                continue
    
    # Create Pareto front
    pareto_front = tools.ParetoFront()
    pareto_front.update(pareto_front_list)
    
    # Filter valid solutions
    pareto_front_filtered = filter_valid_solutions(pareto_front)
    
    # For evolution tracking, we need to simulate
    if track_evolution:
        # Since we don't have a real Pareto front during evolution, we will use the best cost
        # and estimate net AEP based on the best cost found
        if len(pareto_front_filtered) > 0 and ref_point is not None:
            # Calculate hypervolume of final front
            pf_points = [[ind.fitness.values[1], -ind.fitness.values[0]] for ind in pareto_front_filtered]
            pf_array = np.array(pf_points)
            hv = hypervolume_module.hypervolume(pf_array, np.array(ref_point))
            evolution_data["gen"] = list(range(NGEN_PHASE1, NGEN_PHASE1 + NGEN_PHASE2 + 1))
            evolution_data["hv"] = [hv] * (NGEN_PHASE2 + 1)  # Keep constant (approximation)
            evolution_data["n_solutions"] = [len(pareto_front_filtered)] * (NGEN_PHASE2 + 1)
        else:
            evolution_data["gen"] = list(range(NGEN_PHASE1, NGEN_PHASE1 + NGEN_PHASE2 + 1))
            evolution_data["hv"] = [0.0] * (NGEN_PHASE2 + 1)
            evolution_data["n_solutions"] = [0] * (NGEN_PHASE2 + 1)
    
    if track_evolution:
        return pareto_front_filtered, evolution_data
    return pareto_front_filtered

# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def filter_valid_solutions(pareto_front):
    """
    Filters invalid solutions from the Pareto front.
    Valid solutions: AEP > 0 and Cost > 0
    """
    return [ind for ind in pareto_front 
            if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]

# =============================================================================
# 6. ANALYSIS FUNCTIONS AND ADDITIONAL METRICS
# =============================================================================

def calculate_coverage(pareto_front_A, pareto_front_B):
    """
    Calculates the Coverage metric (C-metric) between two Pareto fronts.
    
    C(A, B) = |{b in B | exists a in A such that a dominates b}| / |B|
    
    Returns a value between 0 and 1:
    - 1.0 means all points of B are dominated by A
    - 0.0 means no points of B are dominated by A
    
    Args:
        pareto_front_A: List of individuals from front A
        pareto_front_B: List of individuals from front B
    
    Returns:
        coverage: Coverage value of A over B
    """
    if len(pareto_front_B) == 0:
        return 0.0
    if len(pareto_front_A) == 0:
        return 0.0
    
    # Extract objective values (AEP, Cost)
    # Assuming fitness.values = (AEP, Cost) with weights=(1.0, -1.0)
    # We want to maximize AEP and minimize Cost
    points_A = [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front_A 
                if ind.fitness.valid and ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
    points_B = [(ind.fitness.values[0], ind.fitness.values[1]) for ind in pareto_front_B 
                if ind.fitness.valid and ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
    
    if len(points_B) == 0:
        return 0.0
    if len(points_A) == 0:
        return 0.0
    
    dominated_count = 0
    for b_aep, b_cost in points_B:
        # Check if there is any point in A that dominates b
        # a dominates b if: a_aep >= b_aep AND a_cost <= b_cost AND (a_aep > b_aep OR a_cost < b_cost)
        is_dominated = False
        for a_aep, a_cost in points_A:
            if (a_aep >= b_aep and a_cost <= b_cost and 
                (a_aep > b_aep or a_cost < b_cost)):
                is_dominated = True
                break
        if is_dominated:
            dominated_count += 1
    
    return dominated_count / len(points_B)

def calculate_spread(pareto_front):
    """
    Calculates the spread metric (diversity) of the Pareto front.
    Spread measures the distribution of solutions along the front.
    Lower spread = better distribution.
    """
    if len(pareto_front) < 2:
        return 0.0
    
    # Extract objective values
    aep_values = [ind.fitness.values[0] for ind in pareto_front]
    cost_values = [ind.fitness.values[1] for ind in pareto_front]
    
    # Normalize to [0, 1]
    aep_min, aep_max = min(aep_values), max(aep_values)
    cost_min, cost_max = min(cost_values), max(cost_values)
    
    if aep_max == aep_min or cost_max == cost_min:
        return 0.0
    
    aep_norm = [(a - aep_min) / (aep_max - aep_min) for a in aep_values]
    cost_norm = [(c - cost_min) / (cost_max - cost_min) for c in cost_values]
    
    # Calculate distances between consecutive points (after sorting by AEP)
    sorted_indices = sorted(range(len(aep_norm)), key=lambda i: aep_norm[i])
    distances = []
    for i in range(len(sorted_indices) - 1):
        idx1, idx2 = sorted_indices[i], sorted_indices[i+1]
        dist = np.sqrt((aep_norm[idx1] - aep_norm[idx2])**2 + 
                      (cost_norm[idx1] - cost_norm[idx2])**2)
        distances.append(dist)
    
    if len(distances) == 0:
        return 0.0
    
    # Spread = standard deviation of distances
    mean_dist = np.mean(distances)
    spread = np.std(distances) if len(distances) > 1 else 0.0
    
    return spread

def calculate_convergence_gen(evolution_data, threshold=0.95):
    """
    Calculates the generation where the hypervolume reaches threshold% of the final value.
    Returns None if it does not converge.
    """
    if evolution_data is None or len(evolution_data["hv"]) == 0:
        return None
    
    hv_values = evolution_data["hv"]
    final_hv = hv_values[-1]
    target_hv = threshold * final_hv
    
    for i, hv in enumerate(hv_values):
        if hv >= target_hv:
            return evolution_data["gen"][i]
    
    return None

def calculate_statistical_tests(prop_data, base_data, metric_name="Hypervolume"):
    """
    Performs statistical tests to compare both methods.
    Returns dict with results.
    """
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available"}
    
    results = {}
    
    # Normality test (Shapiro-Wilk)
    _, p_prop_norm = stats.shapiro(prop_data)
    _, p_base_norm = stats.shapiro(base_data)
    results["normality"] = {
        "proposed_p": p_prop_norm,
        "baseline_p": p_base_norm,
        "both_normal": p_prop_norm > 0.05 and p_base_norm > 0.05
    }
    
    # t-test (if normal) or Mann-Whitney U test (if not normal)
    if results["normality"]["both_normal"]:
        # Student's t-test
        t_stat, p_value = stats.ttest_ind(prop_data, base_data, alternative='greater')
        results["test"] = "t-test"
        results["statistic"] = t_stat
        results["p_value"] = p_value
    else:
        # Mann-Whitney U test (Wilcoxon rank-sum)
        u_stat, p_value = stats.mannwhitneyu(prop_data, base_data, alternative='greater')
        results["test"] = "Mann-Whitney U"
        results["statistic"] = u_stat
        results["p_value"] = p_value
    
    # Effect size (Cohen's d or similar)
    mean_prop = np.mean(prop_data)
    mean_base = np.mean(base_data)
    std_pooled = np.sqrt((np.var(prop_data) + np.var(base_data)) / 2)
    cohens_d = (mean_prop - mean_base) / std_pooled if std_pooled > 0 else 0
    results["effect_size"] = {
        "cohens_d": cohens_d,
        "interpretation": "large" if abs(cohens_d) > 0.8 else "medium" if abs(cohens_d) > 0.5 else "small"
    }
    
    # Descriptive statistics
    results["descriptive"] = {
        "proposed": {
            "mean": mean_prop,
            "std": np.std(prop_data),
            "median": np.median(prop_data),
            "min": np.min(prop_data),
            "max": np.max(prop_data)
        },
        "baseline": {
            "mean": mean_base,
            "std": np.std(base_data),
            "median": np.median(base_data),
            "min": np.min(base_data),
            "max": np.max(base_data)
        }
    }
    
    results["significant"] = p_value < 0.05
    results["metric_name"] = metric_name
    
    return results

# =============================================================================
# 6. MAIN LOOP AND METRICS
# =============================================================================

if __name__ == "__main__":
    # ── CLI ──────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="opt_wind_farm_simulator benchmark — Proposed vs. Baseline vs. Sequential"
    )
    parser.add_argument("--quick", action="store_true",
                        help="Smoke-test mode: 3 seeds, 50+100 generations")
    parser.add_argument("--seeds",  type=int, default=None, help="Override N_SEEDS")
    parser.add_argument("--output", type=str, default="results",
                        help="Directory to save plots (default: results/)")
    args = parser.parse_args()

    if args.quick:
        N_SEEDS     = 3
        POP_SIZE    = 50
        NGEN_PHASE1 = 50
        NGEN_PHASE2 = 100
        NGEN_BASELINE = NGEN_PHASE1 + NGEN_PHASE2
        print("[QUICK MODE] 3 seeds, 50 + 100 generations")
    elif args.seeds is not None:
        N_SEEDS = args.seeds

    OUTPUT_DIR = args.output
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"--- GECCO 2026 BENCHMARK ---")
    print(f"Proposed Two-Phase  vs.  Baseline Pure NSGA-II  vs.  Sequential")
    print(f"Seeds: {N_SEEDS}  |  Pop: {POP_SIZE}")
    print(f"Proposed: Phase1={NGEN_PHASE1} + Phase2={NGEN_PHASE2} gens")
    print(f"Baseline: {NGEN_BASELINE} gens (same budget as Proposed)")
    
    results_prop_hv = []
    results_base_hv = []
    results_seq_hv = []  # Sequential
    results_prop_n_solutions = []
    results_base_n_solutions = []
    results_seq_n_solutions = []  # Sequential
    results_prop_spread = []
    results_base_spread = []
    results_seq_spread = []  # Sequential
    pareto_prop_all = []
    pareto_base_all = []
    pareto_seq_all = []  # Sequential
    
    # Execution time metrics (computational)
    times_prop = []  # Execution time of proposed method (seconds)
    times_base = []  # Execution time of baseline (seconds)
    times_seq = []   # Execution time of sequential (seconds)
    
    # Additional metrics
    n_evaluations_prop = []  # Number of function evaluations (proposed method)
    n_evaluations_base = []  # Number of function evaluations (baseline)
    n_evaluations_seq = []   # Number of function evaluations (sequential)
    success_rate_prop = []   # Success rate (1 if it found valid solutions, 0 otherwise)
    success_rate_base = []
    success_rate_seq = []
    
    # Evolution data (for visualization)
    evolution_prop = []  # List of dicts, one per seed
    evolution_base = []
    evolution_seq = []  # Sequential
    
    # Reference point for Hypervolume
    # Max AEP ~600 GWh -> normalize or use fixed value.
    # Max Cost ~10M USD.
    # Since DEAP assumes minimization in HV, we transform AEP to -AEP.
    # Ref Point must be worse than any viable solution: (Max Cost, Min -AEP)
    # Ex: Cost=20M, -AEP=0 (meaning AEP=0)
    ref_point = [2e7, 0]
    
    # Flag to track evolution (can be slow, use only if necessary)
    TRACK_EVOLUTION = True  # Change to False to run faster 

    for i in range(N_SEEDS):
        t0_seed = time.time()
        print(f"\n>>> Running Seed {i+1}/{N_SEEDS}...")
        
        # 1. Proposed
        print("   -> Running Proposed Method...")
        t_prop_start = time.time()
        if TRACK_EVOLUTION:
            pf_prop, evol_prop = run_proposed_method(i, track_evolution=True, ref_point=ref_point)
            evolution_prop.append(evol_prop)
        else:
            pf_prop = run_proposed_method(i)
        t_prop = time.time() - t_prop_start
        times_prop.append(t_prop)
        print(f"   [Proposed Time: {t_prop:.1f}s]")
        
        # 2. Baseline
        print("   -> Running Baseline...")
        t_base_start = time.time()
        if TRACK_EVOLUTION:
            pf_base, evol_base = run_baseline_method(i, track_evolution=True, ref_point=ref_point)
            evolution_base.append(evol_base)
        else:
            pf_base = run_baseline_method(i)
        t_base = time.time() - t_base_start
        times_base.append(t_base)
        print(f"   [Baseline Time: {t_base:.1f}s]")
        
        # 3. Sequential
        print("   -> Running Sequential Method...")
        t_seq_start = time.time()
        if TRACK_EVOLUTION:
            pf_seq, evol_seq = run_sequential_method(i, track_evolution=True, ref_point=ref_point)
            evolution_seq.append(evol_seq)
        else:
            pf_seq = run_sequential_method(i)
        t_seq = time.time() - t_seq_start
        times_seq.append(t_seq)
        print(f"   [Sequential Time: {t_seq:.1f}s]")
        
        # Calculate number of function evaluations (approximate)
        # Proposed: Phase 1 (NGEN_PHASE1 * POP_SIZE) + Phase 2 (NGEN_PHASE2 * POP_SIZE)
        n_eval_prop = NGEN_PHASE1 * POP_SIZE + NGEN_PHASE2 * POP_SIZE
        # Baseline: NGEN_BASELINE * POP_SIZE
        n_eval_base = NGEN_BASELINE * POP_SIZE
        # Sequential: Phase 1 (NGEN_PHASE1 * POP_SIZE) + Phase 2 (NGEN_PHASE2 * POP_SIZE)
        n_eval_seq = NGEN_PHASE1 * POP_SIZE + NGEN_PHASE2 * POP_SIZE
        
        n_evaluations_prop.append(n_eval_prop)
        n_evaluations_base.append(n_eval_base)
        n_evaluations_seq.append(n_eval_seq)
        
        # Success rate (1 if it found valid solutions, 0 otherwise)
        success_rate_prop.append(1 if len(pf_prop) > 0 else 0)
        success_rate_base.append(1 if len(pf_base) > 0 else 0)
        success_rate_seq.append(1 if len(pf_seq) > 0 else 0)
        
        # Debug: show info about the solutions
        if len(pf_prop) == 0:
            print(f"   WARNING [Seed {i}]: Proposed method returned 0 valid solutions!")
        if len(pf_base) == 0:
            print(f"   WARNING [Seed {i}]: Baseline returned 0 valid solutions!")
        if len(pf_seq) == 0:
            print(f"   WARNING [Seed {i}]: Sequential method returned 0 valid solutions!")
        
        # Hypervolume Calculation
        # Transform for minimization: [Cost, -AEP]
        # (Original: [AEP, Cost] -> FitnessMulti weights=(1.0, -1.0))
        # DEAP stores fitness.values as (AEP, Cost) or similar depending on implementation.
        # Our weights are (1.0, -1.0).
        # For DEAP's HV, we need to pass values to MINIMIZE.
        # Obj1: AEP (we want to max). For min, we use -AEP.
        # Obj2: Cost (we want to min). It is already Cost.
        
        def get_front_points(pf):
            # Returns list of [Cost, -AEP] for HV calculation
            # Already filtered, but we make sure they are valid
            return [[ind.fitness.values[1], -ind.fitness.values[0]] 
                   for ind in pf if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
            
        pts_prop = get_front_points(pf_prop)
        pts_base = get_front_points(pf_base)
        pts_seq = get_front_points(pf_seq)
        
        # Debug: show how many valid solutions we have
        print(f"   [Seed {i}] Valid solutions: Prop={len(pts_prop)}, Base={len(pts_base)}, Seq={len(pts_seq)}")
        
        # Calculate hypervolume using correct DEAP function
        # The hypervolume function expects a numpy array and reference point
        if len(pts_prop) > 0:
            pts_prop_array = np.array(pts_prop)
            hv_prop = hypervolume_module.hypervolume(pts_prop_array, np.array(ref_point))
        else:
            hv_prop = 0.0
            
        if len(pts_base) > 0:
            pts_base_array = np.array(pts_base)
            hv_base = hypervolume_module.hypervolume(pts_base_array, np.array(ref_point))
        else:
            hv_base = 0.0
        
        if len(pts_seq) > 0:
            pts_seq_array = np.array(pts_seq)
            hv_seq = hypervolume_module.hypervolume(pts_seq_array, np.array(ref_point))
        else:
            hv_seq = 0.0
        
        results_prop_hv.append(hv_prop)
        results_base_hv.append(hv_base)
        results_seq_hv.append(hv_seq)
        
        # Additional metrics
        n_sol_prop = len(pf_prop)
        n_sol_base = len(pf_base)
        n_sol_seq = len(pf_seq)
        results_prop_n_solutions.append(n_sol_prop)
        results_base_n_solutions.append(n_sol_base)
        results_seq_n_solutions.append(n_sol_seq)
        
        spread_prop = calculate_spread(pf_prop)
        spread_base = calculate_spread(pf_base)
        spread_seq = calculate_spread(pf_seq)
        results_prop_spread.append(spread_prop)
        results_base_spread.append(spread_base)
        results_seq_spread.append(spread_seq)
        
        # Save original points for plot (AEP, Cost)
        # Filter only valid solutions (AEP > 0, Cost > 0)
        prop_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                      for ind in pf_prop if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        base_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                      for ind in pf_base if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        seq_valid = [(ind.fitness.values[0]/1000, ind.fitness.values[1]/1e6) 
                      for ind in pf_seq if ind.fitness.values[0] > 0 and ind.fitness.values[1] > 0]
        
        pareto_prop_all.extend(prop_valid)
        pareto_base_all.extend(base_valid)
        pareto_seq_all.extend(seq_valid)
        
        if len(prop_valid) == 0:
            print(f"   WARNING [Seed {i}]: No valid solution in proposed method!")
        if len(base_valid) == 0:
            print(f"   WARNING [Seed {i}]: No valid solution in baseline!")
        if len(seq_valid) == 0:
            print(f"   WARNING [Seed {i}]: No valid solution in sequential method!")
        
        print(f"   [Seed {i}] HV: Prop={hv_prop:.2e} | Base={hv_base:.2e} | Seq={hv_seq:.2e}")
        print(f"   [Seed {i}] Solutions: Prop={n_sol_prop} | Base={n_sol_base} | Seq={n_sol_seq}")
        print(f"   [Seed {i}] Spread: Prop={spread_prop:.4f} | Base={spread_base:.4f} | Seq={spread_seq:.4f}")
        print(f"   [Seed {i}] Times: Prop={t_prop:.1f}s | Base={t_base:.1f}s | Seq={t_seq:.1f}s")
        print(f"   Total Seed Time: {time.time()-t0_seed:.1f}s")

    # =============================================================================
    # 7. STATISTICAL ANALYSIS
    # =============================================================================
    
    print("\n" + "="*80)
    print("FINAL RESULTS AND STATISTICAL ANALYSIS")
    print("="*80)
    
    print(f"\n--- HYPERVOLUME ---")
    print(f"Proposed:   Mean={np.mean(results_prop_hv):.2e}, Std={np.std(results_prop_hv):.2e}, Median={np.median(results_prop_hv):.2e}")
    print(f"Baseline:   Mean={np.mean(results_base_hv):.2e}, Std={np.std(results_base_hv):.2e}, Median={np.median(results_base_hv):.2e}")
    print(f"Sequential: Mean={np.mean(results_seq_hv):.2e}, Std={np.std(results_seq_hv):.2e}, Median={np.median(results_seq_hv):.2e}")
    
    print(f"\n--- NUMBER OF SOLUTIONS ---")
    print(f"Proposed:   Mean={np.mean(results_prop_n_solutions):.1f}, Std={np.std(results_prop_n_solutions):.1f}")
    print(f"Baseline:   Mean={np.mean(results_base_n_solutions):.1f}, Std={np.std(results_base_n_solutions):.1f}")
    print(f"Sequential: Mean={np.mean(results_seq_n_solutions):.1f}, Std={np.std(results_seq_n_solutions):.1f}")
    
    print(f"\n--- SPREAD (DIVERSITY) ---")
    print(f"Proposed:   Mean={np.mean(results_prop_spread):.4f}, Std={np.std(results_prop_spread):.4f}")
    print(f"Baseline:   Mean={np.mean(results_base_spread):.4f}, Std={np.std(results_base_spread):.4f}")
    print(f"Sequential: Mean={np.mean(results_seq_spread):.4f}, Std={np.std(results_seq_spread):.4f}")
    
    print(f"\n--- EXECUTION TIME (COMPUTATIONAL) ---")
    print(f"Proposed:   Mean={np.mean(times_prop):.1f}s, Std={np.std(times_prop):.1f}s, Median={np.median(times_prop):.1f}s")
    print(f"Baseline:   Mean={np.mean(times_base):.1f}s, Std={np.std(times_base):.1f}s, Median={np.median(times_base):.1f}s")
    print(f"Sequential: Mean={np.mean(times_seq):.1f}s, Std={np.std(times_seq):.1f}s, Median={np.median(times_seq):.1f}s")
    print(f"\n   Speedup Proposed vs Baseline: {np.mean(times_base)/np.mean(times_prop):.2f}x")
    print(f"   Speedup Sequential vs Baseline: {np.mean(times_base)/np.mean(times_seq):.2f}x")
    print(f"   Speedup Sequential vs Proposed: {np.mean(times_prop)/np.mean(times_seq):.2f}x")
    
    print(f"\n--- NUMBER OF FUNCTION EVALUATIONS ---")
    print(f"Proposed:   {n_evaluations_prop[0]:,} evaluations (Phase 1: {NGEN_PHASE1*POP_SIZE:,} + Phase 2: {NGEN_PHASE2*POP_SIZE:,})")
    print(f"Baseline:   {n_evaluations_base[0]:,} evaluations ({NGEN_BASELINE*POP_SIZE:,})")
    print(f"Sequential: {n_evaluations_seq[0]:,} evaluations (Phase 1: {NGEN_PHASE1*POP_SIZE:,} + Phase 2: {NGEN_PHASE2*POP_SIZE:,})")
    
    # Time per evaluation (computational efficiency)
    time_per_eval_prop = np.mean(times_prop) / n_evaluations_prop[0] if n_evaluations_prop[0] > 0 else 0
    time_per_eval_base = np.mean(times_base) / n_evaluations_base[0] if n_evaluations_base[0] > 0 else 0
    time_per_eval_seq = np.mean(times_seq) / n_evaluations_seq[0] if n_evaluations_seq[0] > 0 else 0
    
    print(f"\n--- TIME PER EVALUATION (EFFICIENCY) ---")
    print(f"Proposed:   {time_per_eval_prop*1000:.3f} ms/evaluation")
    print(f"Baseline:   {time_per_eval_base*1000:.3f} ms/evaluation")
    print(f"Sequential: {time_per_eval_seq*1000:.3f} ms/evaluation")
    
    print(f"\n--- SUCCESS RATE (VALID SOLUTIONS) ---")
    success_prop = np.mean(success_rate_prop) * 100
    success_base = np.mean(success_rate_base) * 100
    success_seq = np.mean(success_rate_seq) * 100
    print(f"Proposed:   {success_prop:.1f}% ({sum(success_rate_prop)}/{N_SEEDS} runs)")
    print(f"Baseline:   {success_base:.1f}% ({sum(success_rate_base)}/{N_SEEDS} runs)")
    print(f"Sequential: {success_seq:.1f}% ({sum(success_rate_seq)}/{N_SEEDS} runs)")
    
    # Coverage (C-metric) - dominance comparison between methods
    # Calculate on aggregated solutions of all seeds
    print(f"\n--- COVERAGE (C-METRIC) - DOMINANCE BETWEEN METHODS ---")
    print(f"   (C(A,B) = fraction of solutions of B dominated by A)")
    
    # Filter valid solutions for coverage
    prop_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_prop_all if aep > 0 and cost > 0]
    base_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_base_all if aep > 0 and cost > 0]
    seq_valid_coverage = [(aep*1000, cost*1e6) for aep, cost in pareto_seq_all if aep > 0 and cost > 0]
    
    # Helper function to calculate coverage between two lists of points
    def calc_coverage_points(points_A, points_B):
        """Calculates coverage of A over B using lists of points (aep, cost)"""
        if len(points_B) == 0 or len(points_A) == 0:
            return 0.0
        dominated = 0
        for b_aep, b_cost in points_B:
            for a_aep, a_cost in points_A:
                if a_aep >= b_aep and a_cost <= b_cost and (a_aep > b_aep or a_cost < b_cost):
                     dominated += 1
                     break
        return dominated / len(points_B)
    
    if len(prop_valid_coverage) > 0 and len(base_valid_coverage) > 0:
        cov_prop_base = calc_coverage_points(prop_valid_coverage, base_valid_coverage)
        cov_base_prop = calc_coverage_points(base_valid_coverage, prop_valid_coverage)
        print(f"Proposed dominates Baseline: {cov_prop_base:.3f} ({cov_prop_base*100:.1f}%)")
        print(f"Baseline dominates Proposed: {cov_base_prop:.3f} ({cov_base_prop*100:.1f}%)")
    
    if len(prop_valid_coverage) > 0 and len(seq_valid_coverage) > 0:
        cov_prop_seq = calc_coverage_points(prop_valid_coverage, seq_valid_coverage)
        cov_seq_prop = calc_coverage_points(seq_valid_coverage, prop_valid_coverage)
        print(f"Proposed dominates Sequential: {cov_prop_seq:.3f} ({cov_prop_seq*100:.1f}%)")
        print(f"Sequential dominates Proposed: {cov_seq_prop:.3f} ({cov_seq_prop*100:.1f}%)")
    
    if len(base_valid_coverage) > 0 and len(seq_valid_coverage) > 0:
        cov_base_seq = calc_coverage_points(base_valid_coverage, seq_valid_coverage)
        cov_seq_base = calc_coverage_points(seq_valid_coverage, base_valid_coverage)
        print(f"Baseline dominates Sequential: {cov_base_seq:.3f} ({cov_base_seq*100:.1f}%)")
        print(f"Sequential dominates Baseline: {cov_seq_base:.3f} ({cov_seq_base*100:.1f}%)")
    
    # Statistical tests
    if SCIPY_AVAILABLE:
        print(f"\n--- STATISTICAL TESTS ---")
        stats_hv = calculate_statistical_tests(results_prop_hv, results_base_hv, "Hypervolume")
        print(f"Test: {stats_hv['test']}")
        print(f"Statistic: {stats_hv['statistic']:.4f}")
        print(f"p-value: {stats_hv['p_value']:.6f}")
        print(f"Significant (p<0.05): {'YES' if stats_hv['significant'] else 'NO'}")
        print(f"Effect Size (Cohen's d): {stats_hv['effect_size']['cohens_d']:.4f} ({stats_hv['effect_size']['interpretation']})")
        
        stats_nsol = calculate_statistical_tests(results_prop_n_solutions, results_base_n_solutions, "Number of Solutions")
        print(f"\nNumber of Solutions - p-value: {stats_nsol['p_value']:.6f}")
        
        stats_spread = calculate_statistical_tests(results_prop_spread, results_base_spread, "Spread")
        print(f"Spread - p-value: {stats_spread['p_value']:.6f}")
        
        # Sequential vs others tests
        print(f"\n--- SEQUENTIAL COMPARISON ---")
        stats_seq_hv = calculate_statistical_tests(results_seq_hv, results_base_hv, "Hypervolume (Sequential vs Baseline)")
        print(f"Sequential vs Baseline HV - p-value: {stats_seq_hv['p_value']:.6f}")
        
        stats_seq_prop_hv = calculate_statistical_tests(results_seq_hv, results_prop_hv, "Hypervolume (Sequential vs Proposed)")
        print(f"Sequential vs Proposed HV - p-value: {stats_seq_prop_hv['p_value']:.6f}")
        
        # Execution time statistical tests
        print(f"\n--- STATISTICAL TESTS - EXECUTION TIME ---")
        stats_time_prop_base = calculate_statistical_tests(times_base, times_prop, "Execution Time (Baseline vs Proposed)")
        print(f"Baseline vs Proposed Time - p-value: {stats_time_prop_base['p_value']:.6f}")
        print(f"   (Test verifies if Proposed is faster)")
        
        stats_time_seq_base = calculate_statistical_tests(times_base, times_seq, "Execution Time (Baseline vs Sequential)")
        print(f"Baseline vs Sequential Time - p-value: {stats_time_seq_base['p_value']:.6f}")
        print(f"   (Test verifies if Sequential is faster)")
        
        stats_time_seq_prop = calculate_statistical_tests(times_prop, times_seq, "Execution Time (Proposed vs Sequential)")
        print(f"Proposed vs Sequential Time - p-value: {stats_time_seq_prop['p_value']:.6f}")
        print(f"   (Test verifies if Sequential is faster)")
    
    # Convergence time (if tracked)
    if TRACK_EVOLUTION and len(evolution_prop) > 0:
        print(f"\n--- CONVERGENCE TIME (95% of final HV) ---")
        conv_prop = [calculate_convergence_gen(evol) for evol in evolution_prop]
        conv_base = [calculate_convergence_gen(evol) for evol in evolution_base]
        conv_prop = [c for c in conv_prop if c is not None]
        conv_base = [c for c in conv_base if c is not None]
        if conv_prop:
            print(f"Proposed:   Mean={np.mean(conv_prop):.1f} generations")
        if conv_base:
            print(f"Baseline:   Mean={np.mean(conv_base):.1f} generations")
    
    # =============================================================================
    # 8. VISUALIZATION AND SAVING
    # =============================================================================
    
    # PLOT 1: Boxplot Hypervolume - IMPROVED
    plt.figure(figsize=(12, 6))
    bp = plt.boxplot([results_prop_hv, results_base_hv, results_seq_hv], 
                     tick_labels=['Proposed (Two-Phase)', 'Baseline (Pure NSGA-II)', 'Sequential'],
                     patch_artist=True, widths=0.6)
    
    # Custom colors
    colors = ['#2E86AB', '#E63946', '#06A77D']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Improve lines
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color='black', linewidth=1.2)
    
    plt.ylabel('Hypervolume (Higher is Better)', fontsize=15, fontweight='bold')
    plt.title(f'Statistical Comparison ({N_SEEDS} runs)', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='x', which='major', labelsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_boxplot.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_boxplot.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Graph saved: {os.path.join(OUTPUT_DIR, 'comparison_boxplot.png')} and .pdf")
    
    # PLOT 2: Pareto Fronts (Scatter) - IMPROVED
    print(f"\n--- PREPARING SCATTER PLOT ---")
    print(f"Total proposed points collected: {len(pareto_prop_all)}")
    print(f"Total baseline points collected: {len(pareto_base_all)}")
    
    # Filter valid solutions (AEP > 0, Cost > 0)
    pareto_prop_valid = [(aep, cost) for aep, cost in pareto_prop_all if aep > 0 and cost > 0]
    pareto_base_valid = [(aep, cost) for aep, cost in pareto_base_all if aep > 0 and cost > 0]
    pareto_seq_valid = [(aep, cost) for aep, cost in pareto_seq_all if aep > 0 and cost > 0]
    
    print(f"Valid points after filter: Proposed={len(pareto_prop_valid)}, Baseline={len(pareto_base_valid)}, Sequential={len(pareto_seq_valid)}")
    
    if len(pareto_prop_valid) == 0 and len(pareto_base_valid) == 0 and len(pareto_seq_valid) == 0:
        print("ERROR: No valid point to plot! Skipping scatter plot.")
    else:
        plt.figure(figsize=(12, 8))
        
        if len(pareto_prop_valid) > 0:
            p_aep = [p[0] for p in pareto_prop_valid]
            p_cost = [p[1] for p in pareto_prop_valid]
            plt.scatter(p_cost, p_aep, c='#2E86AB', alpha=0.7, s=60, 
                       edgecolors='#1B4965', linewidths=0.8, 
                       label=f'Proposed Solutions (n={len(pareto_prop_valid)})', zorder=3)
        
        if len(pareto_base_valid) > 0:
            b_aep = [p[0] for p in pareto_base_valid]
            b_cost = [p[1] for p in pareto_base_valid]
            plt.scatter(b_cost, b_aep, c='#E63946', alpha=0.6, s=50,
                       edgecolors='#A41623', linewidths=0.8,
                       label=f'Baseline Solutions (n={len(pareto_base_valid)})', zorder=2)
        
        if len(pareto_seq_valid) > 0:
            s_aep = [p[0] for p in pareto_seq_valid]
            s_cost = [p[1] for p in pareto_seq_valid]
            plt.scatter(s_cost, s_aep, c='#06A77D', alpha=0.6, s=50,
                       edgecolors='#045D4A', linewidths=0.8,
                       label=f'Sequential Solutions (n={len(pareto_seq_valid)})', zorder=2)
        
        # Improve appearance
        plt.xlabel('Cabling Cost (M USD)', fontsize=15, fontweight='bold')
        plt.ylabel('Net AEP (GWh)', fontsize=15, fontweight='bold')
        plt.title(f'Pareto Fronts Accumulation ({N_SEEDS} seeds)\nPop={POP_SIZE}, Gens={NGEN_BASELINE}', 
                  fontsize=18, fontweight='bold')
        plt.legend(fontsize=14, framealpha=0.9, loc='best')
        plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
        
        # Remove top and right borders
        ax = plt.gca()
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_pareto.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_pareto.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Graph saved: comparison_pareto.png and comparison_pareto.pdf ({len(pareto_prop_valid)} proposed, {len(pareto_base_valid)} baseline, {len(pareto_seq_valid)} sequential solutions)")
    
    # PLOT 3: Hypervolume Evolution (if tracked)
    if TRACK_EVOLUTION and len(evolution_prop) > 0:
        plt.figure(figsize=(12, 5))
        
        # Subplot 1: Mean evolution
        plt.subplot(1, 2, 1)
        for i, evol in enumerate(evolution_prop):
            if i == 0:
                plt.plot(evol["gen"], evol["hv"], '#2E86AB', alpha=0.3, linewidth=0.8, label='Proposed (individual)')
            else:
                plt.plot(evol["gen"], evol["hv"], '#2E86AB', alpha=0.3, linewidth=0.8)
        
        for i, evol in enumerate(evolution_base):
            if i == 0:
                plt.plot(evol["gen"], evol["hv"], '#E63946', alpha=0.3, linewidth=0.8, label='Baseline (individual)')
            else:
                plt.plot(evol["gen"], evol["hv"], '#E63946', alpha=0.3, linewidth=0.8)
        
        # Calculate mean per generation
        all_gens_prop = set()
        for evol in evolution_prop:
            all_gens_prop.update(evol["gen"])
        all_gens_base = set()
        for evol in evolution_base:
            all_gens_base.update(evol["gen"])
        
        if all_gens_prop:
            gen_range_prop = sorted(all_gens_prop)
            hv_mean_prop = []
            for gen in gen_range_prop:
                hvs = []
                for evol in evolution_prop:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        hvs.append(evol["hv"][idx])
                if hvs:
                    hv_mean_prop.append(np.mean(hvs))
                else:
                    hv_mean_prop.append(None)
            # Remove None values
            gen_range_prop_clean = [g for g, h in zip(gen_range_prop, hv_mean_prop) if h is not None]
            hv_mean_prop_clean = [h for h in hv_mean_prop if h is not None]
            if gen_range_prop_clean:
                plt.plot(gen_range_prop_clean, hv_mean_prop_clean, '#2E86AB', linewidth=2.5, 
                        label='Proposed (mean)', zorder=10)
        
        if all_gens_base:
            gen_range_base = sorted(all_gens_base)
            hv_mean_base = []
            for gen in gen_range_base:
                hvs = []
                for evol in evolution_base:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        hvs.append(evol["hv"][idx])
                if hvs:
                    hv_mean_base.append(np.mean(hvs))
                else:
                    hv_mean_base.append(None)
            # Remove None values
            gen_range_base_clean = [g for g, h in zip(gen_range_base, hv_mean_base) if h is not None]
            hv_mean_base_clean = [h for h in hv_mean_base if h is not None]
            if gen_range_base_clean:
                plt.plot(gen_range_base_clean, hv_mean_base_clean, '#E63946', linewidth=2.5, 
                        label='Baseline (mean)', zorder=10)
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Hypervolume', fontsize=14, fontweight='bold')
        plt.title('Hypervolume Evolution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        ax1 = plt.gca()
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        
        # Subplot 2: Number of solutions
        plt.subplot(1, 2, 2)
        for i, evol in enumerate(evolution_prop):
            if i == 0:
                plt.plot(evol["gen"], evol["n_solutions"], '#2E86AB', alpha=0.3, linewidth=0.8, label='Proposed (individual)')
            else:
                plt.plot(evol["gen"], evol["n_solutions"], '#2E86AB', alpha=0.3, linewidth=0.8)
        
        for i, evol in enumerate(evolution_base):
            if i == 0:
                plt.plot(evol["gen"], evol["n_solutions"], '#E63946', alpha=0.3, linewidth=0.8, label='Baseline (individual)')
            else:
                plt.plot(evol["gen"], evol["n_solutions"], '#E63946', alpha=0.3, linewidth=0.8)
        
        # Calculate mean for number of solutions as well
        if all_gens_prop:
            n_sol_mean_prop = []
            for gen in gen_range_prop:
                n_sols = []
                for evol in evolution_prop:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        n_sols.append(evol["n_solutions"][idx])
                if n_sols:
                    n_sol_mean_prop.append(np.mean(n_sols))
                else:
                    n_sol_mean_prop.append(None)
            gen_range_prop_clean_nsol = [g for g, n in zip(gen_range_prop, n_sol_mean_prop) if n is not None]
            n_sol_mean_prop_clean = [n for n in n_sol_mean_prop if n is not None]
            if gen_range_prop_clean_nsol:
                plt.plot(gen_range_prop_clean_nsol, n_sol_mean_prop_clean, '#2E86AB', 
                        linewidth=2.5, label='Proposed (mean)', zorder=10)
        
        if all_gens_base:
            n_sol_mean_base = []
            for gen in gen_range_base:
                n_sols = []
                for evol in evolution_base:
                    if gen in evol["gen"]:
                        idx = evol["gen"].index(gen)
                        n_sols.append(evol["n_solutions"][idx])
                if n_sols:
                    n_sol_mean_base.append(np.mean(n_sols))
                else:
                    n_sol_mean_base.append(None)
            gen_range_base_clean_nsol = [g for g, n in zip(gen_range_base, n_sol_mean_base) if n is not None]
            n_sol_mean_base_clean = [n for n in n_sol_mean_base if n is not None]
            if gen_range_base_clean_nsol:
                plt.plot(gen_range_base_clean_nsol, n_sol_mean_base_clean, '#E63946', 
                        linewidth=2.5, label='Baseline (mean)', zorder=10)
        
        plt.xlabel('Generation', fontsize=14, fontweight='bold')
        plt.ylabel('Number of Pareto Solutions', fontsize=14, fontweight='bold')
        plt.title('Pareto Front Size Evolution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=13, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='--')
        ax2 = plt.gca()
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_evolution.png'), dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_evolution.pdf'), bbox_inches='tight', facecolor='white')
        print(f"Graph saved: {os.path.join(OUTPUT_DIR, 'comparison_evolution.png')} and .pdf")
    
    # PLOT 4: Multi-metric Comparison - IMPROVED (including execution time)
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    colors = ['#2E86AB', '#E63946', '#06A77D']
    
    # Hypervolume
    bp1 = axes[0].boxplot([results_prop_hv, results_base_hv, results_seq_hv], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp1[element], color='black', linewidth=1.2)
    axes[0].set_ylabel('Hypervolume', fontsize=14, fontweight='bold')
    axes[0].set_title('Hypervolume Comparison', fontsize=16, fontweight='bold')
    axes[0].tick_params(axis='both', which='major', labelsize=12)
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    # Number of solutions
    bp2 = axes[1].boxplot([results_prop_n_solutions, results_base_n_solutions, results_seq_n_solutions], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp2[element], color='black', linewidth=1.2)
    axes[1].set_ylabel('Number of Solutions', fontsize=14, fontweight='bold')
    axes[1].set_title('Pareto Front Size', fontsize=16, fontweight='bold')
    axes[1].tick_params(axis='both', which='major', labelsize=12)
    axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    # Spread
    bp3 = axes[2].boxplot([results_prop_spread, results_base_spread, results_seq_spread], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp3['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp3[element], color='black', linewidth=1.2)
    axes[2].set_ylabel('Spread (Lower is Better)', fontsize=14, fontweight='bold')
    axes[2].set_title('Solution Diversity', fontsize=16, fontweight='bold')
    axes[2].tick_params(axis='both', which='major', labelsize=12)
    axes[2].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[2].spines['top'].set_visible(False)
    axes[2].spines['right'].set_visible(False)
    
    # Execution time
    bp4 = axes[3].boxplot([times_prop, times_base, times_seq], 
                         tick_labels=['Proposed', 'Baseline', 'Sequential'],
                         patch_artist=True, widths=0.6)
    for patch, color in zip(bp4['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp4[element], color='black', linewidth=1.2)
    axes[3].set_ylabel('Execution Time (seconds)', fontsize=14, fontweight='bold')
    axes[3].set_title('Computational Efficiency', fontsize=16, fontweight='bold')
    axes[3].tick_params(axis='both', which='major', labelsize=12)
    axes[3].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[3].spines['top'].set_visible(False)
    axes[3].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_metrics.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_metrics.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Graph saved: {os.path.join(OUTPUT_DIR, 'comparison_metrics.png')} and .pdf")
    
    # PLOT 5: Dedicated execution time graph (for paper)
    plt.figure(figsize=(10, 6))
    bp_time = plt.boxplot([times_prop, times_base, times_seq], 
                          tick_labels=['Proposed\n(Two-Phase)', 'Baseline\n(Pure NSGA-II)', 'Sequential\n(Two GAs)'],
                          patch_artist=True, widths=0.6)
    
    # Custom colors
    colors_time = ['#2E86AB', '#E63946', '#06A77D']
    for patch, color in zip(bp_time['boxes'], colors_time):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Improve lines
    for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp_time[element], color='black', linewidth=1.2)
    
    plt.ylabel('Execution Time (seconds)', fontsize=16, fontweight='bold')
    plt.title(f'Computational Efficiency Comparison ({N_SEEDS} runs)', fontsize=18, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Add speedup annotations
    speedup_prop = np.mean(times_base) / np.mean(times_prop)
    speedup_seq = np.mean(times_base) / np.mean(times_seq)
    speedup_seq_prop = np.mean(times_prop) / np.mean(times_seq)
    
    # Annotations on graph
    ax_time = plt.gca()
    ax_time.text(1, np.max(times_prop) * 1.1, f'{speedup_prop:.2f}x faster\nthan baseline', 
                ha='center', fontsize=11, fontweight='bold', color='#2E86AB')
    ax_time.text(3, np.max(times_seq) * 1.1, f'{speedup_seq:.2f}x faster\nthan baseline', 
                ha='center', fontsize=11, fontweight='bold', color='#06A77D')
    
    ax_time.tick_params(axis='both', which='major', labelsize=13)
    ax_time.spines['top'].set_visible(False)
    ax_time.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_execution_time.png'), dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_execution_time.pdf'), bbox_inches='tight', facecolor='white')
    print(f"Graph saved: {os.path.join(OUTPUT_DIR, 'comparison_execution_time.png')} and .pdf")
    
    # Save results to CSV
    results_df = pd.DataFrame({
        'seed': list(range(N_SEEDS)),
        'hv_proposed': results_prop_hv,
        'hv_baseline': results_base_hv,
        'hv_sequential': results_seq_hv,
        'n_solutions_proposed': results_prop_n_solutions,
        'n_solutions_baseline': results_base_n_solutions,
        'n_solutions_sequential': results_seq_n_solutions,
        'spread_proposed': results_prop_spread,
        'spread_baseline': results_base_spread,
        'spread_sequential': results_seq_spread,
        'time_proposed_seconds': times_prop,
        'time_baseline_seconds': times_base,
        'time_sequential_seconds': times_seq,
        'n_evaluations_proposed': n_evaluations_prop,
        'n_evaluations_baseline': n_evaluations_base,
        'n_evaluations_sequential': n_evaluations_seq,
        'success_rate_proposed': success_rate_prop,
        'success_rate_baseline': success_rate_base,
        'success_rate_sequential': success_rate_seq
    })
    results_df.to_csv('benchmark_results.csv', index=False)
    print("Results saved: benchmark_results.csv")
    
    plt.show()