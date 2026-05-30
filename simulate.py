#!/usr/bin/env python3
"""
simulate.py — User-Facing Simulation and Co-design Pipeline
===========================================================
Runs a two-phase optimization for layout (AEP) and cabling (CAPEX)
on any user-configured YAML case with automatic plateau detection.

Steps:
  1. Load case configuration from YAML.
  2. Phase 1: Optimize turbine layout for maximum gross AEP (layout-only).
     Stops automatically when AEP converges (plateau detection).
  3. Smart Seeding: Expand genome and seed Phase 2 using Phase 1 results.
  4. Phase 2: Co-optimize layout, substation and cable grouping.
     Stops automatically when net AEP converges.
  5. Generate high-quality evolution GIFs (wake field and cabling layouts).
"""

import os
import sys
import time
import multiprocessing
import yaml
import random
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import io

from deap import base, creator, tools, algorithms

# ── reproducibility setup ────────────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from config.iea37_aepcalc import calcAEP, getTurbLocYAML, getWindRoseYAML, getTurbAtrbtYAML
from core.cabling_v3 import analisar_layout_completo
from core.wfwe import WindFarm

# ── DEAP setup clean-up ───────────────────────────────────────────────────────
if hasattr(creator, "FitnessMax"):
    del creator.FitnessMax
if hasattr(creator, "IndividualPhase1"):
    del creator.IndividualPhase1
if hasattr(creator, "FitnessMulti"):
    del creator.FitnessMulti
if hasattr(creator, "IndividualPhase2"):
    del creator.IndividualPhase2

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)

creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)


# ── helpers ───────────────────────────────────────────────────────────────────
def banner(text: str) -> None:
    width = 70
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def is_within_circle(x, y, radius):
    return x**2 + y**2 <= radius**2


def enforce_circle_p1(individual, radius, n_turb):
    for i in range(n_turb):
        x, y = individual[2*i], individual[2*i+1]
        if not is_within_circle(x, y, radius):
            angle = np.arctan2(y, x)
            individual[2*i] = radius * np.cos(angle)
            individual[2*i+1] = radius * np.sin(angle)
    return individual


def enforce_circle_p2(individual, radius, n_turb, substation_mode):
    # Enforce turbine boundaries
    enforce_circle_p1(individual, radius, n_turb)
    
    # Enforce substation boundaries if optimizing
    n_coords = n_turb * 2
    individual[n_coords] = max(0.0, min(1.0, individual[n_coords]))  # g_norm
    
    if substation_mode == "optimize":
        sub_x, sub_y = individual[n_coords+1], individual[n_coords+2]
        if not is_within_circle(sub_x, sub_y, radius):
            angle = np.arctan2(sub_y, sub_x)
            individual[n_coords+1] = radius * np.cos(angle)
            individual[n_coords+2] = radius * np.sin(angle)
    return individual


def repair_spacing(coords, min_spacing, radius, max_iterations=10):
    """Repels turbines that are too close, keeping them inside the boundary."""
    coords = coords.copy()
    n_turb = len(coords)
    for _ in range(max_iterations):
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(n_turb, k=1)
        violations = dists[i_upper, j_upper] < min_spacing
        
        if not np.any(violations):
            break
            
        for idx in np.where(violations)[0]:
            i, j = i_upper[idx], j_upper[idx]
            dist_ij = dists[i, j]
            if dist_ij < 1e-6:
                angle = random.uniform(0, 2*np.pi)
                direction = np.array([np.cos(angle), np.sin(angle)])
            else:
                direction = (coords[i] - coords[j]) / dist_ij
                
            sep = (min_spacing - dist_ij) / 2.0
            coords[i] += direction * sep
            coords[j] -= direction * sep
            
            # Clamp back to boundary circle
            for k in [i, j]:
                d = np.linalg.norm(coords[k])
                if d > radius:
                    coords[k] = (coords[k] / d) * radius
    return coords


# ── simulation wrapper class ──────────────────────────────────────────────────
class WindFarmSimulator:
    def __init__(self, config_path: str, output_dir: str):
        self.config = load_config(config_path) or {}
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default values for all parameters if missing or None
        defaults = {
            "name": "Wind Farm Simulation",
            "n_turbines": 16,
            "boundary_radius": 1300.0,
            "min_spacing_multiplier": 2.0,
            "turbine_yaml": "config/iea37-335mw.yaml",
            "windrose_yaml": "config/iea37-windrose.yaml",
            "initial_layout_yaml": None,
            "population_size": 300,
            "crossover_probability": 0.95,
            "mutation_probability": 0.7,
            "mutation_sigma": 100.0,
            "mutation_indpb": 0.4,
            "mu": 0.0,
            "tournament_size": 5,
            "alpha": 0.5,
            "plateau_generations_p1": None,
            "plateau_generations_p2": None,
            "plateau_generations": 50,
            "cable_groups": {}
        }
        for key, val in defaults.items():
            if key not in self.config or self.config[key] is None:
                self.config[key] = val

        # Handle substation configuration
        sub_cfg = self.config.get("substation")
        if isinstance(sub_cfg, list):
            self.sub_mode = "fixed"
            self.sub_fixed_pos = np.array(sub_cfg)
        elif isinstance(sub_cfg, str) and sub_cfg.lower() in ("optimize", "optimise"):
            self.sub_mode = "optimize"
            self.sub_fixed_pos = np.array([-1350.0, 0.0])
        elif isinstance(sub_cfg, dict):
            self.sub_mode = sub_cfg.get("mode") or "optimize"
            self.sub_fixed_pos = np.array(sub_cfg.get("fixed_pos") or [-1350.0, 0.0])
        else:
            self.sub_mode = "optimize"
            self.sub_fixed_pos = np.array([-1350.0, 0.0])

        # Handle cable groups defaults
        cable_defaults = {
            "min_groups": 2,
            "max_groups": 16
        }
        if not isinstance(self.config["cable_groups"], dict):
            self.config["cable_groups"] = {}
        for key, val in cable_defaults.items():
            if key not in self.config["cable_groups"] or self.config["cable_groups"][key] is None:
                self.config["cable_groups"][key] = val

        # Load turbine characteristics
        turb_yaml = os.path.join(ROOT, self.config["turbine_yaml"])
        self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr, self.turb_diam = getTurbAtrbtYAML(turb_yaml)
        
        # Load wind rose
        wr_yaml = os.path.join(ROOT, self.config["windrose_yaml"])
        self.wind_dir, self.wind_freq, self.wind_speed = getWindRoseYAML(wr_yaml)
        
        # Geometry
        self.n_turb = self.config["n_turbines"]
        self.radius = self.config["boundary_radius"]
        self.min_spacing = self.config["min_spacing_multiplier"] * self.turb_diam
        
        # Substation and Cable limits
        self.min_groups = self.config["cable_groups"]["min_groups"]
        self.max_groups = self.config["cable_groups"]["max_groups"]
        
        # Load or generate initial layout
        if "initial_layout_yaml" in self.config and self.config["initial_layout_yaml"]:
            layout_yaml = os.path.join(ROOT, self.config["initial_layout_yaml"])
            self.initial_coords = getTurbLocYAML(layout_yaml)
            # Ensure it matches config size
            if len(self.initial_coords) != self.n_turb:
                print(f"[Warning] Initial layout size ({len(self.initial_coords)}) mismatch with config turbines ({self.n_turb}). Truncating/Padding.")
                if len(self.initial_coords) > self.n_turb:
                    self.initial_coords = self.initial_coords[:self.n_turb]
                else:
                    # Pad randomly
                    padding = []
                    for _ in range(self.n_turb - len(self.initial_coords)):
                        r = random.uniform(0, self.radius * 0.9)
                        angle = random.uniform(0, 2*np.pi)
                        padding.append([r*np.cos(angle), r*np.sin(angle)])
                    self.initial_coords = np.vstack([self.initial_coords, padding])
        else:
            # Generate random spacing-repaired initial layout
            coords_list = []
            for _ in range(self.n_turb):
                r = random.uniform(0, self.radius * 0.8)
                angle = random.uniform(0, 2*np.pi)
                coords_list.append([r*np.cos(angle), r*np.sin(angle)])
            self.initial_coords = repair_spacing(np.array(coords_list), self.min_spacing, self.radius)

        print(f"Loaded simulation case: {self.config['name']}")
        print(f"  Turbines: {self.n_turb} | Radius: {self.radius}m | Min Spacing: {self.min_spacing:.1f}m")
        print(f"  Substation mode: {self.sub_mode} | Output directory: {self.output_dir}")

    # ── Phase 1 evaluation: Gross AEP ─────────────────────────────────────────
    def evaluate_p1(self, individual):
        turb_coords = np.array(individual).reshape((self.n_turb, 2))
        
        # Spacing penalty
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(self.n_turb, k=1)
        violations = dists[i_upper, j_upper] < self.min_spacing
        penalty_spacing = np.sum(np.maximum(0, self.min_spacing - dists[i_upper, j_upper][violations])) * 1e6
        
        # Boundary penalty
        dist_from_center = np.linalg.norm(turb_coords, axis=1)
        penalty_circle = np.sum(np.maximum(0, dist_from_center - self.radius)) * 1e6
        
        # Calculate gross AEP
        aep = np.sum(calcAEP(turb_coords, self.wind_freq, self.wind_speed, self.wind_dir,
                             self.turb_diam, self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr))
        
        return aep - penalty_spacing - penalty_circle,

    # ── Phase 2 evaluation: Net AEP vs Cable Cost ─────────────────────────────
    def evaluate_p2(self, individual):
        n_coords = self.n_turb * 2
        turb_coords = np.array(individual[:n_coords]).reshape((self.n_turb, 2))
        g_norm = individual[n_coords]
        
        # Map number of groups
        n_groups = int(np.round(self.min_groups + g_norm * (self.max_groups - self.min_groups)))
        n_groups = max(self.min_groups, min(self.n_turb, n_groups))
        
        # Substation position
        if self.sub_mode == "optimize":
            sub_pos = np.array([individual[n_coords+1], individual[n_coords+2]])
        else:
            sub_pos = self.sub_fixed_pos
            
        # Penalties
        penalty = 0.0
        # 1. Spacing
        diff = turb_coords[:, np.newaxis, :] - turb_coords[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2)
        i_upper, j_upper = np.triu_indices(self.n_turb, k=1)
        violations = dists[i_upper, j_upper] < self.min_spacing
        penalty += np.sum(np.maximum(0, self.min_spacing - dists[i_upper, j_upper][violations])) * 1e6
        
        # 2. Boundary
        dists_center = np.linalg.norm(turb_coords, axis=1)
        penalty += np.sum(np.maximum(0, dists_center - self.radius)) * 1e6
        
        # 3. Substation boundary if optimizing
        if self.sub_mode == "optimize":
            dist_sub = np.linalg.norm(sub_pos)
            penalty += np.maximum(0, dist_sub - self.radius) * 1e6
            
            # Substation spacing to turbines (must be at least 50m)
            d_sub_turb = np.linalg.norm(turb_coords - sub_pos, axis=1)
            penalty += np.sum(np.maximum(0, 50.0 - d_sub_turb)) * 1e6

        # AEP Bruto
        gross_aep = np.sum(calcAEP(turb_coords, self.wind_freq, self.wind_speed, self.wind_dir,
                                   self.turb_diam, self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr))
        
        # Cabling cost and losses
        combined_coords = np.vstack([turb_coords, sub_pos.reshape((1, 2))])
        sub_index = self.n_turb  # Substation is at the last index
        
        try:
            _, res = analisar_layout_completo(combined_coords, sub=sub_index, n_grupos=n_groups)
            capex = res["custo_total_usd"]
            losses = res["perda_anual_mwh"]
        except Exception:
            capex = 1e12
            losses = gross_aep
            
        net_aep = gross_aep - losses - penalty
        cost = capex + penalty
        
        return net_aep, cost

    # ── Phase 1: Optimize Layout ──────────────────────────────────────────────
    def run_phase1(self, max_gens=1000) -> tuple[np.ndarray, list[np.ndarray], list[float], list]:
        banner("PHASE 1: Layout Optimization (maximizing Gross AEP)")
        
        pool = multiprocessing.Pool()
        
        tb = base.Toolbox()
        tb.register("map", pool.map)
        tb.register("individual", lambda: creator.IndividualPhase1(self.initial_coords.flatten().tolist()))
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", self.evaluate_p1)
        tb.register("mate", tools.cxBlend, alpha=self.config.get("alpha", 0.5))
        
        def mutate_p1(ind):
            mu_val = self.config.get("mu", 0.0)
            for i in range(len(ind)):
                if random.random() < self.config["mutation_indpb"]:
                    ind[i] += random.gauss(mu_val, self.config["mutation_sigma"])
            enforce_circle_p1(ind, self.radius, self.n_turb)
            # Apply spacing repair
            coords_arr = np.array(ind).reshape((self.n_turb, 2))
            repaired = repair_spacing(coords_arr, self.min_spacing, self.radius)
            ind[:] = repaired.flatten().tolist()
            return ind,
            
        tb.register("mutate", mutate_p1)
        tb.register("select", tools.selTournament, tournsize=self.config.get("tournament_size", 5))
        
        pop = tb.population(n=self.config["population_size"])
        
        # Evaluate initial population
        fits = tb.map(tb.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit
            
        history_best = []
        p1_frames = []  # Keep best layout from each generation to build the GIF
        
        plateau_gens = self.config.get("plateau_generations_p1", self.config.get("plateau_generations", 50))
        
        gen = 0
        while gen < max_gens:
            gen += 1
            offspring = [tb.clone(ind) for ind in tb.select(pop, len(pop))]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config["crossover_probability"]:
                    tb.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
                    
            for mutant in offspring:
                if random.random() < self.config["mutation_probability"]:
                    tb.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            
            # Garante limites e repara espaçamento de todos os modificados pós crossover/mutação
            for ind in invalid:
                enforce_circle_p1(ind, self.radius, self.n_turb)
                coords_arr = np.array(ind).reshape((self.n_turb, 2))
                repaired = repair_spacing(coords_arr, self.min_spacing, self.radius)
                ind[:] = repaired.flatten().tolist()
                
            fits = tb.map(tb.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
                
            pop = tb.select(pop + offspring, self.config["population_size"])
            
            # Track best
            best_ind = tools.selBest(pop, 1)[0]
            best_aep = best_ind.fitness.values[0]
            history_best.append(best_aep)
            
            # Save coordinates for GIF frame
            best_coords = np.array(best_ind).reshape((self.n_turb, 2))
            p1_frames.append(best_coords)
            
            if gen % 10 == 0 or gen == 1:
                print(f"Gen {gen:>3}/{max_gens} | Best Gross AEP: {best_aep/1e3:.3f} GWh")
                
            # Plateau check: strict zero improvement over plateau_gens
            if gen >= plateau_gens:
                prev_best = history_best[gen - plateau_gens]
                improvement = (best_aep - prev_best) / prev_best
                if improvement <= 1e-7:  # Absolutely no change (allowing tiny numerical threshold)
                    print(f"\n[Convergence Detected] AEP did not improve for {plateau_gens} generations.")
                    print(f"Phase 1 finished at Gen {gen}.")
                    break
                    
        # Return best coordinates, coordinate frames, history, and population
        best_lay = np.array(tools.selBest(pop, 1)[0]).reshape((self.n_turb, 2))
        
        pool.close()
        pool.join()
        
        return best_lay, p1_frames, history_best, pop


    # ── Phase 2: Joint Co-design ──────────────────────────────────────────────
    def run_phase2(self, best_p1_layout, pop_p1, max_gens=1000) -> tuple[tools.ParetoFront, list[tuple[np.ndarray, np.ndarray, int]], list[float]]:
        banner("PHASE 2: Co-design Optimization (maximizing Net AEP & minimizing Cable Cost)")
        
        pool = multiprocessing.Pool()
        
        tb = base.Toolbox()
        tb.register("map", pool.map)
        tb.register("evaluate", self.evaluate_p2)
        tb.register("mate", tools.cxBlend, alpha=self.config.get("alpha", 0.5))
        
        def mutate_p2(ind):
            n_coords = self.n_turb * 2
            mu_val = self.config.get("mu", 0.0)
            # Mutate coordinates
            for i in range(n_coords):
                if random.random() < self.config["mutation_indpb"]:
                    ind[i] += random.gauss(mu_val, self.config["mutation_sigma"])
            # Mutate group gene
            if random.random() < self.config["mutation_indpb"]:
                ind[n_coords] += random.gauss(mu_val, 0.1)
            # Mutate substation if optimizing (with high sigma for wide search)
            if self.sub_mode == "optimize":
                if random.random() < self.config["mutation_indpb"]:
                    sub_sig = self.radius * 0.4
                    ind[n_coords+1] += random.gauss(mu_val, sub_sig)
                    ind[n_coords+2] += random.gauss(mu_val, sub_sig)
                    
            enforce_circle_p2(ind, self.radius, self.n_turb, self.sub_mode)
            
            # Spacing repair
            coords_arr = np.array(ind[:n_coords]).reshape((self.n_turb, 2))
            repaired = repair_spacing(coords_arr, self.min_spacing, self.radius)
            ind[:n_coords] = repaired.flatten().tolist()
            
            return ind,
            
        tb.register("mutate", mutate_p2)
        tb.register("select", tools.selNSGA2)
        
        # Smart Seeding (introducing substation spatial diversity)
        pop_p2 = []
        best_p1_inds = tools.selBest(pop_p1, int(len(pop_p1) * 0.3))
        for idx, ind in enumerate(best_p1_inds):
            coords = list(ind)
            g_norm = random.random()
            if self.sub_mode == "optimize":
                # 50% centroid, 50% random uniform position in circle
                if idx % 2 == 0:
                    centroid = np.mean(np.array(coords).reshape((self.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    r_sub = self.radius * np.sqrt(random.random())
                    theta_sub = random.random() * 2 * np.pi
                    sub_x = r_sub * np.cos(theta_sub)
                    sub_y = r_sub * np.sin(theta_sub)
            else:
                sub_x, sub_y = self.sub_fixed_pos[0], self.sub_fixed_pos[1]
            pop_p2.append(creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y]))
            
        # Complete the rest of the population with mutations for diversity
        while len(pop_p2) < self.config["population_size"]:
            parent = random.choice(best_p1_inds)
            coords = list(parent)
            g_norm = random.random()
            if self.sub_mode == "optimize":
                # 50% centroid, 50% random uniform position in circle
                if random.random() < 0.5:
                    centroid = np.mean(np.array(coords).reshape((self.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    r_sub = self.radius * np.sqrt(random.random())
                    theta_sub = random.random() * 2 * np.pi
                    sub_x = r_sub * np.cos(theta_sub)
                    sub_y = r_sub * np.sin(theta_sub)
            else:
                sub_x, sub_y = self.sub_fixed_pos[0], self.sub_fixed_pos[1]
            child = creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y])
            child, = mutate_p2(child)
            pop_p2.append(child)
            
        # Initial evaluation
        fits = tb.map(tb.evaluate, pop_p2)
        for ind, fit in zip(pop_p2, fits):
            ind.fitness.values = fit
            
        pop_p2 = [ind for ind in pop_p2 if ind.fitness.values[0] > 0]
        if not pop_p2:
            print("Error: No valid solutions after seeding. Regenerating random Phase 2 population.")
            # Fallback
            pop_p2 = []
            for _ in range(self.config["population_size"]):
                coords = self.initial_coords.flatten().tolist()
                g_norm = random.random()
                sub_x, sub_y = self.sub_fixed_pos[0], self.sub_fixed_pos[1]
                pop_p2.append(creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y]))
            fits = tb.map(tb.evaluate, pop_p2)
            for ind, fit in zip(pop_p2, fits):
                ind.fitness.values = fit
        
        hof = tools.ParetoFront()
        hof.update(pop_p2)
        
        n_coords = self.n_turb * 2
        history_net_best = []
        history_capex_best = []
        p2_frames = []  # List of tuples (turb_coords, sub_pos, n_groups)
        
        plateau_gens = self.config.get("plateau_generations_p2", self.config.get("plateau_generations", 50))
        
        gen = 0
        while gen < max_gens:
            gen += 1
            offspring = [tb.clone(ind) for ind in tb.select(pop_p2, len(pop_p2))]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.config["crossover_probability"]:
                    tb.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
                    
            for mutant in offspring:
                if random.random() < self.config["mutation_probability"]:
                    tb.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            
            # Garante limites e repara espaçamento de todos os modificados pós crossover/mutação
            for ind in invalid:
                enforce_circle_p2(ind, self.radius, self.n_turb, self.sub_mode)
                coords_arr = np.array(ind[:n_coords]).reshape((self.n_turb, 2))
                repaired = repair_spacing(coords_arr, self.min_spacing, self.radius)
                ind[:n_coords] = repaired.flatten().tolist()
                
            fits = tb.map(tb.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
                
            pop_p2 = tb.select(pop_p2 + offspring, self.config["population_size"])
            hof.update(pop_p2)
            
            # Track best extremes for monitoring
            hof_valid = [ind for ind in hof if ind.fitness.values[0] > 0]
            if hof_valid:
                best_ind = max(hof_valid, key=lambda x: x.fitness.values[0])
                best_cost_ind = min(hof_valid, key=lambda x: x.fitness.values[1])
                target_list = hof_valid
            else:
                best_ind = max(pop_p2, key=lambda x: x.fitness.values[0])
                best_cost_ind = min(pop_p2, key=lambda x: x.fitness.values[1])
                target_list = pop_p2
                
            best_net_aep = best_ind.fitness.values[0]
            best_capex = best_cost_ind.fitness.values[1]
            
            # Find Knee Point of this generation
            if len(target_list) > 1:
                gen_aeps = np.array([ind.fitness.values[0] for ind in target_list])
                gen_capexs = np.array([ind.fitness.values[1] for ind in target_list])
                
                aep_min, aep_max = np.min(gen_aeps), np.max(gen_aeps)
                capex_min, capex_max = np.min(gen_capexs), np.max(gen_capexs)
                
                aep_range = aep_max - aep_min if aep_max != aep_min else 1.0
                capex_range = capex_max - capex_min if capex_max != capex_min else 1.0
                
                aep_norm = (gen_aeps - aep_min) / aep_range
                capex_norm = (capex_max - gen_capexs) / capex_range
                
                dists = np.sqrt((1.0 - aep_norm)**2 + (1.0 - capex_norm)**2)
                knee_idx_gen = np.argmin(dists)
                visualized_ind = target_list[knee_idx_gen]
            else:
                visualized_ind = target_list[0]
                
            knee_net_aep = visualized_ind.fitness.values[0]
            knee_capex = visualized_ind.fitness.values[1]
            history_net_best.append(knee_net_aep)
            history_capex_best.append(knee_capex)
            
            # Save coordinates and sub position of Knee Point for GIF
            n_coords = self.n_turb * 2
            knee_turb_coords = np.array(visualized_ind[:n_coords]).reshape((self.n_turb, 2))
            g_norm = visualized_ind[n_coords]
            knee_n_groups = int(np.round(self.min_groups + g_norm * (self.max_groups - self.min_groups)))
            knee_n_groups = max(self.min_groups, min(self.n_turb, knee_n_groups))
            
            if self.sub_mode == "optimize":
                knee_sub_pos = np.array([visualized_ind[n_coords+1], visualized_ind[n_coords+2]])
            else:
                knee_sub_pos = self.sub_fixed_pos
                
            p2_frames.append((knee_turb_coords, knee_sub_pos, knee_n_groups))
            
            if gen % 10 == 0 or gen == 1:
                print(f"Gen {gen:>3}/{max_gens} | Best Net AEP: {best_net_aep/1e3:.3f} GWh | Min Cabling: ${best_capex/1e3:.1f} kUSD")
                
            # Plateau check: strict zero improvement in BOTH objectives of the Knee Point over plateau_gens
            if gen >= plateau_gens:
                prev_knee_net = history_net_best[gen - plateau_gens]
                prev_knee_capex = history_capex_best[gen - plateau_gens]
                
                net_improvement = (knee_net_aep - prev_knee_net) / prev_knee_net
                # Cabling is minimized, so improvement is prev_capex > current_capex
                capex_improvement = (prev_knee_capex - knee_capex) / prev_knee_capex
                
                if net_improvement <= 1e-7 and capex_improvement <= 1e-7:
                    print(f"\n[Convergence Detected] Neither Net AEP nor Cabling CAPEX of the Knee Point improved for {plateau_gens} generations.")
                    print(f"Phase 2 finished at Gen {gen}.")
                    break
                    
        pool.close()
        pool.join()
        
        return hof, p2_frames, history_net_best, history_capex_best


    # ── Render animations ────────────────────────────────────────────────────────
    def generate_animations(self, p1_frames, p2_frames, h_p1_aep, h_p2_net, h_p2_capex=None):
        banner("Rendering Animation GIFs")
        
        total_p1 = len(p1_frames)
        total_p2 = len(p2_frames)
        total_frames = total_p1 + total_p2
        
        # Determine sampling rate to keep the GIF fluid and around 150 frames max
        skip_step = max(1, total_frames // 150)
        
        # 1. Physical Evolution GIF
        print("Rendering evolution_layout.gif...")
        frames_layout = []
        for idx in range(0, total_frames, skip_step):
            plt.close('all')
            fig, ax = plt.subplots(figsize=(9.5, 7), facecolor="#0a1628")
            ax.set_facecolor("#0a1628")
            
            boundary = Circle((0, 0), self.radius, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2)
            ax.add_patch(boundary)
            
            ax.set_xlim(-self.radius - 200, self.radius + 200)
            ax.set_ylim(-self.radius - 200, self.radius + 200)
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
                            wind_direction=270.0,
                            wind_speed_free_stream=float(self.wind_speed[0]),
                            turbine_diameter=self.turb_diam)
            limit_val = self.radius + self.turb_diam * 2
            X, Y, V = farm.get_velocity_field(resolution=100, x_bounds=(-limit_val, limit_val), y_bounds=(-limit_val, limit_val))
            ax.contourf(X, Y, V, levels=30, cmap="coolwarm_r", alpha=0.85, zorder=1)
            
            if phase == 2:
                combined_coords = np.vstack([coords, sub_pos.reshape((1, 2))])
                sub_index = self.n_turb
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
                aep = np.sum(calcAEP(coords, self.wind_freq, self.wind_speed, self.wind_dir,
                                     self.turb_diam, self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr))
                ax.set_title("Phase 1 — Layout Optimization", color="white", fontsize=11, fontweight="bold")
                ax.text(1.05, 0.95, f"★ Phase 1 Stats ★\nGross AEP: {aep/1e3:.2f} GWh", transform=ax.transAxes, ha="left", va="top",
                        color="#22c55e", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
            else:
                gross_aep = np.sum(calcAEP(coords, self.wind_freq, self.wind_speed, self.wind_dir,
                                           self.turb_diam, self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr))
                net_aep = gross_aep - losses
                ax.set_title("Phase 2 — Co-design Optimization", color="white", fontsize=11, fontweight="bold")
                
                # Display Phase 1 Stats on top as seed/reference
                p1_final_aep = h_p1_aep[-1]
                ax.text(1.05, 0.95, f"★ Phase 1 Seed ★\nGross AEP: {p1_final_aep/1e3:.2f} GWh", transform=ax.transAxes, ha="left", va="top",
                        color="#22c55e", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
                
                # Display Phase 2 Stats stacked below it
                ax.text(1.05, 0.72, f"★ Phase 2 Stats ★\nNet AEP: {net_aep/1e3:.2f} GWh\nCAPEX: ${capex/1e3:.1f}k\nGroups: {n_groups}", transform=ax.transAxes, ha="left", va="top",
                        color="#f59e0b", fontweight="bold", bbox=dict(facecolor="#0f2038", edgecolor="#1e3050", alpha=0.85))
                
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a1628")
            plt.close(fig)
            buf.seek(0)
            frames_layout.append(Image.open(buf).copy())
            buf.close()
            
        gif_path_layout = os.path.join(self.output_dir, "evolution_layout.gif")
        frames_layout[0].save(gif_path_layout, save_all=True, append_images=frames_layout[1:], duration=150, loop=0, optimize=False, disposal=2)
        print(f"Saved: {gif_path_layout}")
        
        # 2. AEP & CAPEX Evolution Graph GIF
        print("Rendering aep_evolution.gif...")
        frames_aep = []
        x1 = list(range(1, len(h_p1_aep) + 1))
        y1 = [v/1e3 for v in h_p1_aep]
        
        x2 = list(range(len(h_p1_aep), len(h_p1_aep) + len(h_p2_net) + 1))
        y2 = [h_p1_aep[-1]/1e3] + [v/1e3 for v in h_p2_net]
        
        all_y = y1 + y2
        y_min_val, y_max_val = min(all_y), max(all_y)
        
        if h_p2_capex is not None:
            y_cost = [c / 1e3 for c in h_p2_capex]
            y_cost = [y_cost[0]] + y_cost
            cost_min, cost_max = min(y_cost), max(y_cost)
            cost_range = cost_max - cost_min if cost_max != cost_min else 1.0
            cost_ylim = (cost_min - 0.05 * cost_range, cost_max + 0.05 * cost_range)
        
        for idx in range(0, total_frames, skip_step):
            plt.close('all')
            fig, ax = plt.subplots(figsize=(8, 5.0), facecolor="#0a1628")
            ax.set_facecolor("#0f2038")
            
            ax.set_xlim(0, total_frames + 5)
            ax.set_ylim(y_min_val * 0.95, y_max_val * 1.05)
            
            ax_cost = None
            if h_p2_capex is not None:
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
            
            # Combine legends from both axes and place them at the bottom outside the plot
            lines, labels = ax.get_legend_handles_labels()
            if ax_cost is not None:
                lines2, labels2 = ax_cost.get_legend_handles_labels()
                lines += lines2
                labels += labels2
            
            leg = ax.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, facecolor="#0a1628", edgecolor="#1e3050")
            if leg:
                for text in leg.get_texts():
                    text.set_color("white")
            for s in ax.spines.values():
                s.set_edgecolor("#1e3050")
                
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100, bbox_inches="tight", facecolor="#0a1628")
            plt.close(fig)
            buf.seek(0)
            frames_aep.append(Image.open(buf).copy())
            buf.close()
            
        gif_path_aep = os.path.join(self.output_dir, "aep_evolution.gif")
        frames_aep[0].save(gif_path_aep, save_all=True, append_images=frames_aep[1:], duration=150, loop=0, optimize=True)
        print(f"Saved: {gif_path_aep}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Instantiate custom wind farm simulation with cabling co-design.")
    parser.add_argument("--case", type=str, default="cases/case_example.yaml", help="Path to YAML configuration case file")
    parser.add_argument("--output", type=str, default="results/user_run", help="Output directory to save results")
    parser.add_argument("--max-gens", type=int, default=1000, help="Maximum generations per phase (default: 1000)")
    args = parser.parse_args()
    
    t0 = time.time()
    
    sim = WindFarmSimulator(args.case, args.output)
    
    # 1. Run Phase 1
    best_p1, p1_frames, h_p1_aep, pop_p1 = sim.run_phase1(max_gens=args.max_gens)
    
    # 2. Run Phase 2
    hof_p2, p2_frames, h_p2_net, h_p2_capex = sim.run_phase2(best_p1, pop_p1, max_gens=args.max_gens)
    
    # 3. Generate GIF animations
    sim.generate_animations(p1_frames, p2_frames, h_p1_aep, h_p2_net, h_p2_capex)
    
    # 4. Generate AEP & CAPEX Evolution Plot
    print("\nGenerating aep_evolution.png plot...")
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor="#0a1628")
    ax.set_facecolor("#0f2038")
    
    # Phase 1 trace
    x1 = list(range(1, len(h_p1_aep) + 1))
    ax.plot(x1, [v/1e3 for v in h_p1_aep], color="#3b82f6", lw=2.5, label="Phase 1 — Gross AEP")
    
    # Phase 2 trace
    x2 = list(range(len(h_p1_aep), len(h_p1_aep) + len(h_p2_net) + 1))
    y2 = [h_p1_aep[-1]/1e3] + [v/1e3 for v in h_p2_net]
    ax.plot(x2, y2, color="#22c55e", lw=2.5, label="Phase 2 — Net AEP (with cable losses)")
    
    # Cabling Cost trace on twin Y-axis
    ax_cost = ax.twinx()
    y_cost = [c / 1e3 for c in h_p2_capex]
    y_cost = [y_cost[0]] + y_cost
    ax_cost.plot(x2, y_cost, color="#ef4444", lw=2.0, linestyle="-", label="Phase 2 — Cabling CAPEX")
    
    ax_cost.set_ylabel("Cabling CAPEX [kUSD]", color="#ef4444", fontsize=10)
    ax_cost.tick_params(colors="#ef4444", labelsize=8)
    ax_cost.grid(False)
    
    # Phase separator line
    ax.axvline(len(h_p1_aep), color="#f59e0b", lw=1.5, linestyle="--", alpha=0.8)
    
    ax.set_title("Two-Phase Optimization — AEP & CAPEX Evolution", color="white", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Generation", color="white", fontsize=10)
    ax.set_ylabel("AEP [GWh]", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=8)
    ax.grid(True, color="#1e3050", lw=0.5, linestyle=":")
    
    # Combine legends from both axes and place them at the bottom outside the plot
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_cost.get_legend_handles_labels()
    
    leg = ax.legend(lines + lines2, labels + labels2, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, facecolor="#0a1628", edgecolor="#1e3050")
    for text in leg.get_texts():
        text.set_color("white")
        
    for s in ax.spines.values():
        s.set_edgecolor("#1e3050")
        
    plt.tight_layout()
    plot_path = os.path.join(args.output, "aep_evolution.png")
    plt.savefig(plot_path, dpi=150, facecolor="#0a1628")
    plt.close()
    print(f"Saved: {plot_path}")
    
    # 5. Generate Pareto Front Plot and identify Knee Point
    valid_solutions = [ind for ind in hof_p2 if ind.fitness.values[0] > 0]
    if valid_solutions:
        print("\nGenerating pareto_front.png plot...")
        aeps = np.array([ind.fitness.values[0] / 1e3 for ind in valid_solutions]) # GWh
        capexs = np.array([ind.fitness.values[1] for ind in valid_solutions]) # USD
        
        # Identify Knee Point using normalized distance to utopian point (max AEP, min CAPEX)
        if len(valid_solutions) > 1:
            aep_min, aep_max = np.min(aeps), np.max(aeps)
            capex_min, capex_max = np.min(capexs), np.max(capexs)
            
            aep_range = aep_max - aep_min if aep_max != aep_min else 1.0
            capex_range = capex_max - capex_min if capex_max != capex_min else 1.0
            
            aep_norm = (aeps - aep_min) / aep_range
            capex_norm = (capex_max - capexs) / capex_range
            
            dists = np.sqrt((1.0 - aep_norm)**2 + (1.0 - capex_norm)**2)
            knee_idx = np.argmin(dists)
        else:
            knee_idx = 0
            
        knee_ind = valid_solutions[knee_idx]
        knee_aep = aeps[knee_idx]
        knee_capex = capexs[knee_idx]
        
        # Plotting Pareto Front
        fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0a1628")
        ax.set_facecolor("#0f2038")
        
        sort_indices = np.argsort(capexs)
        ax.plot(capexs[sort_indices] / 1e3, aeps[sort_indices], color="#38bdf8", alpha=0.5, lw=2, linestyle="-")
        ax.scatter(capexs / 1e3, aeps, color="#38bdf8", edgecolors="#0f2038", s=60, zorder=3, label="Pareto Solutions")
        
        # Highlight Knee Point
        ax.scatter(knee_capex / 1e3, knee_aep, color="#f59e0b", edgecolors="white", marker="*", s=250, zorder=5, label="Knee Point (Ideal Compromise)")
        ax.axhline(knee_aep, color="#f59e0b", linestyle=":", alpha=0.6, lw=1.2)
        ax.axvline(knee_capex / 1e3, color="#f59e0b", linestyle=":", alpha=0.6, lw=1.2)
        
        # Substation and grouping info for Knee Point
        g_norm_knee = knee_ind[sim.n_turb * 2]
        knee_grp = int(np.round(sim.min_groups + g_norm_knee * (sim.max_groups - sim.min_groups)))
        knee_grp = max(sim.min_groups, min(sim.n_turb, knee_grp))
        
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
            
        plt.tight_layout()
        pareto_path = os.path.join(args.output, "pareto_front.png")
        plt.savefig(pareto_path, dpi=150, facecolor="#0a1628")
        plt.close()
        print(f"Saved: {pareto_path}")
        
        # Generate Knee Point Layout Plot (knee_layout.png)
        print("\nGenerating knee_layout.png plot...")
        n_coords = sim.n_turb * 2
        knee_coords = np.array(knee_ind[:n_coords]).reshape((sim.n_turb, 2))
        
        if sim.sub_mode == "optimize":
            knee_sub_pos = np.array([knee_ind[n_coords+1], knee_ind[n_coords+2]])
        else:
            knee_sub_pos = sim.sub_fixed_pos
            
        fig, ax = plt.subplots(figsize=(10.5, 8), facecolor="#0a1628")
        ax.set_facecolor("#0a1628")
        
        boundary = Circle((0, 0), sim.radius, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2)
        ax.add_patch(boundary)
        
        # Calculate and plot wake velocity field
        farm = WindFarm(knee_coords,
                        wind_direction=270.0,
                        wind_speed_free_stream=float(sim.wind_speed[0]),
                        turbine_diameter=sim.turb_diam)
        limit_val = sim.radius + sim.turb_diam * 2
        X, Y, V = farm.get_velocity_field(resolution=100, x_bounds=(-limit_val, limit_val), y_bounds=(-limit_val, limit_val))
        ax.contourf(X, Y, V, levels=30, cmap="coolwarm_r", alpha=0.85, zorder=1)
        
        # Cable routing
        combined_coords = np.vstack([knee_coords, knee_sub_pos.reshape((1, 2))])
        sub_index = sim.n_turb
        try:
            planta, res = analisar_layout_completo(combined_coords, sub=sub_index, n_grupos=knee_grp)
            capex_val = res["custo_total_usd"]
            losses_val = res["perda_anual_mwh"]
            
            cmap = plt.cm.tab10
            for string_idx, path in enumerate(planta.paths):
                xs = [combined_coords[k, 0] for k in path]
                ys = [combined_coords[k, 1] for k in path]
                ax.plot(xs, ys, "-", color=cmap(string_idx % 10), lw=2.0, zorder=3)
        except Exception as e:
            print(f"Error computing cable routing for knee layout: {e}")
            capex_val = 1e12
            losses_val = 0.0
            
        # Draw turbines
        ax.scatter(knee_coords[:, 0], knee_coords[:, 1], s=50, color="white", edgecolors="#0a1628", linewidths=0.8, zorder=5)
        
        # Draw substation
        ax.scatter(knee_sub_pos[0], knee_sub_pos[1], s=200, color="#f97316", marker="s", edgecolors="white", label="Substation", zorder=10)
        
        # Set bounds and styling
        ax.set_xlim(-sim.radius - 200, sim.radius + 200)
        ax.set_ylim(-sim.radius - 200, sim.radius + 200)
        ax.set_aspect("equal")
        ax.grid(True, color="#1e3050", lw=0.4)
        ax.tick_params(colors="white", labelsize=8)
        ax.set_xlabel("X [m]", color="white", fontsize=9)
        ax.set_ylabel("Y [m]", color="white", fontsize=9)
        for s in ax.spines.values():
            s.set_edgecolor("#1e3050")
            
        # Detailed stats info box (placed on the right side, outside the circular boundary)
        stats_text = (f"★ Knee Point Layout Stats ★\n"
                      f"Net AEP: {knee_aep:.3f} GWh\n"
                      f"Cable CAPEX: ${capex_val:,.2f} USD\n"
                      f"Annual Losses: {losses_val:.2f} MWh\n"
                      f"Cable Groups: {knee_grp}\n"
                      f"Substation: [{knee_sub_pos[0]:.1f}, {knee_sub_pos[1]:.1f}]")
        ax.text(1.05, 0.95, stats_text, transform=ax.transAxes, ha="left", va="top",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="#0f2038", edgecolor="#f59e0b", boxstyle="round,pad=0.6", alpha=0.9), zorder=15)
                
        ax.set_title("Knee Point Co-design Layout (Best Compromise)", color="white", fontsize=12, fontweight="bold", pad=12)
        
        plt.tight_layout()
        knee_layout_path = os.path.join(args.output, "knee_layout.png")
        plt.savefig(knee_layout_path, dpi=150, facecolor="#0a1628")
        plt.close()
        print(f"Saved: {knee_layout_path}")
    else:
        knee_ind = None
        
    # Show final top solutions
    print("\n" + "═"*50)
    print("  SIMULATION COMPLETE — Pareto Front Solutions")
    print("═"*50)
    
    valid_solutions.sort(key=lambda x: x.fitness.values[0], reverse=True)
    for idx, ind in enumerate(valid_solutions[:5]):
        net_aep = ind.fitness.values[0]
        capex = ind.fitness.values[1]
        n_coords = sim.n_turb * 2
        g_norm = ind[n_coords]
        n_grp = int(np.round(sim.min_groups + g_norm * (sim.max_groups - sim.min_groups)))
        n_grp = max(sim.min_groups, min(sim.n_turb, n_grp))
        is_knee = " ★ KNEE POINT (Best Compromise)" if ind == knee_ind else ""
        print(f"  [{idx+1}] Net AEP: {net_aep/1e3:.3f} GWh | Cable CAPEX: ${capex:,.2f} USD | Groups: {n_grp}{is_knee}")
        
    # 6. Save detailed data logs (Pareto solutions and convergence histories)
    import json
    
    pareto_data = []
    for idx, ind in enumerate(valid_solutions):
        net_aep = ind.fitness.values[0]
        capex = ind.fitness.values[1]
        n_coords = sim.n_turb * 2
        turb_coords = np.array(ind[:n_coords]).reshape((sim.n_turb, 2)).tolist()
        g_norm = ind[n_coords]
        n_grp = int(np.round(sim.min_groups + g_norm * (sim.max_groups - sim.min_groups)))
        n_grp = max(sim.min_groups, min(sim.n_turb, n_grp))
        
        if sim.sub_mode == "optimize":
            sub_pos = [ind[n_coords+1], ind[n_coords+2]]
        else:
            sub_pos = sim.sub_fixed_pos.tolist()
            
        is_knee = bool(ind == knee_ind)
        
        pareto_data.append({
            "rank_by_aep": idx + 1,
            "net_aep_gwh": net_aep / 1e3,
            "cable_capex_usd": capex,
            "cable_groups": n_grp,
            "substation_position": sub_pos,
            "turbine_coordinates": turb_coords,
            "is_knee_point": is_knee
        })
        
    json_path = os.path.join(args.output, "pareto_solutions.json")
    with open(json_path, "w") as f:
        json.dump(pareto_data, f, indent=4)
        
    history_data = {
        "phase1_gross_aep_gwh": [v / 1e3 for v in h_p1_aep],
        "phase2_net_aep_gwh": [v / 1e3 for v in h_p2_net]
    }
    history_path = os.path.join(args.output, "convergence_history.json")
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=4)
        
    print(f"\nSaved detailed Pareto data to: {json_path}")
    print(f"Saved convergence history to: {history_path}")
    print(f"\nExecution time: {(time.time() - t0)/60:.2f} minutes")
    print(f"Plots, GIFs and logs saved in directory: {args.output}")

