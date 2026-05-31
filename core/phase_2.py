import os
import sys
import numpy as np
import random
import multiprocessing
from deap import base, creator, tools, algorithms

# Setup path so we can import from config
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.aep import calcAEP, getWindRoseYAML, getTurbAtrbtYAML
from core.boundary import SiteBoundary
from core.cabling_v3 import analisar_layout_completo

def repair_spacing(coords, min_spacing, boundary, max_iterations=10):
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

            # Clamp back inside polygon boundary
            for k in [i, j]:
                coords[k, 0], coords[k, 1] = boundary.enforce(
                    coords[k, 0], coords[k, 1]
                )
    return coords

class Phase2Optimizer:
    def __init__(self, config, best_p1_coords, pop_p1, max_gens=1000):
        self.config = config
        self.best_p1_coords = best_p1_coords
        self.pop_p1 = pop_p1
        self.max_gens = max_gens
        
        # Load paths from config (relative to ROOT)
        turb_yaml = os.path.join(ROOT, self.config["turbine_yaml"])
        wind_yaml = os.path.join(ROOT, self.config["windrose_yaml"])
        geojson_path = os.path.join(ROOT, self.config["boundary_geojson"])

        # Pre-load data
        self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr, self.turb_diam = getTurbAtrbtYAML(turb_yaml)
        self.wind_dir, self.wind_freq, self.wind_speed = getWindRoseYAML(wind_yaml)
        self.boundary = SiteBoundary.from_geojson(geojson_path)

        # Geometry & Substation
        self.n_turb = self.config.get("n_turbines", 16)
        spacing_mult = self.config.get("min_spacing_multiplier", 2.0)
        self.min_spacing = self.turb_diam * spacing_mult
        
        sub_cfg = self.config.get("substation")
        if isinstance(sub_cfg, list):
            self.sub_mode = "fixed"
            self.sub_fixed_pos = np.array(sub_cfg)
        elif isinstance(sub_cfg, dict):
            self.sub_mode = sub_cfg.get("mode", "optimize")
            self.sub_fixed_pos = np.array(sub_cfg.get("fixed_pos", [-1350.0, 0.0]))
        else:
            self.sub_mode = "optimize"
            self.sub_fixed_pos = np.array([-1350.0, 0.0])
            
        cable_cfg = self.config.get("cable_groups", {})
        self.min_groups = cable_cfg.get("min_groups", 2)
        self.max_groups = cable_cfg.get("max_groups", 16)
        
        # Setup DEAP
        if not hasattr(creator, "FitnessMultiP2"):
            creator.create("FitnessMultiP2", base.Fitness, weights=(1.0, -1.0)) # Max Net AEP, Min Cost
        if not hasattr(creator, "IndividualPhase2"):
            creator.create("IndividualPhase2", list, fitness=creator.FitnessMultiP2)
        
    def _evaluate_p2(self, individual):
        n_coords = self.n_turb * 2
        turb_coords = np.array(individual[:n_coords]).reshape((self.n_turb, 2))
        g_norm = individual[n_coords]
        
        n_groups = int(np.round(self.min_groups + g_norm * (self.max_groups - self.min_groups)))
        n_groups = max(self.min_groups, min(self.n_turb, n_groups))
        
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

        # 2. Boundary (polygon)
        outside = np.array([
            not self.boundary.contains(turb_coords[i, 0], turb_coords[i, 1])
            for i in range(self.n_turb)
        ])
        penalty += np.sum(outside) * 1e6

        # 3. Substation boundary
        if self.sub_mode == "optimize":
            if not self.boundary.contains(sub_pos[0], sub_pos[1]):
                penalty += 1e6
            d_sub_turb = np.linalg.norm(turb_coords - sub_pos, axis=1)
            penalty += np.sum(np.maximum(0, 50.0 - d_sub_turb)) * 1e6

        # AEP Bruto
        gross_aep = np.sum(calcAEP(turb_coords, self.wind_freq, self.wind_speed, self.wind_dir,
                                   self.turb_diam, self.turb_ci, self.turb_co, self.rated_ws, self.rated_pwr))
        
        # Cabling
        combined_coords = np.vstack([turb_coords, sub_pos.reshape((1, 2))])
        sub_index = self.n_turb
        
        try:
            planta, res = analisar_layout_completo(combined_coords, sub=sub_index, n_grupos=n_groups)
            capex = res["custo_total_usd"]
            losses = res["perda_anual_mwh"]
            
            # Check for cable intersections with boundaries/holes
            from shapely.geometry import LineString
            for path in planta.paths:
                for i in range(len(path) - 1):
                    p1 = combined_coords[path[i]]
                    p2 = combined_coords[path[i+1]]
                    line = LineString([p1, p2])
                    # If the cable crosses a hole or goes outside the site, penalize heavily
                    if not self.boundary._poly.contains(line):
                        penalty += 1e6
                        
        except Exception:
            capex = 1e12
            losses = gross_aep
            
        net_aep = gross_aep - losses - penalty
        cost = capex + penalty
        
        return net_aep, cost

    def _mutate_p2(self, ind):
        n_coords = self.n_turb * 2
        mu_val = self.config.get("mu", 0.0)

        for i in range(n_coords):
            if random.random() < self.config.get("mutation_indpb", 0.4):
                ind[i] += random.gauss(mu_val, self.config.get("mutation_sigma", 100))

        if random.random() < self.config.get("mutation_indpb", 0.4):
            ind[n_coords] += random.gauss(mu_val, 0.1)

        if self.sub_mode == "optimize":
            if random.random() < self.config.get("mutation_indpb", 0.4):
                xmin, ymin, xmax, ymax = self.boundary.bbox
                sub_range = max(xmax - xmin, ymax - ymin) * 0.4
                ind[n_coords+1] += random.gauss(mu_val, sub_range)
                ind[n_coords+2] += random.gauss(mu_val, sub_range)

        # Enforce polygon boundary for turbines
        for i in range(self.n_turb):
            ind[2*i], ind[2*i+1] = self.boundary.enforce(ind[2*i], ind[2*i+1])

        # Clamp g_norm
        ind[n_coords] = max(0.0, min(1.0, ind[n_coords]))

        # Enforce boundary for substation
        if self.sub_mode == "optimize":
            ind[n_coords+1], ind[n_coords+2] = self.boundary.enforce(
                ind[n_coords+1], ind[n_coords+2]
            )
        return ind,

    def run(self):
        tb = base.Toolbox()
        pool = multiprocessing.Pool()
        tb.register("map", pool.map)
        tb.register("evaluate", self._evaluate_p2)
        tb.register("mate", tools.cxBlend, alpha=self.config.get("alpha", 0.5))
        tb.register("mutate", self._mutate_p2)
        tb.register("select", tools.selNSGA2)
        
        # Smart Seeding
        pop_p2 = []
        best_p1_inds = tools.selBest(self.pop_p1, int(len(self.pop_p1) * 0.3)) if self.pop_p1 else []
        
        # Garante que o hof[0] absoluto da Fase 1 esteja na semente (como índice 0)
        hof_flat = self.best_p1_coords.flatten().tolist()
        best_p1_inds.insert(0, hof_flat)
        
        for idx, ind in enumerate(best_p1_inds):
            coords = list(ind)
            g_norm = random.random()
            if self.sub_mode == "optimize":
                if idx % 2 == 0:
                    centroid = np.mean(np.array(coords).reshape((self.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    sub_x, sub_y = self.boundary.random_point()
            else:
                sub_x, sub_y = self.sub_fixed_pos[0], self.sub_fixed_pos[1]
            pop_p2.append(creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y]))
            
        while len(pop_p2) < self.config.get("population_size", 300):
            parent = random.choice(best_p1_inds)
            coords = list(parent)
            g_norm = random.random()
            if self.sub_mode == "optimize":
                if random.random() < 0.5:
                    centroid = np.mean(np.array(coords).reshape((self.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    sub_x, sub_y = self.boundary.random_point()
            else:
                sub_x, sub_y = self.sub_fixed_pos[0], self.sub_fixed_pos[1]
            child = creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y])
            child, = self._mutate_p2(child)
            pop_p2.append(child)
            
        # Initial evaluation
        fits = tb.map(tb.evaluate, pop_p2)
        for ind, fit in zip(pop_p2, fits):
            ind.fitness.values = fit

        hof = tools.ParetoFront()
        hof.update(pop_p2)
        
        history_net_best = []
        history_capex_best = []
        p2_frames = [] 
        
        plateau_gens = self.config.get("plateau_generations_p2", 200)
        cxpb = self.config.get("crossover_probability", 0.95)
        mutpb = self.config.get("mutation_probability", 0.70)
        pop_size = self.config.get("population_size", 300)
        n_coords = self.n_turb * 2

        gen = 0
        while gen < self.max_gens:
            gen += 1
            offspring = [tb.clone(ind) for ind in tb.select(pop_p2, len(pop_p2))]
            
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb:
                    tb.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
                    
            for mutant in offspring:
                if random.random() < mutpb:
                    tb.mutate(mutant)
                    del mutant.fitness.values
                    
            invalid = [ind for ind in offspring if not ind.fitness.valid]
            
            # Enforce constraints directly
            for ind in invalid:
                # Enforce polygon boundary
                for i in range(self.n_turb):
                    ind[2*i], ind[2*i+1] = self.boundary.enforce(ind[2*i], ind[2*i+1])
                ind[n_coords] = max(0.0, min(1.0, ind[n_coords]))
                if self.sub_mode == "optimize":
                    ind[n_coords+1], ind[n_coords+2] = self.boundary.enforce(
                        ind[n_coords+1], ind[n_coords+2]
                    )
                
            fits = tb.map(tb.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit
                
            pop_p2 = tb.select(pop_p2 + offspring, pop_size)
            hof.update(pop_p2)
            
            # Monitoring
            hof_valid = [ind for ind in hof if ind.fitness.values[0] > 0]
            target_list = hof_valid if hof_valid else pop_p2
            
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
                knee_idx = np.argmin(dists)
                visualized_ind = target_list[knee_idx]
            else:
                visualized_ind = target_list[0]
                
            knee_net_aep = visualized_ind.fitness.values[0]
            knee_capex = visualized_ind.fitness.values[1]
            history_net_best.append(knee_net_aep)
            history_capex_best.append(knee_capex)
            
            # Save frames
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
                print(f"Gen {gen:>3}/{self.max_gens} | Best Net AEP: {knee_net_aep/1e3:.3f} GWh | Min Cabling: ${knee_capex/1e3:.1f} kUSD")
                
            if gen >= plateau_gens:
                prev_net = history_net_best[gen - plateau_gens]
                prev_capex = history_capex_best[gen - plateau_gens]
                
                net_imp = (knee_net_aep - prev_net) / prev_net if prev_net > 0 else 0
                capex_imp = (prev_capex - knee_capex) / prev_capex if prev_capex > 0 else 0
                
                if net_imp <= 1e3 and capex_imp <= 1e-3: # Watts e kUSD
                    print(f"\n[Convergence Detected] Neither Net AEP nor Cabling CAPEX improved for {plateau_gens} generations.")
                    print(f"Phase 2 finished at Gen {gen}.")
                    break
                    
        pool.close()
        pool.join()
        
        return hof, p2_frames, history_net_best, history_capex_best

def run_phase_2(config, best_p1_coords, pop_p1, max_gens=1000):
    optimizer = Phase2Optimizer(config, best_p1_coords, pop_p1, max_gens)
    return optimizer.run()
