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

class Phase1Optimizer:
    def __init__(self, config, max_gens=1000):
        self.config = config
        self.max_gens = max_gens
        
        # Load paths from config (relative to ROOT)
        turb_yaml = os.path.join(ROOT, self.config["turbine_yaml"])
        wind_yaml = os.path.join(ROOT, self.config["windrose_yaml"])
        geojson_path = os.path.join(ROOT, self.config["boundary_geojson"])

        # Pre-load data
        self.turb_atrbt_data = getTurbAtrbtYAML(turb_yaml)
        self.wind_rose_data = getWindRoseYAML(wind_yaml)
        self.boundary = SiteBoundary.from_geojson(geojson_path)

        # Extract constants
        self.n_turb = int(self.config["n_turbines"])
        turb_diam = self.turb_atrbt_data[4]
        spacing_mult = self.config.get("min_spacing_multiplier", 2.0)
        self.min_spacing = turb_diam * spacing_mult
        
        # Setup DEAP
        # We use a unique name for the individual class to avoid conflicts if Phase 2 runs in same process
        if not hasattr(creator, "FitnessMaxP1"):
            creator.create("FitnessMaxP1", base.Fitness, weights=(1.0,))
        if not hasattr(creator, "IndividualP1"):
            creator.create("IndividualP1", list, fitness=creator.FitnessMaxP1)
        
    def _evaluate_otimizado(self, individual):
        # Desempacota os dados previamente carregados
        turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = self.turb_atrbt_data
        wind_dir, wind_freq, wind_speed = self.wind_rose_data

        # Converte o indivíduo para coordenadas de turbinas
        turb_coords = np.array(individual).reshape((self.n_turb, 2))
        
        penalty_outside = 0
        penalty_close_turbines = 0

        # Penaliza turbinas fora da fronteira poligonal
        mask_inside = np.array([
            self.boundary.contains(turb_coords[i, 0], turb_coords[i, 1])
            for i in range(self.n_turb)
        ])
        penalty_outside = np.sum(~mask_inside) * 1e6

        # Penaliza turbinas muito próximas: vetorize o cálculo das distâncias
        if self.n_turb > 1:
            diff = turb_coords.reshape(self.n_turb, 1, 2) - turb_coords.reshape(1, self.n_turb, 2)
            dist_matrix = np.linalg.norm(diff, axis=2)
            i_upper, j_upper = np.triu_indices(self.n_turb, k=1)
            close_mask = dist_matrix[i_upper, j_upper] < self.min_spacing
            penalty_close_turbines = np.sum(close_mask) * 1e6

        # Calcula o AEP
        aep = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                      turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
        
        fitness = np.sum(aep) - penalty_outside - penalty_close_turbines
        return fitness,

    def _mutate(self, individual, mu, sigma, indpb):
        if random.random() < indpb:
            for i in range(len(individual)):
                individual[i] += random.gauss(mu, sigma)
            # Enforce boundary for each turbine
            for i in range(self.n_turb):
                x, y = individual[2*i], individual[2*i + 1]
                individual[2*i], individual[2*i + 1] = self.boundary.enforce(x, y)
        return individual, 
        
    def _generate_grid_layout(self):
        """Generates a regular grid layout of n_turbines inside the polygon."""
        xmin, ymin, xmax, ymax = self.boundary.bbox
        max_dim = max(xmax - xmin, ymax - ymin)
        spacing = max_dim
        
        best_points = []
        step = max_dim / 2.0
        
        # Binary search for the perfect grid spacing
        for _ in range(50):
            xs = np.arange(xmin + spacing/2, xmax, spacing)
            ys = np.arange(ymin + spacing/2, ymax, spacing)
            
            valid_points = []
            for x in xs:
                for y in ys:
                    if self.boundary.contains(x, y):
                        valid_points.append((x, y))
                        
            if len(valid_points) >= self.n_turb:
                best_points = valid_points
                spacing += step # Try a larger spacing to spread them out more
            else:
                spacing -= step # Try a smaller spacing to fit more points
            step /= 2.0
            
        if len(best_points) >= self.n_turb:
            # Sort by distance to center to keep the layout compact and deterministic
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            best_points.sort(key=lambda p: (p[0]-cx)**2 + (p[1]-cy)**2)
            
            coords = []
            for p in best_points[:self.n_turb]:
                coords.extend([p[0], p[1]])
            return coords
        else:
            raise ValueError(f"Could not fit {self.n_turb} turbines in the provided polygon. The area might be too small or the number of turbines too large.")

    def _create_individual(self, coords):
        return creator.IndividualP1(np.array(coords).flatten().tolist())

    def run(self):
        tb = base.Toolbox()
        pool = multiprocessing.Pool()
        tb.register("map", pool.map)
        
        # Generate the single well-spread seed layout
        seed_coords = self._generate_grid_layout()
        
        # Setup population starting entirely from this homogenous seed
        tb.register("individual", self._create_individual, coords=seed_coords)
        tb.register("population", tools.initRepeat, list, tb.individual)
        
        tb.register("evaluate", self._evaluate_otimizado)
        tb.register("mate", tools.cxBlend, alpha=self.config.get("alpha", 0.5))
        
        mu_val = self.config.get("mu", 0.0)
        sig_val = self.config.get("mutation_sigma", 100)
        indpb_val = self.config.get("mutation_indpb", 0.40)
        
        tb.register("mutate", self._mutate, mu=mu_val, sigma=sig_val, indpb=indpb_val)
        tb.register("select", tools.selTournament, tournsize=self.config.get("tournament_size", 5))
        
        pop = tb.population(n=self.config["population_size"])
        
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        cxpb_val = self.config.get("crossover_probability", 0.95)
        mutpb_val = self.config.get("mutation_probability", 0.70)
        
        # --- Motor de Evolução Customizado para suportar Early Stopping (Plateau) ---
        
        # Avaliação Inicial
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fits = tb.map(tb.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits):
            ind.fitness.values = fit
            
        hof.update(pop)
        
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        plateau_gens = self.config.get("plateau_generations_p1", 100)
        plateau_count = 0
        best_aep = hof[0].fitness.values[0] if hof else 0
        
        frames = []
        best_coords = np.array(hof[0]).reshape((self.n_turb, 2))
        frames.append(best_coords.copy())
        
        print(f"Gen   0/{self.max_gens} | Best Gross AEP: {best_aep/1e3:.3f} GWh")
        
        for gen in range(1, self.max_gens + 1):
            # Seleção
            offspring = tb.select(pop, len(pop))
            offspring = [tb.clone(ind) for ind in offspring]
            
            # Crossover
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < cxpb_val:
                    tb.mate(ind1, ind2)
                    del ind1.fitness.values, ind2.fitness.values
                    
            # Mutação
            for mutant in offspring:
                if random.random() < mutpb_val:
                    tb.mutate(mutant)
                    del mutant.fitness.values
                    
            # Avaliação dos novos
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fits = tb.map(tb.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fits):
                ind.fitness.values = fit
                
            hof.update(offspring)
            pop[:] = offspring
            
            record = stats.compile(pop) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            
            current_best = hof[0].fitness.values[0]
            
            best_coords_gen = np.array(hof[0]).reshape((self.n_turb, 2))
            frames.append(best_coords_gen.copy())
            
            # Print de 10 em 10 gerações
            if gen % 10 == 0:
                print(f"Gen {gen:>3}/{self.max_gens} | Best Gross AEP: {current_best/1e3:.3f} GWh")
                
            # Early Stopping (Plateau check)
            if (current_best - best_aep) <= 1e3: # Mega Watts
                plateau_count += 1
            else:
                best_aep = current_best
                plateau_count = 0
                
            if plateau_count >= plateau_gens:
                print(f"\n[Convergence Detected] Gross AEP hasn't improved for {plateau_gens} generations.")
                print(f"Phase 1 finished early at Gen {gen}.")
                break
                
        pool.close()
        pool.join()
        
        # Captura os dados
        best_ind = hof[0]
        best_coords = np.array(best_ind).reshape((self.n_turb, 2))
        
        # Filtra valores negativos para não estragar o gráfico do orquestrador
        history_best = [max(0.0, record['max']) for record in logbook]
        
        return best_coords, history_best, pop, logbook, frames

def run_phase_1(config, max_gens=1000):
    optimizer = Phase1Optimizer(config, max_gens)
    return optimizer.run()
