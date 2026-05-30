"""
worker.py — Background optimization thread for the Wind Farm GUI.

Runs WindFarmSimulator in a QThread, emitting signals to update the UI
without blocking the main event loop.
"""

import os
import sys
import time
import random
import traceback
import numpy as np

from PyQt5.QtCore import QThread, pyqtSignal

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deap import base, creator, tools
from config.iea37_aepcalc import calcAEP
from core.cabling_v3 import analisar_layout_completo
from simulate import (
    WindFarmSimulator,
    repair_spacing,
    enforce_circle_p1,
    enforce_circle_p2,
    load_config,
)


def _ensure_creators():
    """Ensure DEAP creator classes exist (safe for re-import)."""
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualPhase1"):
        creator.create("IndividualPhase1", list, fitness=creator.FitnessMax)
    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "IndividualPhase2"):
        creator.create("IndividualPhase2", list, fitness=creator.FitnessMulti)


class SimWorker(QThread):
    """
    Runs the two-phase wind farm optimization in a background thread.

    Signals
    -------
    sig_log(str)            : plain-text log line for the console widget.
    sig_p1_frame(np.ndarray): turbine coordinates every generation (Phase 1).
    sig_p1_aep(int, float)  : (generation, best_gross_aep_GWh).
    sig_p2_frame(np.ndarray, np.ndarray, int, list):
                              (turb_coords, sub_pos, n_groups, cable_paths).
    sig_p2_metrics(int, float, float):
                              (generation, best_net_aep_GWh, best_capex_kUSD).
    sig_progress(int, int)  : (current_gen, total_gens_this_phase).
    sig_phase_change(int)   : 1 or 2 when a phase starts.
    sig_eta(float)          : estimated seconds remaining.
    sig_done(list, object)  : (pareto_solutions_list, sim_ref) when finished.
    sig_error(str)          : error message if the run crashes.
    """

    sig_log          = pyqtSignal(str)
    sig_p1_frame     = pyqtSignal(object)
    sig_p1_aep       = pyqtSignal(int, float)
    sig_p2_frame     = pyqtSignal(object, object, int, list)
    sig_p2_metrics   = pyqtSignal(int, float, float)
    sig_progress     = pyqtSignal(int, int)
    sig_phase_change = pyqtSignal(int)
    sig_eta          = pyqtSignal(float)
    sig_done         = pyqtSignal(list, object)
    sig_error        = pyqtSignal(str)

    def __init__(self, config: dict, output_dir: str, max_gens: int = 1000):
        super().__init__()
        self.config     = config
        self.output_dir = output_dir
        self.max_gens   = max_gens
        self._stop      = False
        self._pause     = False

    def stop(self):
        self._stop = True

    def pause(self):
        self._pause = True

    def resume(self):
        self._pause = False

    # ------------------------------------------------------------------
    def run(self):
        try:
            _ensure_creators()
            self._run_optimization()
        except Exception:
            self.sig_error.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    def _run_optimization(self):
        import multiprocessing
        from simulate import (
            WindFarmSimulator,
            repair_spacing,
            enforce_circle_p1,
            enforce_circle_p2,
        )

        # Write config to a temp YAML so WindFarmSimulator can load it
        import tempfile, yaml
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, dir=self.output_dir
        ) as tmp:
            yaml.dump(self.config, tmp)
            tmp_path = tmp.name

        sim = WindFarmSimulator(tmp_path, self.output_dir)
        os.unlink(tmp_path)

        # ── PHASE 1 ───────────────────────────────────────────────────
        self.sig_phase_change.emit(1)
        self.sig_log.emit("═" * 60)
        self.sig_log.emit("  PHASE 1 — Layout Optimization (Gross AEP)")
        self.sig_log.emit("═" * 60)

        pool = multiprocessing.Pool()
        tb = base.Toolbox()
        tb.register("map", pool.map)
        tb.register(
            "individual",
            lambda: creator.IndividualPhase1(sim.initial_coords.flatten().tolist()),
        )
        tb.register("population", tools.initRepeat, list, tb.individual)
        tb.register("evaluate", sim.evaluate_p1)
        tb.register("mate", tools.cxBlend, alpha=sim.config.get("alpha", 0.5))

        def mutate_p1(ind):
            mu_val = sim.config.get("mu", 0.0)
            for i in range(len(ind)):
                if random.random() < sim.config["mutation_indpb"]:
                    ind[i] += random.gauss(mu_val, sim.config["mutation_sigma"])
            enforce_circle_p1(ind, sim.radius, sim.n_turb)
            coords_arr = np.array(ind).reshape((sim.n_turb, 2))
            repaired = repair_spacing(coords_arr, sim.min_spacing, sim.radius)
            ind[:] = repaired.flatten().tolist()
            return (ind,)

        tb.register("mutate", mutate_p1)
        tb.register(
            "select",
            tools.selTournament,
            tournsize=sim.config.get("tournament_size", 5),
        )

        pop = tb.population(n=sim.config["population_size"])
        fits = tb.map(tb.evaluate, pop)
        for ind, fit in zip(pop, fits):
            ind.fitness.values = fit

        history_best = []
        plateau_gens = sim.config.get(
            "plateau_generations_p1", sim.config.get("plateau_generations", 50)
        )
        gen_times = []

        for gen in range(1, self.max_gens + 1):
            if self._stop:
                self.sig_log.emit("[STOPPED by user]")
                pool.close()
                pool.join()
                return
            while self._pause:
                time.sleep(0.2)

            t_gen = time.time()

            offspring = [tb.clone(ind) for ind in tb.select(pop, len(pop))]
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < sim.config["crossover_probability"]:
                    tb.mate(i1, i2)
                    del i1.fitness.values, i2.fitness.values
            for mutant in offspring:
                if random.random() < sim.config["mutation_probability"]:
                    tb.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                enforce_circle_p1(ind, sim.radius, sim.n_turb)
                c = np.array(ind).reshape((sim.n_turb, 2))
                ind[:] = repair_spacing(c, sim.min_spacing, sim.radius).flatten().tolist()

            fits = tb.map(tb.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop = tb.select(pop + offspring, sim.config["population_size"])
            best_ind = tools.selBest(pop, 1)[0]
            best_aep = best_ind.fitness.values[0]
            history_best.append(best_aep)

            best_coords = np.array(best_ind).reshape((sim.n_turb, 2))
            self.sig_p1_frame.emit(best_coords.copy())
            self.sig_p1_aep.emit(gen, best_aep / 1e3)
            self.sig_progress.emit(gen, self.max_gens)

            gen_times.append(time.time() - t_gen)
            if len(gen_times) > 10:
                gen_times.pop(0)
            eta = np.mean(gen_times) * (self.max_gens - gen)
            self.sig_eta.emit(eta)

            if gen % 5 == 0 or gen == 1:
                self.sig_log.emit(
                    f"  Gen {gen:>4} | Best Gross AEP: {best_aep/1e3:.3f} GWh"
                )

            if gen >= plateau_gens:
                prev = history_best[gen - plateau_gens]
                if prev > 0 and (best_aep - prev) / prev <= 1e-7:
                    self.sig_log.emit(
                        f"\n[Convergence] No AEP gain in {plateau_gens} gens. "
                        f"Phase 1 done at gen {gen}."
                    )
                    break

        best_p1 = np.array(tools.selBest(pop, 1)[0]).reshape((sim.n_turb, 2))
        pop_p1 = pop
        pool.close()
        pool.join()

        # ── PHASE 2 ───────────────────────────────────────────────────
        self.sig_phase_change.emit(2)
        self.sig_log.emit("═" * 60)
        self.sig_log.emit("  PHASE 2 — Co-design (Net AEP & Cable CAPEX)")
        self.sig_log.emit("═" * 60)

        pool2 = multiprocessing.Pool()
        tb2 = base.Toolbox()
        tb2.register("map", pool2.map)
        tb2.register("evaluate", sim.evaluate_p2)
        tb2.register("mate", tools.cxBlend, alpha=sim.config.get("alpha", 0.5))
        n_coords = sim.n_turb * 2

        def mutate_p2(ind):
            mu_val = sim.config.get("mu", 0.0)
            for i in range(n_coords):
                if random.random() < sim.config["mutation_indpb"]:
                    ind[i] += random.gauss(mu_val, sim.config["mutation_sigma"])
            if random.random() < sim.config["mutation_indpb"]:
                ind[n_coords] += random.gauss(mu_val, 0.1)
            if sim.sub_mode == "optimize":
                if random.random() < sim.config["mutation_indpb"]:
                    sub_sig = sim.radius * 0.4
                    ind[n_coords + 1] += random.gauss(mu_val, sub_sig)
                    ind[n_coords + 2] += random.gauss(mu_val, sub_sig)
            enforce_circle_p2(ind, sim.radius, sim.n_turb, sim.sub_mode)
            c = np.array(ind[:n_coords]).reshape((sim.n_turb, 2))
            ind[:n_coords] = repair_spacing(c, sim.min_spacing, sim.radius).flatten().tolist()
            return (ind,)

        tb2.register("mutate", mutate_p2)
        tb2.register("select", tools.selNSGA2)

        # Smart seeding
        pop_p2 = []
        best_p1_inds = tools.selBest(pop_p1, int(len(pop_p1) * 0.3))
        for idx, ind in enumerate(best_p1_inds):
            coords = list(ind)
            g_norm = random.random()
            if sim.sub_mode == "optimize":
                if idx % 2 == 0:
                    centroid = np.mean(np.array(coords).reshape((sim.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    r_sub = sim.radius * np.sqrt(random.random())
                    theta = random.random() * 2 * np.pi
                    sub_x, sub_y = r_sub * np.cos(theta), r_sub * np.sin(theta)
            else:
                sub_x, sub_y = sim.sub_fixed_pos[0], sim.sub_fixed_pos[1]
            pop_p2.append(creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y]))

        while len(pop_p2) < sim.config["population_size"]:
            parent = random.choice(best_p1_inds)
            coords = list(parent)
            g_norm = random.random()
            if sim.sub_mode == "optimize":
                if random.random() < 0.5:
                    centroid = np.mean(np.array(coords).reshape((sim.n_turb, 2)), axis=0)
                    sub_x, sub_y = centroid[0], centroid[1]
                else:
                    r_sub = sim.radius * np.sqrt(random.random())
                    theta = random.random() * 2 * np.pi
                    sub_x, sub_y = r_sub * np.cos(theta), r_sub * np.sin(theta)
            else:
                sub_x, sub_y = sim.sub_fixed_pos[0], sim.sub_fixed_pos[1]
            child = creator.IndividualPhase2(coords + [g_norm, sub_x, sub_y])
            child, = mutate_p2(child)
            pop_p2.append(child)

        fits = tb2.map(tb2.evaluate, pop_p2)
        for ind, fit in zip(pop_p2, fits):
            ind.fitness.values = fit
        pop_p2 = [ind for ind in pop_p2 if ind.fitness.values[0] > 0] or pop_p2

        hof = tools.ParetoFront()
        hof.update(pop_p2)

        history_net, history_capex = [], []
        plateau_gens2 = sim.config.get(
            "plateau_generations_p2", sim.config.get("plateau_generations", 50)
        )
        gen_times2 = []

        for gen in range(1, self.max_gens + 1):
            if self._stop:
                self.sig_log.emit("[STOPPED by user]")
                break
            while self._pause:
                time.sleep(0.2)

            t_gen = time.time()

            offspring = [tb2.clone(ind) for ind in tb2.select(pop_p2, len(pop_p2))]
            for i1, i2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < sim.config["crossover_probability"]:
                    tb2.mate(i1, i2)
                    del i1.fitness.values, i2.fitness.values
            for mutant in offspring:
                if random.random() < sim.config["mutation_probability"]:
                    tb2.mutate(mutant)
                    del mutant.fitness.values

            invalid = [ind for ind in offspring if not ind.fitness.valid]
            for ind in invalid:
                enforce_circle_p2(ind, sim.radius, sim.n_turb, sim.sub_mode)
                c = np.array(ind[:n_coords]).reshape((sim.n_turb, 2))
                ind[:n_coords] = repair_spacing(c, sim.min_spacing, sim.radius).flatten().tolist()

            fits = tb2.map(tb2.evaluate, invalid)
            for ind, fit in zip(invalid, fits):
                ind.fitness.values = fit

            pop_p2 = tb2.select(pop_p2 + offspring, sim.config["population_size"])
            hof.update(pop_p2)

            valid_hof = [ind for ind in hof if ind.fitness.values[0] > 0] or pop_p2
            best_aep_ind = max(valid_hof, key=lambda x: x.fitness.values[0])
            best_cost_ind = min(valid_hof, key=lambda x: x.fitness.values[1])
            best_net_aep = best_aep_ind.fitness.values[0]
            best_capex   = best_cost_ind.fitness.values[1]

            # Knee point for live layout frame
            if len(valid_hof) > 1:
                aeps   = np.array([i.fitness.values[0] for i in valid_hof])
                capexs = np.array([i.fitness.values[1] for i in valid_hof])
                a_range = (aeps.max() - aeps.min()) or 1.0
                c_range = (capexs.max() - capexs.min()) or 1.0
                dists = np.sqrt(
                    ((aeps.max() - aeps) / a_range) ** 2
                    + ((capexs - capexs.min()) / c_range) ** 2
                )
                knee = valid_hof[np.argmin(dists)]
            else:
                knee = valid_hof[0]

            knee_coords = np.array(knee[:n_coords]).reshape((sim.n_turb, 2))
            g_norm      = knee[n_coords]
            n_groups    = int(np.round(sim.min_groups + g_norm * (sim.max_groups - sim.min_groups)))
            n_groups    = max(sim.min_groups, min(sim.n_turb, n_groups))
            sub_pos     = (
                np.array([knee[n_coords + 1], knee[n_coords + 2]])
                if sim.sub_mode == "optimize"
                else sim.sub_fixed_pos
            )

            # Compute cable paths for live drawing
            cable_paths = []
            try:
                combined = np.vstack([knee_coords, sub_pos.reshape((1, 2))])
                planta, _ = analisar_layout_completo(combined, sub=sim.n_turb, n_grupos=n_groups)
                cable_paths = planta.paths
            except Exception:
                pass

            history_net.append(best_net_aep)
            history_capex.append(best_capex)

            self.sig_p2_frame.emit(
                knee_coords.copy(), sub_pos.copy(), n_groups, cable_paths
            )
            self.sig_p2_metrics.emit(gen, best_net_aep / 1e3, best_capex / 1e3)
            self.sig_progress.emit(gen, self.max_gens)

            gen_times2.append(time.time() - t_gen)
            if len(gen_times2) > 10:
                gen_times2.pop(0)
            self.sig_eta.emit(np.mean(gen_times2) * (self.max_gens - gen))

            if gen % 5 == 0 or gen == 1:
                self.sig_log.emit(
                    f"  Gen {gen:>4} | Net AEP: {best_net_aep/1e3:.3f} GWh"
                    f" | CAPEX: ${best_capex/1e3:.1f} kUSD"
                )

            if gen >= plateau_gens2:
                pn = history_net[gen - plateau_gens2]
                pc = history_capex[gen - plateau_gens2]
                if pn > 0 and (best_net_aep - pn) / pn <= 1e-7:
                    if pc > 0 and (pc - best_capex) / pc <= 1e-7:
                        self.sig_log.emit(
                            f"\n[Convergence] Phase 2 done at gen {gen}."
                        )
                        break

        pool2.close()
        pool2.join()

        # Build result list for Pareto tab
        valid_solutions = [ind for ind in hof if ind.fitness.values[0] > 0]
        self.sig_done.emit(valid_solutions, sim)
