# `optimizer/` — NSGA-II Multi-Objective Optimizer Benchmarking

This directory contains the benchmarking suite for comparing different optimization strategies under the Two-Phase Hierarchical Framework.

| Script | Purpose |
|---|---|
| `benchmark.py` | **GECCO paper benchmark** — compares 3 methods over N seeds with statistical tests |

---


## `benchmark.py` — GECCO 2026 Comparison

Compares **three methods** over N independent seeds with full statistical rigour:

| Method | Description |
|---|---|
| **Proposed** | Two-phase warm-start: Phase 1 (layout GA) → Phase 2 (joint NSGA-II) |
| **Baseline** | Pure NSGA-II from random init, same evaluation budget |
| **Sequential** | Phase 1 layout GA, then separate cable GA with turbines fixed |

### Metrics computed

- **Hypervolume** — Convergence and diversity (primary metric)
- **C-metric** — Fraction of opponent's solutions dominated
- **Spread** — Diversity along the Pareto front
- **Statistical tests** — Mann-Whitney U or t-test + Cohen's d
- **Computational efficiency** — Time per evaluation, speedup factors

### Target case study

By default, the benchmark is configured to optimize the standard **16-turbine IEA Task 37 case study** with:
- **Number of Turbines (`IND_SIZE`):** `16`
- **Initial Layout (`main_yaml_path`):** `"config/iea37-ex16.yaml"`
- **Boundary Radius (`CIRCLE_RADIUS`):** `5000` meters
- **Minimum Turbine Spacing (`N_DIAMETERS`):** `260` meters (2 * rotor diameter)

To evaluate other benchmark configurations (e.g., 36 or 64 turbines), you must open `optimizer/benchmark.py` and modify these variables at the top of the script (around lines 80-98) to match your target layout coordinates and turbine count.

### Running

```bash
# Full benchmark — matches GECCO paper (20 seeds, ~hours)
python optimizer/benchmark.py

# Quick smoke-test (3 seeds, 50+100 gens — minutes)
python optimizer/benchmark.py --quick

# Custom seeds & output dir
python optimizer/benchmark.py --seeds 5 --output results/my_benchmark
```

### Output files (saved to `results/` by default)

| File | Description |
|---|---|
| `comparison_boxplot.png/.pdf` | Hypervolume boxplot (3 methods) |
| `comparison_pareto.png/.pdf` | Accumulated Pareto fronts scatter |
| `comparison_evolution.png/.pdf` | Hypervolume convergence curves |
| `comparison_metrics.png/.pdf` | 4-panel: HV, #solutions, spread, time |
| `comparison_execution_time.png/.pdf` | Computational efficiency boxplot |
