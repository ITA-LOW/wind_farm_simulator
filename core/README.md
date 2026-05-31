# `core/` — Physics & Optimization Engine

This directory contains the computational backbone of the wind farm simulator, covering site definitions, aerodynamic modelling, electrical routing, and evolutionary algorithms.

---

## `boundary.py` — Site-Agnostic Engine

Handles the loading, projection, and enforcement of arbitrary polygonal boundaries.
- **GeoJSON parsing**: Reads standard `.geojson` files.
- **Cartesian Projection**: Automatically converts WGS84 coordinates to a local Azimuthal Equidistant projection (meters).
- **Geometric enforcement**: Ensures turbines and substations remain within the valid area and respects forbidden zones (holes).

---

## `wind_rose.py` — Meteorological Integration & Diagnostics

Manages dynamic wind data retrieval and site-suitability analysis.
- **ERA5 API Integration**: Automatically fetches the latest 2 years of hourly wind data from Open-Meteo based on the WGS84 coordinates derived from the `boundary.py` GeoJSON centroid.
- **Power Law Extrapolation**: Dynamically extrapolates 100m wind speeds to the specific turbine's `hub_height`.
- **Site Suitability Diagnostics**: Evaluates the raw hourly data array against the turbine's specific power curve attributes (`cut_in`, `cut_out`, `rated_ws`) to generate automated reports (e.g., % time operating in cubic region vs rated power).

---

## `aep.py` & `wfwe.py` — Aerodynamic & Wake Models

### `aep.py`
The highly-optimized Annual Energy Production (AEP) calculator used inside the Genetic Algorithm loop. Implements the **simplified Bastankhah Gaussian wake model** combined with **Katic quadratic superposition** across a full wind rose probability distribution.

### `wfwe.py` (Wind Farm Wake Engine)
Used primarily for high-resolution validation and visualisations. It exposes a full 2D velocity-field heat map (contour plots) for specific wind directions.

---

## `cabling_v3.py` — SAP Cabling & Cost Model

Implements **Strict Angular Partitioning (SAP)** — the novel algorithm that guarantees planar, non-crossing cable networks in O(N log N).

### Algorithm
```text
1. Compute polar angle of each turbine w.r.t. the substation
2. Sort turbines by angle → contiguous angular sectors
3. Split into n_groups equal-size angular slices
4. Within each slice, sort radially (far → near)
5. Connect: T_far → ... → T_near → Substation
```

**Key property:** Because groups are defined by *contiguous angular sectors*, cables from different groups can never cross. This eliminates the need for expensive MILP-based cable routing solvers.

### Cost model
Cable CAPEX is computed using the **NREL linear cost model** (Nakhai et al., 2023):

$$
\text{Cost } [\text{USD/m}] = 0.3476 \cdot \text{CSA } [\text{mm}^2] \cdot N_{\text{conductors}}
$$

Joule losses (annual, MWh):
$$
P_J = 3 \cdot I^2 \cdot R = 3 \cdot I^2 \cdot R_{\text{km}} \cdot \left(\frac{L}{1000}\right)
$$

---

## `phase_1.py` & `phase_2.py` — Two-Phase Optimization

The core evolutionary logic running the layout and co-design optimizations using DEAP.

- **`phase_1.py` (Layout):** Uses a standard Genetic Algorithm (GA) to maximize gross AEP. It automatically calculates a perfect grid spacing to generate a homogenously spread initial generation inside the GeoJSON boundary, preventing destructive crossover (`cxBlend`) effects.
- **`phase_2.py` (Co-design):** Uses NSGA-II to perform multi-objective optimization (Net AEP vs. CAPEX), simultaneously optimizing the placement of the substation, routing of cables, and micro-siting of turbines.

---

## `plot.py` — Visualisation Utilities

Helper functions for publication-quality figures:

| Function | Output |
|---|---|
| `generate_evolution_gifs` | Animated GIFs of the layout optimization process and dynamic AEP/CAPEX convergence graphs |
| `plot_pareto_front` | Pareto front scatter (Net AEP vs CAPEX) |
| `plot_cabling_layout` | Colour-coded inter-array cable network layout |
| `plot_wake_fields` | High-res wake field heatmap (wraps `wfwe.py`) |
