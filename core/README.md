# `core/` — Physics & Optimisation Engine

The three modules here form the computational backbone of opt_wind_farm_simulator.

---

## `wfwe.py` — Wind Farm Wake Engine

Implements the **Bastankhah Gaussian wake model** for a single wind direction
and exposes a full velocity-field visualisation.

### Key class: `WindFarm`

```python
from core.wfwe import WindFarm

farm = WindFarm(
    turbine_coords,          # (N, 2) array of (x, y) positions [m]
    wind_direction=270.0,    # meteorological convention (N=0, CW)
    wind_speed_free_stream=9.8,
    turbine_diameter=130.0,
    wake_k=0.0324555,        # turbulence parameter
    ct_coeff=8/9,            # thrust coefficient
)

farm.calculate_wake_effects()   # compute effective speed at each turbine
farm.summarize_results()        # print table
X, Y, V = farm.get_velocity_field(resolution=300)  # heatmap grid
farm.plot_layout_with_wake_field(save_path="wake.png")
```

### Wake model

The Bastankhah deficit at point (x, y) from a turbine at (x₀, y₀):

$$
\sigma_y(x) = k_y (x - x_0) + \frac{D}{\sqrt{8}}
$$

$$
\frac{\Delta U}{U_\infty} = \left[ 1 - \sqrt{1 - \frac{C_T}{8 (\sigma_y / D)^2}} \right] \exp\left( -\frac{(y - y_0)^2}{2 \sigma_y^2} \right)
$$

Multiple wakes combined via **Katic quadratic superposition**.

---

## `cabling_v3.py` — SAP Cabling & Cost Model

Implements **Strict Angular Partitioning (SAP)** — the novel algorithm
that guarantees planar, non-crossing cable networks in O(N log N).

### Algorithm

```
1. Compute polar angle of each turbine w.r.t. the substation
2. Sort turbines by angle → contiguous angular sectors
3. Split into n_groups equal-size angular slices
4. Within each slice, sort radially (far → near)
5. Connect: T_far → ... → T_near → Substation
```

**Key property:** Because groups are defined by *contiguous angular sectors*,
cables from different groups can never cross. This eliminates the need for
expensive MILP-based cable routing solvers.

### Cost model

Cable CAPEX is computed using the **NREL linear cost model**
(Nakhai et al., 2023):

$$
\text{Cost } [\text{USD/m}] = 0.3476 \cdot \text{CSA } [\text{mm}^2] \cdot N_{\text{conductors}}
$$

For a 3-phase inter-array system with commercially available cross-sections
(50–240 mm²).  The required cross-section is determined by the peak current:

$$
I = \frac{P_{\text{accumulated}}}{\sqrt{3} \cdot V_n \cdot \cos \varphi}
$$

$$
A_{\text{min}} = \frac{I}{J_{\text{max}}} \quad \left(J_{\text{max}} = 2.3 \text{ A/mm}^2\right)
$$

Joule losses (annual, MWh):

$$
P_J = 3 \cdot I^2 \cdot R = 3 \cdot I^2 \cdot R_{\text{km}} \cdot \left(\frac{L}{1000}\right)
$$

$$
\text{Losses}_{\text{annual}} = \frac{P_J \cdot 8760}{10^6} \quad [\text{MWh/yr}]
$$

### Public API

```python
from core.cabling_v3 import analisar_layout_completo

planta, results = analisar_layout_completo(
    coords,        # (N+1, 2) — turbines + substation
    sub=N,         # index of the substation
    n_grupos=4,    # number of cable strings
    Vn=33e3,       # nominal voltage [V]
    P_turb=3.35e6, # turbine rated power [W]
)

print(results["custo_total_usd"])   # Cabling CAPEX [USD]
print(results["perda_anual_mwh"])   # Annual Joule losses [MWh]
print(results["comprimento_total_m"]) # Total cable length [m]
```

---

## `plot.py` — Visualisation Utilities

Helper functions for publication-quality figures:

| Function | Output |
|---|---|
| `plot_solution_circle` | Turbine layout inside farm boundary |
| `plot_fitness` | Fitness vs generations curve |
| `plot_benchmark_results` | Side-by-side: fitness + layout |
| `plot_pareto_front` | Pareto front scatter (AEP vs CAPEX) |
| `plot_cabling_layout` | Colour-coded inter-array cable layout |
| `plot_wake_fields` | Wake field heatmap (wraps `WindFarm`) |
