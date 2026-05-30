# `cases/` —  Custom Case Configuration

This directory contains the simulation configurations for wind farms in YAML format. You can define your own simulation setups by creating a new `.yaml` file in this directory.

---

## YAML File Structure

Below is the standard configuration structure (e.g., `case_example.yaml`) with every option commented:

```yaml
name: "Simulation # 1"                        # Name of the simulation run
n_turbines: 12                                # Number of turbines to optimize
boundary_radius: 1000.0                       # Wind farm boundary circle radius in meters
min_spacing_multiplier: 2.0                   # Min turbine spacing as factor of rotor diameter

# Input files (paths relative to repository root)
turbine_yaml: "config/iea37-335mw.yaml"         # Turbine specifications YAML path
windrose_yaml: "config/iea37-windrose.yaml"     # Wind rose resource YAML path
initial_layout_yaml: "config/iea37-ex12.yaml"   # Preset starting turbine coordinates (default: null/random layout)

# Genetic Algorithm parameters
population_size: 300                      # GA population size
crossover_probability: 0.95               # GA crossover probability [0.0 - 1.0]
mutation_probability: 0.7                 # GA mutation probability [0.0 - 1.0]
mutation_sigma: 100.0                      # Gaussian mutation deviation in meters (default: 100.0)
mu: 0.0                                    # Mean deviation of Gaussian mutation (default: 0.0)
mutation_indpb: 0.4                        # Probability of mutating each coordinate gene [0.0 - 1.0] (default: 0.4)
tournament_size: 5                         # Tournament size for selection (default: 5)
alpha: 0.5                                 # Blend crossover blending parameter (default: 0.5)

# Plateau convergence criteria
plateau_generations_p1: 50                 # Phase 1 gens without improvement to trigger stop (default: 50)
plateau_generations_p2: 200                # Phase 2 gens without improvement to trigger stop (default: 50)

# Substation: "optimize" to co-optimize coordinates, or [X, Y] list to keep it fixed (default: "optimize")
substation: "optimize" # substation: [300, 200] for fixed substation for example

# Inter-array cabling groups (SAP strings)
cable_groups:
  min_groups: 2                            # Lower bound of independent SAP cable strings (default: 2)
  max_groups: 50                           # Upper bound of independent SAP cable strings (default: 16)
```


---

## How to create new cases

1. Copy `case_example.yaml` as a base for a new file, e.g., `my_layout.yaml`.
2. Modify the number of turbines (`n_turbines`), the boundary radius (`boundary_radius`), or the wind rose and turbine files.
3. If you do not have an initial layout YAML file for your farm, remove or comment out the `initial_layout_yaml` line. The optimizer will automatically generate a valid random initial layout within the boundary circle.
4. Run your custom case using the simulation runner:
   ```bash
   python simulate.py --case cases/my_layout.yaml
   ```
