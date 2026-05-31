# `cases/` —  Custom Case Configuration

This directory contains the simulation configurations for wind farms in YAML format. You can define your own simulation setups by creating a new `.yaml` file in this directory.

---

## YAML File Structure

Below is the standard configuration structure (e.g., `case_example.yaml`) with every option commented:

```yaml
# Simulation configuration
name: "My wind farm simulation"               # Name of the simulation run
n_turbines: 27                                # Number of turbines to optimize
min_spacing_multiplier: 3.0                   # Min turbine spacing as factor of rotor diameter

# Input files (paths relative to repository root)
turbine_yaml: "config/turbines/iea37-335mw.yaml"   # Turbine specifications YAML path
windrose_yaml: "config/windrose/iea37-windrose.yaml" # Wind rose resource YAML path
boundary_geojson: "config/boundaries/site_boundary.geojson" # Site polygonal boundary

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
plateau_generations_p1: 200                # Phase 1 gens without improvement to trigger stop
plateau_generations_p2: 200                # Phase 2 gens without improvement to trigger stop

# Substation: "optimize" to co-optimize coordinates, or [X, Y] list to keep it fixed (default: "optimize")
substation: "optimize" # substation: [300, 200] for fixed substation for example

# Inter-array cabling groups (SAP strings)
cable_groups:
  min_groups: 4                            # Lower bound of independent SAP cable strings
  max_groups: 15                           # Upper bound of independent SAP cable strings
```


---

## How to create new cases

1. **Draw your site:** Use a tool like [geojson.io](https://geojson.io/) to draw your wind farm site (can be any irregular polygon). Save the exported file in `config/boundaries/`.
2. **Setup your case:** Copy `case_example.yaml` as a base for a new file, e.g., `my_layout.yaml`.
3. **Configure the layout:** Point `boundary_geojson` to your new file and define the number of turbines (`n_turbines`).
   > **Note:** The optimizer will automatically calculate the perfect grid spacing to generate an evenly-spread initial layout inside your polygon without requiring a manual start layout.
4. **Run your custom case** using the orchestrator:
   ```bash
   python orchestrator.py --case cases/my_layout.yaml --output results/my_test --max-gens 200
   ```
