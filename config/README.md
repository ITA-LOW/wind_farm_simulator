# `config/` — Configuration & Data

This directory contains all input data required to evaluate a wind farm layout, organized into subdirectories for boundaries, turbines, and wind resources.

---

## Directory Structure

| Directory | Description |
|---|---|
| `boundaries/` | Contains site definitions in GeoJSON format (e.g., `site_boundary.geojson`). Define any polygonal area where turbines can be placed. |
| `turbines/` | Contains turbine specifications (power curve, rotor diameter, etc.), such as `iea37-335mw.yaml`. |
| `windrose/` | Contains wind resource definitions (directions, frequencies, speeds), such as `iea37-windrose.yaml`. |
---

## Customizing Inputs & Data

If you need to simulate custom wind farm configurations for a real-world location, you can create your own data files by copying and modifying the templates in the respective subdirectories.

### 1. Custom Boundaries (GeoJSON)
To model a specific wind farm site:
1. Use a tool like [geojson.io](https://geojson.io/) to draw your polygon.
2. Export it as GeoJSON and save it in `config/boundaries/my_site.geojson`.
3. The optimizer will automatically read the vertices, project them to local coordinates (meters), and ensure all turbines are placed perfectly inside this area.

### 2. Custom Turbine Specifications
To model a different wind turbine:
1. Copy `config/turbines/iea37-335mw.yaml` and rename it (e.g., `config/turbines/my_custom_turbine.yaml`).
2. Modify the key attributes under `definitions.wind_turbine`:
   * **`rotor_diameter`**: The rotor diameter in meters (affects wake width and spacing requirements).
   * **`hub_height`**: The height of the tower hub in meters.
   * **`power_curve`**: The wind speed bins (`wind_speed` in m/s) and corresponding power output values (`electrical_power` in W).

### 3. Custom Wind Rose Resource
To model a different wind location:
1. Copy `config/windrose/iea37-windrose.yaml` and rename it (e.g., `config/windrose/my_custom_windrose.yaml`).
2. Modify the properties under `definitions.wind_resource`:
   * **`direction.bins`**: List of directional sectors. You can define **any number of sectors/bins** (e.g., 8, 12, 16, or 36 bins).
   * **`speed`**: The annual average wind speed (in m/s) for each bin. These **do not have to be uniform**.
   * **`probability`**: The frequency of wind coming from each sector. Ensure the sum of all probabilities equals exactly `1.0`.

### 4. Linking to your Case Configuration
Once your new files are saved, reference them in your custom case YAML file inside the `cases/` directory:
```yaml
turbine_yaml: "config/turbines/my_custom_turbine.yaml"
windrose_yaml: "config/windrose/my_custom_windrose.yaml"
boundary_geojson: "config/boundaries/my_site.geojson"
```

## Reference

> N. F. Baker et al. (2019). *Best practices for wake model and optimization
> algorithm selection in wind energy.* AIAA SciTech Forum.

> I. F. D. Silva, T. B. Lazzarin, L. Schmitz and A. R. Panisson, "Genetic
> Algorithm-Based Optimization Framework for Offshore Wind Farm Layout Design," 
> in IEEE Access, vol. 13, pp. 170081-170094, 2025, doi: 
> 10.1109/ACCESS.2025.3614516.
