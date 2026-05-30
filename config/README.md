# `config/` — Wind Farm Benchmark Data

This directory contains all input data required to evaluate a wind farm layout:
the **IEA Wind Task 37** benchmark cases, turbine specifications, and wind resource.

---

## Files

| File | Description |
|---|---|
| `iea37_aepcalc.py` | AEP calculator using the simplified Bastankhah Gaussian wake model |
| `iea37-ex16.yaml` | 16-turbine benchmark layout (IEA Task 37 Case Study) |
| `iea37-335mw.yaml` | 3.35 MW turbine power curve attributes |
| `iea37-windrose.yaml` | 16-directional wind rose (IEA37 standard) |

---

## Wind Rose

The IEA37 wind rose uses **16 directional bins** of 22.5° each.  
The dominant wind direction is **270° (West)**, with 21.3% frequency.

```
Wind bins: [0°, 22.5°, 45°, ..., 337.5°]
Speeds   : 9.8 m/s (uniform across bins, IEA37 standard)
```

## Turbine Model (IEA37 3.35 MW)

| Parameter | Value |
|---|---|
| Rated power | 3.35 MW |
| Rotor diameter | 130 m |
| Cut-in wind speed | 4 m/s |
| Rated wind speed | 9.8 m/s |
| Cut-out wind speed | 25 m/s |

## Wake Model

The `iea37_aepcalc.py` implements the **simplified Bastankhah Gaussian wake model**
as defined in the IEA Task 37 combined case study:

$$\sigma(x) = k \cdot x + \frac{D}{\sqrt{8}}$$

$$\frac{\Delta U}{U} = \left(1 - \sqrt{1 - \frac{C_T}{8 \sigma(x)^2 / D^2}}\right) \cdot \exp\left(-\frac{y^2}{2\sigma(x)^2}\right)$$

Multiple upstream wakes are combined using **quadratic superposition** (Katic):

$$total_{deficit} = \sqrt{\sum_{i} \text{deficit}_i^2}$$

## Customizing Inputs & Data

If you need to simulate custom wind farm configurations (e.g., using a different turbine model or a specific wind resource for a real-world location), you can create your own data files by copying and modifying the templates in this directory.

### 1. Custom Turbine Specifications
To model a different wind turbine:
1. Copy `config/iea37-335mw.yaml` and rename it (e.g., `config/my_custom_turbine.yaml`).
2. Modify the key attributes under `definitions.wind_turbine`:
   * **`rotor_diameter`**: The rotor diameter in meters (affects wake width and spacing requirements).
   * **`hub_height`**: The height of the tower hub in meters.
   * **`power_curve`**: The wind speed bins (`wind_speed` in m/s) and corresponding power output values (`electrical_power` in W).

### 2. Custom Wind Rose Resource
To model a different wind location:
1. Copy `config/iea37-windrose.yaml` and rename it (e.g., `config/my_custom_windrose.yaml`).
2. Modify the properties under `definitions.wind_resource`:
   * **`direction.bins`**: List of directional sectors. You can define **any number of sectors/bins** (e.g., 8, 12, 16, or 36 bins).
   * **`speed`**: The annual average wind speed (in m/s) for each bin. These **do not have to be uniform**; you can specify a unique average speed for each wind sector.
   * **`probability`**: The frequency of wind coming from each sector. Ensure the sum of all probabilities equals exactly `1.0`.

### 3. Linking to your Case Configuration
Once your new files are saved in `config/`, reference them in your custom case YAML file inside the `cases/` directory:
```yaml
turbine_yaml: "config/my_custom_turbine.yaml"
windrose_yaml: "config/my_custom_windrose.yaml"
```

## Reference

> N. F. Baker et al. (2019). *Best practices for wake model and optimization
> algorithm selection in wind energy.* AIAA SciTech Forum.

> I. F. D. Silva, T. B. Lazzarin, L. Schmitz and A. R. Panisson, "Genetic
> Algorithm-Based Optimization Framework for Offshore Wind Farm Layout Design," 
> in IEEE Access, vol. 13, pp. 170081-170094, 2025, doi: 
> 10.1109/ACCESS.2025.3614516.
