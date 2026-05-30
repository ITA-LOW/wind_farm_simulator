import math
import numpy as np

# =============================================================================
# 1. HELPER FUNCTIONS
# =============================================================================

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points using math.hypot."""
    return math.hypot(x2 - x1, y2 - y1)


# =============================================================================
# 2. ELECTRICAL CLASSES
# =============================================================================

class Cable:
    """Represents a cable segment in the wind farm collection system.

    Each cable segment transports cumulative power from upstream turbines
    and requires an adequate cross-sectional area to support the current load.
    """
    # Typical power factor for modern wind turbines
    POWER_FACTOR = 0.95  # cos(phi)

    # Electrical resistance per commercially available cross-section (Ohm/km)
    SECTION_TABLE = {
        50: 0.49, 70: 0.34, 95: 0.25, 120: 0.20,
        150: 0.16, 185: 0.13, 240: 0.10,
    }

    def __init__(self, lc, Vn, Pn):
        """Initialize a cable segment.

        Args:
            lc: Cable length (meters)
            Vn: Nominal system voltage (Volts) - e.g., 33kV
            Pn: Cumulative power carried by this segment (Watts)
        """
        self.lc = lc
        self.Vn = Vn
        self.Pn = Pn
        self.dI = 2.3  # Maximum allowable current density (A/mm2)

        # Calculate current: I = P / (sqrt(3) * V * cos(phi))
        self.I = self.Pn / (math.sqrt(3) * self.Vn * Cable.POWER_FACTOR)

        # Minimum required continuous cross-sectional area (mm2)
        self.A_continuous = self.I / self.dI

        # Placeholders for assigned properties
        self.A = None      # Selected cross-sectional area (mm2)
        self.R_km = None   # Resistance per km (Ohm/km)
        self.R = None      # Total resistance (Ohm)
        self.Pj = None     # Joule losses (Watts)
        self.C = 0         # Cost per meter (USD/m)
        self.Ctot = 0      # Total segment cost (USD)

    def assign_section(self, section):
        """Assign a commercial cross-section and compute electrical properties."""
        self.A = section
        self.R_km = self.SECTION_TABLE[section]
        self.R = self.R_km * (self.lc / 1000.0)
        # Joule losses for a 3-phase system: Pj = 3 * I^2 * R
        self.Pj = 3.0 * (self.I ** 2) * self.R


class Turbine:
    """Represents a wind turbine's rated power and position."""
    def __init__(self, Pt, x, y):
        self.P = Pt
        self.x = x
        self.y = y


class Plant:
    """Represents the cable connection network of the wind farm."""
    # NREL cost model multiplier: 0.3476 USD/m per mm2 * 3 conductors
    NREL_UNIT_COST = 0.3476 * 3

    # Discrete commercial cable cost database (USD/m) mapping
    INDUSTRIAL_CABLE_COSTS = {
        50: 52.14,
        70: 72.99,
        95: 99.07,
        120: 125.14,
        150: 156.42,
        185: 192.92,
        240: 250.27,
        300: 312.84,
        400: 400 * 1.0428,
        500: 500 * 1.0428,
        630: 630 * 1.0428,
        800: 800 * 1.0428
    }

    def __init__(self, Vn, Tr, paths):
        """Initialize the cabling plant and compute electrical properties.

        Args:
            Vn: Nominal system voltage (Volts)
            Tr: List of Turbine objects (including the substation)
            paths: List of cable strings (each path is a list of turbine indices)
        """
        self.Vn = Vn
        self.Tr = Tr
        self.paths = paths
        self.Cb = []
        self.cables_flat = []
        self.Pjtot = 0.0
        self.Ctot = 0.0

        self.lay_cables()
        self.uniform_section()
        self.calculate_losses()
        self.calculate_cost()

    def lay_cables(self):
        """Create Cable objects for all connected segments in the layout paths."""
        self.Cb = []
        for path in self.paths:
            cable_path = []
            Pacc = 0.0
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                Pacc += self.Tr[a].P
                L = calculate_distance(
                    self.Tr[a].x, self.Tr[a].y,
                    self.Tr[b].x, self.Tr[b].y
                )
                cable_path.append(Cable(L, self.Vn, Pacc))
            self.Cb.append(cable_path)
        self.cables_flat = [c for p in self.Cb for c in p]

    def uniform_section(self):
        """Assign a uniform commercial cable section to the entire plant."""
        Amax = max(c.A_continuous for c in self.cables_flat)
        chosen = max(Cable.SECTION_TABLE)

        # Select the smallest commercial size that satisfies current constraints
        for sec in sorted(Cable.SECTION_TABLE):
            if sec >= Amax:
                chosen = sec
                break

        for c in self.cables_flat:
            c.assign_section(chosen)

    def calculate_losses(self):
        """Compute the total Joule losses in the plant."""
        self.Pjtot = sum(c.Pj for c in self.cables_flat)

    def calculate_cost(self):
        """Compute the total cable CAPEX using the NREL cost model database."""
        sec = self.cables_flat[0].A
        custo_m = self.INDUSTRIAL_CABLE_COSTS[sec]

        self.Ctot = 0.0
        for c in self.cables_flat:
            c.C = custo_m
            c.Ctot = c.lc * custo_m
            self.Ctot += c.Ctot

    def get_max_calculated_section(self):
        return self.cables_flat[0].A


# =============================================================================
# 3. STRICT ANGULAR PARTITIONING (SAP)
# =============================================================================

def agrupar_por_setor_angular(coords, sub, n_grupos):
    """Partition turbines into contiguous angular groups relative to the substation.

    This ensures no cable crossings occur between different groups.
    """
    v = coords - coords[sub]
    ang = np.arctan2(v[:, 1], v[:, 0])

    # Filter out substation index
    indices_turbines = np.array([i for i in range(len(coords)) if i != sub])
    angles_turbines = ang[indices_turbines]

    # Sort indices counter-clockwise
    idx_sorted = indices_turbines[np.argsort(angles_turbines)]

    # Split into contiguous angular groups
    groups = np.array_split(idx_sorted, n_grupos)
    return [list(g) for g in groups if len(g) > 0]


# =============================================================================
# 4. MAIN INTERFACE
# =============================================================================

def analisar_layout_completo(coords, sub, n_grupos=15, Vn=33e3, P_turb=3.35e6):
    """Analyze the complete electrical layout using SAP cabling.

    Args:
        coords: Coordinates array of shape (N_turbines + 1, 2)
        sub: Index of the substation within coords
        n_grupos: Target number of cable strings
        Vn: Nominal system voltage (Volts)
        P_turb: Nominal power of each turbine (Watts)

    Returns:
        planta: Plant object with detailed electrical metrics
        resultados: Summary metrics dict (CAPEX, length, losses, etc.)
    """
    # Step 1: Sector partitioning
    groups = agrupar_por_setor_angular(coords, sub, n_grupos)

    # Step 2: Radial sorting (cascading far-to-near connections)
    paths = []
    for g in groups:
        if len(g) == 0:
            continue
        distances = [np.linalg.norm(coords[t] - coords[sub]) for t in g]
        sorted_g = [t for _, t in sorted(zip(distances, g), reverse=True)]
        paths.append(sorted_g + [sub])

    # Step 3: Instantiate Plant and compute parameters
    turbines = [Turbine(P_turb, x, y) for x, y in coords]
    planta = Plant(Vn, turbines, paths)

    # Step 4: Aggregate metrics
    COT_DOLAR = 0.1722  # Historically calibrated exchange rate constant
    total_length = sum(c.lc for c in planta.cables_flat)
    annual_losses_mwh = planta.Pjtot * 8760 / 1e6
    total_losses_kw = planta.Pjtot / 1e3
    section = planta.get_max_calculated_section()

    custo_total = planta.Ctot
    custo_total_usd = custo_total * COT_DOLAR

    resultados = {
        "custo_total_usd": custo_total_usd,
        "comprimento_total_m": total_length,
        "perda_total_kw": total_losses_kw,
        "perda_anual_mwh": annual_losses_mwh,
        "secao_cabo_mm2": section,
        "custo_total": custo_total,
        "secao_mm2": section
    }

    return planta, resultados


if __name__ == "__main__":
    print("Cabling evaluation module loaded successfully.")
