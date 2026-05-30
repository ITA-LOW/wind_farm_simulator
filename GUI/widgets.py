"""
widgets.py — Reusable Matplotlib-based PyQt5 canvas widgets.

Provides:
  - FarmCanvas        : live wind-farm layout (turbines + cables + wake).
  - AEPCanvas         : live AEP / CAPEX evolution curves.
  - WindRoseCanvas    : polar wind-rose preview.
  - PowerCurveCanvas  : turbine power-curve preview.
  - ParetoCanvas      : interactive Pareto-front with pick events.
  - LayoutDetailCanvas: static layout renderer for a selected Pareto solution.
"""

import numpy as np
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import Circle
from matplotlib.figure import Figure

# Dark-theme palette shared by all widgets
BG_DARK   = "#0a1628"
BG_MID    = "#0f2038"
GRID_COL  = "#1e3050"
CLR_P1    = "#3b82f6"   # Phase-1 AEP (blue)
CLR_P2    = "#22c55e"   # Phase-2 Net AEP (green)
CLR_CAP   = "#ef4444"   # CAPEX (red)
CLR_TRB   = "#e2e8f0"   # turbine dots
CLR_SUB   = "#f97316"   # substation square
CLR_KNEEL = "#f59e0b"   # knee / highlight


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(BG_MID)
    ax.tick_params(colors="white", labelsize=8)
    ax.set_xlabel(xlabel, color="white", fontsize=9)
    ax.set_ylabel(ylabel, color="white", fontsize=9)
    ax.set_title(title, color="white", fontsize=10, fontweight="bold")
    ax.grid(True, color=GRID_COL, lw=0.5, linestyle=":")
    for s in ax.spines.values():
        s.set_edgecolor(GRID_COL)


# ─────────────────────────────────────────────────────────────────────────────
class FarmCanvas(FigureCanvas):
    """Live wind-farm layout canvas — updated every generation."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Wind Farm Layout", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        self._radius = 1300.0
        self._phase  = 1
        self._boundary = None
        self._draw_empty()

    def _draw_empty(self):
        self.ax.clear()
        _style_ax(self.ax, "Wind Farm Layout", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        r = self._radius
        self._boundary = Circle(
            (0, 0), r, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2
        )
        self.ax.add_patch(self._boundary)
        self.ax.set_xlim(-r - 200, r + 200)
        self.ax.set_ylim(-r - 200, r + 200)
        self.draw_idle()

    def set_radius(self, radius: float):
        self._radius = radius
        self._draw_empty()

    def update_p1(self, turb_coords: np.ndarray):
        """Update layout with Phase-1 turbine positions."""
        self._phase = 1
        self.ax.clear()
        _style_ax(self.ax, "Phase 1 — Layout Optimization", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        r = self._radius
        self.ax.add_patch(
            Circle((0, 0), r, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2)
        )
        self.ax.set_xlim(-r - 200, r + 200)
        self.ax.set_ylim(-r - 200, r + 200)
        self.ax.scatter(
            turb_coords[:, 0], turb_coords[:, 1],
            s=50, color=CLR_TRB, edgecolors=BG_DARK, linewidths=0.8, zorder=5
        )
        self.draw_idle()

    def update_p2(
        self,
        turb_coords: np.ndarray,
        sub_pos: np.ndarray,
        n_groups: int,
        cable_paths: list,
        combined_coords: np.ndarray = None,
    ):
        """Update layout with Phase-2 turbines, substation and cables."""
        self._phase = 2
        self.ax.clear()
        _style_ax(self.ax, "Phase 2 — Co-design Optimization", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        r = self._radius
        self.ax.add_patch(
            Circle((0, 0), r, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2)
        )
        self.ax.set_xlim(-r - 200, r + 200)
        self.ax.set_ylim(-r - 200, r + 200)

        # Draw cables
        if cable_paths and combined_coords is not None:
            cmap = plt.cm.tab10
            for idx, path in enumerate(cable_paths):
                xs = [combined_coords[k, 0] for k in path]
                ys = [combined_coords[k, 1] for k in path]
                self.ax.plot(xs, ys, "-", color=cmap(idx % 10), lw=1.8, zorder=3, alpha=0.85)

        self.ax.scatter(
            turb_coords[:, 0], turb_coords[:, 1],
            s=50, color=CLR_TRB, edgecolors=BG_DARK, linewidths=0.8, zorder=5
        )
        self.ax.scatter(
            sub_pos[0], sub_pos[1],
            s=160, color=CLR_SUB, marker="s", edgecolors="white", zorder=10,
            label="Substation"
        )
        leg = self.ax.legend(loc="lower right", facecolor=BG_DARK, edgecolor=GRID_COL, fontsize=8)
        for t in leg.get_texts():
            t.set_color("white")
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
class AEPCanvas(FigureCanvas):
    """Live AEP & CAPEX evolution curves canvas."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Optimization Evolution", "Generation", "AEP [GWh]")
        self.ax_cost = None
        self._x1, self._y1 = [], []
        self._x2, self._y2, self._y_cap = [], [], []
        self._p1_done = False
        self.draw_idle()

    def add_p1_point(self, gen: int, aep_gwh: float):
        self._x1.append(gen)
        self._y1.append(aep_gwh)
        self._refresh()

    def add_p2_point(self, gen: int, net_gwh: float, capex_kusd: float):
        self._p1_done = True
        offset = self._x1[-1] if self._x1 else 0
        self._x2.append(offset + gen)
        self._y2.append(net_gwh)
        self._y_cap.append(capex_kusd)
        self._refresh()

    def reset(self):
        self._x1, self._y1 = [], []
        self._x2, self._y2, self._y_cap = [], [], []
        self._p1_done = False
        self._refresh()

    def _refresh(self):
        self.ax.clear()
        if self.ax_cost:
            self.ax_cost.remove()
            self.ax_cost = None

        _style_ax(self.ax, "Optimization Evolution", "Generation", "AEP [GWh]")

        if self._x1:
            self.ax.plot(self._x1, self._y1, color=CLR_P1, lw=2.0, label="Phase 1 — Gross AEP")

        if self._x2:
            self.ax.axvline(self._x2[0], color=CLR_KNEEL, lw=1.2, linestyle="--", alpha=0.7)
            self.ax.plot(self._x2, self._y2, color=CLR_P2, lw=2.0, label="Phase 2 — Net AEP")

            self.ax_cost = self.ax.twinx()
            self.ax_cost.set_ylabel("CAPEX [kUSD]", color=CLR_CAP, fontsize=9)
            self.ax_cost.tick_params(colors=CLR_CAP, labelsize=8)
            self.ax_cost.grid(False)
            self.ax_cost.spines["right"].set_edgecolor(CLR_CAP)
            self.ax_cost.plot(
                self._x2, self._y_cap, color=CLR_CAP, lw=1.8, linestyle="-",
                label="Phase 2 — Cabling CAPEX"
            )

        lines, labels = self.ax.get_legend_handles_labels()
        if self.ax_cost:
            l2, lab2 = self.ax_cost.get_legend_handles_labels()
            lines += l2; labels += lab2
        if lines:
            leg = self.ax.legend(
                lines, labels, loc="lower right", facecolor=BG_DARK,
                edgecolor=GRID_COL, fontsize=7
            )
            for t in leg.get_texts():
                t.set_color("white")

        self.fig.tight_layout()
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
class WindRoseCanvas(FigureCanvas):
    """Polar wind-rose preview canvas, updates on data change."""

    DIRS_DEG = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5,
                180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4, 4), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="polar")
        self.ax.set_facecolor(BG_MID)
        self._freqs  = [1 / 16] * 16
        self._speeds = [9.8] * 16
        self.refresh()

    def set_data(self, freqs: list, speeds: list):
        self._freqs  = list(freqs)
        self._speeds = list(speeds)
        self.refresh()

    def refresh(self):
        self.ax.clear()
        self.ax.set_facecolor(BG_MID)
        angles = [np.deg2rad(d) for d in self.DIRS_DEG]
        width  = np.deg2rad(22.5)

        # Color bars by wind speed
        speeds = np.array(self._speeds)
        s_min, s_max = speeds.min(), speeds.max()
        norm = plt.Normalize(s_min, s_max + 0.01)
        cmap = plt.cm.plasma

        for angle, freq, speed in zip(angles, self._freqs, self._speeds):
            color = cmap(norm(speed))
            self.ax.bar(angle, freq, width=width, bottom=0.0, color=color, alpha=0.85, edgecolor="none")

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(-1)
        self.ax.tick_params(colors="white", labelsize=8)
        self.ax.set_yticklabels([])
        self.ax.set_title("Wind Rose", color="white", fontsize=10, fontweight="bold", pad=14)
        self.ax.spines["polar"].set_color(GRID_COL)

        # Colorbar for speed
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        try:
            self.fig.colorbar(sm, ax=self.ax, pad=0.12, shrink=0.7,
                              label="Wind Speed [m/s]").ax.yaxis.label.set_color("white")
        except Exception:
            pass

        self.fig.tight_layout()
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
class PowerCurveCanvas(FigureCanvas):
    """Turbine power-curve + Ct curve preview canvas."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(4.5, 3.5), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax  = self.fig.add_subplot(111)
        self.ax2 = self.ax.twinx()
        _style_ax(self.ax, "Power Curve", "Wind Speed [m/s]", "Power [MW]")
        self.ax2.tick_params(colors=CLR_KNEEL, labelsize=8)
        self.ax2.set_ylabel("Thrust Coeff. Ct", color=CLR_KNEEL, fontsize=9)
        self.refresh(rated_pwr=3.35, ci=4.0, co=25.0, rated_ws=9.8)

    def refresh(self, rated_pwr: float, ci: float, co: float, rated_ws: float):
        ws = np.linspace(0, co + 2, 200)
        pwr = np.where(
            (ws >= ci) & (ws <= rated_ws),
            rated_pwr * ((ws - ci) / (rated_ws - ci)) ** 3,
            np.where(ws > rated_ws, rated_pwr, 0.0),
        )
        # Simplified Ct
        ct = np.where(
            (ws >= ci) & (ws <= rated_ws), 0.8,
            np.where(ws > rated_ws, 0.3, 0.0)
        )

        self.ax.clear()
        self.ax2.clear()
        _style_ax(self.ax, "Power & Thrust Curves", "Wind Speed [m/s]", "Power [MW]")
        self.ax2.tick_params(colors=CLR_KNEEL, labelsize=8)
        self.ax2.set_ylabel("Thrust Coeff. Ct", color=CLR_KNEEL, fontsize=9)
        self.ax2.grid(False)

        self.ax.axvline(ci, color="gray", linestyle=":", lw=0.9, alpha=0.7)
        self.ax.axvline(co, color="gray", linestyle=":", lw=0.9, alpha=0.7)
        self.ax.axvline(rated_ws, color=CLR_P1, linestyle=":", lw=0.9, alpha=0.7)

        self.ax.plot(ws, pwr, color=CLR_P2, lw=2.0, label=f"Power (max {rated_pwr:.2f} MW)")
        self.ax2.plot(ws, ct, color=CLR_KNEEL, lw=1.6, linestyle="--", label="Ct")

        l1, lab1 = self.ax.get_legend_handles_labels()
        l2, lab2 = self.ax2.get_legend_handles_labels()
        leg = self.ax.legend(l1 + l2, lab1 + lab2, loc="upper left", facecolor=BG_DARK,
                             edgecolor=GRID_COL, fontsize=8)
        for t in leg.get_texts():
            t.set_color("white")

        self.fig.tight_layout()
        self.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
class ParetoCanvas(FigureCanvas):
    """Interactive Pareto-front canvas with click-to-select."""

    # Emits the index into the solutions list when the user clicks a point
    from PyQt5.QtCore import pyqtSignal as _pqs
    solution_selected = _pqs(int)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Pareto Front", "Cable CAPEX [kUSD]", "Net AEP [GWh]")
        self._solutions = []
        self._aeps      = np.array([])
        self._capexs    = np.array([])
        self._scatter   = None
        self.mpl_connect("pick_event", self._on_pick)

    def set_solutions(self, solutions: list, n_turb: int, n_coord: int):
        """Populate the canvas with Pareto solutions."""
        self._solutions = solutions
        self._n_turb    = n_turb
        self._n_coord   = n_coord

        if not solutions:
            return

        self._aeps   = np.array([s.fitness.values[0] / 1e3 for s in solutions])
        self._capexs = np.array([s.fitness.values[1] / 1e3 for s in solutions])

        self.ax.clear()
        _style_ax(self.ax, "Pareto Front  (click a point!)",
                  "Cable CAPEX [kUSD]", "Net AEP [GWh]")

        sort_idx = np.argsort(self._capexs)
        self.ax.plot(
            self._capexs[sort_idx], self._aeps[sort_idx],
            color="#38bdf8", alpha=0.4, lw=1.5
        )
        self._scatter = self.ax.scatter(
            self._capexs, self._aeps,
            color="#38bdf8", edgecolors=BG_MID, s=70, zorder=5,
            picker=8, label="Pareto solutions"
        )

        # Mark Knee Point
        if len(solutions) > 1:
            a_range = (self._aeps.max() - self._aeps.min()) or 1.0
            c_range = (self._capexs.max() - self._capexs.min()) or 1.0
            dists = np.sqrt(
                ((self._aeps.max() - self._aeps) / a_range) ** 2
                + ((self._capexs - self._capexs.min()) / c_range) ** 2
            )
            ki = np.argmin(dists)
            self.ax.scatter(
                self._capexs[ki], self._aeps[ki],
                color=CLR_KNEEL, edgecolors="white", marker="*", s=260, zorder=10,
                label="Knee Point"
            )

        leg = self.ax.legend(loc="lower right", facecolor=BG_DARK, edgecolor=GRID_COL, fontsize=8)
        for t in leg.get_texts():
            t.set_color("white")

        self.fig.tight_layout()
        self.draw_idle()

    def _on_pick(self, event):
        if event.artist is not self._scatter:
            return
        if len(event.ind) == 0:
            return
        idx = int(event.ind[0])
        # Highlight selection
        colors = ["#38bdf8"] * len(self._solutions)
        colors[idx] = CLR_KNEEL
        self._scatter.set_facecolors(colors)
        self.draw_idle()
        self.solution_selected.emit(idx)


# ─────────────────────────────────────────────────────────────────────────────
class LayoutDetailCanvas(FigureCanvas):
    """Static layout renderer for a selected Pareto solution."""

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 4.5), facecolor=BG_DARK)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)
        _style_ax(self.ax, "Selected Solution Layout", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        self.draw_idle()

    def render(
        self,
        turb_coords: np.ndarray,
        sub_pos: np.ndarray,
        radius: float,
        cable_paths: list,
        combined_coords: np.ndarray = None,
    ):
        self.ax.clear()
        _style_ax(self.ax, "Selected Solution Layout", "X [m]", "Y [m]")
        self.ax.set_aspect("equal")
        self.ax.add_patch(
            Circle((0, 0), radius, fill=False, linestyle="--", color="white", alpha=0.3, lw=1.2)
        )
        self.ax.set_xlim(-radius - 200, radius + 200)
        self.ax.set_ylim(-radius - 200, radius + 200)

        # Cables
        if cable_paths and combined_coords is not None:
            cmap = plt.cm.tab10
            for idx, path in enumerate(cable_paths):
                xs = [combined_coords[k, 0] for k in path]
                ys = [combined_coords[k, 1] for k in path]
                self.ax.plot(xs, ys, "-", color=cmap(idx % 10), lw=2.0, zorder=3)

        self.ax.scatter(
            turb_coords[:, 0], turb_coords[:, 1],
            s=55, color=CLR_TRB, edgecolors=BG_DARK, linewidths=0.8, zorder=5,
            label=f"Turbines (n={len(turb_coords)})"
        )
        self.ax.scatter(
            sub_pos[0], sub_pos[1],
            s=160, color=CLR_SUB, marker="s", edgecolors="white", zorder=10,
            label="Substation"
        )
        leg = self.ax.legend(loc="lower right", facecolor=BG_DARK, edgecolor=GRID_COL, fontsize=8)
        for t in leg.get_texts():
            t.set_color("white")

        self.fig.tight_layout()
        self.draw_idle()
