"""
tab_pareto.py — Pareto-front analysis tab for the Wind Farm GUI.

Instantiated by app.py and inserted as the third QTabWidget tab.
"""

import os
import sys
import numpy as np
import yaml

from PyQt5.QtCore    import Qt
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QPushButton, QFileDialog, QGroupBox, QTextEdit,
)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from GUI.widgets import ParetoCanvas, LayoutDetailCanvas

try:
    from core.cabling_v3 import analisar_layout_completo
    _CABLING_OK = True
except Exception:
    _CABLING_OK = False

# ── dark-palette helpers ────────────────────────────────────────────────────
_DARK  = "#0a1628"
_MID   = "#0f2038"
_BORD  = "#1e3050"
_WHITE = "#e2e8f0"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"
_RED   = "#ef4444"


def _label(text, bold=False, color=_WHITE):
    lbl = QLabel(text)
    style = f"color:{color}; font-size:13px;"
    if bold:
        style += " font-weight:bold;"
    lbl.setStyleSheet(style)
    return lbl


def _btn(text, color="#1d4ed8", hover="#2563eb"):
    b = QPushButton(text)
    b.setStyleSheet(
        f"QPushButton{{background:{color};color:white;border-radius:6px;"
        f"padding:6px 14px;font-weight:bold;font-size:13px;}}"
        f"QPushButton:hover{{background:{hover};}}"
        f"QPushButton:disabled{{background:#334155;color:#64748b;}}"
    )
    return b


# ── Tab widget ───────────────────────────────────────────────────────────────
class ParetoTab(QWidget):
    """Pareto-front exploration tab."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._solutions  = []
        self._sim        = None
        self._selected   = None          # currently selected solution object

        # ── main layout ───────────────────────────────────────────────
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        # Title
        title = _label("🎯  Pareto Front — Interactive Solution Explorer", bold=True, color=_AMBER)
        title.setStyleSheet(title.styleSheet() + "font-size:15px;")
        root_layout.addWidget(title)

        hint = _label(
            "Click any point on the Pareto Front to inspect its layout, metrics and export it.",
            color="#94a3b8"
        )
        root_layout.addWidget(hint)

        # ── horizontal splitter: Pareto | Detail ─────────────────────
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet(f"QSplitter::handle{{background:{_BORD};}}")

        # Left – Pareto canvas
        left = QWidget()
        left_v = QVBoxLayout(left)
        left_v.setContentsMargins(0, 0, 4, 0)
        self.pareto_canvas = ParetoCanvas()
        self.pareto_canvas.solution_selected.connect(self._on_solution_selected)
        left_v.addWidget(_label("Pareto Front", bold=True))
        left_v.addWidget(self.pareto_canvas, stretch=1)
        splitter.addWidget(left)

        # Right – detail layout + info
        right = QWidget()
        right_v = QVBoxLayout(right)
        right_v.setContentsMargins(4, 0, 0, 0)
        self.detail_canvas = LayoutDetailCanvas()
        right_v.addWidget(_label("Selected Layout", bold=True))
        right_v.addWidget(self.detail_canvas, stretch=1)

        # Metrics card
        info_box = QGroupBox("Solution Metrics")
        info_box.setStyleSheet(
            f"QGroupBox{{color:{_AMBER};border:1px solid {_BORD};"
            f"border-radius:6px;margin-top:8px;padding:6px;}}"
            f"QGroupBox::title{{subcontrol-origin:margin;left:10px;}}"
        )
        info_v = QVBoxLayout(info_box)
        self.lbl_aep    = _label("Net AEP:  —")
        self.lbl_capex  = _label("CAPEX:    —")
        self.lbl_groups = _label("Groups:   —")
        self.lbl_cable  = _label("Cable len:—")
        for lbl in (self.lbl_aep, self.lbl_capex, self.lbl_groups, self.lbl_cable):
            info_v.addWidget(lbl)
        right_v.addWidget(info_box)

        # Export buttons
        btn_row = QHBoxLayout()
        self.btn_export_yaml  = _btn("💾  Export as YAML",  color="#065f46", hover="#047857")
        self.btn_export_img   = _btn("🖼️  Save Layout PNG",  color="#6b21a8", hover="#7e22ce")
        self.btn_export_yaml.setEnabled(False)
        self.btn_export_img.setEnabled(False)
        self.btn_export_yaml.clicked.connect(self._export_yaml)
        self.btn_export_img.clicked.connect(self._export_img)
        btn_row.addWidget(self.btn_export_yaml)
        btn_row.addWidget(self.btn_export_img)
        right_v.addLayout(btn_row)

        splitter.addWidget(right)
        splitter.setSizes([480, 480])
        root_layout.addWidget(splitter, stretch=1)

        # ── placeholder label ─────────────────────────────────────────
        self.placeholder = _label(
            "Run the simulation to populate the Pareto Front.",
            color="#475569"
        )
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setStyleSheet(self.placeholder.styleSheet() + "font-size:14px;")
        root_layout.addWidget(self.placeholder)

    # ── public API ────────────────────────────────────────────────────
    def load_results(self, solutions: list, sim):
        """Called by app.py when the worker emits sig_done."""
        self._solutions = solutions
        self._sim       = sim
        if not solutions:
            self.placeholder.setText("No valid Pareto solutions found.")
            return
        self.placeholder.hide()
        n_coord = sim.n_turb * 2
        self.pareto_canvas.set_solutions(solutions, sim.n_turb, n_coord)

    # ── slot ─────────────────────────────────────────────────────────
    def _on_solution_selected(self, idx: int):
        if not self._solutions or self._sim is None:
            return
        sol = self._solutions[idx]
        self._selected = sol
        sim  = self._sim
        n_c  = sim.n_turb * 2
        g_n  = sol[n_c]

        n_groups = int(round(sim.min_groups + g_n * (sim.max_groups - sim.min_groups)))
        n_groups = max(sim.min_groups, min(sim.n_turb, n_groups))

        turb_coords = np.array(sol[:n_c]).reshape((sim.n_turb, 2))
        sub_pos = (
            np.array([sol[n_c + 1], sol[n_c + 2]])
            if sim.sub_mode == "optimize"
            else sim.sub_fixed_pos
        )

        aep_gwh   = sol.fitness.values[0] / 1e3
        capex_usd = sol.fitness.values[1]

        # Cable paths
        cable_paths    = []
        cable_len_m    = 0.0
        combined_coords = np.vstack([turb_coords, sub_pos.reshape((1, 2))])
        if _CABLING_OK:
            try:
                planta, res = analisar_layout_completo(
                    combined_coords, sub=sim.n_turb, n_grupos=n_groups
                )
                cable_paths = planta.paths
                cable_len_m = res.get("comprimento_total_m", 0.0)
            except Exception:
                pass

        # Render
        self.detail_canvas.render(
            turb_coords, sub_pos, sim.radius, cable_paths, combined_coords
        )

        # Update metrics
        self.lbl_aep.setText(f"Net AEP:    <b>{aep_gwh:.3f} GWh</b>")
        self.lbl_capex.setText(f"CAPEX:     <b>${capex_usd/1e3:,.1f} kUSD</b>")
        self.lbl_groups.setText(f"Groups:    <b>{n_groups}</b>")
        self.lbl_cable.setText(f"Cable len: <b>{cable_len_m/1e3:.2f} km</b>")
        for lbl in (self.lbl_aep, self.lbl_capex, self.lbl_groups, self.lbl_cable):
            lbl.setTextFormat(Qt.RichText)

        self.btn_export_yaml.setEnabled(True)
        self.btn_export_img.setEnabled(True)

    # ── export helpers ────────────────────────────────────────────────
    def _export_yaml(self):
        if self._selected is None or self._sim is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Solution as YAML", "solution.yaml", "YAML files (*.yaml)"
        )
        if not path:
            return
        sim = self._sim
        n_c = sim.n_turb * 2
        g_n = self._selected[n_c]
        n_groups = int(round(sim.min_groups + g_n * (sim.max_groups - sim.min_groups)))
        turb_coords = np.array(self._selected[:n_c]).reshape((sim.n_turb, 2))
        sub_pos = (
            np.array([self._selected[n_c + 1], self._selected[n_c + 2]])
            if sim.sub_mode == "optimize"
            else sim.sub_fixed_pos
        )
        out = {
            **sim.config,
            "initial_layout_yaml": None,
            "turbine_positions": turb_coords.tolist(),
            "substation": sub_pos.tolist(),
            "cable_groups": {"min_groups": n_groups, "max_groups": n_groups},
        }
        with open(path, "w") as f:
            yaml.dump(out, f)

    def _export_img(self):
        if self._sim is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Layout Image", "layout.png",
            "PNG Image (*.png);;PDF (*.pdf)"
        )
        if not path:
            return
        self.detail_canvas.fig.savefig(
            path, dpi=200, bbox_inches="tight", facecolor="#0a1628"
        )
