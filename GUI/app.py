"""
app.py — Main PyQt5 window for the Wind Farm Optimizer GUI.
Run from the repository root:  python GUI/app.py
"""
import os, sys, time, copy, yaml
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from PyQt5.QtCore    import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QLabel, QPushButton, QLineEdit, QSpinBox, QDoubleSpinBox,
    QComboBox, QFileDialog, QProgressBar, QTextEdit,
    QGroupBox, QTableWidget, QTableWidgetItem, QHeaderView,
    QCheckBox, QScrollArea, QSizePolicy, QMessageBox,
)

from GUI.widgets    import FarmCanvas, AEPCanvas, WindRoseCanvas, PowerCurveCanvas
from GUI.tab_pareto import ParetoTab
from GUI.worker     import SimWorker

# ── palette ──────────────────────────────────────────────────────────────────
_BG    = "#0a1628"
_MID   = "#0f2038"
_BORD  = "#1e3050"
_WHITE = "#e2e8f0"
_AMBER = "#f59e0b"
_GREEN = "#22c55e"
_RED   = "#ef4444"
_BLUE  = "#3b82f6"

_GLOBAL_SS = f"""
QWidget      {{ background:{_BG}; color:{_WHITE}; font-family:'Segoe UI',sans-serif; font-size:13px; }}
QTabWidget::pane {{ border:1px solid {_BORD}; }}
QTabBar::tab  {{ background:{_MID}; color:#94a3b8; padding:8px 18px; border-radius:4px 4px 0 0; }}
QTabBar::tab:selected {{ background:{_BG}; color:{_WHITE}; border-bottom:2px solid {_BLUE}; }}
QGroupBox     {{ border:1px solid {_BORD}; border-radius:6px; margin-top:10px; padding:8px; color:{_AMBER}; }}
QGroupBox::title {{ subcontrol-origin:margin; left:10px; }}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background:{_MID}; color:{_WHITE}; border:1px solid {_BORD}; border-radius:4px; padding:3px 6px; }}
QScrollBar:vertical {{ background:{_MID}; width:8px; }}
QScrollBar::handle:vertical {{ background:{_BORD}; border-radius:4px; }}
QTableWidget  {{ background:{_MID}; gridline-color:{_BORD}; color:{_WHITE}; }}
QHeaderView::section {{ background:{_BG}; color:{_AMBER}; border:1px solid {_BORD}; padding:4px; }}
QTextEdit     {{ background:{_MID}; color:{_WHITE}; border:1px solid {_BORD}; border-radius:4px; font-family:monospace; }}
QProgressBar  {{ background:{_MID}; border:1px solid {_BORD}; border-radius:4px; text-align:center; color:{_WHITE}; }}
QProgressBar::chunk {{ background:{_BLUE}; border-radius:4px; }}
"""

def _btn(text, color=_BLUE, hover="#2563eb", w=None):
    b = QPushButton(text)
    s = (f"QPushButton{{background:{color};color:white;border-radius:6px;"
         f"padding:6px 16px;font-weight:bold;}}"
         f"QPushButton:hover{{background:{hover};}}"
         f"QPushButton:disabled{{background:#334155;color:#64748b;}}")
    b.setStyleSheet(s)
    if w:
        b.setFixedWidth(w)
    return b

def _lbl(text, bold=False, color=_WHITE):
    l = QLabel(text)
    s = f"color:{color};font-size:13px;"
    if bold:
        s += "font-weight:bold;"
    l.setStyleSheet(s)
    return l

def _spin(lo, hi, val, dec=0, step=1):
    if dec == 0:
        w = QSpinBox(); w.setRange(int(lo), int(hi)); w.setValue(int(val))
    else:
        w = QDoubleSpinBox(); w.setRange(lo, hi); w.setDecimals(dec)
        w.setSingleStep(step); w.setValue(val)
    return w

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT config mirroring cases/case_example.yaml
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "name": "My wind farm simulation",
    "n_turbines": 16,
    "boundary_radius": 1300.0,
    "min_spacing_multiplier": 2.0,
    "turbine_yaml": "config/iea37-335mw.yaml",
    "windrose_yaml": "config/iea37-windrose.yaml",
    "initial_layout_yaml": "config/iea37-ex16.yaml",
    "population_size": 300,
    "crossover_probability": 0.95,
    "mutation_probability": 0.7,
    "mutation_sigma": 100.0,
    "mu": 0.0,
    "mutation_indpb": 0.4,
    "tournament_size": 5,
    "alpha": 0.5,
    "plateau_generations_p1": 100,
    "plateau_generations_p2": 200,
    "substation": "optimize",
    "cable_groups": {"min_groups": 2, "max_groups": 7},
    # wind rose (16 bins)
    "_wr_freqs":  [.025,.024,.029,.036,.063,.065,.100,.122,
                   .063,.038,.039,.083,.213,.046,.032,.022],
    "_wr_speeds": [9.8]*16,
    # turbine specs
    "_turb_rated_mw": 3.35,
    "_turb_diam_m":   130.0,
    "_turb_ci":       4.0,
    "_turb_co":       25.0,
    "_turb_rated_ws": 9.8,
}

WR_DIRS = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
           "S","SSW","SW","WSW","W","WNW","NW","NNW"]

# ─────────────────────────────────────────────────────────────────────────────
class ConfigTab(QWidget):
    """Aba 1 — Configuration with sub-tabs for GA params, Wind Rose, Turbine."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.cfg = copy.deepcopy(DEFAULT_CFG)
        self._build_ui()

    # ── build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)

        # Top bar — load / save YAML
        top = QHBoxLayout()
        top.addWidget(_lbl("📁  Case file:", bold=True))
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("No file loaded — using defaults")
        self.path_edit.setReadOnly(True)
        top.addWidget(self.path_edit, stretch=1)
        b_load = _btn("Load YAML", color="#065f46", hover="#047857")
        b_save = _btn("Save YAML", color="#1e3a5f", hover="#1d4ed8")
        b_load.clicked.connect(self._load_yaml)
        b_save.clicked.connect(self._save_yaml)
        top.addWidget(b_load); top.addWidget(b_save)
        root.addLayout(top)

        # Sub-tabs
        self.sub_tabs = QTabWidget()
        self.sub_tabs.addTab(self._build_ga_tab(),  "⚙️  GA Parameters")
        self.sub_tabs.addTab(self._build_wr_tab(),   "🌬️  Wind Rose")
        self.sub_tabs.addTab(self._build_turb_tab(), "🔧  Turbine Specs")
        root.addWidget(self.sub_tabs, stretch=1)

    # ── GA params sub-tab ─────────────────────────────────────────────
    def _build_ga_tab(self):
        w = QScrollArea(); w.setWidgetResizable(True)
        inner = QWidget(); gl = QGridLayout(inner)
        gl.setSpacing(8)

        rows = [
            ("Simulation name",        "name",                  "str"),
            ("Turbines",               "n_turbines",            (1, 100, 16)),
            ("Boundary radius [m]",    "boundary_radius",       (100., 10000., 1300., 1, 50.)),
            ("Min spacing multiplier", "min_spacing_multiplier",(0.5, 10., 2., 2, 0.1)),
            ("Population size",        "population_size",       (10, 2000, 300)),
            ("Crossover prob.",        "crossover_probability", (0., 1., 0.95, 2, 0.01)),
            ("Mutation prob.",         "mutation_probability",  (0., 1., 0.7,  2, 0.01)),
            ("Mutation sigma",         "mutation_sigma",        (1., 2000., 100., 1, 10.)),
            ("Mutation indpb",         "mutation_indpb",        (0., 1., 0.4,  2, 0.01)),
            ("Tournament size",        "tournament_size",       (2, 20, 5)),
            ("Alpha (blend CX)",       "alpha",                 (0., 1., 0.5,  2, 0.05)),
            ("Plateau gens P1",        "plateau_generations_p1",(5, 1000, 100)),
            ("Plateau gens P2",        "plateau_generations_p2",(5, 1000, 200)),
            ("Min cable groups",       "_min_groups",           (1, 50, 2)),
            ("Max cable groups",       "_max_groups",           (1, 50, 7)),
            ("Substation mode",        "_sub_mode",             ["optimize","fixed"]),
            ("Substation X [m]",       "_sub_x",                (-5000., 5000., -1350., 1, 50.)),
            ("Substation Y [m]",       "_sub_y",                (-5000., 5000., 0., 1, 50.)),
        ]

        self._ga_widgets = {}
        for r, (label, key, spec) in enumerate(rows):
            gl.addWidget(_lbl(label), r, 0)
            if spec == "str":
                e = QLineEdit(str(self.cfg.get(key,"")))
                e.editingFinished.connect(lambda k=key, w2=e: self.cfg.update({k: w2.text()}))
                gl.addWidget(e, r, 1)
                self._ga_widgets[key] = e
            elif isinstance(spec, list):
                cb = QComboBox(); cb.addItems(spec)
                sub = self.cfg.get("substation","optimize")
                cb.setCurrentText(sub if isinstance(sub,str) else "fixed")
                gl.addWidget(cb, r, 1)
                self._ga_widgets[key] = cb
            else:
                ww = _spin(*spec)
                gl.addWidget(ww, r, 1)
                self._ga_widgets[key] = ww

        gl.setRowStretch(len(rows), 1)
        w.setWidget(inner)
        return w

    # ── wind rose sub-tab ─────────────────────────────────────────────
    def _build_wr_tab(self):
        w = QWidget(); h = QHBoxLayout(w)

        # Table
        left = QWidget(); lv = QVBoxLayout(left)
        lv.addWidget(_lbl("Edit wind rose values (16 directions):", bold=True))
        self.wr_table = QTableWidget(16, 3)
        self.wr_table.setHorizontalHeaderLabels(["Direction","Frequency","Speed [m/s]"])
        self.wr_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.wr_table.setFixedWidth(380)
        for i, (d, f, s) in enumerate(zip(
                WR_DIRS,
                self.cfg["_wr_freqs"],
                self.cfg["_wr_speeds"])):
            self.wr_table.setItem(i, 0, QTableWidgetItem(d))
            self.wr_table.setItem(i, 1, QTableWidgetItem(f"{f:.4f}"))
            self.wr_table.setItem(i, 2, QTableWidgetItem(f"{s:.2f}"))
            self.wr_table.item(i, 0).setFlags(Qt.ItemIsEnabled)  # direction not editable
        self.wr_table.cellChanged.connect(self._wr_changed)
        lv.addWidget(self.wr_table, stretch=1)
        h.addWidget(left)

        # Canvas
        right = QWidget(); rv = QVBoxLayout(right)
        rv.addWidget(_lbl("Live Preview:", bold=True))
        self.wr_canvas = WindRoseCanvas()
        rv.addWidget(self.wr_canvas, stretch=1)
        h.addWidget(right, stretch=1)
        return w

    # ── turbine sub-tab ───────────────────────────────────────────────
    def _build_turb_tab(self):
        w = QWidget(); h = QHBoxLayout(w)

        left = QGroupBox("Turbine Specifications")
        gl = QGridLayout(left)
        specs = [
            ("Rated power [MW]", "_turb_rated_mw", (0.1, 20., 3.35, 2, 0.05)),
            ("Rotor diameter [m]","_turb_diam_m",  (10., 300., 130., 1, 5.)),
            ("Cut-in  [m/s]",     "_turb_ci",      (0.,  10.,  4.,  1, 0.5)),
            ("Cut-out [m/s]",     "_turb_co",      (10., 50.,  25., 1, 1.)),
            ("Rated WS [m/s]",    "_turb_rated_ws",(4.,  30.,  9.8, 1, 0.1)),
        ]
        self._turb_widgets = {}
        for r, (lbl, key, sp) in enumerate(specs):
            gl.addWidget(_lbl(lbl), r, 0)
            ww = _spin(*sp)
            ww.valueChanged.connect(self._turb_changed)
            gl.addWidget(ww, r, 1)
            self._turb_widgets[key] = ww
        gl.setRowStretch(len(specs), 1)
        h.addWidget(left)

        right = QWidget(); rv = QVBoxLayout(right)
        rv.addWidget(_lbl("Power & Ct Curves:", bold=True))
        self.pc_canvas = PowerCurveCanvas()
        rv.addWidget(self.pc_canvas, stretch=1)
        h.addWidget(right, stretch=1)
        return w

    # ── slots ─────────────────────────────────────────────────────────
    def _wr_changed(self, row, col):
        try:
            if col == 1:
                self.cfg["_wr_freqs"][row] = float(self.wr_table.item(row,1).text())
            elif col == 2:
                self.cfg["_wr_speeds"][row] = float(self.wr_table.item(row,2).text())
        except (ValueError, TypeError):
            return
        self.wr_canvas.set_data(self.cfg["_wr_freqs"], self.cfg["_wr_speeds"])

    def _turb_changed(self):
        for k, ww in self._turb_widgets.items():
            self.cfg[k] = ww.value()
        self.pc_canvas.refresh(
            self.cfg["_turb_rated_mw"],
            self.cfg["_turb_ci"],
            self.cfg["_turb_co"],
            self.cfg["_turb_rated_ws"],
        )

    # ── YAML I/O ──────────────────────────────────────────────────────
    def _load_yaml(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Case YAML",
            os.path.join(ROOT, "cases"), "YAML files (*.yaml *.yml)"
        )
        if not path:
            return
        with open(path) as f:
            loaded = yaml.safe_load(f) or {}
        self.path_edit.setText(path)

        # Merge into default config preserving private keys
        for k, v in loaded.items():
            self.cfg[k] = v
        if isinstance(self.cfg.get("cable_groups"), dict):
            self.cfg["_min_groups"] = self.cfg["cable_groups"].get("min_groups", 2)
            self.cfg["_max_groups"] = self.cfg["cable_groups"].get("max_groups", 7)
        sub = self.cfg.get("substation", "optimize")
        if isinstance(sub, list):
            self.cfg["_sub_mode"] = "fixed"
            self.cfg["_sub_x"]    = sub[0]
            self.cfg["_sub_y"]    = sub[1]
        else:
            self.cfg["_sub_mode"] = "optimize"
        self._sync_widgets_from_cfg()

    def _sync_widgets_from_cfg(self):
        """Push cfg values back into UI widgets after a YAML load."""
        for key, ww in self._ga_widgets.items():
            val = self.cfg.get(key)
            if val is None:
                continue
            if isinstance(ww, (QSpinBox, QDoubleSpinBox)):
                try:
                    ww.setValue(float(val))
                except Exception:
                    pass
            elif isinstance(ww, QLineEdit):
                ww.setText(str(val))
            elif isinstance(ww, QComboBox):
                ww.setCurrentText(str(val))
        # wind rose
        self.wr_canvas.set_data(self.cfg["_wr_freqs"], self.cfg["_wr_speeds"])
        for i in range(16):
            self.wr_table.blockSignals(True)
            self.wr_table.item(i,1).setText(f"{self.cfg['_wr_freqs'][i]:.4f}")
            self.wr_table.item(i,2).setText(f"{self.cfg['_wr_speeds'][i]:.2f}")
            self.wr_table.blockSignals(False)
        # turbine
        for k, ww in self._turb_widgets.items():
            if k in self.cfg:
                ww.blockSignals(True)
                ww.setValue(float(self.cfg[k]))
                ww.blockSignals(False)
        self.pc_canvas.refresh(
            self.cfg["_turb_rated_mw"], self.cfg["_turb_ci"],
            self.cfg["_turb_co"],       self.cfg["_turb_rated_ws"],
        )

    def _save_yaml(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Case YAML", os.path.join(ROOT,"cases","new_case.yaml"),
            "YAML files (*.yaml)"
        )
        if not path:
            return
        out = self.build_sim_config()
        with open(path,"w") as f:
            yaml.dump(out, f)

    def build_sim_config(self) -> dict:
        """Collect all widget values and return a clean config dict."""
        cfg = {}
        for key, ww in self._ga_widgets.items():
            if key.startswith("_"):
                continue
            if isinstance(ww, (QSpinBox, QDoubleSpinBox)):
                cfg[key] = ww.value()
            elif isinstance(ww, QLineEdit):
                cfg[key] = ww.text()
            elif isinstance(ww, QComboBox):
                cfg[key] = ww.currentText()

        cfg["turbine_yaml"]   = self.cfg.get("turbine_yaml",  "config/iea37-335mw.yaml")
        cfg["windrose_yaml"]  = self.cfg.get("windrose_yaml", "config/iea37-windrose.yaml")
        cfg["initial_layout_yaml"] = self.cfg.get("initial_layout_yaml")

        sub_mode = self._ga_widgets["_sub_mode"].currentText()
        if sub_mode == "fixed":
            sx = self._ga_widgets["_sub_x"].value()
            sy = self._ga_widgets["_sub_y"].value()
            cfg["substation"] = [sx, sy]
        else:
            cfg["substation"] = "optimize"

        cfg["cable_groups"] = {
            "min_groups": int(self._ga_widgets["_min_groups"].value()),
            "max_groups": int(self._ga_widgets["_max_groups"].value()),
        }
        # keep private turbine/wr data in cfg for worker
        cfg["_wr_freqs"]     = self.cfg["_wr_freqs"]
        cfg["_wr_speeds"]    = self.cfg["_wr_speeds"]
        cfg["_turb_rated_mw"]= self.cfg["_turb_rated_mw"]
        cfg["_turb_diam_m"]  = self.cfg["_turb_diam_m"]
        cfg["_turb_ci"]      = self.cfg["_turb_ci"]
        cfg["_turb_co"]      = self.cfg["_turb_co"]
        cfg["_turb_rated_ws"]= self.cfg["_turb_rated_ws"]
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
class SimTab(QWidget):
    """Aba 2 — Live simulation view: layout, AEP curves, progress, log."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker   = None
        self._t_start  = None
        self._max_gens = 1000
        self._build_ui()

    # ── build ─────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        # ── control bar ───────────────────────────────────────────────
        ctrl = QHBoxLayout()
        self.btn_start  = _btn("▶  Start",  color="#065f46", hover="#047857", w=130)
        self.btn_pause  = _btn("⏸  Pause",  color="#92400e", hover="#b45309", w=130)
        self.btn_stop   = _btn("⏹  Stop",   color="#7f1d1d", hover="#991b1b", w=130)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.lbl_phase = _lbl("Phase: —", bold=True, color=_AMBER)
        self.lbl_gen   = _lbl("Gen: —",  color=_WHITE)
        self.lbl_eta   = _lbl("ETA: —",  color="#94a3b8")

        for w in (self.btn_start, self.btn_pause, self.btn_stop,
                  _lbl("  "), self.lbl_phase, self.lbl_gen, self.lbl_eta):
            ctrl.addWidget(w)
        ctrl.addStretch()
        root.addLayout(ctrl)

        # ── progress bar ──────────────────────────────────────────────
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFixedHeight(18)
        root.addWidget(self.progress)

        # ── splitter: charts | log ────────────────────────────────────
        main_split = QSplitter(Qt.Horizontal)
        main_split.setStyleSheet(f"QSplitter::handle{{background:{_BORD};}}")

        # Left: farm + AEP stacked
        left_split = QSplitter(Qt.Vertical)
        left_split.setStyleSheet(f"QSplitter::handle{{background:{_BORD};}}")

        farm_wrap = QWidget(); fv = QVBoxLayout(farm_wrap); fv.setContentsMargins(0,0,0,0)
        fv.addWidget(_lbl("🗺️  Wind Farm Layout", bold=True))
        self.farm_canvas = FarmCanvas()
        fv.addWidget(self.farm_canvas, stretch=1)
        left_split.addWidget(farm_wrap)

        aep_wrap = QWidget(); av = QVBoxLayout(aep_wrap); av.setContentsMargins(0,0,0,0)
        av.addWidget(_lbl("📈  AEP & CAPEX Evolution", bold=True))
        self.aep_canvas = AEPCanvas()
        av.addWidget(self.aep_canvas, stretch=1)
        left_split.addWidget(aep_wrap)
        left_split.setSizes([400, 300])
        main_split.addWidget(left_split)

        # Right: live stats + console
        right = QWidget(); rv = QVBoxLayout(right)
        rv.addWidget(_lbl("📊  Live Statistics", bold=True, color=_AMBER))

        stats_box = QGroupBox("Current Best")
        sg = QGridLayout(stats_box)
        self.lbl_aep_val   = _lbl("—", color=_GREEN)
        self.lbl_capex_val = _lbl("—", color=_RED)
        self.lbl_elapsed   = _lbl("—", color="#94a3b8")
        for r, (lbl_text, w2) in enumerate([
            ("Net AEP [GWh]:", self.lbl_aep_val),
            ("CAPEX [kUSD]:",  self.lbl_capex_val),
            ("Elapsed:",       self.lbl_elapsed),
        ]):
            sg.addWidget(_lbl(lbl_text), r, 0)
            sg.addWidget(w2, r, 1)
        rv.addWidget(stats_box)

        rv.addWidget(_lbl("🖥️  Log Console", bold=True, color=_AMBER))
        self.console = QTextEdit()
        self.console.setReadOnly(True)
        self.console.setMinimumWidth(280)
        rv.addWidget(self.console, stretch=1)
        main_split.addWidget(right)
        main_split.setSizes([700, 280])

        root.addWidget(main_split, stretch=1)

        # Elapsed timer
        self._timer = QTimer()
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick_elapsed)

        # Connect buttons
        self.btn_pause.clicked.connect(self._toggle_pause)
        self.btn_stop.clicked.connect(self._stop)

    # ── public API ────────────────────────────────────────────────────
    def start(self, cfg: dict, output_dir: str, max_gens: int, on_done, on_error):
        """Called by MainWindow to start the worker."""
        self.aep_canvas.reset()
        self.farm_canvas.set_radius(cfg.get("boundary_radius", 1300.0))
        self.console.clear()
        self._log("Starting simulation …")
        self._max_gens = max_gens
        self._t_start  = time.time()
        self._timer.start()

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)

        self._worker = SimWorker(cfg, output_dir, max_gens)
        self._worker.sig_log.connect(self._log)
        self._worker.sig_p1_frame.connect(self.farm_canvas.update_p1)
        self._worker.sig_p1_aep.connect(self._on_p1_aep)
        self._worker.sig_p2_frame.connect(self._on_p2_frame)
        self._worker.sig_p2_metrics.connect(self._on_p2_metrics)
        self._worker.sig_progress.connect(self._on_progress)
        self._worker.sig_phase_change.connect(self._on_phase)
        self._worker.sig_eta.connect(self._on_eta)
        self._worker.sig_done.connect(on_done)
        self._worker.sig_done.connect(self._on_worker_done)
        self._worker.sig_error.connect(on_error)
        self._worker.sig_error.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, *_):
        self._timer.stop()
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._log("Simulation finished.")

    # ── slots ─────────────────────────────────────────────────────────
    def _log(self, text: str):
        self.console.append(text)

    def _on_p1_aep(self, gen: int, aep_gwh: float):
        self.aep_canvas.add_p1_point(gen, aep_gwh)
        self.lbl_aep_val.setText(f"{aep_gwh:.3f} GWh")
        self.lbl_gen.setText(f"Gen: {gen}")

    def _on_p2_frame(self, turb, sub, n_groups, paths):
        import numpy as np
        combined = np.vstack([turb, sub.reshape((1, 2))])
        self.farm_canvas.update_p2(turb, sub, n_groups, paths, combined)

    def _on_p2_metrics(self, gen: int, net_gwh: float, capex_kusd: float):
        self.aep_canvas.add_p2_point(gen, net_gwh, capex_kusd)
        self.lbl_aep_val.setText(f"{net_gwh:.3f} GWh")
        self.lbl_capex_val.setText(f"${capex_kusd:,.1f} kUSD")
        self.lbl_gen.setText(f"Gen: {gen}")

    def _on_progress(self, gen: int, total: int):
        pct = int(gen / max(total, 1) * 100)
        self.progress.setValue(pct)

    def _on_phase(self, phase: int):
        self.lbl_phase.setText(f"Phase: {phase}")
        color = _BLUE if phase == 1 else _GREEN
        self.lbl_phase.setStyleSheet(f"color:{color};font-size:13px;font-weight:bold;")

    def _on_eta(self, secs: float):
        if secs < 3600:
            self.lbl_eta.setText(f"ETA: {int(secs//60)}m {int(secs%60)}s")
        else:
            self.lbl_eta.setText(f"ETA: {secs/3600:.1f}h")

    def _tick_elapsed(self):
        if self._t_start:
            e = int(time.time() - self._t_start)
            self.lbl_elapsed.setText(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def _toggle_pause(self):
        if not self._worker:
            return
        if self._worker._pause:
            self._worker.resume()
            self.btn_pause.setText("⏸  Pause")
            self._timer.start()
        else:
            self._worker.pause()
            self.btn_pause.setText("▶  Resume")
            self._timer.stop()

    def _stop(self):
        if self._worker:
            self._worker.stop()


# ─────────────────────────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    """Top-level window with 3 tabs: Config | Simulation | Pareto."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wind Farm Optimizer — GUI")
        self.resize(1280, 820)
        self.setStyleSheet(_GLOBAL_SS)

        central = QWidget(); self.setCentralWidget(central)
        ml = QVBoxLayout(central); ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        # Header banner row (banner label + Run button side by side)
        header = QWidget()
        header.setStyleSheet(f"background:{_MID};border-bottom:2px solid {_BORD};")
        header.setFixedHeight(52)
        hr = QHBoxLayout(header); hr.setContentsMargins(16, 0, 12, 0)
        banner = QLabel("🌬️  Wind Farm Optimizer")
        banner.setStyleSheet(
            f"color:{_WHITE};font-size:18px;font-weight:bold;background:transparent;"
        )
        self.btn_run = _btn("🚀  Run Optimization", color="#065f46", hover="#047857", w=200)
        self.btn_run.clicked.connect(self._run)
        hr.addWidget(banner, stretch=1)
        hr.addWidget(self.btn_run)
        ml.addWidget(header)

        self.tabs = QTabWidget()
        self.tab_config = ConfigTab()
        self.tab_sim    = SimTab()
        self.tab_pareto = ParetoTab()
        self.tabs.addTab(self.tab_config,  "⚙️  Configuration")
        self.tabs.addTab(self.tab_sim,     "🚀  Simulation")
        self.tabs.addTab(self.tab_pareto,  "🎯  Pareto Results")
        ml.addWidget(self.tabs, stretch=1)

    # ── run ───────────────────────────────────────────────────────────
    def _run(self):
        cfg = self.tab_config.build_sim_config()
        output_dir = os.path.join(ROOT, "results", "gui_run")
        os.makedirs(output_dir, exist_ok=True)
        self.tabs.setCurrentIndex(1)
        self.tab_sim.start(
            cfg, output_dir,
            max_gens=1000,
            on_done=self._on_done,
            on_error=self._on_error,
        )

    def _on_done(self, solutions: list, sim):
        self.tab_pareto.load_results(solutions, sim)
        self.tabs.setCurrentIndex(2)

    def _on_error(self, tb_text: str):
        QMessageBox.critical(self, "Simulation Error", tb_text)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
