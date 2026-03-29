"""
Epoch Panel (PySide6)
---------------------
GUI tab for epoch analysis. Sits inside the main QTabWidget in analysis_widget.py.

Layout:
    Left column  — controls (events, epoch params, condition selector) in QScrollArea
    Right column — plot area
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QLabel, QPushButton, QLineEdit, QCheckBox, QComboBox,
    QTreeWidget, QTreeWidgetItem, QFrame, QMessageBox,
    QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from epoch_analyzer import EpochAnalyzer
from visualisation import plot_manager


class EpochPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.analyzer = None
        self.visualize = plot_manager()
        self._canvas_widget = None
        self.on_epochs_created = None   # callback — set from analysis_widget

        self._current_trial = 0
        self._condition_vars = {}   # label -> QCheckBox

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 8, 8)
        main_layout.setSpacing(0)

        # ── Scrollable left column ──────────────────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(320)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(6, 4, 6, 4)
        left_layout.setSpacing(4)
        scroll_area.setWidget(left_widget)
        main_layout.addWidget(scroll_area)

        # ── Right plot area ─────────────────────────────────────────────────
        right_widget = QWidget()
        right_widget.setStyleSheet("background: white;")
        self._plot_layout = QVBoxLayout(right_widget)
        self._plot_layout.setContentsMargins(4, 4, 4, 4)
        self._plot_frame = right_widget
        main_layout.addWidget(right_widget, stretch=1)

        # ── Section 1: Load events ──────────────────────────────────────────
        self._section_label(left_layout, "1. Events")

        self._load_btn = QPushButton("Load Events from Recording")
        self._load_btn.clicked.connect(self._load_events)
        left_layout.addWidget(self._load_btn)

        # Event table (QTreeWidget as two-column list)
        self._event_table = QTreeWidget()
        self._event_table.setHeaderLabels(["Condition", "Trials"])
        self._event_table.setColumnWidth(0, 180)
        self._event_table.setColumnWidth(1, 70)
        self._event_table.setMaximumHeight(130)
        self._event_table.setRootIsDecorated(False)
        left_layout.addWidget(self._event_table)

        self._add_separator(left_layout)

        # ── Section 2: Epoch parameters ────────────────────────────────────
        self._section_label(left_layout, "2. Epoch Parameters")

        cond_lbl = QLabel("Select conditions:")
        cond_lbl.setFont(QFont("Arial", 9))
        left_layout.addWidget(cond_lbl)

        self._condition_frame = QWidget()
        self._condition_frame_layout = QVBoxLayout(self._condition_frame)
        self._condition_frame_layout.setContentsMargins(0, 0, 0, 0)
        self._condition_frame_layout.setSpacing(2)
        left_layout.addWidget(self._condition_frame)

        # Epoch param fields
        params_widget = QWidget()
        params_layout = QVBoxLayout(params_widget)
        params_layout.setContentsMargins(0, 2, 0, 2)
        params_layout.setSpacing(3)

        self._tmin_edit       = self._param_row(params_layout, "Epoch start (s):",       "-0.2")
        self._tmax_edit       = self._param_row(params_layout, "Epoch end (s):",          "0.8")
        self._baseline_start_edit = self._param_row(params_layout, "Baseline start (s):", "-0.2")
        self._baseline_end_edit   = self._param_row(params_layout, "Baseline end (s):",   "0.0")
        self._reject_edit     = self._param_row(params_layout, "Reject threshold (µV):", "100")

        left_layout.addWidget(params_widget)

        create_btn = QPushButton("Create Epochs")
        create_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "font-weight: bold; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        create_btn.clicked.connect(self._create_epochs)
        left_layout.addWidget(create_btn)

        self._status_label = QLabel("No epochs created.")
        self._status_label.setFont(QFont("Arial", 9))
        self._status_label.setStyleSheet("color: #555;")
        self._status_label.setWordWrap(True)
        left_layout.addWidget(self._status_label)

        self._add_separator(left_layout)

        # ── Section 3: Analysis ─────────────────────────────────────────────
        self._section_label(left_layout, "3. Analysis")

        lbl_a = QLabel("Condition A:")
        lbl_a.setFont(QFont("Arial", 9))
        left_layout.addWidget(lbl_a)
        self._cond_a_combo = QComboBox()
        self._cond_a_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_layout.addWidget(self._cond_a_combo)

        lbl_b = QLabel("Condition B (for comparison):")
        lbl_b.setFont(QFont("Arial", 9))
        left_layout.addWidget(lbl_b)
        self._cond_b_combo = QComboBox()
        self._cond_b_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_layout.addWidget(self._cond_b_combo)

        lbl_ch = QLabel("Channel (for trial grid):")
        lbl_ch.setFont(QFont("Arial", 9))
        left_layout.addWidget(lbl_ch)
        self._channel_combo = QComboBox()
        self._channel_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        left_layout.addWidget(self._channel_combo)

        btn_configs = [
            ("Plot ERP (A)",          self._plot_erp),
            ("Plot ERP Comparison",   self._plot_erp_comparison),
            ("Trial Grid (A)",        self._plot_trial_grid),
            ("Band Power Comparison", self._plot_band_power),
        ]
        for label, cmd in btn_configs:
            btn = QPushButton(label)
            btn.clicked.connect(cmd)
            left_layout.addWidget(btn)

        self._add_separator(left_layout)

        # ── Section 4: Trial browser ────────────────────────────────────────
        self._section_label(left_layout, "4. Trial Browser")

        nav_widget = QWidget()
        nav_layout = QHBoxLayout(nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)

        prev_btn = QPushButton("◀ Prev")
        prev_btn.clicked.connect(self._prev_trial)
        nav_layout.addWidget(prev_btn)

        self._trial_label = QLabel("Trial —/—")
        self._trial_label.setFont(QFont("Arial", 9))
        nav_layout.addWidget(self._trial_label)

        next_btn = QPushButton("Next ▶")
        next_btn.clicked.connect(self._next_trial)
        nav_layout.addWidget(next_btn)

        left_layout.addWidget(nav_widget)

        self._trial_info = QLabel("")
        self._trial_info.setFont(QFont("Arial", 9))
        self._trial_info.setStyleSheet("color: #555;")
        self._trial_info.setWordWrap(True)
        left_layout.addWidget(self._trial_info)

        left_layout.addStretch()

    def _section_label(self, layout, text):
        lbl = QLabel(text)
        font = QFont("Arial", 10)
        font.setBold(True)
        lbl.setFont(font)
        layout.addWidget(lbl)

    def _add_separator(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

    def _param_row(self, layout, label_text, default):
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setFont(QFont("Arial", 9))
        lbl.setFixedWidth(170)
        row_layout.addWidget(lbl)
        edit = QLineEdit(default)
        edit.setFixedWidth(80)
        row_layout.addWidget(edit)
        row_layout.addStretch()
        layout.addWidget(row_widget)
        return edit

    # -----------------------------------------------------------------------
    # Public — called from analysis_widget when preprocessor is ready
    # -----------------------------------------------------------------------

    def set_preprocessor(self, preprocessor):
        """Called after preprocessing completes to attach the raw data."""
        if preprocessor is None or preprocessor.cleaned_raw is None:
            return

        if self.analyzer is None:
            self.analyzer = EpochAnalyzer(preprocessor.cleaned_raw)
        else:
            self.analyzer.raw = preprocessor.cleaned_raw

        ch_names = list(preprocessor.cleaned_raw.ch_names)
        self._channel_combo.clear()
        self._channel_combo.addItems(ch_names)
        if ch_names:
            self._channel_combo.setCurrentText(ch_names[0])

    def mark_epochs_stale(self):
        """
        Called when ICA exclusion changes after epochs were created.
        Clears epochs and shows a red warning.
        """
        if self.analyzer is None:
            return

        epochs_existed = self.analyzer.epochs is not None
        self.analyzer.epochs = None

        if epochs_existed:
            self._status_label.setText(
                "⚠  ICA components changed — epoch results are stale.\n"
                "Re-create epochs before running any analysis."
            )
            self._status_label.setStyleSheet("color: #b71c1c;")
            self._current_trial = 0
            self._update_trial_label()
            self._trial_info.setText("")
            self._clear_plot()

    # -----------------------------------------------------------------------
    # Section 1 — Load events
    # -----------------------------------------------------------------------

    def _load_events(self):
        if self.analyzer is None:
            QMessageBox.critical(self, "Error", "Run preprocessing first.")
            return

        try:
            counts = self.analyzer.read_events()

            if not counts:
                QMessageBox.warning(
                    self,
                    "No events",
                    "No events found in this recording.\n\n"
                    "This can happen if the .set file has no annotations, "
                    "or if the events are stored in a format MNE cannot read automatically."
                )
                return

            # Clear and repopulate table
            self._event_table.clear()
            for label, n in counts.items():
                item = QTreeWidgetItem([label, str(n)])
                item.setTextAlignment(1, Qt.AlignmentFlag.AlignCenter)
                self._event_table.addTopLevelItem(item)

            # Build condition checkboxes
            while self._condition_frame_layout.count():
                child = self._condition_frame_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
            self._condition_vars = {}

            for label in counts:
                cb = QCheckBox(f"{label}  ({counts[label]} trials)")
                cb.setChecked(True)
                cb.setFont(QFont("Arial", 9))
                self._condition_vars[label] = cb
                self._condition_frame_layout.addWidget(cb)

            # Populate analysis dropdowns
            condition_list = list(counts.keys())
            self._cond_a_combo.clear()
            self._cond_b_combo.clear()
            self._cond_a_combo.addItems(condition_list)
            self._cond_b_combo.addItems(condition_list)
            if len(condition_list) >= 1:
                self._cond_a_combo.setCurrentText(condition_list[0])
            if len(condition_list) >= 2:
                self._cond_b_combo.setCurrentText(condition_list[1])

        except Exception as e:
            QMessageBox.critical(self, "Event loading error", str(e))

    # -----------------------------------------------------------------------
    # Section 2 — Create epochs
    # -----------------------------------------------------------------------

    def _create_epochs(self):
        if self.analyzer is None:
            QMessageBox.critical(self, "Error", "Run preprocessing first.")
            return

        if not self._condition_vars:
            QMessageBox.critical(self, "Error", "Load events first.")
            return

        selected = [l for l, v in self._condition_vars.items() if v.isChecked()]
        if not selected:
            QMessageBox.critical(self, "Error", "Select at least one condition.")
            return

        try:
            tmin      = float(self._tmin_edit.text())
            tmax      = float(self._tmax_edit.text())
            bl_start  = float(self._baseline_start_edit.text())
            bl_end    = float(self._baseline_end_edit.text())
            threshold = float(self._reject_edit.text())
        except ValueError:
            QMessageBox.critical(self, "Error", "Invalid parameter — check epoch window values.")
            return

        try:
            self.analyzer.create_epochs(
                event_labels=selected,
                tmin=tmin,
                tmax=tmax,
                baseline=(bl_start, bl_end),
                reject_threshold_uv=threshold
            )

            report   = self.analyzer.get_epoch_report()
            kept     = report["n_trials_kept"]
            rejected = report["n_trials_rejected"]
            total    = report["n_trials_total"]

            self._status_label.setText(
                f"{kept}/{total} trials kept  ({rejected} rejected > {threshold} µV)"
            )
            if rejected == 0:
                self._status_label.setStyleSheet("color: #2e7d32;")
            else:
                self._status_label.setStyleSheet("color: #e65100;")

            self._current_trial = 0
            self._update_trial_label()

            if self.on_epochs_created:
                self.on_epochs_created(self.analyzer)

        except Exception as e:
            QMessageBox.critical(self, "Epoch creation error", str(e))

    # -----------------------------------------------------------------------
    # Section 3 — Plot buttons
    # -----------------------------------------------------------------------

    def _plot_erp(self):
        if not self._epochs_ready():
            return
        try:
            cond = self._cond_a_combo.currentText() or None
            erp  = self.analyzer.get_erp(condition=cond)
            fig  = self.visualize.plot_erp(erp)
            self._show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "ERP plot error", str(e))

    def _plot_erp_comparison(self):
        if not self._epochs_ready():
            return
        cond_a = self._cond_a_combo.currentText()
        cond_b = self._cond_b_combo.currentText()
        if not cond_a or not cond_b:
            QMessageBox.critical(self, "Error", "Select both Condition A and B.")
            return
        if cond_a == cond_b:
            QMessageBox.critical(self, "Error", "Condition A and B must be different.")
            return
        try:
            comparison = self.analyzer.get_erp_comparison(cond_a, cond_b)
            fig = self.visualize.plot_erp_comparison(comparison)
            self._show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "ERP comparison error", str(e))

    def _plot_trial_grid(self):
        if not self._epochs_ready():
            return
        try:
            ch_name  = self._channel_combo.currentText()
            ch_names = list(self.analyzer.epochs.info["ch_names"])
            ch_idx   = ch_names.index(ch_name) if ch_name in ch_names else 0
            cond     = self._cond_a_combo.currentText() or None
            grid     = self.analyzer.get_trial_grid(channel_index=ch_idx, condition=cond)
            fig      = self.visualize.plot_trial_grid(grid)
            self._show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "Trial grid error", str(e))

    def _plot_band_power(self):
        if not self._epochs_ready():
            return
        cond_a = self._cond_a_combo.currentText()
        cond_b = self._cond_b_combo.currentText()
        if not cond_a or not cond_b:
            QMessageBox.critical(self, "Error", "Select both Condition A and B.")
            return
        try:
            bp_a = self.analyzer.get_band_power(condition=cond_a)
            bp_b = self.analyzer.get_band_power(condition=cond_b)
            fig  = self.visualize.plot_epoch_band_power(bp_a, bp_b, cond_a, cond_b)
            self._show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "Band power error", str(e))

    # -----------------------------------------------------------------------
    # Section 4 — Trial browser
    # -----------------------------------------------------------------------

    def _prev_trial(self):
        if self.analyzer is None or self.analyzer.epochs is None:
            return
        self._current_trial = max(0, self._current_trial - 1)
        self._show_trial(self._current_trial)

    def _next_trial(self):
        if self.analyzer is None or self.analyzer.epochs is None:
            return
        n = len(self.analyzer.epochs)
        self._current_trial = min(n - 1, self._current_trial + 1)
        self._show_trial(self._current_trial)

    def _show_trial(self, index: int):
        if self.analyzer is None or self.analyzer.epochs is None:
            return

        quality = self.analyzer.get_trial_quality()
        n       = quality["n_trials"]

        if index >= n:
            return

        ptp   = quality["ptp_per_trial_uv"][index]
        rms   = quality["rms_per_trial_uv"][index]
        flag  = quality["flags"][index]
        label = quality["trial_labels"][index]
        times = np.array(quality["times"])

        data = self.analyzer.epochs.get_data(picks="eeg")[index]
        mean = data.mean(axis=0) * 1e6

        fig = Figure(figsize=(8, 3), dpi=100)
        ax  = fig.add_subplot(111)
        color = "#2e7d32" if flag == "clean" else "#b71c1c"
        ax.plot(times * 1000, mean, linewidth=1.2, color=color)
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.set_title(
            f"Trial {index + 1}/{n}  [{label}]  —  "
            f"{'✓ Clean' if flag == 'clean' else '✗ Rejected'}",
            color=color
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mean amplitude (µV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._show_figure(fig)
        self._update_trial_label(index, n)
        self._trial_info.setText(
            f"Condition: {label}\n"
            f"Peak-to-peak: {round(ptp, 2)} µV\n"
            f"RMS: {round(rms, 2)} µV\n"
            f"Status: {flag}"
        )

    def _update_trial_label(self, index=0, n=None):
        if self.analyzer and self.analyzer.epochs is not None:
            n = n or len(self.analyzer.epochs)
            self._trial_label.setText(f"Trial {index + 1}/{n}")
        else:
            self._trial_label.setText("Trial —/—")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _epochs_ready(self) -> bool:
        if self.analyzer is None or self.analyzer.epochs is None:
            QMessageBox.critical(self, "Error", "Create epochs first.")
            return False
        return True

    def _clear_plot(self):
        while self._plot_layout.count():
            child = self._plot_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def _show_figure(self, fig: Figure):
        self._clear_plot()
        canvas = FigureCanvasQTAgg(fig)
        self._plot_layout.addWidget(canvas)
        canvas.draw()
        self._canvas_widget = canvas
