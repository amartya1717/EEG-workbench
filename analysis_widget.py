"""
Analysis Widget (PySide6)
-------------------------
Main widget translated from EEG_analysis_tool.py (tkinter analysis_frame).
"""

import os
import threading

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QTextEdit, QTabWidget, QFileDialog,
    QMessageBox, QDialog, QLineEdit, QRadioButton,
    QButtonGroup, QListWidget, QSplitter, QScrollArea,
    QCheckBox, QComboBox, QFrame, QSizePolicy,
    QDialogButtonBox
)
from PySide6.QtCore import QObject, Signal, Qt as _Qt, QTimer
from PySide6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor

from preprocessing import EEGPreprocessor
from interpreter import EEGInterpreter
from settings_panel import SettingsPanel
from epoch_panel import EpochPanel
from hypothesis_panel import HypothesisPanel
from visualisation import plot_manager


# ---------------------------------------------------------------------------
# Thread-safe bridge — use instead of self.after(0, fn)
# ---------------------------------------------------------------------------

class _Bridge(QObject):
    _invoke = Signal(object)

    def __init__(self):
        super().__init__()
        self._invoke.connect(lambda fn: fn(), _Qt.QueuedConnection)

    def call(self, fn):
        self._invoke.emit(fn)


_bridge = _Bridge()


# ---------------------------------------------------------------------------
# Tag styles for coloured text in QTextEdit
# ---------------------------------------------------------------------------

TAG_STYLES = {
    "good":       {"color": "#2e7d32", "bold": True,  "size": 10},
    "medium":     {"color": "#e65100", "bold": True,  "size": 10},
    "high":       {"color": "#b71c1c", "bold": True,  "size": 10},
    "info":       {"color": "#1565c0", "bold": True,  "size": 10},
    "heading":    {"bold": True, "size": 11, "underline": True},
    "subtext":    {"color": "#555555", "size": 9},
    "action":     {"color": "#4a148c", "italic": True, "size": 9},
    "body":       {"size": 10},
    "subheading": {"bold": True, "size": 10},
    "dim":        {"color": "#666", "size": 9},
    "bullet":     {"size": 10},
    "methods_box": {"color": "#1a237e", "size": 9, "family": "Courier", "bg": "#e8eaf6"},
}


def _insert(widget: QTextEdit, text: str, tag: str = "body"):
    style = TAG_STYLES.get(tag, {})
    cursor = widget.textCursor()
    cursor.movePosition(QTextCursor.MoveOperation.End)
    fmt = QTextCharFormat()
    if "color" in style:
        fmt.setForeground(QColor(style["color"]))
    if style.get("bold"):
        fmt.setFontWeight(QFont.Weight.Bold)
    if style.get("italic"):
        fmt.setFontItalic(True)
    if style.get("underline"):
        fmt.setFontUnderline(True)
    if "size" in style:
        fmt.setFontPointSize(style["size"])
    if "family" in style:
        fmt.setFontFamily(style["family"])
    if "bg" in style:
        fmt.setBackground(QColor(style["bg"]))
    cursor.insertText(text, fmt)
    widget.setTextCursor(cursor)


# ---------------------------------------------------------------------------
# Main analysis widget
# ---------------------------------------------------------------------------

class AnalysisWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)

        self.import_complete = False
        self.run_complete    = False
        self.ica_popup       = None   # QDialog reference
        self.ica_entry       = None   # QLineEdit inside ica_popup
        self.preprocessor    = None
        self.visualize       = plot_manager()
        self._last_interp    = None
        self.selected_file   = None

        # Pipeline option state (plain attributes — no tkinter vars)
        from config import config as _cfg
        p = _cfg.get_section("pipeline")
        self._step_bandpass  = True
        self._step_notch     = True
        self._interpolate    = bool(p.get("interpolate_bads", True))
        self._fit_ica        = bool(p.get("fit_ica", True))
        self._l_freq         = str(p.get("l_freq",           1.0))
        self._h_freq         = str(p.get("h_freq",          40.0))
        self._notch          = str(p.get("notch_freq",       50.0))
        self._n_ica          = str(p.get("n_ica_components",   20))

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(4)

        # Title
        title_lbl = QLabel("EEG Analysis")
        title_font = QFont("Arial", 24)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setAlignment(_Qt.AlignCenter)
        main_layout.addWidget(title_lbl)

        # Top button bar
        btn_bar = QWidget()
        btn_bar_layout = QHBoxLayout(btn_bar)
        btn_bar_layout.setContentsMargins(0, 0, 0, 0)
        btn_bar_layout.setSpacing(4)

        def _btn(text, slot):
            b = QPushButton(text)
            b.clicked.connect(slot)
            return b

        btn_bar_layout.addWidget(_btn("Import .set File",      self.import_file))
        btn_bar_layout.addWidget(_btn("Run Preprocessing",     self.run_preprocessing))
        btn_bar_layout.addWidget(_btn("⚙ Pipeline Options",   self.show_pipeline_options))
        btn_bar_layout.addWidget(_btn("Plot Raw Channel",      self.call_plot_raw_channel))
        btn_bar_layout.addWidget(_btn("Plot PSD",              self.call_plot_psd))

        self.ica_button = QPushButton("ICA ▼")
        self.ica_button.clicked.connect(self.show_ica_menu)
        btn_bar_layout.addWidget(self.ica_button)

        self.plot_comparison_button = QPushButton("Plot Comparison Graph")
        self.plot_comparison_button.clicked.connect(self.call_plot_comparision_graph_menu)
        btn_bar_layout.addWidget(self.plot_comparison_button)

        btn_bar_layout.addStretch()

        btn_bar_layout.addWidget(_btn("History",    self.show_history_menu))
        btn_bar_layout.addWidget(_btn("Export",     self.export_menu))
        btn_bar_layout.addWidget(_btn("⚙ Settings", self.open_settings))

        main_layout.addWidget(btn_bar)

        # File label
        self.file_label = QLabel("No file selected")
        self.file_label.setAlignment(_Qt.AlignCenter)
        main_layout.addWidget(self.file_label)

        # Audience selector
        audience_bar = QWidget()
        audience_bar_layout = QHBoxLayout(audience_bar)
        audience_bar_layout.setContentsMargins(0, 0, 0, 0)
        audience_bar_layout.setSpacing(6)
        audience_bar_layout.addWidget(QLabel("Report style:"))

        self.audience_group = QButtonGroup(self)
        for val, label in [("researcher", "Researcher"),
                           ("clinician",  "Clinician"),
                           ("general",    "General")]:
            rb = QRadioButton(label)
            rb.setProperty("value", val)
            if val == "researcher":
                rb.setChecked(True)
            self.audience_group.addButton(rb)
            audience_bar_layout.addWidget(rb)
            rb.toggled.connect(self._refresh_interpretation)

        audience_bar_layout.addStretch()
        main_layout.addWidget(audience_bar)

        # Status bar
        status_bar = QWidget()
        status_bar.setStyleSheet("background: #f0f0f0;")
        status_bar.setFixedHeight(26)
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(8, 2, 8, 2)
        self.progress_label = QLabel("Ready.")
        self.progress_label.setFont(QFont("Arial", 9))
        self.progress_label.setStyleSheet("color: #555; background: transparent;")
        status_bar_layout.addWidget(self.progress_label)
        main_layout.addWidget(status_bar)

        # Notebook (QTabWidget)
        self.notebook = QTabWidget()
        main_layout.addWidget(self.notebook, stretch=1)

        # ── Tab 1: Preprocessing Report ─────────────────────────────────────
        tab_report = QWidget()
        tab_report_layout = QVBoxLayout(tab_report)
        tab_report_layout.setContentsMargins(2, 2, 2, 2)
        self.result_text = QTextEdit()
        self.result_text.setFont(QFont("Courier", 10))
        self.result_text.setReadOnly(True)
        tab_report_layout.addWidget(self.result_text)
        self.notebook.addTab(tab_report, "Preprocessing Report")

        # ── Tab 2: Interpretation ────────────────────────────────────────────
        tab_interp = QWidget()
        tab_interp_layout = QHBoxLayout(tab_interp)
        tab_interp_layout.setContentsMargins(2, 2, 2, 2)
        tab_interp_layout.setSpacing(4)

        self.interp_text = QTextEdit()
        self.interp_text.setFont(QFont("Arial", 10))
        self.interp_text.setReadOnly(True)
        tab_interp_layout.addWidget(self.interp_text, stretch=1)

        # Plot frame (right side of interpretation tab)
        self.plot_frame = QWidget()
        self.plot_frame.setStyleSheet("background: white;")
        self.plot_frame_layout = QVBoxLayout(self.plot_frame)
        self.plot_frame_layout.setContentsMargins(2, 2, 2, 2)
        tab_interp_layout.addWidget(self.plot_frame, stretch=1)

        self.notebook.addTab(tab_interp, "Interpretation")

        # ── Tab 3: Methods Text ──────────────────────────────────────────────
        tab_methods = QWidget()
        tab_methods_layout = QVBoxLayout(tab_methods)
        tab_methods_layout.setContentsMargins(2, 2, 2, 2)
        self.methods_text = QTextEdit()
        self.methods_text.setFont(QFont("Arial", 10))
        self.methods_text.setReadOnly(True)
        tab_methods_layout.addWidget(self.methods_text)
        methods_hint = QLabel("Copy this paragraph directly into your methods section.")
        methods_hint.setFont(QFont("Arial", 9))
        methods_hint.setStyleSheet("color: #555; font-style: italic;")
        tab_methods_layout.addWidget(methods_hint)
        self.notebook.addTab(tab_methods, "Methods Text")

        # ── Tab 4: Epoch Analysis ────────────────────────────────────────────
        self._epoch_panel = EpochPanel()
        self.notebook.addTab(self._epoch_panel, "Epoch Analysis")

        # ── Tab 5: Hypothesis Report ─────────────────────────────────────────
        self._hypothesis_panel = HypothesisPanel()
        self.notebook.addTab(self._hypothesis_panel, "Hypothesis Report")

    # -----------------------------------------------------------------------
    # ICA popup
    # -----------------------------------------------------------------------

    def show_ica_menu(self):
        if self.ica_popup is not None and self.ica_popup.isVisible():
            self.ica_popup.raise_()
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("ICA Tools")
        dlg.resize(300, 300)
        dlg.setSizeGripEnabled(False)
        self.ica_popup = dlg

        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)
        layout.setContentsMargins(12, 12, 12, 12)

        title = QLabel("ICA Tools")
        title_font = QFont("Arial", 11)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(_Qt.AlignCenter)
        layout.addWidget(title)

        comp_btn = QPushButton("Plot ICA Components")
        comp_btn.setFixedWidth(200)
        comp_btn.clicked.connect(self.call_plot_ica_components)
        layout.addWidget(comp_btn, alignment=_Qt.AlignCenter)

        src_btn = QPushButton("Plot ICA Sources")
        src_btn.setFixedWidth(200)
        src_btn.clicked.connect(self.call_plot_ica_sources)
        layout.addWidget(src_btn, alignment=_Qt.AlignCenter)

        excl_lbl = QLabel("Exclude components (e.g. 0,2,5):")
        excl_lbl.setAlignment(_Qt.AlignCenter)
        layout.addWidget(excl_lbl)

        self.ica_entry = QLineEdit()
        self.ica_entry.setFixedWidth(160)
        layout.addWidget(self.ica_entry, alignment=_Qt.AlignCenter)

        prop_btn = QPushButton("Plot ICA Properties")
        prop_btn.setFixedWidth(200)
        prop_btn.clicked.connect(self.call_plot_ica_properties_from_entry)
        layout.addWidget(prop_btn, alignment=_Qt.AlignCenter)

        apply_btn = QPushButton("Apply ICA Exclusion")
        apply_btn.setFixedWidth(200)
        apply_btn.clicked.connect(self.apply_ica_from_entry)
        layout.addWidget(apply_btn, alignment=_Qt.AlignCenter)

        auto_btn = QPushButton("Auto Detect ICA Artifacts")
        auto_btn.setFixedWidth(200)
        auto_btn.clicked.connect(self.auto_detect_ica_components)
        layout.addWidget(auto_btn, alignment=_Qt.AlignCenter)

        layout.addStretch()
        dlg.show()

    # -----------------------------------------------------------------------
    # History menu
    # -----------------------------------------------------------------------

    def show_history_menu(self):
        if self.preprocessor is None or not self.preprocessor.history.snapshots:
            QMessageBox.information(self, "History", "No history snapshots available.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("History Viewer")
        dlg.resize(420, 320)
        dlg_layout = QVBoxLayout(dlg)

        title = QLabel("Processing History")
        title_font = QFont("Arial", 11)
        title_font.setBold(True)
        title.setFont(title_font)
        title.setAlignment(_Qt.AlignCenter)
        dlg_layout.addWidget(title)

        list_widget = QListWidget()
        list_widget.setFont(QFont("Arial", 10))
        dlg_layout.addWidget(list_widget)

        snapshots = self.preprocessor.history.snapshots
        for i, snap in enumerate(snapshots):
            label     = snap.get("label", f"Snapshot {i}")
            step_type = snap.get("step_type", "unknown")
            list_widget.addItem(f"{i + 1}. [{step_type}] {label}")

        def on_select():
            row = list_widget.currentRow()
            if row < 0:
                return
            snapshot  = self.preprocessor.history.get_snapshot(row)
            info_text = self.format_snapshot(snapshot)
            self.result_text.setReadOnly(False)
            cursor = self.result_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.insertText(info_text)
            self.result_text.setTextCursor(cursor)
            self.result_text.setReadOnly(True)

        list_widget.itemClicked.connect(lambda item: on_select())
        dlg.exec()

    def format_snapshot(self, snapshot):
        text = f"\nLabel: {snapshot['label']}\n"
        text += f"Step: {snapshot['step_type']}\n"
        details = snapshot.get("details", {})
        if details:
            text += "\nDetails:\n"
            for k, v in details.items():
                text += f"  - {k}: {v}\n"
        return text

    # -----------------------------------------------------------------------
    # Comparison graph menu
    # -----------------------------------------------------------------------

    def call_plot_comparision_graph_menu(self):
        if self.preprocessor is None or not self.preprocessor.history.snapshots:
            QMessageBox.information(self, "Info",
                "Perform preprocessing first or complete ICA exclusions.")
            return

        snapshots = self.preprocessor.history.snapshots
        if len(snapshots) < 2:
            QMessageBox.information(self, "Info", "Need at least 2 snapshots to compare.")
            return

        win = QDialog(self)
        win.setWindowTitle("Graph Comparison")
        win.resize(900, 520)
        win_layout = QVBoxLayout(win)
        win_layout.setContentsMargins(10, 10, 10, 10)
        win_layout.setSpacing(6)

        title_lbl = QLabel("Graph Comparison")
        title_font = QFont("Arial", 12)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setAlignment(_Qt.AlignCenter)
        win_layout.addWidget(title_lbl)

        hint_lbl = QLabel(
            "Select exactly 2 snapshots to compare, choose a mode and channel, then click Plot."
        )
        hint_lbl.setFont(QFont("Arial", 9))
        hint_lbl.setStyleSheet("color: #555;")
        hint_lbl.setAlignment(_Qt.AlignCenter)
        win_layout.addWidget(hint_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        win_layout.addWidget(sep)

        # Splitter: left controls | right plot
        splitter = QSplitter(_Qt.Horizontal)
        win_layout.addWidget(splitter, stretch=1)

        # ── Left controls column (scrollable) ──────────────────────────────
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(280)
        left_scroll.setHorizontalScrollBarPolicy(_Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(8, 4, 8, 4)
        left_layout.setSpacing(4)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)

        # ── Right plot area ──────────────────────────────────────────────────
        plot_area = QWidget()
        plot_area.setStyleSheet("background: white;")
        plot_area_layout = QVBoxLayout(plot_area)
        plot_area_layout.setContentsMargins(4, 4, 4, 4)
        splitter.addWidget(plot_area)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Snapshot checkboxes
        snap_title = QLabel("Snapshots  (pick exactly 2)")
        snap_title_font = QFont("Arial", 10)
        snap_title_font.setBold(True)
        snap_title.setFont(snap_title_font)
        left_layout.addWidget(snap_title)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep2)

        snap_checks = {}
        for snap in snapshots:
            label     = snap.get("label", "")
            step_type = snap.get("step_type", "")
            cb = QCheckBox(f"[{step_type}]  {label}")
            cb.setFont(QFont("Arial", 9))
            snap_checks[label] = cb
            left_layout.addWidget(cb)

        sep3 = QFrame()
        sep3.setFrameShape(QFrame.Shape.HLine)
        sep3.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep3)

        # Mode selector
        mode_lbl = QLabel("Plot mode")
        mode_lbl_font = QFont("Arial", 10)
        mode_lbl_font.setBold(True)
        mode_lbl.setFont(mode_lbl_font)
        left_layout.addWidget(mode_lbl)

        mode_group = QButtonGroup(win)
        mode_buttons = {}
        for mode in ["PSD", "Raw Overlay", "Difference signal", "Band Power"]:
            rb = QRadioButton(mode)
            rb.setFont(QFont("Arial", 9))
            if mode == "PSD":
                rb.setChecked(True)
            mode_group.addButton(rb)
            mode_buttons[mode] = rb
            left_layout.addWidget(rb)

        sep4 = QFrame()
        sep4.setFrameShape(QFrame.Shape.HLine)
        sep4.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep4)

        # Channel selector
        ch_lbl = QLabel("Channel")
        ch_lbl_font = QFont("Arial", 10)
        ch_lbl_font.setBold(True)
        ch_lbl.setFont(ch_lbl_font)
        left_layout.addWidget(ch_lbl)

        ch_mode_group = QButtonGroup(win)
        rb_auto = QRadioButton("Auto (max difference)")
        rb_auto.setFont(QFont("Arial", 9))
        rb_auto.setChecked(True)
        ch_mode_group.addButton(rb_auto)
        left_layout.addWidget(rb_auto)

        rb_manual = QRadioButton("Manually select")
        rb_manual.setFont(QFont("Arial", 9))
        ch_mode_group.addButton(rb_manual)
        left_layout.addWidget(rb_manual)

        ch_names = list(self.preprocessor.cleaned_raw.ch_names) \
            if self.preprocessor and self.preprocessor.cleaned_raw else []
        manual_ch_combo = QComboBox()
        manual_ch_combo.addItems(ch_names)
        if ch_names:
            manual_ch_combo.setCurrentText(ch_names[0])
        left_layout.addWidget(manual_ch_combo)

        sep5 = QFrame()
        sep5.setFrameShape(QFrame.Shape.HLine)
        sep5.setFrameShadow(QFrame.Shadow.Sunken)
        left_layout.addWidget(sep5)

        # Status
        status_lbl = QLabel("")
        status_lbl.setFont(QFont("Arial", 9))
        status_lbl.setStyleSheet("color: #b71c1c;")
        status_lbl.setWordWrap(True)
        left_layout.addWidget(status_lbl)

        # Plot button
        def do_plot():
            selected = [lbl for lbl, cb in snap_checks.items() if cb.isChecked()]
            if len(selected) != 2:
                status_lbl.setText("⚠  Select exactly 2 snapshots.")
                return

            checked_mode = mode_group.checkedButton()
            if checked_mode is None:
                status_lbl.setText("⚠  Select a plot mode.")
                return
            mode = checked_mode.text()

            plotting_data = []
            for lbl in selected:
                for snap in snapshots:
                    if snap.get("label") == lbl:
                        plotting_data.append((snap["raw"], lbl))
                        break

            if len(plotting_data) < 2:
                status_lbl.setText("⚠  Could not retrieve snapshot data.")
                return

            raw1, label1 = plotting_data[0]
            raw2, label2 = plotting_data[1]

            channel_index = 0
            if rb_auto.isChecked():
                try:
                    d1   = raw1.get_data(picks="eeg")
                    d2   = raw2.get_data(picks="eeg")
                    diff = np.abs(d1 - d2).mean(axis=1)
                    channel_index = int(np.argmax(diff))
                except Exception:
                    channel_index = 0
            else:
                ch = manual_ch_combo.currentText()
                channel_index = ch_names.index(ch) if ch in ch_names else 0

            try:
                if mode == "PSD":
                    fig = self.visualize.plot_comparision_psd(raw1, raw2, label1, label2)
                elif mode == "Raw Overlay":
                    fig = self.visualize.plot_raw_overlay_comparision(
                        raw1, raw2, label1, label2, channel_index)
                elif mode == "Difference signal":
                    fig = self.visualize.plot_difference_signal(
                        raw1, raw2, label1, label2, channel_index)
                elif mode == "Band Power":
                    fig = self.visualize.plot_band_power_comparision(raw1, raw2, label1, label2)
                else:
                    status_lbl.setText("⚠  Unknown mode.")
                    return
            except Exception as e:
                status_lbl.setText(f"Plot error: {e}")
                return

            while plot_area_layout.count():
                child = plot_area_layout.takeAt(0)
                if child.widget():
                    child.widget().deleteLater()

            canvas = FigureCanvasQTAgg(fig)
            plot_area_layout.addWidget(canvas)
            canvas.draw()
            status_lbl.setText("")

        plot_btn = QPushButton("Plot")
        plot_btn.setFixedWidth(150)
        plot_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "font-weight: bold; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        plot_btn.clicked.connect(do_plot)
        left_layout.addWidget(plot_btn)

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(150)
        close_btn.clicked.connect(win.accept)
        left_layout.addWidget(close_btn)

        left_layout.addStretch()
        win.exec()

    # -----------------------------------------------------------------------
    # Import file
    # -----------------------------------------------------------------------

    def import_file(self):
        if self.run_complete:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("New Dataset")
            msg_box.setText(
                "Preprocessing has already been run on the current file.\n\n"
                "Importing a new file will reset all results including:\n"
                "  • Preprocessing report and interpretation\n"
                "  • Methods text\n"
                "  • Epoch analysis and trial data\n"
                "  • Hypothesis report\n\n"
                "Do you want to continue and reset everything?"
            )
            msg_box.setIcon(QMessageBox.Icon.Warning)
            msg_box.addButton(QMessageBox.StandardButton.Yes)
            msg_box.addButton(QMessageBox.StandardButton.No)
            msg_box.addButton(QMessageBox.StandardButton.Cancel)
            answer = msg_box.exec()
            if answer != QMessageBox.StandardButton.Yes:
                return
            self._reset_all()

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG File",
            "",
            "EEG files (*.edf *.csv *.txt *.set);;All files (*.*)"
        )

        if file_path:
            self.selected_file   = file_path
            file_name            = os.path.basename(file_path)
            self.file_label.setText(f"Selected: {file_name}  —  {file_path}")
            self.import_complete = True

    # -----------------------------------------------------------------------
    # Reset all
    # -----------------------------------------------------------------------

    def _reset_all(self):
        self.preprocessor    = None
        self.run_complete    = False
        self.import_complete = False
        self._last_interp    = None
        self.selected_file   = None
        self.file_label.setText("No file selected")
        self.progress_label.setText("Ready.")
        self.progress_label.setStyleSheet("color: #555; background: transparent;")

        # Tab 1
        self.result_text.setReadOnly(False)
        self.result_text.clear()
        self.result_text.setReadOnly(True)

        # Tab 2
        self.interp_text.setReadOnly(False)
        self.interp_text.clear()
        self.interp_text.setReadOnly(True)

        # Tab 3
        self.methods_text.setReadOnly(False)
        self.methods_text.clear()
        self.methods_text.setReadOnly(True)

        # Tab 4 — Epoch panel
        ep = self._epoch_panel
        if ep.analyzer is not None:
            ep.analyzer.epochs = None
            ep.analyzer.events = None
            ep.analyzer        = None

        ep._status_label.setText("No epochs created.")
        ep._status_label.setStyleSheet("color: #555;")
        ep._trial_label.setText("Trial —/—")
        ep._trial_info.setText("")

        ep._event_table.clear()

        while ep._condition_frame_layout.count():
            child = ep._condition_frame_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        ep._condition_vars = {}

        ep._cond_a_combo.clear()
        ep._cond_b_combo.clear()
        ep._channel_combo.clear()

        ep._clear_plot()

        # Tab 5 — Hypothesis panel
        hp = self._hypothesis_panel
        hp._analyzer              = None
        hp._methods_text_cache    = ""
        hp._cond_a_combo.clear()
        hp._cond_b_combo.clear()
        hp._report_text.setReadOnly(False)
        hp._report_text.clear()
        hp._report_text.setReadOnly(True)
        hp._status.setText("")

    # -----------------------------------------------------------------------
    # Run preprocessing
    # -----------------------------------------------------------------------

    def run_preprocessing(self):
        if not self.import_complete:
            QMessageBox.critical(self, "Error", "No EEG file selected.")
            return

        # Read GUI state on main thread before launching background thread
        run_bandpass = self._step_bandpass
        run_notch    = self._step_notch
        interpolate  = self._interpolate
        fit_ica      = self._fit_ica

        try:
            l_freq = float(self._l_freq)
            h_freq = float(self._h_freq)
            notch  = float(self._notch)
            n_ica  = int(self._n_ica)
        except ValueError:
            QMessageBox.critical(
                self, "Error",
                "Invalid pipeline parameter — check Pipeline Options values."
            )
            return

        def task():
            try:
                _bridge.call(lambda: self.progress_label.setText("⏳  Loading file..."))
                _bridge.call(lambda: self.progress_label.setStyleSheet(
                    "color: #1565c0; background: transparent;"))

                self.preprocessor = EEGPreprocessor(self.selected_file)

                _bridge.call(lambda: self.progress_label.setText("⏳  Running pipeline..."))

                report = self.preprocessor.run_basic_pipeline(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    notch_freq=notch,
                    interpolate_bads=interpolate,
                    fit_ica=fit_ica,
                    run_bandpass=run_bandpass,
                    run_notch=run_notch,
                    run_reference=True,
                    run_bad_channels=True,
                    run_metrics=True,
                    n_ica_components=n_ica,
                    progress_callback=lambda msg: _bridge.call(
                        lambda m=msg: (
                            self.progress_label.setText(f"⏳  {m}"),
                            self.progress_label.setStyleSheet(
                                "color: #1565c0; background: transparent;")
                        )
                    )
                )

                _bridge.call(lambda r=report: self._on_preprocessing_complete(r))

            except Exception as e:
                _bridge.call(lambda err=e: self._on_preprocessing_error(str(err)))

        threading.Thread(target=task, daemon=True).start()

    def _on_preprocessing_complete(self, report):
        bads  = report.get("bad_channels", [])
        n_ch  = report.get("n_channels", "?")
        sfreq = report.get("sfreq", "?")
        self.progress_label.setText(
            f"✓  Preprocessing complete — "
            f"{n_ch} channels, {sfreq} Hz, "
            f"{len(bads)} bad channel(s) detected"
        )
        self.progress_label.setStyleSheet("color: #2e7d32; background: transparent;")
        self.display_report(report)
        self.run_complete = True
        self._run_interpretation()
        self._epoch_panel.set_preprocessor(self.preprocessor)
        self._epoch_panel.on_epochs_created = self._on_epochs_created

    def _on_epochs_created(self, analyzer):
        checked = self.audience_group.checkedButton()
        audience = checked.property("value") if checked else "researcher"
        self._hypothesis_panel.set_analyzer(analyzer, audience=audience)

    def _on_preprocessing_error(self, error_msg):
        self.progress_label.setText(f"✗  Error: {error_msg[:120]}")
        self.progress_label.setStyleSheet("color: #b71c1c; background: transparent;")
        QMessageBox.critical(self, "Preprocessing Error", error_msg)

    # -----------------------------------------------------------------------
    # Interpretation
    # -----------------------------------------------------------------------

    def _run_interpretation(self):
        if self.preprocessor is None:
            return
        checked  = self.audience_group.checkedButton()
        audience = checked.property("value") if checked else "researcher"
        interp   = EEGInterpreter(self.preprocessor.report, audience=audience)
        self._last_interp = interp.interpret()
        self._render_interpretation(self._last_interp)
        self._render_methods(self._last_interp)

    def _refresh_interpretation(self):
        if self._last_interp is None or self.preprocessor is None:
            return
        checked  = self.audience_group.checkedButton()
        audience = checked.property("value") if checked else "researcher"
        interp   = EEGInterpreter(self.preprocessor.report, audience=audience)
        self._last_interp = interp.interpret()
        self._render_interpretation(self._last_interp)
        self._render_methods(self._last_interp)

    def _render_interpretation(self, result):
        t = self.interp_text
        t.setReadOnly(False)
        t.clear()

        quality = result["quality"]
        noise   = result["noise"]
        pipe    = result["pipeline"]

        ICONS = {"good": "✓", "medium": "⚠", "high": "✗", "info": "ℹ"}

        score  = quality["score"]
        grade  = quality["grade"]
        _insert(t, "OVERALL SUMMARY\n", "heading")
        _insert(t, f"{result['summary']}\n\n", "subtext")

        filled = int(score / 10)
        bar    = "█" * filled + "░" * (10 - filled)
        colour = "good" if score >= 80 else ("medium" if score >= 60 else "high")
        _insert(t, "Quality Score:  ")
        _insert(t, f"{bar}  {score}/100 — {grade}\n\n", colour)

        _insert(t, "DATA QUALITY\n", "heading")
        for f in quality["findings"]:
            sev  = f["severity"]
            icon = ICONS.get(sev, "•")
            _insert(t, f"  {icon} {f['title']}\n", sev)
            _insert(t, f"     {f['explanation']}\n", "subtext")
            if f.get("action"):
                _insert(t, f"     → {f['action']}\n", "action")
        _insert(t, "\n")

        _insert(t, "NOISE ANALYSIS\n", "heading")
        if noise.get("note"):
            _insert(t, f"  ℹ {noise['note']}\n\n", "info")
        else:
            for f in noise.get("findings", []):
                sev      = f.get("severity", "info")
                icon     = ICONS.get(sev, "•")
                comps    = f.get("components", [])
                comp_str = f"  Components: {comps}" if comps else ""
                _insert(t, f"  {icon} {f['name']}  [{f.get('status','')}]\n", sev)
                if comp_str:
                    _insert(t, f"     {comp_str}\n", "subtext")
                _insert(t, f"     {f.get('explanation','')}\n", "subtext")
                if f.get("action"):
                    _insert(t, f"     → {f['action']}\n", "action")
            _insert(t, "\n")

        _insert(t, "PROCESSING STEPS\n", "heading")
        for step in pipe["steps"]:
            _insert(t, f"  • {step}\n", "subtext")
        _insert(t, "\n")

        t.setReadOnly(True)
        cursor = t.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        t.setTextCursor(cursor)

    def _render_methods(self, result):
        t = self.methods_text
        t.setReadOnly(False)
        t.clear()
        t.insertPlainText(result["methods_text"])
        t.setReadOnly(True)

    # -----------------------------------------------------------------------
    # Display preprocessing report
    # -----------------------------------------------------------------------

    def display_report(self, report):
        self.result_text.setReadOnly(False)
        lines = []
        lines.append("Preprocessing Report")
        lines.append("=" * 50)
        lines.append(f"Loaded: {report.get('loaded')}")
        lines.append(f"Channels: {report.get('n_channels')}")
        lines.append(f"Sampling Rate: {report.get('sfreq')} Hz")
        lines.append(f"Duration: {round(report.get('duration_sec', 0), 2)} sec")

        bads = report.get("bad_channels", [])
        lines.append(f"Bad Channels: {', '.join(bads) if bads else 'None'}")

        lines.append("\nSteps:")
        for step in report.get("steps", []):
            lines.append(f"  - {step}")

        metrics = report.get("artifact_metrics", {})
        if metrics:
            lines.append("\nArtifact Metrics:")
            for key, value in metrics.items():
                lines.append(f"  - {key}: {value}")

        interpolation = report.get("interpolation", {})
        if interpolation:
            lines.append("\nInterpolation:")
            lines.append(f"  - Performed: {interpolation.get('performed', False)}")
            channels = interpolation.get("channels", [])
            lines.append(
                f"  - Channels: {', '.join(channels) if channels else 'None'}"
            )
            if "n_interpolated" in interpolation:
                lines.append(
                    f"  - Number Interpolated: {interpolation.get('n_interpolated')}"
                )
            if "reason" in interpolation:
                lines.append(f"  - Reason: {interpolation.get('reason')}")

        ica = report.get("ica", {})
        if ica:
            lines.append("\nICA:")
            lines.append(f"  - Fitted: {ica.get('fitted', False)}")
            lines.append(f"  - Requested Components: {ica.get('n_components_requested', 'NA')}")
            lines.append(f"  - Fitted Components: {ica.get('n_components_fitted', 'NA')}")
            excluded = ica.get("excluded_components", [])
            lines.append(
                f"  - Excluded Components: {', '.join(map(str, excluded)) if excluded else 'None'}"
            )
            lines.append(f"  - Applied: {ica.get('applied', False)}")

        notes = report.get("notes", [])
        if notes:
            lines.append("\nNotes:")
            for note in notes:
                lines.append(f"  - {note}")

        cursor = self.result_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n".join(lines))
        self.result_text.setTextCursor(cursor)
        self.result_text.setReadOnly(True)

    # -----------------------------------------------------------------------
    # Show figure in interpretation plot frame
    # -----------------------------------------------------------------------

    def show_figure(self, fig):
        while self.plot_frame_layout.count():
            child = self.plot_frame_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        canvas = FigureCanvasQTAgg(fig)
        self.plot_frame_layout.addWidget(canvas)
        canvas.draw()

    # -----------------------------------------------------------------------
    # Plot actions
    # -----------------------------------------------------------------------

    def call_plot_raw_channel(self):
        if not self.run_complete:
            QMessageBox.critical(self, "Error", "Run preprocessing first.")
            return
        try:
            fig = self.visualize.plot_raw_channel(self.preprocessor.cleaned_raw)
            self.show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "Raw channel plot error", str(e))

    def call_plot_psd(self):
        if not self.run_complete:
            QMessageBox.critical(self, "Error", "Run preprocessing first.")
            return
        try:
            fig = self.visualize.plot_psd(self.preprocessor.cleaned_raw)
            self.show_figure(fig)
        except Exception as e:
            QMessageBox.critical(self, "PSD plot error", str(e))

    def call_plot_ica_components(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            QMessageBox.critical(self, "Error", "Run preprocessing with ICA first.")
            return
        try:
            self.visualize.plot_ica_components(self.preprocessor.ica)
        except Exception as e:
            QMessageBox.critical(self, "ICA Plot Error", str(e))

    def call_plot_ica_sources(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            QMessageBox.critical(self, "Error", "Run preprocessing with ICA first.")
            return
        try:
            self.visualize.plot_ica_sources(
                self.preprocessor.ica, self.preprocessor.cleaned_raw)
        except Exception as e:
            QMessageBox.critical(self, "ICA Plot Error", str(e))

    def call_plot_ica_properties_from_entry(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            QMessageBox.critical(self, "Error", "ICA has not been fitted yet.")
            return
        if self.ica_entry is None or not self.ica_entry.text().strip():
            QMessageBox.critical(self, "Error", "Enter component numbers like 0,1,2")
            return
        try:
            text = self.ica_entry.text().strip()
            self.visualize.plot_ica_properties_from_entry(
                text, self.preprocessor.cleaned_raw, self.preprocessor.ica)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def apply_ica_from_entry(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            QMessageBox.critical(self, "Error", "ICA has not been fitted yet.")
            return
        if self.ica_entry is None:
            return
        text = self.ica_entry.text().strip()
        if not text:
            QMessageBox.critical(self, "Error", "Enter components like 0,2,5")
            return
        try:
            self.preprocessor.apply_ica_exclusion(text)
            self.display_report(self.preprocessor.report)
            self._run_interpretation()
            self._epoch_panel.set_preprocessor(self.preprocessor)
            self._epoch_panel.mark_epochs_stale()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def auto_detect_ica_components(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            QMessageBox.critical(self, "Error", "ICA has not been fitted yet.")
            return
        try:
            result    = self.preprocessor.auto_detect_ica_artifacts()
            suggested = result.get("suggested_exclude", [])

            if self.ica_entry is not None:
                self.ica_entry.setText(",".join(map(str, suggested)))

            lines = []
            lines.append("=" * 50)
            lines.append("ICA AUTO-DETECTION REPORT")
            lines.append("=" * 50)

            eog_indices = result.get("eog_indices", [])
            lines.append(
                f"EOG-related components: "
                f"{', '.join(map(str, eog_indices)) if eog_indices else 'None'}"
            )

            ecg_indices = result.get("ecg_indices", [])
            lines.append(
                f"ECG-related components: "
                f"{', '.join(map(str, ecg_indices)) if ecg_indices else 'None'}"
            )

            lines.append(
                f"Suggested components to exclude: "
                f"{', '.join(map(str, suggested)) if suggested else 'None'}"
            )

            if result.get("eog_scores"):
                lines.append(f"EOG scores: {result['eog_scores']}")
            if result.get("ecg_scores"):
                lines.append(f"ECG scores: {result['ecg_scores']}")
            if "eog_error" in result:
                lines.append(f"EOG detection error: {result['eog_error']}")
            if "ecg_error" in result:
                lines.append(f"ECG detection error: {result['ecg_error']}")

            self.append_to_report_widget(lines)

        except Exception as e:
            QMessageBox.critical(self, "Auto Detect Error", str(e))

    def append_to_report_widget(self, lines):
        self.result_text.setReadOnly(False)
        cursor = self.result_text.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText("\n" + "\n".join(lines) + "\n")
        self.result_text.setTextCursor(cursor)
        self.result_text.ensureCursorVisible()
        self.result_text.setReadOnly(True)

    # -----------------------------------------------------------------------
    # Pipeline options dialog
    # -----------------------------------------------------------------------

    def show_pipeline_options(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Pipeline Options")
        dlg.setFixedSize(380, 480)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(14, 14, 14, 14)
        dlg_layout.setSpacing(6)

        title_lbl = QLabel("Pipeline Options")
        title_font = QFont("Arial", 12)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setAlignment(_Qt.AlignCenter)
        dlg_layout.addWidget(title_lbl)

        hint_lbl = QLabel(
            "Choose which steps run and set their parameters.\n"
            "Greyed-out steps are always required."
        )
        hint_lbl.setFont(QFont("Arial", 9))
        hint_lbl.setStyleSheet("color: #555;")
        hint_lbl.setAlignment(_Qt.AlignCenter)
        dlg_layout.addWidget(hint_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        dlg_layout.addWidget(sep)

        # Local copies of current values
        step_bandpass  = [self._step_bandpass]
        step_notch     = [self._step_notch]
        interpolate    = [self._interpolate]
        fit_ica        = [self._fit_ica]
        l_freq_val     = [self._l_freq]
        h_freq_val     = [self._h_freq]
        notch_val      = [self._notch]
        n_ica_val      = [self._n_ica]

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(3)
        dlg_layout.addWidget(content)

        def _sep_line(parent_layout):
            s = QFrame()
            s.setFrameShape(QFrame.Shape.HLine)
            s.setFrameShadow(QFrame.Shadow.Sunken)
            parent_layout.addWidget(s)

        def _step_row(label, checked, enabled=True):
            cb = QCheckBox(label)
            cb.setFont(QFont("Arial", 10))
            cb.setChecked(checked)
            cb.setEnabled(enabled)
            content_layout.addWidget(cb)
            return cb

        def _param_row(label, current_val, unit=""):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(20, 0, 0, 0)
            lbl = QLabel(label)
            lbl.setFont(QFont("Arial", 9))
            lbl.setStyleSheet("color: #444;")
            lbl.setFixedWidth(140)
            row_layout.addWidget(lbl)
            edit = QLineEdit(str(current_val))
            edit.setFont(QFont("Arial", 9))
            edit.setFixedWidth(70)
            row_layout.addWidget(edit)
            if unit:
                u_lbl = QLabel(unit)
                u_lbl.setFont(QFont("Arial", 9))
                u_lbl.setStyleSheet("color: #888;")
                row_layout.addWidget(u_lbl)
            row_layout.addStretch()
            content_layout.addWidget(row)
            return edit

        # Step 1 — always on
        _step_row("1.  Load .set file", True, enabled=False)
        _sep_line(content_layout)

        # Step 2 — Bandpass
        cb_bp   = _step_row("2.  Bandpass filter", self._step_bandpass)
        e_lfreq = _param_row("Low cutoff:",  self._l_freq, "Hz")
        e_hfreq = _param_row("High cutoff:", self._h_freq, "Hz")
        _sep_line(content_layout)

        # Step 3 — Notch
        cb_notch  = _step_row("3.  Notch filter", self._step_notch)
        e_notch   = _param_row("Frequency:", self._notch, "Hz")
        _sep_line(content_layout)

        # Step 4 — Average reference (always on)
        _step_row("4.  Average reference", True, enabled=False)
        _sep_line(content_layout)

        # Step 5 — Bad channel detection (always on)
        _step_row("5.  Detect bad channels", True, enabled=False)
        _sep_line(content_layout)

        # Step 6 — Interpolation
        cb_interp = _step_row("6.  Interpolate bad channels", self._interpolate)
        _sep_line(content_layout)

        # Step 7 — ICA
        cb_ica  = _step_row("7.  Fit ICA", self._fit_ica)
        e_n_ica = _param_row("Components:", self._n_ica, "")
        _sep_line(content_layout)

        # Step 8 — Metrics (always on)
        _step_row("8.  Compute artifact metrics", True, enabled=False)

        dlg_layout.addStretch()

        # Buttons
        btn_row = QWidget()
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0, 6, 0, 0)

        ok_btn = QPushButton("OK")
        ok_btn.setFixedWidth(100)
        ok_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "font-weight: bold; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )

        def _on_ok():
            self._step_bandpass = cb_bp.isChecked()
            self._step_notch    = cb_notch.isChecked()
            self._interpolate   = cb_interp.isChecked()
            self._fit_ica       = cb_ica.isChecked()
            self._l_freq        = e_lfreq.text()
            self._h_freq        = e_hfreq.text()
            self._notch         = e_notch.text()
            self._n_ica         = e_n_ica.text()
            dlg.accept()

        ok_btn.clicked.connect(_on_ok)
        btn_row_layout.addWidget(ok_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(90)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row_layout.addWidget(cancel_btn)

        btn_row_layout.addStretch()
        dlg_layout.addWidget(btn_row)

        dlg.exec()

    # -----------------------------------------------------------------------
    # Settings
    # -----------------------------------------------------------------------

    def open_settings(self):
        dlg = SettingsPanel(self, on_save_callback=self._refresh_interpretation)
        dlg.exec()

    # -----------------------------------------------------------------------
    # Export
    # -----------------------------------------------------------------------

    def export_menu(self):
        if self.preprocessor is None or not self.run_complete:
            QMessageBox.critical(self, "Error", "Run preprocessing before exporting.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Export")
        dlg.setFixedSize(260, 140)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(12, 12, 12, 12)
        dlg_layout.setSpacing(8)

        title_lbl = QLabel("Export Options")
        title_font = QFont("Arial", 11)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        title_lbl.setAlignment(_Qt.AlignCenter)
        dlg_layout.addWidget(title_lbl)

        fif_btn = QPushButton("Save Cleaned EEG (.fif)")
        fif_btn.setFixedWidth(220)
        fif_btn.clicked.connect(lambda: self._export_fif(dlg))
        dlg_layout.addWidget(fif_btn, alignment=_Qt.AlignCenter)

        json_btn = QPushButton("Save Report (.json)")
        json_btn.setFixedWidth(220)
        json_btn.clicked.connect(lambda: self._export_json(dlg))
        dlg_layout.addWidget(json_btn, alignment=_Qt.AlignCenter)

        dlg.exec()

    def _export_fif(self, parent_dlg):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save cleaned EEG",
            "",
            "FIF files (*.fif)"
        )
        if path:
            try:
                self.preprocessor.save_cleaned_file(path)
                QMessageBox.information(self, "Saved", f"Cleaned EEG saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_json(self, parent_dlg):
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save preprocessing report",
            "",
            "JSON files (*.json)"
        )
        if path:
            try:
                self.preprocessor.save_report_json(path)
                QMessageBox.information(self, "Saved", f"Report saved to:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
