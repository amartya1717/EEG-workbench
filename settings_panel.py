"""
Settings Panel (PySide6)
------------------------
A QDialog that exposes all ConfigManager values as editable fields.
Organised into labelled sections matching the config structure.

Usage:
    from settings_panel import SettingsPanel
    dlg = SettingsPanel(parent, on_save_callback=self._refresh_interpretation)
    dlg.exec()
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QLineEdit, QPushButton, QScrollArea,
    QWidget, QFrame, QMessageBox, QSizePolicy
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from config import config, DEFAULTS


# ---------------------------------------------------------------------------
# Field definitions
# Each entry: (config_section, config_key, label, unit_hint, field_type)
# field_type: "float" | "int" | "bool"
# ---------------------------------------------------------------------------

THRESHOLD_FIELDS = [
    ("thresholds", "bad_channel_ratio_good",   "Bad channel ratio — good threshold",   "0–1",    "float"),
    ("thresholds", "bad_channel_ratio_medium",  "Bad channel ratio — medium threshold",  "0–1",    "float"),
    ("thresholds", "mean_rms_good_uv",          "Mean RMS — good threshold",             "µV",     "float"),
    ("thresholds", "mean_rms_medium_uv",        "Mean RMS — medium threshold",           "µV",     "float"),
    ("thresholds", "max_ptp_good_uv",           "Max peak-to-peak — good threshold",     "µV",     "float"),
    ("thresholds", "max_ptp_medium_uv",         "Max peak-to-peak — medium threshold",   "µV",     "float"),
    ("thresholds", "duration_good_sec",         "Duration — good threshold",             "sec",    "float"),
    ("thresholds", "duration_medium_sec",       "Duration — medium threshold",           "sec",    "float"),
]

SCORING_FIELDS = [
    ("scoring", "bad_channel_high",   "Bad channels (high severity) deduction",   "pts", "int"),
    ("scoring", "bad_channel_medium", "Bad channels (medium severity) deduction", "pts", "int"),
    ("scoring", "rms_high",           "RMS high severity deduction",              "pts", "int"),
    ("scoring", "rms_medium",         "RMS medium severity deduction",            "pts", "int"),
    ("scoring", "ptp_medium",         "Peak-to-peak medium severity deduction",   "pts", "int"),
    ("scoring", "duration_high",      "Duration high severity deduction",         "pts", "int"),
    ("scoring", "duration_medium",    "Duration medium severity deduction",       "pts", "int"),
]

GRADE_FIELDS = [
    ("grades", "good",       "Score threshold for 'Good' grade",       "0–100", "int"),
    ("grades", "acceptable", "Score threshold for 'Acceptable' grade", "0–100", "int"),
    ("grades", "poor",       "Score threshold for 'Poor' grade",       "0–100", "int"),
]

PIPELINE_FIELDS = [
    ("pipeline", "l_freq",              "Bandpass low cutoff",         "Hz",  "float"),
    ("pipeline", "h_freq",              "Bandpass high cutoff",        "Hz",  "float"),
    ("pipeline", "notch_freq",          "Notch filter frequency",      "Hz",  "float"),
    ("pipeline", "n_ica_components",    "ICA components",              "",    "int"),
    ("pipeline", "variance_z_threshold","Bad channel variance z-score","",    "float"),
    ("pipeline", "flat_std_threshold",  "Flat channel std threshold",  "V",   "float"),
]


class SettingsPanel(QDialog):

    def __init__(self, parent=None, on_save_callback=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(620, 680)
        self.setMinimumSize(400, 400)
        self.on_save_callback = on_save_callback

        self._entries = {}   # (section, key) -> QLineEdit

        self._build_ui()
        self._load_values()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 14, 14, 14)
        main_layout.setSpacing(6)

        # Header
        title_label = QLabel("Settings")
        title_font = QFont("Arial", 13)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        info_label = QLabel(
            "Changes take effect immediately after saving.\n"
            "Settings are persisted to eeg_config.json next to this tool."
        )
        info_font = QFont("Arial", 9)
        info_label.setFont(info_font)
        info_label.setStyleSheet("color: #555;")
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # Scrollable main area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._scroll_layout = QVBoxLayout(scroll_widget)
        self._scroll_layout.setSpacing(4)
        self._scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_area.setWidget(scroll_widget)
        main_layout.addWidget(scroll_area, stretch=1)

        # Build sections
        self._build_section("Quality Thresholds",
                            THRESHOLD_FIELDS,
                            "Boundaries between Good / Medium / Poor verdicts.")
        self._build_section("Score Deductions",
                            SCORING_FIELDS,
                            "Points subtracted from 100 for each quality issue.")
        self._build_section("Grade Boundaries",
                            GRADE_FIELDS,
                            "Minimum score required for each grade label.")
        self._build_section("Pipeline Defaults",
                            PIPELINE_FIELDS,
                            "Default values pre-filled in the preprocessing pipeline.")

        self._scroll_layout.addStretch()

        # Button row
        btn_layout = QHBoxLayout()
        btn_layout.setContentsMargins(0, 6, 0, 0)

        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(120)
        save_btn.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; font-size: 10pt; padding: 5px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #388e3c; }"
        )
        save_btn.clicked.connect(self._save)
        btn_layout.addWidget(save_btn)

        reset_btn = QPushButton("Reset to Defaults")
        reset_btn.setFixedWidth(140)
        reset_btn.clicked.connect(self._reset)
        btn_layout.addWidget(reset_btn)

        btn_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedWidth(90)
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        main_layout.addLayout(btn_layout)

    def _build_section(self, title, fields, description):
        layout = self._scroll_layout

        # Section header
        title_lbl = QLabel(title)
        title_font = QFont("Arial", 10)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        layout.addWidget(title_lbl)

        desc_lbl = QLabel(description)
        desc_lbl.setFont(QFont("Arial", 8))
        desc_lbl.setStyleSheet("color: #666;")
        layout.addWidget(desc_lbl)

        # Separator line
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Field grid
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(6, 2, 6, 2)
        grid.setSpacing(4)

        for row_idx, (section, key, label, unit, ftype) in enumerate(fields):
            lbl = QLabel(label)
            lbl.setFont(QFont("Arial", 9))
            grid.addWidget(lbl, row_idx, 0, Qt.AlignLeft)

            entry = QLineEdit()
            entry.setFixedWidth(110)
            entry.setFont(QFont("Arial", 9))
            grid.addWidget(entry, row_idx, 1)

            if unit:
                unit_lbl = QLabel(unit)
                unit_lbl.setFont(QFont("Arial", 8))
                unit_lbl.setStyleSheet("color: #888;")
                grid.addWidget(unit_lbl, row_idx, 2, Qt.AlignLeft)

            self._entries[(section, key)] = entry

        layout.addWidget(grid_widget)

    # -----------------------------------------------------------------------
    # Load / save / reset
    # -----------------------------------------------------------------------

    def _load_values(self):
        for (section, key), entry in self._entries.items():
            val = config.get(section, key)
            if val is None:
                val = DEFAULTS.get(section, {}).get(key, "")
            entry.setText(str(val))

    def _save(self):
        errors = []

        for (section, key), entry in self._entries.items():
            raw = entry.text().strip()
            ftype = self._get_ftype(section, key)

            try:
                if ftype == "float":
                    value = float(raw)
                elif ftype == "int":
                    value = int(raw)
                elif ftype == "bool":
                    value = raw.lower() in ("true", "1", "yes")
                else:
                    value = raw
            except ValueError:
                errors.append(f"{section}/{key}: expected {ftype}, got '{raw}'")
                continue

            config.set(section, key, value)

        if errors:
            QMessageBox.critical(
                self,
                "Validation errors",
                "The following fields have invalid values:\n\n" + "\n".join(errors)
            )
            return

        config.save()
        QMessageBox.information(self, "Saved", "Settings saved to eeg_config.json.")

        if self.on_save_callback:
            self.on_save_callback()

        self.accept()

    def _reset(self):
        reply = QMessageBox.question(
            self,
            "Reset to defaults",
            "This will reset all settings to factory defaults.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        config.reset_to_defaults()
        self._load_values()
        QMessageBox.information(
            self, "Reset", "All settings reset to defaults.\nPress Save to persist."
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_ftype(self, section, key):
        all_fields = THRESHOLD_FIELDS + SCORING_FIELDS + GRADE_FIELDS + PIPELINE_FIELDS
        for s, k, _, _, ftype in all_fields:
            if s == section and k == key:
                return ftype
        return "float"
