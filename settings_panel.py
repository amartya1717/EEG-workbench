"""
Settings Panel
--------------
A Toplevel window that exposes all ConfigManager values as editable fields.
Organised into labelled sections matching the config structure.

Usage (from analysis_frame):
    from settings_panel import SettingsPanel
    SettingsPanel(parent, on_save_callback=self._refresh_interpretation)
"""

import tkinter as tk
from tkinter import ttk, messagebox
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


class SettingsPanel(tk.Toplevel):

    def __init__(self, parent, on_save_callback=None):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("620x680")
        self.resizable(True, True)
        self.on_save_callback = on_save_callback

        self._entries = {}   # (section, key) -> tk.StringVar

        self._build_ui()
        self._load_values()

        self.grab_set()  # modal

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Header
        tk.Label(
            self, text="Settings",
            font=("Arial", 13, "bold")
        ).pack(pady=(12, 4))

        tk.Label(
            self,
            text="Changes take effect immediately after saving.\n"
                 "Settings are persisted to eeg_config.json next to this tool.",
            font=("Arial", 9), fg="#555"
        ).pack(pady=(0, 8))

        # Scrollable main area
        container = tk.Frame(self)
        container.pack(fill="both", expand=True, padx=14)

        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self._scroll_frame = tk.Frame(canvas)

        self._scroll_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(e):
            try:
                canvas.yview_scroll(-1 * (e.delta // 120), "units")
            except tk.TclError:
                pass

        canvas.bind("<Enter>", lambda e: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))
        self._canvas = canvas

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

        # Button row
        btn_frame = tk.Frame(self)
        btn_frame.pack(fill="x", padx=14, pady=10)

        tk.Button(
            btn_frame, text="Save", width=14, bg="#2e7d32", fg="white",
            font=("Arial", 10, "bold"),
            command=self._save
        ).pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="Reset to Defaults", width=16,
            command=self._reset
        ).pack(side="left", padx=4)

        tk.Button(
            btn_frame, text="Cancel", width=10,
            command=self._on_close
        ).pack(side="right", padx=4)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        try:
            self._canvas.unbind_all("<MouseWheel>")
        except Exception:
            pass
        self.destroy()

    def _build_section(self, title, fields, description):
        f = self._scroll_frame

        # Section header
        tk.Label(
            f, text=title,
            font=("Arial", 10, "bold"), anchor="w"
        ).pack(fill="x", pady=(14, 0))

        tk.Label(
            f, text=description,
            font=("Arial", 8), fg="#666", anchor="w"
        ).pack(fill="x", pady=(0, 4))

        ttk.Separator(f, orient="horizontal").pack(fill="x", pady=2)

        # Field grid
        grid = tk.Frame(f)
        grid.pack(fill="x", pady=4)

        for row_idx, (section, key, label, unit, ftype) in enumerate(fields):
            tk.Label(
                grid, text=label, anchor="w",
                font=("Arial", 9)
            ).grid(row=row_idx, column=0, sticky="w", padx=6, pady=3)

            var = tk.StringVar()
            entry = tk.Entry(grid, textvariable=var, width=12, font=("Arial", 9))
            entry.grid(row=row_idx, column=1, padx=6, pady=3)

            if unit:
                tk.Label(
                    grid, text=unit, fg="#888",
                    font=("Arial", 8)
                ).grid(row=row_idx, column=2, sticky="w", padx=2)

            self._entries[(section, key)] = var

    # -----------------------------------------------------------------------
    # Load / save / reset
    # -----------------------------------------------------------------------

    def _load_values(self):
        for (section, key), var in self._entries.items():
            val = config.get(section, key)
            if val is None:
                val = DEFAULTS.get(section, {}).get(key, "")
            var.set(str(val))

    def _save(self):
        errors = []

        for (section, key), var in self._entries.items():
            raw = var.get().strip()

            # Find expected type
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
            messagebox.showerror(
                "Validation errors",
                "The following fields have invalid values:\n\n" + "\n".join(errors)
            )
            return

        config.save()
        messagebox.showinfo("Saved", "Settings saved to eeg_config.json.")

        if self.on_save_callback:
            self.on_save_callback()

        self.destroy()

    def _reset(self):
        if not messagebox.askyesno(
            "Reset to defaults",
            "This will reset all settings to factory defaults.\nContinue?"
        ):
            return

        config.reset_to_defaults()
        self._load_values()
        messagebox.showinfo("Reset", "All settings reset to defaults.\nPress Save to persist.")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_ftype(self, section, key):
        all_fields = THRESHOLD_FIELDS + SCORING_FIELDS + GRADE_FIELDS + PIPELINE_FIELDS
        for s, k, _, _, ftype in all_fields:
            if s == section and k == key:
                return ftype
        return "float"