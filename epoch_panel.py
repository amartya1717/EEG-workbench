"""
Epoch Panel
-----------
GUI tab for epoch analysis. Sits inside the main notebook in EEG_analysis_tool.py.

Layout:
    Left column  — controls (events, epoch params, condition selector)
    Right column — plot area

Flow:
    1. Load Events     → shows event table
    2. Select events + set window/baseline/threshold → Create Epochs
    3. Browse trials / plot ERP / compare conditions / view trial grid
"""

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from epoch_analyzer import EpochAnalyzer
from visualisation import plot_manager


class EpochPanel(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg="white")

        self.analyzer  = None
        self.visualize = plot_manager()
        self._canvas_widget = None
        self.on_epochs_created = None   # callback — set from analysis_frame

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Two-column layout
        left_container = tk.Frame(self, bg="white", width=310)
        left_container.pack(side="left", fill="y", padx=0, pady=8)
        left_container.pack_propagate(False)

        # Scrollable left column
        left_canvas = tk.Canvas(left_container, bg="white",
                                highlightthickness=0, width=295)
        left_scrollbar = tk.Scrollbar(left_container, orient="vertical",
                                      command=left_canvas.yview)
        left = tk.Frame(left_canvas, bg="white")

        left.bind(
            "<Configure>",
            lambda e: left_canvas.configure(
                scrollregion=left_canvas.bbox("all")
            )
        )

        left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        # Mousewheel scroll — scoped to when mouse is over the left panel
        def _mw(e):
            try:
                left_canvas.yview_scroll(-1 * (e.delta // 120), "units")
            except tk.TclError:
                pass

        left_canvas.bind("<Enter>", lambda e: left_canvas.bind_all("<MouseWheel>", _mw))
        left_canvas.bind("<Leave>", lambda e: left_canvas.unbind_all("<MouseWheel>"))

        right = tk.Frame(self, bg="white")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)

        self._plot_frame = right

        # ---- Section 1: Load events ----
        self._section_label(left, "1. Events")

        self._load_btn = tk.Button(
            left, text="Load Events from Recording",
            command=self._load_events, width=28
        )
        self._load_btn.pack(pady=4, padx=6)

        # Event table
        tbl_frame = tk.Frame(left)
        tbl_frame.pack(fill="x", pady=4)

        self._event_table = ttk.Treeview(
            tbl_frame,
            columns=("condition", "trials"),
            show="headings", height=5
        )
        self._event_table.heading("condition", text="Condition")
        self._event_table.heading("trials",    text="Trials")
        self._event_table.column("condition",  width=160)
        self._event_table.column("trials",     width=60, anchor="center")
        self._event_table.pack(fill="x")

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # ---- Section 2: Epoch parameters ----
        self._section_label(left, "2. Epoch Parameters")

        # Condition checkboxes — populated dynamically after events load
        tk.Label(left, text="Select conditions:", bg="white",
                 font=("Arial", 9)).pack(anchor="w")
        self._condition_frame = tk.Frame(left, bg="white")
        self._condition_frame.pack(fill="x", pady=2)
        self._condition_vars = {}   # label → BooleanVar

        # Window
        params_grid = tk.Frame(left, bg="white")
        params_grid.pack(fill="x", pady=4)

        def _row(label, default, row):
            tk.Label(params_grid, text=label, bg="white",
                     font=("Arial", 9), anchor="w").grid(
                row=row, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=default)
            tk.Entry(params_grid, textvariable=var, width=10).grid(
                row=row, column=1, padx=6, pady=2)
            return var

        self._tmin_var      = _row("Epoch start (s):",          "-0.2", 0)
        self._tmax_var      = _row("Epoch end (s):",             "0.8",  1)
        self._baseline_start = _row("Baseline start (s):",      "-0.2", 2)
        self._baseline_end   = _row("Baseline end (s):",         "0.0",  3)
        self._reject_var    = _row("Reject threshold (µV):",    "100",  4)

        tk.Button(
            left, text="Create Epochs", width=28,
            bg="#1565c0", fg="white", font=("Arial", 9, "bold"),
            command=self._create_epochs
        ).pack(pady=6)

        # Epoch status
        self._status_label = tk.Label(
            left, text="No epochs created.", bg="white",
            font=("Arial", 9), fg="#555"
        )
        self._status_label.pack()

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # ---- Section 3: Analysis ----
        self._section_label(left, "3. Analysis")

        # Single condition selector
        tk.Label(left, text="Condition A:", bg="white",
                 font=("Arial", 9)).pack(anchor="w")
        self._cond_a_var = tk.StringVar()
        self._cond_a_combo = ttk.Combobox(
            left, textvariable=self._cond_a_var, state="readonly", width=26)
        self._cond_a_combo.pack(anchor="w", pady=2)

        tk.Label(left, text="Condition B (for comparison):", bg="white",
                 font=("Arial", 9)).pack(anchor="w", pady=(6, 0))
        self._cond_b_var = tk.StringVar()
        self._cond_b_combo = ttk.Combobox(
            left, textvariable=self._cond_b_var, state="readonly", width=26)
        self._cond_b_combo.pack(anchor="w", pady=2)

        # Channel selector
        tk.Label(left, text="Channel (for trial grid):", bg="white",
                 font=("Arial", 9)).pack(anchor="w", pady=(6, 0))
        self._channel_var = tk.StringVar()
        self._channel_combo = ttk.Combobox(
            left, textvariable=self._channel_var, state="readonly", width=26)
        self._channel_combo.pack(anchor="w", pady=2)

        # Plot buttons
        btn_cfg = [
            ("Plot ERP (A)",             self._plot_erp),
            ("Plot ERP Comparison",      self._plot_erp_comparison),
            ("Trial Grid (A)",           self._plot_trial_grid),
            ("Band Power Comparison",    self._plot_band_power),
        ]
        for label, cmd in btn_cfg:
            tk.Button(left, text=label, width=28, command=cmd).pack(pady=2)

        ttk.Separator(left, orient="horizontal").pack(fill="x", pady=8)

        # ---- Section 4: Trial browser ----
        self._section_label(left, "4. Trial Browser")

        nav_frame = tk.Frame(left, bg="white")
        nav_frame.pack()

        tk.Button(nav_frame, text="◀ Prev", command=self._prev_trial).pack(side="left", padx=4)
        self._trial_label = tk.Label(nav_frame, text="Trial —/—", bg="white", font=("Arial", 9))
        self._trial_label.pack(side="left", padx=6)
        tk.Button(nav_frame, text="Next ▶", command=self._next_trial).pack(side="left", padx=4)

        self._trial_info = tk.Label(
            left, text="", bg="white", font=("Arial", 9), fg="#555",
            wraplength=270, justify="left"
        )
        self._trial_info.pack(pady=4)

        self._current_trial = 0

    def _section_label(self, parent, text):
        tk.Label(
            parent, text=text, bg="white",
            font=("Arial", 10, "bold"), anchor="w"
        ).pack(fill="x", pady=(8, 2), padx=6)

    # -----------------------------------------------------------------------
    # Public — called from analysis_frame when preprocessor is ready
    # -----------------------------------------------------------------------

    def set_preprocessor(self, preprocessor):
        """Called after preprocessing completes to attach the raw data."""
        if preprocessor is None or preprocessor.cleaned_raw is None:
            return

        # Rebind analyzer to the current cleaned_raw.
        # If epochs exist they are kept — caller is responsible for
        # calling mark_epochs_stale() if the signal has changed.
        if self.analyzer is None:
            self.analyzer = EpochAnalyzer(preprocessor.cleaned_raw)
        else:
            self.analyzer.raw = preprocessor.cleaned_raw

        # Populate channel dropdown
        ch_names = preprocessor.cleaned_raw.ch_names
        self._channel_combo["values"] = ch_names
        if ch_names:
            self._channel_combo.set(ch_names[0])

    def mark_epochs_stale(self):
        """
        Called when ICA exclusion changes after epochs were created.
        Clears the epoch object and shows a red warning so the researcher
        knows they must re-create epochs before trusting any results.
        """
        if self.analyzer is None:
            return

        epochs_existed = self.analyzer.epochs is not None
        self.analyzer.epochs = None

        if epochs_existed:
            self._status_label.config(
                text=(
                    "⚠  ICA components changed — epoch results are stale.\n"
                    "Re-create epochs before running any analysis."
                ),
                fg="#b71c1c"
            )
            # Clear trial browser state
            self._current_trial = 0
            self._update_trial_label()
            self._trial_info.config(text="")

            # Clear plot area
            for widget in self._plot_frame.winfo_children():
                widget.destroy()

    # -----------------------------------------------------------------------
    # Section 1 — Load events
    # -----------------------------------------------------------------------

    def _load_events(self):
        if self.analyzer is None:
            messagebox.showerror("Error", "Run preprocessing first.")
            return

        try:
            counts = self.analyzer.read_events()

            if not counts:
                messagebox.showwarning(
                    "No events",
                    "No events found in this recording.\n\n"
                    "This can happen if the .set file has no annotations, "
                    "or if the events are stored in a format MNE cannot read automatically."
                )
                return

            # Clear and repopulate table
            for row in self._event_table.get_children():
                self._event_table.delete(row)
            for label, n in counts.items():
                self._event_table.insert("", "end", values=(label, n))

            # Build condition checkboxes
            for widget in self._condition_frame.winfo_children():
                widget.destroy()
            self._condition_vars = {}

            for label in counts:
                var = tk.BooleanVar(value=True)
                self._condition_vars[label] = var
                tk.Checkbutton(
                    self._condition_frame, text=f"{label}  ({counts[label]} trials)",
                    variable=var, bg="white", anchor="w"
                ).pack(fill="x")

            # Populate analysis dropdowns
            condition_list = list(counts.keys())
            self._cond_a_combo["values"] = condition_list
            self._cond_b_combo["values"] = condition_list
            if len(condition_list) >= 1:
                self._cond_a_combo.set(condition_list[0])
            if len(condition_list) >= 2:
                self._cond_b_combo.set(condition_list[1])

        except Exception as e:
            messagebox.showerror("Event loading error", str(e))

    # -----------------------------------------------------------------------
    # Section 2 — Create epochs
    # -----------------------------------------------------------------------

    def _create_epochs(self):
        if self.analyzer is None:
            messagebox.showerror("Error", "Run preprocessing first.")
            return

        if not self._condition_vars:
            messagebox.showerror("Error", "Load events first.")
            return

        selected = [l for l, v in self._condition_vars.items() if v.get()]
        if not selected:
            messagebox.showerror("Error", "Select at least one condition.")
            return

        try:
            tmin      = float(self._tmin_var.get())
            tmax      = float(self._tmax_var.get())
            bl_start  = float(self._baseline_start.get())
            bl_end    = float(self._baseline_end.get())
            threshold = float(self._reject_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid parameter — check epoch window values.")
            return

        try:
            epochs = self.analyzer.create_epochs(
                event_labels=selected,
                tmin=tmin,
                tmax=tmax,
                baseline=(bl_start, bl_end),
                reject_threshold_uv=threshold
            )

            report = self.analyzer.get_epoch_report()
            kept     = report["n_trials_kept"]
            rejected = report["n_trials_rejected"]
            total    = report["n_trials_total"]

            self._status_label.config(
                text=f"{kept}/{total} trials kept  ({rejected} rejected >  {threshold} µV)",
                fg="#2e7d32" if rejected == 0 else "#e65100"
            )
            self._current_trial = 0
            self._update_trial_label()

            # Notify hypothesis panel
            if self.on_epochs_created:
                self.on_epochs_created(self.analyzer)

        except Exception as e:
            messagebox.showerror("Epoch creation error", str(e))

    # -----------------------------------------------------------------------
    # Section 3 — Plot buttons
    # -----------------------------------------------------------------------

    def _plot_erp(self):
        if not self._epochs_ready():
            return
        try:
            cond = self._cond_a_var.get() or None
            erp  = self.analyzer.get_erp(condition=cond)
            fig  = self.visualize.plot_erp(erp)
            self._show_figure(fig)
        except Exception as e:
            messagebox.showerror("ERP plot error", str(e))

    def _plot_erp_comparison(self):
        if not self._epochs_ready():
            return
        cond_a = self._cond_a_var.get()
        cond_b = self._cond_b_var.get()
        if not cond_a or not cond_b:
            messagebox.showerror("Error", "Select both Condition A and B.")
            return
        if cond_a == cond_b:
            messagebox.showerror("Error", "Condition A and B must be different.")
            return
        try:
            comparison = self.analyzer.get_erp_comparison(cond_a, cond_b)
            fig = self.visualize.plot_erp_comparison(comparison)
            self._show_figure(fig)
        except Exception as e:
            messagebox.showerror("ERP comparison error", str(e))

    def _plot_trial_grid(self):
        if not self._epochs_ready():
            return
        try:
            ch_name = self._channel_var.get()
            ch_names = self.analyzer.epochs.info["ch_names"]
            ch_idx   = ch_names.index(ch_name) if ch_name in ch_names else 0
            cond     = self._cond_a_var.get() or None
            grid     = self.analyzer.get_trial_grid(channel_index=ch_idx, condition=cond)
            fig      = self.visualize.plot_trial_grid(grid)
            self._show_figure(fig)
        except Exception as e:
            messagebox.showerror("Trial grid error", str(e))

    def _plot_band_power(self):
        if not self._epochs_ready():
            return
        cond_a = self._cond_a_var.get()
        cond_b = self._cond_b_var.get()
        if not cond_a or not cond_b:
            messagebox.showerror("Error", "Select both Condition A and B.")
            return
        try:
            bp_a = self.analyzer.get_band_power(condition=cond_a)
            bp_b = self.analyzer.get_band_power(condition=cond_b)
            fig  = self.visualize.plot_epoch_band_power(bp_a, bp_b, cond_a, cond_b)
            self._show_figure(fig)
        except Exception as e:
            messagebox.showerror("Band power error", str(e))

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

        ptp     = quality["ptp_per_trial_uv"][index]
        rms     = quality["rms_per_trial_uv"][index]
        flag    = quality["flags"][index]
        label   = quality["trial_labels"][index]
        times   = np.array(quality["times"])

        # Get single trial data (all EEG channels)
        data = self.analyzer.epochs.get_data(picks="eeg")[index]  # (n_ch, n_times)
        mean = data.mean(axis=0) * 1e6  # average across channels, µV

        fig = Figure(figsize=(8, 3), dpi=100)
        ax  = fig.add_subplot(111)
        ax.plot(times * 1000, mean, linewidth=1.2,
                color="#2e7d32" if flag == "clean" else "#b71c1c")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.4, alpha=0.3)
        ax.set_title(
            f"Trial {index + 1}/{n}  [{label}]  —  "
            f"{'✓ Clean' if flag == 'clean' else '✗ Rejected'}",
            color="#2e7d32" if flag == "clean" else "#b71c1c"
        )
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Mean amplitude (µV)")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        self._show_figure(fig)
        self._update_trial_label(index, n)
        self._trial_info.config(
            text=f"Condition: {label}\n"
                 f"Peak-to-peak: {round(ptp, 2)} µV\n"
                 f"RMS: {round(rms, 2)} µV\n"
                 f"Status: {flag}"
        )

    def _update_trial_label(self, index=0, n=None):
        if self.analyzer and self.analyzer.epochs is not None:
            n = n or len(self.analyzer.epochs)
            self._trial_label.config(text=f"Trial {index + 1}/{n}")
        else:
            self._trial_label.config(text="Trial —/—")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _epochs_ready(self) -> bool:
        if self.analyzer is None or self.analyzer.epochs is None:
            messagebox.showerror("Error", "Create epochs first.")
            return False
        return True

    def _show_figure(self, fig: Figure):
        for widget in self._plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self._plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._canvas_widget = canvas