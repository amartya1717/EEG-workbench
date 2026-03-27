import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import os
import threading
from tkinter import messagebox
from preprocessing import EEGPreprocessor
from interpreter import EEGInterpreter
from settings_panel import SettingsPanel
from epoch_panel import EpochPanel
from hypothesis_panel import HypothesisPanel
import numpy as np
import mne
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visualisation import plot_manager




class analysis_frame(tk.Frame):
    def __init__(self,parent):
        super().__init__(parent, bg="white")

        tk.Label(
            self,
            text="EEG Analysis",
            font=("Arial", 24, "bold"),
            bg="white"
        ).pack(pady=30)

         # Top buttons
        button_bar = tk.Frame(self)
        button_bar.pack(fill="x", pady=5)

        tk.Button(button_bar, text="Import .set File", command=self.import_file).pack(side="left", padx=5)
        tk.Button(button_bar, text="Run Preprocessing", command=self.run_preprocessing).pack(side="left", padx=5)
        tk.Button(button_bar, text="⚙ Pipeline Options", command=self.show_pipeline_options).pack(side="left", padx=5)

        tk.Button(button_bar, text="Plot Raw Channel", command=self.call_plot_raw_channel).pack(side="left", padx=5)
        tk.Button(button_bar, text="Plot PSD", command=self.call_plot_psd).pack(side="left", padx=5)

        self.ica_button = tk.Button(button_bar, text="ICA ▼", command=self.show_ica_menu)
        self.ica_button.pack(side="left", padx=5)

        self.plot_comparision_button = tk.Button (button_bar, text = "Plot Comaprision Graph", command = self.call_plot_comparision_graph_menu)
        self.plot_comparision_button.pack(side = "left",padx = 5)

        self.history_button = tk.Button(button_bar,text = "History", command = self.show_history_menu)
        self.history_button.pack(side="right", padx=5)

        tk.Button(button_bar, text="Export", command=self.export_menu).pack(side="right", padx=5)
        tk.Button(button_bar, text="⚙ Settings", command=self.open_settings).pack(side="right", padx=5)

        # tk.Button(button_bar, text="Plot ICA Components", command=self.plot_ica_components).pack(side="left", padx=5)
        # tk.Button(button_bar, text="Plot ICA Sources", command=self.plot_ica_sources).pack(side="left", padx=5)
        

        self.file_label = tk.Label(self, text="No file selected")
        self.file_label.pack(pady=10)

        # Audience selector
        audience_bar = tk.Frame(self, bg="white")
        audience_bar.pack(fill="x", padx=10)
        tk.Label(audience_bar, text="Report style:", bg="white").pack(side="left")
        self.audience_var = tk.StringVar(value="researcher")
        for val, label in [("researcher","Researcher"), ("clinician","Clinician"), ("general","General")]:
            tk.Radiobutton(
                audience_bar, text=label, variable=self.audience_var,
                value=val, bg="white",
                command=self._refresh_interpretation
            ).pack(side="left", padx=6)

        # Status bar — always visible between controls and notebook
        status_bar = tk.Frame(self, bg="#f0f0f0", height=26)
        status_bar.pack(fill="x", padx=5, pady=(2, 0))
        status_bar.pack_propagate(False)
        self.progress_label = tk.Label(
            status_bar, text="Ready.",
            fg="#555", bg="#f0f0f0",
            font=("Arial", 9), anchor="w"
        )
        self.progress_label.pack(fill="x", padx=8, pady=4)

        # Notebook
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)
        # Tab 1 — Raw preprocessing report
        tab_report = tk.Frame(self.notebook)
        self.notebook.add(tab_report, text="Preprocessing Report")

        result_frame = tk.Frame(tab_report)
        result_frame.pack(fill="both", expand=True)
        x_scroll = tk.Scrollbar(result_frame, orient="horizontal")
        x_scroll.pack(side="bottom", fill="x")
        y_scroll = tk.Scrollbar(result_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")
        self.result_text = tk.Text(
            result_frame, wrap="none",
            xscrollcommand=x_scroll.set,
            yscrollcommand=y_scroll.set,
            font=("Courier", 10)
        )
        self.result_text.pack(side="left", fill="both", expand=True)
        x_scroll.config(command=self.result_text.xview)
        y_scroll.config(command=self.result_text.yview)

        # Tab 2 — Interpretation panel
        tab_interp = tk.Frame(self.notebook)
        self.notebook.add(tab_interp, text="Interpretation")

        interp_left = tk.Frame(tab_interp)
        interp_left.pack(side="left", fill="both", expand=True)

        interp_y = tk.Scrollbar(interp_left, orient="vertical")
        interp_y.pack(side="right", fill="y")
        self.interp_text = tk.Text(
            interp_left, wrap="word",
            yscrollcommand=interp_y.set,
            font=("Arial", 10), padx=10, pady=8,
            state="disabled"
        )
        self.interp_text.pack(fill="both", expand=True)
        interp_y.config(command=self.interp_text.yview)

        # Colour tags for traffic light system
        self.interp_text.tag_config("good",    foreground="#2e7d32", font=("Arial", 10, "bold"))
        self.interp_text.tag_config("medium",  foreground="#e65100", font=("Arial", 10, "bold"))
        self.interp_text.tag_config("high",    foreground="#b71c1c", font=("Arial", 10, "bold"))
        self.interp_text.tag_config("info",    foreground="#1565c0", font=("Arial", 10, "bold"))
        self.interp_text.tag_config("heading", font=("Arial", 11, "bold"), underline=True)
        self.interp_text.tag_config("subtext", foreground="#555555", font=("Arial", 9))
        self.interp_text.tag_config("action",  foreground="#4a148c", font=("Arial", 9, "italic"))

        self.plot_frame = tk.Frame(tab_interp, bg="white")
        self.plot_frame.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        # Tab 3 — Auto-generated methods text
        tab_methods = tk.Frame(self.notebook)
        self.notebook.add(tab_methods, text="Methods Text")

        methods_frame = tk.Frame(tab_methods)
        methods_frame.pack(fill="both", expand=True)
        methods_y = tk.Scrollbar(methods_frame, orient="vertical")
        methods_y.pack(side="right", fill="y")
        self.methods_text = tk.Text(
            methods_frame, wrap="word",
            yscrollcommand=methods_y.set,
            font=("Arial", 10), padx=12, pady=10,
            state="disabled"
        )
        self.methods_text.pack(fill="both", expand=True)
        methods_y.config(command=self.methods_text.yview)

        tk.Label(
            tab_methods,
            text="Copy this paragraph directly into your methods section.",
            font=("Arial", 9, "italic"), fg="#555"
        ).pack(pady=4)

        # Tab 4 — Epoch analysis
        self._epoch_panel = EpochPanel(self.notebook)
        self.notebook.add(self._epoch_panel, text="Epoch Analysis")

        # Tab 5 — Hypothesis assistant
        self._hypothesis_panel = HypothesisPanel(self.notebook)
        self.notebook.add(self._hypothesis_panel, text="Hypothesis Report")

        self.import_complete = False
        self.run_complete    = False
        self.ica_popup       = None
        self.preprocessor    = None
        self.visualize       = plot_manager()
        self._last_interp    = None

        # Pipeline option vars — driven by Pipeline Options window
        from config import config as _cfg
        p = _cfg.get_section("pipeline")
        self.step_bandpass_var   = tk.BooleanVar(value=True)
        self.step_notch_var      = tk.BooleanVar(value=True)
        self.step_reference_var  = tk.BooleanVar(value=True)
        self.step_badch_var      = tk.BooleanVar(value=True)
        self.interpolate_var     = tk.BooleanVar(value=p.get("interpolate_bads", True))
        self.ica_var             = tk.BooleanVar(value=p.get("fit_ica", True))
        self.step_metrics_var    = tk.BooleanVar(value=True)

        self.l_freq_var  = tk.StringVar(value=str(p.get("l_freq",          1.0)))
        self.h_freq_var  = tk.StringVar(value=str(p.get("h_freq",         40.0)))
        self.notch_var   = tk.StringVar(value=str(p.get("notch_freq",     50.0)))
        self.n_ica_var   = tk.StringVar(value=str(p.get("n_ica_components", 20)))
        
    def show_ica_menu(self):
        if self.ica_popup is not None and self.ica_popup.winfo_exists():
            self.ica_popup.lift()
            return

        self.ica_popup = tk.Toplevel(self)
        self.ica_popup.title("ICA Tools")
        self.ica_popup.geometry("300x280")
        self.ica_popup.resizable(False, False)

        x = self.ica_button.winfo_rootx()
        y = self.ica_button.winfo_rooty() + self.ica_button.winfo_height()
        self.ica_popup.geometry(f"300x280+{x}+{y}")

        tk.Label(
            self.ica_popup,
            text="ICA Tools",
            font=("Arial", 11, "bold")
        ).pack(pady=8)

        tk.Button(
            self.ica_popup,
            text="Plot ICA Components",
            width=22,
            command=self.call_plot_ica_components
        ).pack(pady=4)

        tk.Button(
            self.ica_popup,
            text="Plot ICA Sources",
            width=22,
            command=self.call_plot_ica_sources
        ).pack(pady=4)

        tk.Label(self.ica_popup, text="Exclude components (e.g. 0,2,5)").pack(pady=(10, 2))
        self.ica_entry = tk.Entry(self.ica_popup, width=20)
        self.ica_entry.pack(pady=4)

        tk.Button(
        self.ica_popup,
        text="Plot ICA Properties",
        width = 22,
        command=self.plot_ica_properties_from_entry
        ).pack(pady=4)

        tk.Button(
            self.ica_popup,
            text="Apply ICA Exclusion",
            width=22,
            command=self.apply_ica_from_entry
        ).pack(pady=6)

        tk.Button(
            self.ica_popup,
            text="Auto Detect ICA Artifacts",
            width = 22,
            command=self.auto_detect_ica_components
        ).pack(pady=6)

    def show_history_menu(self):

        if self.preprocessor is None or not self.preprocessor.history.snapshots:
            messagebox.showinfo("History", "No history snapshots available.")
            return

        self.history_win = tk.Toplevel(self)
        self.history_win.title("History Viewer")
        self.history_win.geometry("420x320")
        self.history_win.resizable(True, True)

        tk.Label(
            self.history_win,
            text="Processing History",
            font=("Arial", 11, "bold")
        ).pack(pady=8)

        history_list_frame = tk.Frame(self.history_win)
        history_list_frame.pack(fill="both", expand=True, padx=10, pady=10)

        y_scroll = tk.Scrollbar(history_list_frame, orient="vertical")
        y_scroll.pack(side="right", fill="y")

        self.history_listbox = tk.Listbox(
            history_list_frame,
            yscrollcommand=y_scroll.set,
            font=("Arial", 10)
        )
        self.history_listbox.pack(side="left", fill="both", expand=True)

        y_scroll.config(command=self.history_listbox.yview)

        snapshots = self.preprocessor.history.snapshots

        for i, snap in enumerate(snapshots):
            label = snap.get("label", f"Snapshot {i}")
            step_type = snap.get("step_type", "unknown")
            self.history_listbox.insert(tk.END, f"{i + 1}. [{step_type}] {label}")

        self.history_listbox.bind("<<ListboxSelect>>",self. on_select)

        # tk.Button(
        #     self.history_win,
        #     text="Display Selected",
        #     command=self.on_select
        # ).pack(pady=6)

    def on_select(self,event=None):
            selection = self.history_listbox.curselection()
            if not selection:
                return

            index = selection[0]
            snapshot = self.preprocessor.history.get_snapshot(index)
            info_text = self.format_snapshot(snapshot)

            self.result_text.config(state="normal")
            # self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, info_text)
            self.result_text.config(state="disabled")

    def format_snapshot(self, snapshot):
        text = f"\nLabel: {snapshot['label']}\n"
        text += f"Step: {snapshot['step_type']}\n"

        details = snapshot.get("details", {})
        if details:
            text += "\nDetails:\n"
            for k, v in details.items():
                text += f"  - {k}: {v}\n"

        return text       

    def call_plot_comparision_graph_menu(self):
        if self.preprocessor is None or not self.preprocessor.history.snapshots:
            messagebox.showinfo("Info", "Perform preprocessing first or complete ICA exclusions.")
            return

        # Collect all snapshots (not just ICA — now every step has one)
        snapshots = self.preprocessor.history.snapshots
        if len(snapshots) < 2:
            messagebox.showinfo("Info", "Need at least 2 snapshots to compare.")
            return

        # Build window
        win = tk.Toplevel(self)
        win.title("Graph Comparison")
        win.geometry("900x520")
        win.resizable(True, True)

        # ── Title ──────────────────────────────────────────────────────────
        tk.Label(
            win, text="Graph Comparison",
            font=("Arial", 12, "bold")
        ).pack(pady=(10, 4))

        tk.Label(
            win,
            text="Select exactly 2 snapshots to compare, choose a mode and channel, then click Plot.",
            font=("Arial", 9), fg="#555"
        ).pack(pady=(0, 6))

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=14)

        # ── Two-column body ────────────────────────────────────────────────
        body = tk.Frame(win)
        body.pack(fill="both", expand=True, padx=10, pady=8)

        # Left — scrollable controls column
        left_container = tk.Frame(body, width=270)
        left_container.pack(side="left", fill="y", padx=(0, 8))
        left_container.pack_propagate(False)

        left_canvas = tk.Canvas(left_container, highlightthickness=0)
        left_sb = tk.Scrollbar(left_container, orient="vertical",
                               command=left_canvas.yview)
        left = tk.Frame(left_canvas)

        left.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        )
        left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_sb.set)

        left_sb.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        def _mw(e):
            try:
                left_canvas.yview_scroll(-1 * (e.delta // 120), "units")
            except tk.TclError:
                pass

        left_canvas.bind("<Enter>",
                         lambda e: left_canvas.bind_all("<MouseWheel>", _mw))
        left_canvas.bind("<Leave>",
                         lambda e: left_canvas.unbind_all("<MouseWheel>"))
        win.protocol("WM_DELETE_WINDOW", lambda: [
            left_canvas.unbind_all("<MouseWheel>"), win.destroy()
        ])

        # Right plot area
        plot_area = tk.Frame(body, bg="white")
        plot_area.pack(side="right", fill="both", expand=True)

        # ── Snapshot checkboxes ────────────────────────────────────────────
        tk.Label(left, text="Snapshots  (pick exactly 2)",
                 font=("Arial", 10, "bold"), anchor="w").pack(
            fill="x", pady=(6, 2), padx=6)
        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=6, pady=2)

        filtered_list_state = {}
        for snap in snapshots:
            label = snap.get("label", "")
            step  = snap.get("step_type", "")
            var   = tk.BooleanVar(value=False)
            filtered_list_state[label] = var
            tk.Checkbutton(
                left,
                text=f"[{step}]  {label}",
                variable=var, anchor="w",
                font=("Arial", 9)
            ).pack(fill="x", padx=10, pady=1)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=6, pady=8)

        # ── Mode selector ──────────────────────────────────────────────────
        tk.Label(left, text="Plot mode", font=("Arial", 10, "bold"),
                 anchor="w").pack(fill="x", padx=6)

        mode_var = tk.StringVar(value="PSD")
        for mode in ["PSD", "Raw Overlay", "Difference signal", "Band Power"]:
            tk.Radiobutton(
                left, text=mode, variable=mode_var,
                value=mode, anchor="w", font=("Arial", 9)
            ).pack(fill="x", padx=16)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=6, pady=8)

        # ── Channel selector ───────────────────────────────────────────────
        tk.Label(left, text="Channel", font=("Arial", 10, "bold"),
                 anchor="w").pack(fill="x", padx=6)

        channel_mode_var = tk.StringVar(value="Auto (max difference)")
        tk.Radiobutton(
            left, text="Auto (max difference)",
            variable=channel_mode_var,
            value="Auto (max difference)",
            anchor="w", font=("Arial", 9)
        ).pack(fill="x", padx=16)
        tk.Radiobutton(
            left, text="Manually select",
            variable=channel_mode_var,
            value="Manually select",
            anchor="w", font=("Arial", 9)
        ).pack(fill="x", padx=16)

        ch_names = self.preprocessor.cleaned_raw.ch_names \
            if self.preprocessor.cleaned_raw else []
        manual_ch_var = tk.StringVar(value=ch_names[0] if ch_names else "")
        ttk.Combobox(
            left, textvariable=manual_ch_var,
            values=ch_names, state="readonly",
            width=22, font=("Arial", 9)
        ).pack(anchor="w", padx=24, pady=4)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=6, pady=8)

        # ── Status label ───────────────────────────────────────────────────
        status_label = tk.Label(
            left, text="", font=("Arial", 9),
            fg="#b71c1c", wraplength=230, justify="left"
        )
        status_label.pack(fill="x", padx=8)

        # ── Plot button ────────────────────────────────────────────────────
        def do_plot():
            selected = [lbl for lbl, v in filtered_list_state.items() if v.get()]

            if len(selected) != 2:
                status_label.config(text="⚠  Select exactly 2 snapshots.")
                return

            mode = mode_var.get()
            if mode == "None":
                status_label.config(text="⚠  Select a plot mode.")
                return

            # Retrieve raw objects from snapshots
            plotting_data = []
            for lbl in selected:
                for snap in snapshots:
                    if snap.get("label") == lbl:
                        plotting_data.append((snap["raw"], lbl))
                        break

            if len(plotting_data) < 2:
                status_label.config(text="⚠  Could not retrieve snapshot data.")
                return

            raw1, label1 = plotting_data[0]
            raw2, label2 = plotting_data[1]

            # Resolve channel index
            channel_index = 0
            if channel_mode_var.get() == "Auto (max difference)":
                try:
                    d1 = raw1.get_data(picks="eeg")
                    d2 = raw2.get_data(picks="eeg")
                    diff = np.abs(d1 - d2).mean(axis=1)
                    channel_index = int(np.argmax(diff))
                except Exception:
                    channel_index = 0
            else:
                ch = manual_ch_var.get()
                channel_index = ch_names.index(ch) if ch in ch_names else 0

            # Generate figure
            try:
                if mode == "PSD":
                    fig = self.visualize.plot_comparision_psd(raw1, raw2, label1, label2)
                elif mode == "Raw Overlay":
                    fig = self.visualize.plot_raw_overlay_comparision(raw1, raw2, label1, label2, channel_index)
                elif mode == "Difference signal":
                    fig = self.visualize.plot_difference_signal(raw1, raw2, label1, label2, channel_index)
                elif mode == "Band Power":
                    fig = self.visualize.plot_band_power_comparision(raw1, raw2, label1, label2)
                else:
                    status_label.config(text="⚠  Unknown mode.")
                    return
            except Exception as e:
                status_label.config(text=f"Plot error: {e}")
                return

            # Render into right panel
            for widget in plot_area.winfo_children():
                widget.destroy()
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            canvas = FigureCanvasTkAgg(fig, master=plot_area)
            canvas.draw()
            canvas.get_tk_widget().pack(fill="both", expand=True)
            status_label.config(text="")

        tk.Button(
            left, text="Plot", width=18,
            bg="#1565c0", fg="white",
            font=("Arial", 10, "bold"),
            command=do_plot
        ).pack(pady=8)

        tk.Button(
            left, text="Close", width=18,
            command=win.destroy
        ).pack(pady=2)



    def import_file(self):
        # If preprocessing already done — warn before wiping everything
        if self.run_complete:
            answer = messagebox.askyesnocancel(
                "New Dataset",
                "Preprocessing has already been run on the current file.\n\n"
                "Importing a new file will reset all results including:\n"
                "  • Preprocessing report and interpretation\n"
                "  • Methods text\n"
                "  • Epoch analysis and trial data\n"
                "  • Hypothesis report\n\n"
                "Do you want to continue and reset everything?",
                icon="warning"
            )
            if not answer:   # No or Cancel
                return
            self._reset_all()

        file_path = filedialog.askopenfilename(
            title="Select EEG File",
            filetypes=[
                ("EEG files", "*.edf *.csv *.txt *.set"),
                ("All files", "*.*")
            ]
        )

        if file_path:
            self.selected_file = file_path
            file_name = os.path.basename(file_path)
            self.file_label.config(text=f"Selected: {file_name}  —  {file_path}")
            self.import_complete = True

    def _reset_all(self):
        """
        Reset every tab and all state back to initial — called when a new
        file is imported after preprocessing has already been run.
        """
        # ── Core state ──────────────────────────────────────────────────────
        self.preprocessor    = None
        self.run_complete    = False
        self.import_complete = False
        self._last_interp    = None
        self.selected_file   = None
        self.file_label.config(text="No file selected")
        self.progress_label.config(text="Ready.", fg="#555")

        # ── Tab 1 — Preprocessing report ────────────────────────────────────
        self.result_text.config(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.config(state="disabled")

        # ── Tab 2 — Interpretation ───────────────────────────────────────────
        self.interp_text.config(state="normal")
        self.interp_text.delete("1.0", tk.END)
        self.interp_text.config(state="disabled")

        # ── Tab 3 — Methods text ─────────────────────────────────────────────
        self.methods_text.config(state="normal")
        self.methods_text.delete("1.0", tk.END)
        self.methods_text.config(state="disabled")

        # ── Tab 4 — Epoch panel ──────────────────────────────────────────────
        # Clear epoch data and status
        if self._epoch_panel.analyzer is not None:
            self._epoch_panel.analyzer.epochs  = None
            self._epoch_panel.analyzer.events  = None
            self._epoch_panel.analyzer         = None

        self._epoch_panel._status_label.config(
            text="No epochs created.", fg="#555"
        )
        self._epoch_panel._trial_label.config(text="Trial —/—")
        self._epoch_panel._trial_info.config(text="")

        # Clear event table
        for row in self._epoch_panel._event_table.get_children():
            self._epoch_panel._event_table.delete(row)

        # Clear condition checkboxes
        for widget in self._epoch_panel._condition_frame.winfo_children():
            widget.destroy()
        self._epoch_panel._condition_vars = {}

        # Clear condition dropdowns
        self._epoch_panel._cond_a_combo.set("")
        self._epoch_panel._cond_a_combo["values"] = []
        self._epoch_panel._cond_b_combo.set("")
        self._epoch_panel._cond_b_combo["values"] = []
        self._epoch_panel._channel_combo.set("")
        self._epoch_panel._channel_combo["values"] = []

        # Clear plot area
        for widget in self._epoch_panel._plot_frame.winfo_children():
            widget.destroy()

        # ── Tab 5 — Hypothesis panel ─────────────────────────────────────────
        self._hypothesis_panel._analyzer = None
        self._hypothesis_panel._methods_text_cache = ""

        # Clear condition dropdowns
        self._hypothesis_panel._cond_a_combo.set("")
        self._hypothesis_panel._cond_a_combo["values"] = []
        self._hypothesis_panel._cond_b_combo.set("")
        self._hypothesis_panel._cond_b_combo["values"] = []

        # Clear report text
        self._hypothesis_panel._report_text.config(state="normal")
        self._hypothesis_panel._report_text.delete("1.0", tk.END)
        self._hypothesis_panel._report_text.config(state="disabled")

        # Clear status labels
        self._hypothesis_panel._status.config(text="")
        

    def run_preprocessing(self):
        if self.import_complete is False:
            messagebox.showerror("Error", "No EEG file selected.")
            return

        def task():
            try:
                # Read GUI vars on the main thread before the thread starts
                # (tkinter vars are not thread-safe to read from background threads)
                run_bandpass  = self.step_bandpass_var.get()
                run_notch     = self.step_notch_var.get()
                interpolate   = self.interpolate_var.get()
                fit_ica       = self.ica_var.get()

                try:
                    l_freq  = float(self.l_freq_var.get())
                    h_freq  = float(self.h_freq_var.get())
                    notch   = float(self.notch_var.get())
                    n_ica   = int(self.n_ica_var.get())
                except ValueError:
                    self.after(0, lambda: self._on_preprocessing_error(
                        "Invalid pipeline parameter — check Pipeline Options values."
                    ))
                    return

                self.after(0, lambda: self.progress_label.config(
                    text="⏳  Loading file...", fg="#1565c0"))
                self.preprocessor = EEGPreprocessor(self.selected_file)

                self.after(0, lambda: self.progress_label.config(
                    text="⏳  Running pipeline...", fg="#1565c0"))
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
                    progress_callback=lambda msg: self.after(
                        0, lambda m=msg: self.progress_label.config(
                            text=f"⏳  {m}", fg="#1565c0")
                    )
                )

                self.after(0, lambda: self._on_preprocessing_complete(report))

            except Exception as e:
                self.after(0, lambda: self._on_preprocessing_error(str(e)))

        threading.Thread(target=task, daemon=True).start()

    def _on_preprocessing_complete(self, report):
        bads = report.get("bad_channels", [])
        n_ch = report.get("n_channels", "?")
        sfreq = report.get("sfreq", "?")
        self.progress_label.config(
            text=f"✓  Preprocessing complete — "
                 f"{n_ch} channels, {sfreq} Hz, "
                 f"{len(bads)} bad channel(s) detected",
            fg="#2e7d32"
        )
        self.display_report(report)
        self.run_complete = True
        self._run_interpretation()
        self._epoch_panel.set_preprocessor(self.preprocessor)
        self._epoch_panel.on_epochs_created = self._on_epochs_created

    def _on_epochs_created(self, analyzer):
        """Called by epoch panel when epochs are successfully created."""
        audience = self.audience_var.get()
        self._hypothesis_panel.set_analyzer(analyzer, audience=audience)

    def _on_preprocessing_error(self, error_msg):
        self.progress_label.config(
            text=f"✗  Error: {error_msg[:120]}",
            fg="#b71c1c"
        )
        messagebox.showerror("Preprocessing Error", error_msg)

    def _run_interpretation(self):
        if self.preprocessor is None:
            return
        audience = self.audience_var.get()
        interp = EEGInterpreter(self.preprocessor.report, audience=audience)
        self._last_interp = interp.interpret()
        self._render_interpretation(self._last_interp)
        self._render_methods(self._last_interp)

    def _refresh_interpretation(self):
        """Called when audience radio button changes."""
        if self._last_interp is None or self.preprocessor is None:
            return
        audience = self.audience_var.get()
        interp = EEGInterpreter(self.preprocessor.report, audience=audience)
        self._last_interp = interp.interpret()
        self._render_interpretation(self._last_interp)
        self._render_methods(self._last_interp)

    def _render_interpretation(self, result):
        t = self.interp_text
        t.config(state="normal")
        t.delete("1.0", tk.END)

        quality = result["quality"]
        noise   = result["noise"]
        pipe    = result["pipeline"]

        ICONS = {"good": "✓", "medium": "⚠", "high": "✗", "info": "ℹ"}

        # --- Summary bar ---
        score = quality["score"]
        grade = quality["grade"]
        t.insert(tk.END, "OVERALL SUMMARY\n", "heading")
        t.insert(tk.END, f"{result['summary']}\n\n", "subtext")

        # Score bar (ASCII)
        filled = int(score / 10)
        bar = "█" * filled + "░" * (10 - filled)
        colour = "good" if score >= 80 else ("medium" if score >= 60 else "high")
        t.insert(tk.END, f"Quality Score:  ")
        t.insert(tk.END, f"{bar}  {score}/100 — {grade}\n\n", colour)

        # --- Data quality findings ---
        t.insert(tk.END, "DATA QUALITY\n", "heading")
        for f in quality["findings"]:
            sev  = f["severity"]
            icon = ICONS.get(sev, "•")
            t.insert(tk.END, f"  {icon} {f['title']}\n", sev)
            t.insert(tk.END, f"     {f['explanation']}\n", "subtext")
            if f.get("action"):
                t.insert(tk.END, f"     → {f['action']}\n", "action")
        t.insert(tk.END, "\n")

        # --- Noise findings ---
        t.insert(tk.END, "NOISE ANALYSIS\n", "heading")
        if noise.get("note"):
            t.insert(tk.END, f"  ℹ {noise['note']}\n\n", "info")
        else:
            for f in noise.get("findings", []):
                sev  = f.get("severity", "info")
                icon = ICONS.get(sev, "•")
                comps = f.get("components", [])
                comp_str = f"  Components: {comps}" if comps else ""
                t.insert(tk.END, f"  {icon} {f['name']}  [{f.get('status','')}]\n", sev)
                if comp_str:
                    t.insert(tk.END, f"     {comp_str}\n", "subtext")
                t.insert(tk.END, f"     {f.get('explanation','')}\n", "subtext")
                if f.get("action"):
                    t.insert(tk.END, f"     → {f['action']}\n", "action")
            t.insert(tk.END, "\n")

        # --- Pipeline summary ---
        t.insert(tk.END, "PROCESSING STEPS\n", "heading")
        for step in pipe["steps"]:
            t.insert(tk.END, f"  • {step}\n", "subtext")
        t.insert(tk.END, "\n")

        t.config(state="disabled")
        t.see("1.0")

    def _render_methods(self, result):
        t = self.methods_text
        t.config(state="normal")
        t.delete("1.0", tk.END)
        t.insert(tk.END, result["methods_text"])
        t.config(state="disabled")

    def display_report(self, report):
        self.result_text.config(state="normal")
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
                lines.append(f"  - Number Interpolated: {interpolation.get('n_interpolated')}")

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

        # self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(lines))
        self.result_text.config(state="disabled")
        # self.clear_plot()


    def clear_plot(self):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        # self.show_figure()

    def show_figure(self, fig):
        self.clear_plot()
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def call_plot_raw_channel(self):
        if self.run_complete is False:
            messagebox.showerror("Error", "Run preprocessing first.")
            return
        try:
            raw = self.preprocessor.cleaned_raw

            fig = self.visualize.plot_raw_channel(raw)

            self.show_figure(fig)
        except Exception as e:
            messagebox.showerror("Raw channel plot error",str(e))

    def call_plot_psd(self):
        if self.run_complete is False:
            messagebox.showerror("Error", "Run preprocessing first.")
            return
        try:
            raw = self.preprocessor.cleaned_raw
            
            fig = self.visualize.plot_psd(raw)

            self.show_figure(fig)
        except Exception as e:
            messagebox.showerror("Psd plot error", str(e))

    def call_plot_ica_components(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            messagebox.showerror("Error", "Run preprocessing with ICA first.")
            return

        try:
            data = self.preprocessor.ica
            self.visualize.plot_ica_components(data)
            # self.preprocessor.ica.plot_sources()
        except Exception as e:
            messagebox.showerror("ICA Plot Error", str(e))

    def call_plot_ica_sources(self):
        c_raw = self.preprocessor.cleaned_raw
        if self.preprocessor is None or self.preprocessor.ica is None:
            messagebox.showerror("Error", "Run preprocessing with ICA first.")
            return

        try:
            data = self.preprocessor.ica
            self.visualize.plot_ica_sources(data,c_raw)
        except Exception as e:
            messagebox.showerror("ICA Plot Error", str(e))

    def plot_ica_properties_from_entry(self):
        c_raw = self.preprocessor.cleaned_raw
        if self.preprocessor is None or self.preprocessor.ica is None:
            messagebox.showerror("Error", "ICA has not been fitted yet.")
            return
        
        if not self.ica_entry.get().strip():
            messagebox.showerror("Error", "Enter component numbers like 0,1,2")
            return
        try:
            data = self.preprocessor.ica
            self.visualize.plot_ica_properties_from_entry(self.ica_entry,c_raw,data)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def apply_ica_from_entry(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            messagebox.showerror("Error", "ICA has not been fitted yet.")
            return

        text = self.ica_entry.get().strip()

        if not text:
            messagebox.showerror("Error", "Enter components like 0,2,5")
            return

        try:
            
            self.preprocessor.apply_ica_exclusion(text)
            self.display_report(self.preprocessor.report)
            self._run_interpretation()
            self._epoch_panel.set_preprocessor(self.preprocessor)
            self._epoch_panel.mark_epochs_stale()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def auto_detect_ica_components(self):
        if self.preprocessor is None or self.preprocessor.ica is None:
            messagebox.showerror("Error", "ICA has not been fitted yet.")
            return

        try:
            result = self.preprocessor.auto_detect_ica_artifacts()
            suggested = result.get("suggested_exclude", [])

            self.ica_entry.delete(0, tk.END)
            self.ica_entry.insert(0, ",".join(map(str, suggested)))

            lines = []
            lines.append("=" * 50)
            lines.append("ICA AUTO-DETECTION REPORT")
            lines.append("=" * 50)

            eog_indices = result.get("eog_indices", [])
            lines.append(
                f"EOG-related components: {', '.join(map(str, eog_indices)) if eog_indices else 'None'}"
            )

            ecg_indices = result.get("ecg_indices", [])
            lines.append(
                f"ECG-related components: {', '.join(map(str, ecg_indices)) if ecg_indices else 'None'}"
            )

            lines.append(
                f"Suggested components to exclude: {', '.join(map(str, suggested)) if suggested else 'None'}"
            )

            eog_scores = result.get("eog_scores", [])
            if eog_scores:
                lines.append(f"EOG scores: {eog_scores}")

            ecg_scores = result.get("ecg_scores", [])
            if ecg_scores:
                lines.append(f"ECG scores: {ecg_scores}")

            if "eog_error" in result:
                lines.append(f"EOG detection error: {result['eog_error']}")

            if "ecg_error" in result:
                lines.append(f"ECG detection error: {result['ecg_error']}")

            self.append_to_report_widget(lines)

        except Exception as e:
            messagebox.showerror("Auto Detect Error", str(e))

    def append_to_report_widget(self, lines):
        self.result_text.config(state="normal")
        self.result_text.insert(tk.END, "\n" + "\n".join(lines) + "\n")
        self.result_text.config(state="disabled")
        self.result_text.see(tk.END)

    def show_pipeline_options(self):
        win = tk.Toplevel(self)
        win.title("Pipeline Options")
        win.geometry("360x460")
        win.resizable(False, False)
        win.grab_set()

        tk.Label(
            win, text="Pipeline Options",
            font=("Arial", 12, "bold")
        ).pack(pady=(12, 4))

        tk.Label(
            win,
            text="Choose which steps run and set their parameters.\n"
                 "Greyed-out steps are always required.",
            font=("Arial", 9), fg="#555"
        ).pack(pady=(0, 8))

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=14)

        # Helper to build a step row
        def step_row(parent, label, var, enabled=True):
            row = tk.Frame(parent)
            row.pack(fill="x", padx=18, pady=3)
            cb = tk.Checkbutton(
                row, text=label, variable=var,
                font=("Arial", 10),
                state="normal" if enabled else "disabled"
            )
            cb.pack(side="left")
            return row

        # Helper to build a parameter sub-row
        def param_row(parent, label, var, unit=""):
            row = tk.Frame(parent)
            row.pack(fill="x", padx=36, pady=1)
            tk.Label(row, text=label, font=("Arial", 9),
                     fg="#444", width=22, anchor="w").pack(side="left")
            tk.Entry(row, textvariable=var, width=8,
                     font=("Arial", 9)).pack(side="left", padx=4)
            if unit:
                tk.Label(row, text=unit, font=("Arial", 9),
                         fg="#888").pack(side="left")

        f = tk.Frame(win)
        f.pack(fill="both", expand=True, pady=6)

        # Step 1 — Load file (always on)
        step_row(f, "1.  Load .set file", tk.BooleanVar(value=True), enabled=False)

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 2 — Bandpass filter
        step_row(f, "2.  Bandpass filter", self.step_bandpass_var)
        param_row(f, "Low cutoff:",  self.l_freq_var, "Hz")
        param_row(f, "High cutoff:", self.h_freq_var, "Hz")

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 3 — Notch filter
        step_row(f, "3.  Notch filter", self.step_notch_var)
        param_row(f, "Frequency:", self.notch_var, "Hz")

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 4 — Average reference (always on)
        step_row(f, "4.  Average reference", self.step_reference_var, enabled=False)

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 5 — Bad channel detection (always on)
        step_row(f, "5.  Detect bad channels", self.step_badch_var, enabled=False)

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 6 — Interpolation (optional)
        step_row(f, "6.  Interpolate bad channels", self.interpolate_var)

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 7 — ICA (optional)
        step_row(f, "7.  Fit ICA", self.ica_var)
        param_row(f, "Components:", self.n_ica_var, "")

        ttk.Separator(f, orient="horizontal").pack(fill="x", padx=14, pady=2)

        # Step 8 — Artifact metrics (always on)
        step_row(f, "8.  Compute artifact metrics", self.step_metrics_var, enabled=False)

        ttk.Separator(win, orient="horizontal").pack(fill="x", padx=14, pady=6)

        # Buttons
        btn_row = tk.Frame(win)
        btn_row.pack(pady=6)

        tk.Button(
            btn_row, text="OK", width=12,
            bg="#1565c0", fg="white", font=("Arial", 9, "bold"),
            command=win.destroy
        ).pack(side="left", padx=8)

        tk.Button(
            btn_row, text="Cancel", width=10,
            command=win.destroy
        ).pack(side="left", padx=8)

    def open_settings(self):
        SettingsPanel(self, on_save_callback=self._refresh_interpretation)

    def export_menu(self):
        if self.preprocessor is None or not self.run_complete:
            messagebox.showerror("Error", "Run preprocessing before exporting.")
            return

        win = tk.Toplevel(self)
        win.title("Export")
        win.geometry("260x140")
        win.resizable(False, False)

        tk.Label(win, text="Export Options", font=("Arial", 11, "bold")).pack(pady=10)

        tk.Button(
            win, text="Save Cleaned EEG (.fif)", width=26,
            command=lambda: self._export_fif(win)
        ).pack(pady=4)

        tk.Button(
            win, text="Save Report (.json)", width=26,
            command=lambda: self._export_json(win)
        ).pack(pady=4)

    def _export_fif(self, parent_win):
        path = filedialog.asksaveasfilename(
            title="Save cleaned EEG",
            defaultextension=".fif",
            filetypes=[("FIF files", "*.fif")]
        )
        if path:
            try:
                self.preprocessor.save_cleaned_file(path)
                messagebox.showinfo("Saved", f"Cleaned EEG saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))

    def _export_json(self, parent_win):
        path = filedialog.asksaveasfilename(
            title="Save preprocessing report",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")]
        )
        if path:
            try:
                self.preprocessor.save_report_json(path)
                messagebox.showinfo("Saved", f"Report saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Export Error", str(e))