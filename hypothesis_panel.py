"""
Hypothesis Panel
----------------
GUI tab for hypothesis-driven analysis.
Sits inside the main notebook in EEG_analysis_tool.py.

Layout:
    Left  — hypothesis input + analysis parameters
    Right — structured report output (scrollable text with colour tags)

Flow:
    1. User types a hypothesis
    2. Selects condition A vs B, time window, audience
    3. Clicks Generate Report
    4. Report renders with verdict, ERP findings, band power, limits, next steps
    5. Methods paragraph available to copy
"""

import tkinter as tk
from tkinter import ttk, messagebox

from hypothesis_reporter import HypothesisReporter


class HypothesisPanel(tk.Frame):

    def __init__(self, parent):
        super().__init__(parent, bg="white")

        self._analyzer  = None   # EpochAnalyzer — set from outside
        self._audience  = "researcher"

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        # Two-column layout
        left_container = tk.Frame(self, bg="white", width=320)
        left_container.pack(side="left", fill="y", padx=0, pady=8)
        left_container.pack_propagate(False)

        # Scrollable left column
        lc = tk.Canvas(left_container, bg="white", highlightthickness=0)
        lsb = tk.Scrollbar(left_container, orient="vertical", command=lc.yview)
        left = tk.Frame(lc, bg="white")

        left.bind("<Configure>",
                  lambda e: lc.configure(scrollregion=lc.bbox("all")))

        # Keep inner frame width locked to canvas width so nothing clips right
        lc.bind("<Configure>",
                lambda e: lc.itemconfig(lc_window, width=e.width))

        lc_window = lc.create_window((0, 0), window=left, anchor="nw")
        lc.configure(yscrollcommand=lsb.set)

        lsb.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)

        def _mw(e):
            try:
                lc.yview_scroll(-1 * (e.delta // 120), "units")
            except tk.TclError:
                pass

        lc.bind("<Enter>", lambda e: lc.bind_all("<MouseWheel>", _mw))
        lc.bind("<Leave>", lambda e: lc.unbind_all("<MouseWheel>"))

        # Right — report output
        right = tk.Frame(self, bg="white")
        right.pack(side="right", fill="both", expand=True, padx=8, pady=8)
        self._build_report_panel(right)

        # ── Left contents ──────────────────────────────────────────────────
        pad = {"padx": 10}

        # Section: Hypothesis
        self._section(left, "Your Hypothesis")
        tk.Label(
            left, text="State what you expect to find:",
            bg="white", font=("Arial", 9), fg="#555", anchor="w"
        ).pack(fill="x", **pad)

        self._hyp_text = tk.Text(
            left, height=4, wrap="word",
            font=("Arial", 10), relief="solid", bd=1
        )
        self._hyp_text.pack(fill="x", padx=10, pady=4)
        self._hyp_text.insert("1.0",
            "e.g. Target stimuli will produce a larger P300 than distractors")
        self._sep(left)

        # Section: Conditions
        self._section(left, "Conditions")

        tk.Label(left, text="Condition A (primary):",
                 bg="white", font=("Arial", 9)).pack(anchor="w", **pad)
        self._cond_a_var = tk.StringVar()
        self._cond_a_combo = ttk.Combobox(
            left, textvariable=self._cond_a_var,
            state="readonly")
        self._cond_a_combo.pack(fill="x", padx=14, pady=2)

        tk.Label(left, text="Condition B (comparison):",
                 bg="white", font=("Arial", 9)).pack(anchor="w", **pad)
        self._cond_b_var = tk.StringVar()
        self._cond_b_combo = ttk.Combobox(
            left, textvariable=self._cond_b_var,
            state="readonly")
        self._cond_b_combo.pack(fill="x", padx=14, pady=2)

        self._sep(left)

        # Section: Time window
        self._section(left, "Time Window of Interest")
        tk.Label(
            left,
            text="The latency range where you expect an effect (ms):",
            bg="white", font=("Arial", 9), fg="#555", wraplength=280
        ).pack(anchor="w", **pad)

        win_frame = tk.Frame(left, bg="white")
        win_frame.pack(anchor="w", padx=14, pady=4)

        tk.Label(win_frame, text="From:", bg="white",
                 font=("Arial", 9)).grid(row=0, column=0, padx=4)
        self._tlo_var = tk.StringVar(value="250")
        tk.Entry(win_frame, textvariable=self._tlo_var,
                 width=6, font=("Arial", 9)).grid(row=0, column=1)
        tk.Label(win_frame, text="ms    To:", bg="white",
                 font=("Arial", 9)).grid(row=0, column=2, padx=4)
        self._thi_var = tk.StringVar(value="500")
        tk.Entry(win_frame, textvariable=self._thi_var,
                 width=6, font=("Arial", 9)).grid(row=0, column=3)
        tk.Label(win_frame, text="ms", bg="white",
                 font=("Arial", 9)).grid(row=0, column=4, padx=2)

        self._sep(left)

        # Section: Audience
        self._section(left, "Report Style")
        self._audience_var = tk.StringVar(value="researcher")
        for val, lbl in [
            ("researcher", "Researcher"),
            ("clinician",  "Clinician"),
            ("general",    "General audience")
        ]:
            tk.Radiobutton(
                left, text=lbl, variable=self._audience_var,
                value=val, bg="white", font=("Arial", 9)
            ).pack(anchor="w", padx=14)

        self._sep(left)

        # Status
        self._status = tk.Label(
            left, text="", bg="white",
            font=("Arial", 9), fg="#b71c1c",
            wraplength=280, justify="left"
        )
        self._status.pack(fill="x", **pad)

        # Generate button
        tk.Button(
            left, text="Generate Report",
            bg="#1565c0", fg="white",
            font=("Arial", 10, "bold"),
            command=self._generate
        ).pack(fill="x", pady=8, padx=10)

        # Copy methods button
        tk.Button(
            left, text="Copy Methods Paragraph",
            command=self._copy_methods
        ).pack(fill="x", pady=2, padx=10)

    def _build_report_panel(self, parent):
        # Title bar
        tk.Label(
            parent, text="Analysis Report",
            font=("Arial", 11, "bold"), bg="white", anchor="w"
        ).pack(fill="x", pady=(0, 2))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=2)

        # ── Metadata disclaimer notice ─────────────────────────────────────
        notice_frame = tk.Frame(parent, bg="#fff8e1", bd=1, relief="solid")
        notice_frame.pack(fill="x", pady=(4, 6))

        tk.Label(
            notice_frame,
            text="ℹ  Important — please read before interpreting results",
            font=("Arial", 9, "bold"), bg="#fff8e1", fg="#e65100",anchor="w"
        ).pack(fill="x", padx=8, pady=(6, 2))

        tk.Label(
            notice_frame,
            text=(f"The hypothesis text above is used as a label only — it is metadata. "
                "The tool does not read, parse, or interpret the words you type.\n\n"
                "All findings, verdicts, and conclusions in this report are computed "
                "solely from the Condition A vs Condition B selection and the time "
                "window you specified. Two different hypothesis statements with the "
                "same conditions and window will produce identical results.\n\n"
                "Review all findings critically before drawing conclusions. "
                "This report is a structured summary of descriptive statistics — "
                "not a statistical inference or a clinical judgement.\n\nSuggested "
                "next steps are rule-based, not hypothesis-aware. Steps such as "
                "run statistical testing and replicate with more participants "
                "appear on every report regardless of findings. Steps triggered by "
                "data (time window adjustment, alpha follow-up) are based on "
                "numerical thresholds only — not on what your hypothesis predicted. "
                "Treat next steps as a general checklist to consider, not as a personalised research roadmap."),
            font=("Arial", 8), bg="#fff8e1", fg="#555",
            wraplength=1080, justify="left",anchor="w"
        ).pack(fill="x", padx=8, pady=(0, 8))

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=2)

        # Scrollable text area
        txt_frame = tk.Frame(parent)
        txt_frame.pack(fill="both", expand=True)

        sb = tk.Scrollbar(txt_frame, orient="vertical")
        sb.pack(side="right", fill="y")

        self._report_text = tk.Text(
            txt_frame, wrap="word",
            yscrollcommand=sb.set,
            font=("Arial", 10), padx=12, pady=10,
            state="disabled", relief="flat"
        )
        self._report_text.pack(fill="both", expand=True)
        sb.config(command=self._report_text.yview)

        # Colour tags
        self._report_text.tag_config(
            "heading", font=("Arial", 11, "bold"), spacing1=8)
        self._report_text.tag_config(
            "subheading", font=("Arial", 10, "bold"), spacing1=6)
        self._report_text.tag_config(
            "good",   foreground="#2e7d32", font=("Arial", 10, "bold"))
        self._report_text.tag_config(
            "medium", foreground="#e65100", font=("Arial", 10, "bold"))
        self._report_text.tag_config(
            "high",   foreground="#b71c1c", font=("Arial", 10, "bold"))
        self._report_text.tag_config(
            "body",   font=("Arial", 10))
        self._report_text.tag_config(
            "dim",    foreground="#666", font=("Arial", 9))
        self._report_text.tag_config(
            "bullet", font=("Arial", 10), lmargin1=20, lmargin2=30)
        self._report_text.tag_config(
            "methods_box",
            font=("Courier", 9),
            foreground="#1a237e",
            background="#e8eaf6",
            lmargin1=10, lmargin2=10,
            spacing1=4, spacing3=4
        )

        self._methods_text_cache = ""

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def set_analyzer(self, analyzer, audience: str = "researcher"):
        """Called from analysis_frame when epochs are created."""
        self._analyzer = analyzer
        self._audience_var.set(audience)

        if analyzer is None or analyzer.epochs is None:
            return

        conditions = list(analyzer.epochs.event_id.keys())
        self._cond_a_combo["values"] = conditions
        self._cond_b_combo["values"] = conditions
        if len(conditions) >= 1:
            self._cond_a_combo.set(conditions[0])
        if len(conditions) >= 2:
            self._cond_b_combo.set(conditions[1])

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------

    def _generate(self):
        if self._analyzer is None or self._analyzer.epochs is None:
            self._status.config(
                text="⚠  Create epochs in the Epoch Analysis tab first.")
            return

        hyp = self._hyp_text.get("1.0", tk.END).strip()
        if not hyp or hyp.startswith("e.g."):
            self._status.config(text="⚠  Enter your hypothesis first.")
            return

        cond_a = self._cond_a_var.get()
        cond_b = self._cond_b_var.get()

        if not cond_a or not cond_b:
            self._status.config(text="⚠  Select both conditions.")
            return
        if cond_a == cond_b:
            self._status.config(text="⚠  Conditions must be different.")
            return

        try:
            t_lo = float(self._tlo_var.get())
            t_hi = float(self._thi_var.get())
            if t_lo >= t_hi:
                self._status.config(text="⚠  Time window start must be before end.")
                return
        except ValueError:
            self._status.config(text="⚠  Enter valid numbers for the time window.")
            return

        audience = self._audience_var.get()

        try:
            erp_a        = self._analyzer.get_erp(condition=cond_a)
            erp_b        = self._analyzer.get_erp(condition=cond_b)
            band_power_a = self._analyzer.get_band_power(condition=cond_a)
            band_power_b = self._analyzer.get_band_power(condition=cond_b)
            trial_quality = self._analyzer.get_trial_quality()
        except Exception as e:
            self._status.config(text=f"Error computing results: {e}")
            return

        try:
            reporter = HypothesisReporter(
                hypothesis    = hyp,
                condition_a   = cond_a,
                condition_b   = cond_b,
                time_window   = (t_lo, t_hi),
                erp_a         = erp_a,
                erp_b         = erp_b,
                band_power_a  = band_power_a,
                band_power_b  = band_power_b,
                trial_quality = trial_quality,
                audience      = audience
            )
            result = reporter.generate()
        except Exception as e:
            self._status.config(text=f"Report generation error: {e}")
            return

        self._status.config(text="")
        self._render_report(result)

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------

    def _render_report(self, result: dict):
        t = self._report_text
        t.config(state="normal")
        t.delete("1.0", tk.END)

        def w(text, tag="body"):
            t.insert(tk.END, text, tag)

        def nl(n=1):
            t.insert(tk.END, "\n" * n)

        ICONS = {"good": "✓", "medium": "⚠", "high": "✗", "info": "ℹ"}

        # ── Header ──────────────────────────────────────────────────────────
        w("ANALYSIS REPORT\n", "heading")
        w(f"Hypothesis: ", "subheading")
        w(f"{result['hypothesis']}\n", "body")
        w(f"Conditions: {result['condition_a']}  vs  {result['condition_b']}\n", "dim")
        w(f"Time window: {result['time_window'][0]}–{result['time_window'][1]} ms\n", "dim")
        nl()

        # ── Trial quality ────────────────────────────────────────────────────
        tq = result["trial_summary"]
        w("DATA QUALITY\n", "subheading")
        colour = {"good": "good", "acceptable": "medium", "poor": "high"}[tq["quality"]]
        w(f"  {ICONS[colour]}  {tq['n_clean']}/{tq['n_total']} trials retained "
          f"({tq['pct_kept']}%)\n", colour)
        w(f"  {tq['note']}\n", "dim")
        nl()

        # ── Verdict ──────────────────────────────────────────────────────────
        v = result["verdict"]
        w("VERDICT\n", "subheading")
        w(f"  {ICONS.get(v['colour'], '•')}  {v['verdict']}\n", v["colour"])
        w(f"\n  {v['rationale']}\n", "body")
        nl()

        # ── ERP findings ──────────────────────────────────────────────────────
        erp = result["erp_findings"]
        w("ERP FINDINGS\n", "subheading")
        w(f"  {erp['direction_text']}\n", "body")
        nl()

        if erp["component_notes"]:
            w("  Component analysis:\n", "dim")
            for note in erp["component_notes"]:
                w(f"  •  {note}\n", "bullet")
        nl()

        # ── Band power ────────────────────────────────────────────────────────
        band = result["band_findings"]
        if band["findings"]:
            w("FREQUENCY BAND ANALYSIS\n", "subheading")
            for f in band["findings"]:
                sev  = f["severity"]
                icon = ICONS.get(sev, "•")
                w(f"  {icon}  {f['band']}: {f['direction']}\n", sev if sev != "info" else "dim")
                w(f"     {f['note']}\n", "dim")
            nl()

        # ── Limits ───────────────────────────────────────────────────────────
        w("WHAT THE DATA CANNOT TELL YOU\n", "subheading")
        for lim in result["limits"]:
            w(f"  •  {lim}\n", "bullet")
        nl()

        # ── Next steps ───────────────────────────────────────────────────────
        w("SUGGESTED NEXT STEPS\n", "subheading")
        for i, step in enumerate(result["next_steps"], 1):
            w(f"  {i}.  {step}\n", "bullet")
        nl()

        # ── Methods paragraph ─────────────────────────────────────────────────
        w("METHODS PARAGRAPH\n", "subheading")
        w("  (Copy this directly into your methods section)\n\n", "dim")
        methods = result["methods"]
        self._methods_text_cache = methods
        w(f"  {methods}\n", "methods_box")

        t.config(state="disabled")
        t.see("1.0")

    # -----------------------------------------------------------------------
    # Copy methods
    # -----------------------------------------------------------------------

    def _copy_methods(self):
        if not self._methods_text_cache:
            messagebox.showinfo("Nothing to copy",
                                "Generate a report first.")
            return
        self.clipboard_clear()
        self.clipboard_append(self._methods_text_cache)
        messagebox.showinfo("Copied",
                            "Methods paragraph copied to clipboard.")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _section(self, parent, text):
        tk.Label(
            parent, text=text, bg="white",
            font=("Arial", 10, "bold"), anchor="w"
        ).pack(fill="x", padx=10, pady=(10, 2))

    def _sep(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(
            fill="x", padx=10, pady=6)