"""
Hypothesis Panel (PySide6)
--------------------------
GUI tab for hypothesis-driven analysis.
Sits inside the main QTabWidget in analysis_widget.py.

Layout:
    Left  — hypothesis input + analysis parameters (scrollable)
    Right — structured report output (QTextEdit with coloured text)
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QScrollArea,
    QLabel, QPushButton, QTextEdit, QComboBox, QLineEdit,
    QFrame, QRadioButton, QButtonGroup, QApplication,
    QMessageBox, QSizePolicy, QGridLayout
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor, QTextCharFormat, QColor

from hypothesis_reporter import HypothesisReporter


# ---------------------------------------------------------------------------
# Shared coloured-text helper
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


class HypothesisPanel(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self._analyzer = None
        self._audience = "researcher"
        self._methods_text_cache = ""

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI
    # -----------------------------------------------------------------------

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 8, 8)
        main_layout.setSpacing(0)

        # ── Scrollable left column ──────────────────────────────────────────
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(330)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left_widget = QWidget()
        self._left_layout = QVBoxLayout(left_widget)
        self._left_layout.setContentsMargins(8, 4, 8, 4)
        self._left_layout.setSpacing(4)
        scroll_area.setWidget(left_widget)
        main_layout.addWidget(scroll_area)

        # ── Right report panel ──────────────────────────────────────────────
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(8, 4, 4, 4)
        right_layout.setSpacing(4)
        main_layout.addWidget(right_widget, stretch=1)

        self._build_report_panel(right_layout)

        # ── Left contents ───────────────────────────────────────────────────
        ll = self._left_layout

        # Section: Hypothesis
        self._section(ll, "Your Hypothesis")
        hint_lbl = QLabel("State what you expect to find:")
        hint_lbl.setFont(QFont("Arial", 9))
        hint_lbl.setStyleSheet("color: #555;")
        ll.addWidget(hint_lbl)

        self._hyp_text = QTextEdit()
        self._hyp_text.setFont(QFont("Arial", 10))
        self._hyp_text.setFixedHeight(80)
        self._hyp_text.setPlaceholderText(
            "e.g. Target stimuli will produce a larger P300 than distractors"
        )
        ll.addWidget(self._hyp_text)
        self._sep(ll)

        # Section: Conditions
        self._section(ll, "Conditions")
        lbl_a = QLabel("Condition A (primary):")
        lbl_a.setFont(QFont("Arial", 9))
        ll.addWidget(lbl_a)
        self._cond_a_combo = QComboBox()
        self._cond_a_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ll.addWidget(self._cond_a_combo)

        lbl_b = QLabel("Condition B (comparison):")
        lbl_b.setFont(QFont("Arial", 9))
        ll.addWidget(lbl_b)
        self._cond_b_combo = QComboBox()
        self._cond_b_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        ll.addWidget(self._cond_b_combo)
        self._sep(ll)

        # Section: Time window
        self._section(ll, "Time Window of Interest")
        win_hint = QLabel("The latency range where you expect an effect (ms):")
        win_hint.setFont(QFont("Arial", 9))
        win_hint.setStyleSheet("color: #555;")
        win_hint.setWordWrap(True)
        ll.addWidget(win_hint)

        win_row = QWidget()
        win_row_layout = QHBoxLayout(win_row)
        win_row_layout.setContentsMargins(0, 0, 0, 0)
        win_row_layout.addWidget(QLabel("From:"))
        self._tlo_edit = QLineEdit("250")
        self._tlo_edit.setFixedWidth(55)
        win_row_layout.addWidget(self._tlo_edit)
        win_row_layout.addWidget(QLabel("ms    To:"))
        self._thi_edit = QLineEdit("500")
        self._thi_edit.setFixedWidth(55)
        win_row_layout.addWidget(self._thi_edit)
        win_row_layout.addWidget(QLabel("ms"))
        win_row_layout.addStretch()
        ll.addWidget(win_row)
        self._sep(ll)

        # Section: Audience
        self._section(ll, "Report Style")
        self._audience_group = QButtonGroup(self)
        for val, lbl_text in [
            ("researcher", "Researcher"),
            ("clinician",  "Clinician"),
            ("general",    "General audience")
        ]:
            rb = QRadioButton(lbl_text)
            rb.setFont(QFont("Arial", 9))
            rb.setProperty("value", val)
            if val == "researcher":
                rb.setChecked(True)
            self._audience_group.addButton(rb)
            ll.addWidget(rb)
        self._sep(ll)

        # Status
        self._status = QLabel("")
        self._status.setFont(QFont("Arial", 9))
        self._status.setStyleSheet("color: #b71c1c;")
        self._status.setWordWrap(True)
        ll.addWidget(self._status)

        # Generate button
        gen_btn = QPushButton("Generate Report")
        gen_btn.setStyleSheet(
            "QPushButton { background-color: #1565c0; color: white; "
            "font-weight: bold; font-size: 10pt; padding: 6px; border-radius: 3px; }"
            "QPushButton:hover { background-color: #1976d2; }"
        )
        gen_btn.clicked.connect(self._generate)
        ll.addWidget(gen_btn)

        copy_btn = QPushButton("Copy Methods Paragraph")
        copy_btn.clicked.connect(self._copy_methods)
        ll.addWidget(copy_btn)

        ll.addStretch()

    def _build_report_panel(self, layout):
        title_lbl = QLabel("Analysis Report")
        title_font = QFont("Arial", 11)
        title_font.setBold(True)
        title_lbl.setFont(title_font)
        layout.addWidget(title_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)

        # Disclaimer notice
        notice_frame = QFrame()
        notice_frame.setStyleSheet(
            "QFrame { background-color: #fff8e1; border: 1px solid #ffe082; border-radius: 3px; }"
        )
        notice_layout = QVBoxLayout(notice_frame)
        notice_layout.setContentsMargins(8, 6, 8, 8)

        notice_title = QLabel("ℹ  Important — please read before interpreting results")
        notice_title_font = QFont("Arial", 9)
        notice_title_font.setBold(True)
        notice_title.setFont(notice_title_font)
        notice_title.setStyleSheet("color: #e65100; background: transparent;")
        notice_layout.addWidget(notice_title)

        notice_body = QLabel(
            "The hypothesis text above is used as a label only — it is metadata. "
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
            "Treat next steps as a general checklist to consider, not as a personalised research roadmap."
        )
        notice_body.setFont(QFont("Arial", 8))
        notice_body.setStyleSheet("color: #555; background: transparent;")
        notice_body.setWordWrap(True)
        notice_layout.addWidget(notice_body)

        layout.addWidget(notice_frame)

        sep2 = QFrame()
        sep2.setFrameShape(QFrame.Shape.HLine)
        sep2.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep2)

        # Report text area
        self._report_text = QTextEdit()
        self._report_text.setFont(QFont("Arial", 10))
        self._report_text.setReadOnly(True)
        layout.addWidget(self._report_text, stretch=1)

    # -----------------------------------------------------------------------
    # Public
    # -----------------------------------------------------------------------

    def set_analyzer(self, analyzer, audience: str = "researcher"):
        """Called from analysis_widget when epochs are created."""
        self._analyzer = analyzer

        # Set audience radio
        for btn in self._audience_group.buttons():
            if btn.property("value") == audience:
                btn.setChecked(True)
                break

        if analyzer is None or analyzer.epochs is None:
            return

        conditions = list(analyzer.epochs.event_id.keys())
        self._cond_a_combo.clear()
        self._cond_b_combo.clear()
        self._cond_a_combo.addItems(conditions)
        self._cond_b_combo.addItems(conditions)
        if len(conditions) >= 1:
            self._cond_a_combo.setCurrentText(conditions[0])
        if len(conditions) >= 2:
            self._cond_b_combo.setCurrentText(conditions[1])

    # -----------------------------------------------------------------------
    # Generate
    # -----------------------------------------------------------------------

    def _generate(self):
        if self._analyzer is None or self._analyzer.epochs is None:
            self._status.setText("⚠  Create epochs in the Epoch Analysis tab first.")
            return

        hyp = self._hyp_text.toPlainText().strip()
        if not hyp:
            self._status.setText("⚠  Enter your hypothesis first.")
            return

        cond_a = self._cond_a_combo.currentText()
        cond_b = self._cond_b_combo.currentText()

        if not cond_a or not cond_b:
            self._status.setText("⚠  Select both conditions.")
            return
        if cond_a == cond_b:
            self._status.setText("⚠  Conditions must be different.")
            return

        try:
            t_lo = float(self._tlo_edit.text())
            t_hi = float(self._thi_edit.text())
            if t_lo >= t_hi:
                self._status.setText("⚠  Time window start must be before end.")
                return
        except ValueError:
            self._status.setText("⚠  Enter valid numbers for the time window.")
            return

        checked = self._audience_group.checkedButton()
        audience = checked.property("value") if checked else "researcher"

        try:
            erp_a         = self._analyzer.get_erp(condition=cond_a)
            erp_b         = self._analyzer.get_erp(condition=cond_b)
            band_power_a  = self._analyzer.get_band_power(condition=cond_a)
            band_power_b  = self._analyzer.get_band_power(condition=cond_b)
            trial_quality = self._analyzer.get_trial_quality()
        except Exception as e:
            self._status.setText(f"Error computing results: {e}")
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
            self._status.setText(f"Report generation error: {e}")
            return

        self._status.setText("")
        self._render_report(result)

    # -----------------------------------------------------------------------
    # Render
    # -----------------------------------------------------------------------

    def _render_report(self, result: dict):
        t = self._report_text
        t.setReadOnly(False)
        t.clear()

        def w(text, tag="body"):
            _insert(t, text, tag)

        def nl(n=1):
            _insert(t, "\n" * n, "body")

        ICONS = {"good": "✓", "medium": "⚠", "high": "✗", "info": "ℹ"}

        # Header
        w("ANALYSIS REPORT\n", "heading")
        w("Hypothesis: ", "subheading")
        w(f"{result['hypothesis']}\n", "body")
        w(f"Conditions: {result['condition_a']}  vs  {result['condition_b']}\n", "dim")
        w(f"Time window: {result['time_window'][0]}–{result['time_window'][1]} ms\n", "dim")
        nl()

        # Trial quality
        tq = result["trial_summary"]
        w("DATA QUALITY\n", "subheading")
        colour = {"good": "good", "acceptable": "medium", "poor": "high"}[tq["quality"]]
        w(f"  {ICONS[colour]}  {tq['n_clean']}/{tq['n_total']} trials retained "
          f"({tq['pct_kept']}%)\n", colour)
        w(f"  {tq['note']}\n", "dim")
        nl()

        # Verdict
        v = result["verdict"]
        w("VERDICT\n", "subheading")
        w(f"  {ICONS.get(v['colour'], '•')}  {v['verdict']}\n", v["colour"])
        w(f"\n  {v['rationale']}\n", "body")
        nl()

        # ERP findings
        erp = result["erp_findings"]
        w("ERP FINDINGS\n", "subheading")
        w(f"  {erp['direction_text']}\n", "body")
        nl()

        if erp["component_notes"]:
            w("  Component analysis:\n", "dim")
            for note in erp["component_notes"]:
                w(f"  •  {note}\n", "bullet")
        nl()

        # Band power
        band = result["band_findings"]
        if band["findings"]:
            w("FREQUENCY BAND ANALYSIS\n", "subheading")
            for f in band["findings"]:
                sev  = f["severity"]
                icon = ICONS.get(sev, "•")
                w(f"  {icon}  {f['band']}: {f['direction']}\n", sev if sev != "info" else "dim")
                w(f"     {f['note']}\n", "dim")
            nl()

        # Limits
        w("WHAT THE DATA CANNOT TELL YOU\n", "subheading")
        for lim in result["limits"]:
            w(f"  •  {lim}\n", "bullet")
        nl()

        # Next steps
        w("SUGGESTED NEXT STEPS\n", "subheading")
        for i, step in enumerate(result["next_steps"], 1):
            w(f"  {i}.  {step}\n", "bullet")
        nl()

        # Methods paragraph
        w("METHODS PARAGRAPH\n", "subheading")
        w("  (Copy this directly into your methods section)\n\n", "dim")
        methods = result["methods"]
        self._methods_text_cache = methods
        w(f"  {methods}\n", "methods_box")

        t.setReadOnly(True)
        cursor = t.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        t.setTextCursor(cursor)

    # -----------------------------------------------------------------------
    # Copy methods
    # -----------------------------------------------------------------------

    def _copy_methods(self):
        if not self._methods_text_cache:
            QMessageBox.information(self, "Nothing to copy", "Generate a report first.")
            return
        QApplication.clipboard().setText(self._methods_text_cache)
        QMessageBox.information(self, "Copied", "Methods paragraph copied to clipboard.")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _section(self, layout, text):
        lbl = QLabel(text)
        font = QFont("Arial", 10)
        font.setBold(True)
        lbl.setFont(font)
        layout.addWidget(lbl)

    def _sep(self, layout):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(sep)
