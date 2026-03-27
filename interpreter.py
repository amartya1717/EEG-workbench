"""
EEG Interpretation Layer
------------------------
Takes the raw report dict from EEGPreprocessor and produces structured
findings with quality scores, noise profiles, and audience-aware language.

Does NOT touch MNE or signal data directly — only reads the report dict
and history snapshots. Keeps processing and interpretation cleanly separated.
"""

from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Noise artifact profiles
# Each profile maps an artifact type to plain language at three audience levels
# ---------------------------------------------------------------------------

NOISE_PROFILES = {
    "eog": {
        "name": "Eye Movement Artifact",
        "researcher": (
            "Ocular artifact detected via frontal topography and slow-wave source. "
            "Accounts for variance in Fp1/Fp2 and adjacent channels. "
            "Standard exclusion via ICA is appropriate."
        ),
        "clinician": (
            "Eye blinks and movements are contaminating the frontal electrode signals. "
            "This is very common and has been flagged for removal."
        ),
        "general": (
            "Every time the participant blinked or moved their eyes, it created a large "
            "electrical signal that contaminated the brain data recorded near the forehead. "
            "This needs to be removed before any analysis."
        ),
        "visual_signature": "Frontal topography, large slow irregular waveform",
        "action": "Exclude this ICA component. Eye artifacts are the most common EEG contaminant and removal is standard practice.",
        "severity": "high"
    },
    "ecg": {
        "name": "Heartbeat Artifact (ECG)",
        "researcher": (
            "Cardiac artifact identified — regular ~1 Hz pulses consistent with heartbeat. "
            "Typically presents as a sharp biphasic waveform in ICA sources. "
            "Verify by cross-referencing with heart rate before excluding."
        ),
        "clinician": (
            "The participant's heartbeat is creating a regular electrical interference "
            "in the EEG signal. This has been flagged for review."
        ),
        "general": (
            "The heart produces strong electrical signals, and sometimes the EEG sensors "
            "pick these up. It shows up as a very regular repeated pulse in the data — "
            "once per heartbeat. This should be removed."
        ),
        "visual_signature": "Regular ~1Hz sharp peaks in source waveform",
        "action": "Exclude if the source waveform clearly matches a cardiac rhythm. Always verify before removing.",
        "severity": "medium"
    },
    "muscle": {
        "name": "Muscle Artifact (EMG)",
        "researcher": (
            "High-frequency broadband noise consistent with EMG contamination. "
            "Typically >20 Hz, peripheral or diffuse topography. "
            "Be conservative — muscle components can overlap with gamma-band activity."
        ),
        "clinician": (
            "Tension in facial or neck muscles is creating high-frequency electrical "
            "noise in the recording. This is common and depends on participant relaxation."
        ),
        "general": (
            "When muscles tighten — even slightly — they produce electrical signals. "
            "If the participant clenched their jaw or tensed their neck, this shows up "
            "as a fuzzy high-frequency noise across the sensors."
        ),
        "visual_signature": "Peripheral/diffuse topography, broadband high-frequency source",
        "action": "Only exclude if peripheral topography is confirmed. Muscle and gamma brain signals can look similar.",
        "severity": "medium"
    },
    "line_noise": {
        "name": "Line Noise (50/60 Hz)",
        "researcher": (
            "Power line interference at 50 or 60 Hz. Should be addressed at the "
            "filtering stage via notch filter, not ICA."
        ),
        "clinician": (
            "Electrical interference from power lines is present in the signal. "
            "This is addressed during the filtering step."
        ),
        "general": (
            "The electrical wiring in the room creates a faint hum that the sensors "
            "can sometimes pick up. This is removed using a notch filter during processing."
        ),
        "visual_signature": "Narrow spike at exactly 50 or 60 Hz in PSD",
        "action": "Address with notch filter during preprocessing, not ICA exclusion.",
        "severity": "low"
    }
}


# ---------------------------------------------------------------------------
# Quality thresholds
# These are the numerical boundaries that map metrics to verdicts
# ---------------------------------------------------------------------------

# Thresholds are now in config.py — not hardcoded here.


# ---------------------------------------------------------------------------
# Main interpreter class
# ---------------------------------------------------------------------------

class EEGInterpreter:
    """
    Reads a preprocessor report dict and produces structured findings.
    All quality thresholds and scoring weights are read from config at
    runtime — no values are hardcoded here.

    Usage:
        interpreter = EEGInterpreter(report, audience="researcher")
        result = interpreter.interpret()
    """

    def __init__(self, report: Dict[str, Any], audience: str = "researcher"):
        from config import config as _cfg
        self.report   = report
        self.audience = audience
        self.t        = _cfg.get_section("thresholds")  # threshold values
        self.s        = _cfg.get_section("scoring")     # score deductions
        self.g        = _cfg.get_section("grades")      # grade boundaries

    def interpret(self) -> Dict[str, Any]:
        """Run full interpretation. Returns structured findings dict."""
        quality  = self._assess_quality()
        noise    = self._assess_noise()
        pipeline = self._summarise_pipeline()
        methods  = self._generate_methods_text()
        summary  = self._generate_summary(quality, noise)

        return {
            "quality":      quality,
            "noise":        noise,
            "pipeline":     pipeline,
            "methods_text": methods,
            "summary":      summary,
            "audience":     self.audience
        }

    # -----------------------------------------------------------------------
    # Quality assessment
    # -----------------------------------------------------------------------

    def _assess_quality(self) -> Dict[str, Any]:
        score    = 100
        findings = []
        flags    = []

        t = self.t
        s = self.s

        n_channels   = self.report.get("n_channels") or 1
        bad_channels = self.report.get("bad_channels", [])
        bad_ratio    = len(bad_channels) / n_channels

        metrics     = self.report.get("artifact_metrics", {})
        mean_rms_v  = metrics.get("mean_rms", 0)
        mean_rms_uv = mean_rms_v * 1e6
        max_ptp_v   = metrics.get("max_peak_to_peak", 0)
        max_ptp_uv  = max_ptp_v * 1e6
        duration    = self.report.get("duration_sec", 0)

        # --- Bad channel ratio ---
        if bad_ratio > t["bad_channel_ratio_medium"]:
            score -= s["bad_channel_high"]
            flags.append("high")
            findings.append(self._finding(
                severity="high",
                title=f"{len(bad_channels)} bad channels detected ({round(bad_ratio * 100)}% of montage)",
                researcher=(
                    f"Channels flagged by variance z-score or flat std: "
                    f"{', '.join(bad_channels)}. Exceeds {round(t['bad_channel_ratio_medium']*100)}% threshold."
                ),
                clinician=(
                    f"{len(bad_channels)} electrode sites showed poor signal quality. "
                    f"Interpolation has been applied where possible."
                ),
                general=(
                    f"{len(bad_channels)} out of {n_channels} sensors were too noisy or "
                    f"completely flat. The tool estimated their values from nearby sensors."
                ),
                action="Review these channels manually. Consider re-recording if >20% are bad."
            ))
        elif bad_ratio > t["bad_channel_ratio_good"]:
            score -= s["bad_channel_medium"]
            flags.append("medium")
            findings.append(self._finding(
                severity="medium",
                title=f"{len(bad_channels)} bad channels detected ({round(bad_ratio * 100)}%)",
                researcher=f"Channels flagged: {', '.join(bad_channels)}. Within acceptable range.",
                clinician=f"{len(bad_channels)} electrode sites were interpolated from neighbours.",
                general=f"{len(bad_channels)} sensors had poor signal quality and were reconstructed.",
                action="Interpolation is standard practice for this level of channel loss."
            ))
        else:
            findings.append(self._finding(
                severity="good",
                title="Channel quality: good" + (f" ({len(bad_channels)} interpolated)" if bad_channels else " (no bad channels)"),
                researcher="No significant channel loss detected.",
                clinician="All electrode sites showed acceptable signal quality.",
                general="All sensors were recording cleanly.",
                action=None
            ))

        # --- RMS amplitude ---
        if mean_rms_uv > t["mean_rms_medium_uv"]:
            score -= s["rms_high"]
            flags.append("high")
            findings.append(self._finding(
                severity="high",
                title=f"High mean RMS amplitude: {round(mean_rms_uv, 2)} µV",
                researcher=(
                    f"Mean RMS of {round(mean_rms_uv, 2)} µV exceeds threshold "
                    f"({t['mean_rms_medium_uv']} µV). Likely residual artifact or movement noise."
                ),
                clinician="Signal amplitude is higher than expected, suggesting noise contamination.",
                general="The overall signal strength is higher than normal, which may mean there is still noise in the data.",
                action="Check for residual muscle or movement artifact. Consider stricter epoch rejection."
            ))
        elif mean_rms_uv > t["mean_rms_good_uv"]:
            score -= s["rms_medium"]
            flags.append("medium")
            findings.append(self._finding(
                severity="medium",
                title=f"Moderate RMS amplitude: {round(mean_rms_uv, 2)} µV",
                researcher=f"Mean RMS {round(mean_rms_uv, 2)} µV is slightly elevated but within acceptable range post-ICA.",
                clinician="Signal amplitude is slightly elevated but acceptable.",
                general="Signal strength is a little higher than ideal but within normal limits.",
                action="Monitor after ICA. If still elevated, consider epoch-level rejection."
            ))
        else:
            findings.append(self._finding(
                severity="good",
                title=f"RMS amplitude normal: {round(mean_rms_uv, 2)} µV",
                researcher=f"Mean RMS {round(mean_rms_uv, 2)} µV is within expected range for clean EEG.",
                clinician="Signal amplitude is within normal range.",
                general="The signal strength looks clean and normal.",
                action=None
            ))

        # --- Peak-to-peak ---
        if max_ptp_uv > t["max_ptp_medium_uv"]:
            score -= s["ptp_medium"]
            flags.append("medium")
            findings.append(self._finding(
                severity="medium",
                title=f"Large peak-to-peak excursion: {round(max_ptp_uv, 1)} µV",
                researcher=(
                    f"Max peak-to-peak of {round(max_ptp_uv, 1)} µV exceeds "
                    f"{t['max_ptp_medium_uv']} µV threshold. May be residual movement artifact."
                ),
                clinician="One or more channels show large voltage swings, possibly from movement.",
                general="At least one sensor recorded a very large voltage spike, which may be from movement.",
                action="Identify the channel with max peak-to-peak and inspect its raw signal."
            ))

        # --- Duration ---
        if duration < t["duration_medium_sec"]:
            score -= s["duration_high"]
            flags.append("high")
            findings.append(self._finding(
                severity="high",
                title=f"Short recording: {round(duration, 1)} sec",
                researcher=(
                    f"Recording duration of {round(duration, 1)}s is below the "
                    f"{t['duration_medium_sec']}s minimum recommended for reliable spectral or ICA analysis."
                ),
                clinician="The recording is shorter than recommended for reliable analysis.",
                general="The recording is quite short. Longer recordings give more reliable results.",
                action="Consider whether the recording length is sufficient for your analysis goals."
            ))
        elif duration < t["duration_good_sec"]:
            score -= s["duration_medium"]
            findings.append(self._finding(
                severity="medium",
                title=f"Recording duration borderline: {round(duration, 1)} sec",
                researcher=f"{round(duration, 1)}s is marginal for ICA. Results may be less stable.",
                clinician="Recording length is acceptable but on the short side.",
                general="The recording length is okay but longer would give more reliable results.",
                action=None
            ))
        else:
            findings.append(self._finding(
                severity="good",
                title=f"Recording duration sufficient: {round(duration, 1)} sec",
                researcher=f"{round(duration, 1)}s provides adequate data for spectral and ICA analysis.",
                clinician="Recording length is sufficient for reliable analysis.",
                general="The recording is long enough to give reliable results.",
                action=None
            ))

        score = max(0, score)

        if score >= self.g["good"]:
            grade = "Good"
        elif score >= self.g["acceptable"]:
            grade = "Acceptable"
        elif score >= self.g["poor"]:
            grade = "Poor — review recommended"
        else:
            grade = "Problematic — consider re-recording"

        return {
            "score":    score,
            "grade":    grade,
            "findings": findings,
            "flags":    flags
        }

    # -----------------------------------------------------------------------
    # Noise assessment
    # -----------------------------------------------------------------------

    def _assess_noise(self) -> Dict[str, Any]:
        findings = []
        auto     = self.report.get("ica_auto_detection", {})
        ica      = self.report.get("ica", {})

        if not ica.get("fitted"):
            return {
                "findings": [],
                "note": "ICA was not fitted. Run ICA to enable noise component analysis."
            }

        excluded = ica.get("excluded_components", [])

        # EOG
        eog_indices = auto.get("eog_indices", [])
        if eog_indices:
            profile = NOISE_PROFILES["eog"]
            findings.append({
                "type":              "eog",
                "name":              profile["name"],
                "components":        eog_indices,
                "status":            "excluded" if any(c in excluded for c in eog_indices) else "detected — not yet excluded",
                "severity":          profile["severity"],
                "explanation":       profile[self.audience],
                "visual_signature":  profile["visual_signature"],
                "action":            profile["action"],
                "scores":            auto.get("eog_scores", [])
            })
        
        if auto.get("eog_error"):
            findings.append({
                "type":        "eog",
                "name":        "Eye Movement Detection",
                "status":      "could not auto-detect",
                "severity":    "info",
                "explanation": (
                    "Auto-detection requires a dedicated EOG channel (e.g. EOG061). "
                    "If none is present in this recording, inspect component topographies manually."
                    if self.audience == "researcher" else
                    "Eye movement artifacts could not be detected automatically. Manual inspection is needed."
                ),
                "action": "Inspect ICA component topographies manually for frontal distributions.",
                "components": []
            })

        # ECG
        ecg_indices = auto.get("ecg_indices", [])
        if ecg_indices:
            profile = NOISE_PROFILES["ecg"]
            findings.append({
                "type":             "ecg",
                "name":             profile["name"],
                "components":       ecg_indices,
                "status":           "excluded" if any(c in excluded for c in ecg_indices) else "detected — not yet excluded",
                "severity":         profile["severity"],
                "explanation":      profile[self.audience],
                "visual_signature": profile["visual_signature"],
                "action":           profile["action"],
                "scores":           auto.get("ecg_scores", [])
            })

        if auto.get("ecg_error"):
            findings.append({
                "type":        "ecg",
                "name":        "Heartbeat Detection",
                "status":      "could not auto-detect",
                "severity":    "info",
                "explanation": (
                    "ECG auto-detection requires a cardiac channel or clear QRS morphology in EEG sources. "
                    "Inspect sources manually for regular ~1Hz pulses."
                    if self.audience == "researcher" else
                    "Heartbeat artifacts could not be detected automatically. Manual inspection is needed."
                ),
                "action": "Inspect ICA sources manually for a regular heartbeat-like pattern.",
                "components": []
            })

        # If ICA was applied, report what was removed
        if ica.get("applied") and excluded:
            findings.append({
                "type":        "applied",
                "name":        "ICA Exclusion Applied",
                "components":  excluded,
                "status":      "removed",
                "severity":    "good",
                "explanation": (
                    f"Components {excluded} were excluded and the ICA was applied to the data. "
                    f"The cleaned signal was reconstructed from the remaining {ica.get('n_components_fitted', '?') - len(excluded)} components."
                    if self.audience == "researcher" else
                    f"{len(excluded)} noise component(s) were identified and removed from the signal."
                ),
                "action": None,
                "components": excluded
            })

        if not findings:
            findings.append({
                "type":        "none",
                "name":        "No artifacts auto-detected",
                "status":      "clean",
                "severity":    "good",
                "explanation": (
                    "No EOG or ECG components were automatically identified. "
                    "This may mean the recording is clean, or that dedicated artifact channels were absent. "
                    "Manual inspection of ICA topographies is still recommended."
                    if self.audience == "researcher" else
                    "No major noise sources were automatically detected. Always confirm with manual inspection."
                ),
                "action": "Manually inspect ICA component topographies and sources to confirm.",
                "components": []
            })

        return {"findings": findings}

    # -----------------------------------------------------------------------
    # Pipeline summary
    # -----------------------------------------------------------------------

    def _summarise_pipeline(self) -> Dict[str, Any]:
        steps    = self.report.get("steps", [])
        ica      = self.report.get("ica", {})
        interp   = self.report.get("interpolation", {})
        bads     = self.report.get("bad_channels", [])

        return {
            "steps":           steps,
            "n_steps":         len(steps),
            "filter_applied":  any("bandpass" in s.lower() for s in steps),
            "notch_applied":   any("notch" in s.lower() for s in steps),
            "reference":       "average" if any("average" in s.lower() for s in steps) else "unknown",
            "interpolated":    interp.get("performed", False),
            "n_interpolated":  interp.get("n_interpolated", 0),
            "ica_fitted":      ica.get("fitted", False),
            "ica_applied":     ica.get("applied", False),
            "n_excluded":      len(ica.get("excluded_components", [])),
            "bad_channels":    bads
        }

    # -----------------------------------------------------------------------
    # Auto-generated methods paragraph
    # -----------------------------------------------------------------------

    def _generate_methods_text(self) -> str:
        r       = self.report
        ica     = r.get("ica", {})
        interp  = r.get("interpolation", {})
        bads    = r.get("bad_channels", [])
        steps   = r.get("steps", [])

        # Extract filter params from steps
        l_freq, h_freq, notch = 1.0, 40.0, 50.0
        for s in steps:
            if "bandpass" in s.lower():
                try:
                    parts = s.split(":")[1].strip().replace("Hz","").strip().split("-")
                    l_freq = float(parts[0])
                    h_freq = float(parts[1])
                except Exception:
                    pass
            if "notch" in s.lower():
                try:
                    notch = float(s.split("at")[1].strip().replace("Hz","").strip())
                except Exception:
                    pass

        lines = []
        lines.append(
            f"Continuous EEG data were bandpass filtered ({l_freq}–{h_freq} Hz, "
            f"FIR windowed sinc, zero-phase) and notch filtered at {notch} Hz to remove "
            f"power line interference."
        )
        lines.append(
            "Data were re-referenced to the common average."
        )

        if bads:
            method = "spherical spline interpolation" if interp.get("performed") else "flagged but not interpolated"
            lines.append(
                f"Bad channels ({', '.join(bads)}) were identified using variance z-score "
                f"and flat signal criteria, and addressed via {method}."
            )
        else:
            lines.append("No bad channels were identified.")

        if ica.get("fitted"):
            n_comp   = ica.get("n_components_requested", 20)
            excluded = ica.get("excluded_components", [])
            lines.append(
                f"Independent Component Analysis (FastICA, {n_comp} components) was applied "
                f"to identify and remove artifactual sources."
            )
            if excluded:
                lines.append(
                    f"Components {excluded} were rejected based on topographic and "
                    f"temporal characteristics consistent with ocular and/or cardiac artifact."
                )
            else:
                lines.append(
                    "No components were rejected following manual inspection."
                )

        return " ".join(lines)

    # -----------------------------------------------------------------------
    # Plain language summary
    # -----------------------------------------------------------------------

    def _generate_summary(self, quality: Dict, noise: Dict) -> str:
        grade    = quality["grade"]
        score    = quality["score"]
        n_issues = len([f for f in quality["findings"] if f["severity"] in ("high", "medium")])

        if score >= 80:
            opening = "The recording is in good shape for analysis."
        elif score >= 60:
            opening = "The recording is usable but has some issues worth noting."
        else:
            opening = "The recording has significant quality concerns that should be reviewed before proceeding."

        noise_findings = [f for f in noise.get("findings", []) if f.get("severity") in ("high", "medium")]
        if noise_findings:
            noise_summary = f"{len(noise_findings)} noise source(s) were detected in the ICA components."
        else:
            noise_summary = "No major noise sources were automatically detected in the ICA components."

        return f"{opening} Quality score: {score}/100 ({grade}). {noise_summary}"

    # -----------------------------------------------------------------------
    # Helper
    # -----------------------------------------------------------------------

    def _finding(
        self,
        severity: str,
        title: str,
        researcher: str,
        clinician: str,
        general: str,
        action: str
    ) -> Dict[str, Any]:
        explanations = {
            "researcher": researcher,
            "clinician":  clinician,
            "general":    general
        }
        return {
            "severity":    severity,
            "title":       title,
            "explanation": explanations[self.audience],
            "all_explanations": explanations,
            "action":      action
        }