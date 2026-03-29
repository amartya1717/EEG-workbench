"""
Hypothesis Reporter
-------------------
Takes a user-stated hypothesis, structured condition/window/channel
selections, and computed epoch results — produces a structured
interpretive report that links findings back to the hypothesis.

No GUI code here. Pure logic and text generation.

Usage:
    reporter = HypothesisReporter(
        hypothesis   = "Picture stimuli will produce enhanced P300",
        condition_a  = "animal_target",
        condition_b  = "animal_distractor",
        time_window  = (250, 500),
        erp_a        = analyzer.get_erp("animal_target"),
        erp_b        = analyzer.get_erp("animal_distractor"),
        band_power_a = analyzer.get_band_power("animal_target"),
        band_power_b = analyzer.get_band_power("animal_distractor"),
        trial_quality= analyzer.get_trial_quality(),
        audience     = "researcher"
    )
    result = reporter.generate()
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Band descriptions — used in plain language summaries
# ---------------------------------------------------------------------------

BAND_ROLES = {
    "Delta (1-4 Hz)":   "slow wave activity, often associated with deep processing or artifact",
    "Theta (4-8 Hz)":   "memory encoding and frontal cognitive load",
    "Alpha (8-12 Hz)":  "cortical inhibition — suppression indicates active processing",
    "Beta (12-30 Hz)":  "active cognitive processing and motor preparation",
    "Gamma (30-45 Hz)": "high-level feature binding and conscious perception",
}

# ERP components and what they index
COMPONENT_ROLES = {
    "P1":   "early visual processing — reflects initial cortical response to visual input",
    "N1":   "attention-dependent sensory processing",
    "N170": "structural encoding of objects and faces",
    "P2":   "early categorisation and stimulus evaluation",
    "N2":   "conflict detection and mismatch processing",
    "P300": "attention, context updating, and working memory engagement",
    "N400": "semantic processing and expectation violation",
}


class HypothesisReporter:

    def __init__(
        self,
        hypothesis:    str,
        condition_a:   str,
        condition_b:   str,
        time_window:   Tuple[float, float],   # ms
        erp_a:         Dict[str, Any],
        erp_b:         Dict[str, Any],
        band_power_a:  Dict[str, float],
        band_power_b:  Dict[str, float],
        trial_quality: Dict[str, Any],
        audience:      str = "researcher"
    ):
        self.hypothesis    = hypothesis.strip()
        self.cond_a        = condition_a
        self.cond_b        = condition_b
        self.time_window   = time_window      # (t_lo_ms, t_hi_ms)
        self.erp_a         = erp_a
        self.erp_b         = erp_b
        self.bp_a          = band_power_a
        self.bp_b          = band_power_b
        self.trial_quality = trial_quality
        self.audience      = audience

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def generate(self) -> Dict[str, Any]:
        trial_summary  = self._assess_trial_quality()
        erp_findings   = self._assess_erp()
        band_findings  = self._assess_band_power()
        verdict        = self._generate_verdict(erp_findings, band_findings)
        limits         = self._generate_limits()
        next_steps     = self._generate_next_steps(erp_findings, band_findings)
        methods        = self._generate_epoch_methods()

        return {
            "hypothesis":    self.hypothesis,
            "condition_a":   self.cond_a,
            "condition_b":   self.cond_b,
            "time_window":   self.time_window,
            "trial_summary": trial_summary,
            "erp_findings":  erp_findings,
            "band_findings": band_findings,
            "verdict":       verdict,
            "limits":        limits,
            "next_steps":    next_steps,
            "methods":       methods,
            "audience":      self.audience
        }

    # -----------------------------------------------------------------------
    # Trial quality summary
    # -----------------------------------------------------------------------

    def _assess_trial_quality(self) -> Dict[str, Any]:
        tq       = self.trial_quality
        n        = tq.get("n_trials", 0)
        flags    = tq.get("flags", [])
        n_clean  = flags.count("clean")
        n_reject = flags.count("rejected")
        pct_kept = round(n_clean / n * 100, 1) if n else 0

        if pct_kept >= 90:
            quality = "good"
            note    = "Trial retention is high — the average is based on reliable data."
        elif pct_kept >= 75:
            quality = "acceptable"
            note    = "Trial retention is acceptable but worth noting in your methods."
        else:
            quality = "poor"
            note    = "More than 25% of trials were rejected. Interpret averages with caution."

        return {
            "n_total":   n,
            "n_clean":   n_clean,
            "n_rejected": n_reject,
            "pct_kept":  pct_kept,
            "quality":   quality,
            "note":      note
        }

    # -----------------------------------------------------------------------
    # ERP findings
    # -----------------------------------------------------------------------

    def _assess_erp(self) -> Dict[str, Any]:
        t_lo, t_hi = self.time_window
        times_a = np.array(self.erp_a["times"]) * 1000
        times_b = np.array(self.erp_b["times"]) * 1000
        mean_a  = np.array(self.erp_a["mean_uv"])
        mean_b  = np.array(self.erp_b["mean_uv"])

        # Extract window
        mask_a  = (times_a >= t_lo) & (times_a <= t_hi)
        mask_b  = (times_b >= t_lo) & (times_b <= t_hi)

        win_a   = mean_a[mask_a] if np.any(mask_a) else np.array([0.0])
        win_b   = mean_b[mask_b] if np.any(mask_b) else np.array([0.0])

        peak_a  = float(win_a[np.argmax(np.abs(win_a))]) if len(win_a) else 0.0
        peak_b  = float(win_b[np.argmax(np.abs(win_b))]) if len(win_b) else 0.0
        diff    = peak_a - peak_b

        # Detected components in window
        def components_in_window(erp):
            return [
                p for p in erp.get("peaks", [])
                if t_lo <= p["latency_ms"] <= t_hi
            ]

        comps_a = components_in_window(self.erp_a)
        comps_b = components_in_window(self.erp_b)

        # Direction assessment
        if abs(diff) < 0.1:
            direction = "no difference"
            direction_text = self._phrase(
                r=f"No meaningful amplitude difference between {self.cond_a} and {self.cond_b} "
                  f"in the {t_lo}–{t_hi}ms window (Δ={round(diff,3)}µV).",
                c=f"No clear difference between the two conditions in the selected time window.",
                g=f"The brain's response was similar for both conditions in the time period you selected."
            )
        elif diff > 0:
            direction = "A > B"
            direction_text = self._phrase(
                r=f"{self.cond_a} showed larger amplitude than {self.cond_b} "
                  f"in the {t_lo}–{t_hi}ms window "
                  f"(A peak: {round(peak_a,3)}µV, B peak: {round(peak_b,3)}µV, Δ={round(diff,3)}µV).",
                c=f"Condition {self.cond_a} produced a stronger brain response than {self.cond_b} "
                  f"in the selected time window.",
                g=f"The brain responded more strongly to {self.cond_a} than to {self.cond_b} "
                  f"during the time period you were interested in."
            )
        else:
            direction = "B > A"
            direction_text = self._phrase(
                r=f"{self.cond_b} showed larger amplitude than {self.cond_a} "
                  f"in the {t_lo}–{t_hi}ms window "
                  f"(A peak: {round(peak_a,3)}µV, B peak: {round(peak_b,3)}µV, Δ={round(diff,3)}µV).",
                c=f"Condition {self.cond_b} produced a stronger brain response than {self.cond_a} "
                  f"in the selected time window.",
                g=f"The brain responded more strongly to {self.cond_b} than to {self.cond_a} "
                  f"during the time period you were interested in."
            )

        # Component narratives
        comp_notes = []
        for p in comps_a:
            name = p["name"]
            role = COMPONENT_ROLES.get(name, "")
            amp  = round(p["amplitude_uv"], 3)
            lat  = round(p["latency_ms"], 1)
            comp_notes.append(self._phrase(
                r=f"{name} detected at {lat}ms ({amp}µV) in {self.cond_a}. "
                  f"This component reflects {role}.",
                c=f"A {name} component was present at {lat}ms in {self.cond_a}, "
                  f"suggesting {role}.",
                g=f"At {lat}ms after the stimulus, a brain signal called {name} was detected "
                  f"in {self.cond_a}. This is associated with {role}."
            ))

        for p in comps_b:
            name = p["name"]
            amp  = round(p["amplitude_uv"], 3)
            lat  = round(p["latency_ms"], 1)
            # Only note if not already noted for A
            if not any(ca["name"] == name for ca in comps_a):
                comp_notes.append(self._phrase(
                    r=f"{name} detected at {lat}ms ({amp}µV) in {self.cond_b} but not {self.cond_a}.",
                    c=f"A {name} component appeared in {self.cond_b} but was absent in {self.cond_a}.",
                    g=f"The {name} brain signal was present for {self.cond_b} but not for {self.cond_a}."
                ))

        return {
            "direction":       direction,
            "direction_text":  direction_text,
            "peak_a":          peak_a,
            "peak_b":          peak_b,
            "difference":      diff,
            "components_a":    comps_a,
            "components_b":    comps_b,
            "component_notes": comp_notes,
            "window_ms":       self.time_window
        }

    # -----------------------------------------------------------------------
    # Band power findings
    # -----------------------------------------------------------------------

    def _assess_band_power(self) -> Dict[str, Any]:
        findings = []

        for band, role in BAND_ROLES.items():
            pa = self.bp_a.get(band, 0)
            pb = self.bp_b.get(band, 0)

            if pa == 0 and pb == 0:
                continue

            # Percent difference relative to B
            if pb > 0:
                pct_diff = ((pa - pb) / pb) * 100
            else:
                pct_diff = 0

            if abs(pct_diff) < 10:
                direction = "similar"
                severity  = "info"
            elif pct_diff > 0:
                direction = f"A higher by {round(abs(pct_diff))}%"
                severity  = "medium"
            else:
                direction = f"B higher by {round(abs(pct_diff))}%"
                severity  = "medium"

            findings.append({
                "band":      band,
                "role":      role,
                "power_a":   pa,
                "power_b":   pb,
                "pct_diff":  pct_diff,
                "direction": direction,
                "severity":  severity,
                "note":      self._phrase(
                    r=f"{band}: {self.cond_a}={pa:.2e}, {self.cond_b}={pb:.2e} "
                      f"({direction}). {band} reflects {role}.",
                    c=f"{band} power was {direction} between the two conditions. "
                      f"This band is associated with {role}.",
                    g=f"In the {band} frequency range, the two conditions showed {direction}. "
                      f"This frequency is linked to {role}."
                )
            })

        return {"findings": findings}

    # -----------------------------------------------------------------------
    # Verdict
    # -----------------------------------------------------------------------

    def _generate_verdict(
        self,
        erp: Dict[str, Any],
        band: Dict[str, Any]
    ) -> Dict[str, Any]:

        direction = erp["direction"]
        diff      = erp["difference"]
        comps_a   = erp["components_a"]

        # Simple rules for verdict
        has_effect    = abs(diff) >= 0.1
        has_component = len(comps_a) > 0

        if has_effect and has_component:
            verdict   = "Consistent with hypothesis"
            colour    = "good"
            rationale = self._phrase(
                r=(
                    f"The data shows a measurable amplitude difference between conditions "
                    f"in the specified window ({round(diff,3)}µV), and at least one ERP component "
                    f"was detected in {self.cond_a}. This pattern is consistent with a condition-specific "
                    f"neural response."
                ),
                c=(
                    f"The brain responses differ between the two conditions in the time window "
                    f"you specified, and a recognised brain signal component was detected. "
                    f"This supports your hypothesis."
                ),
                g=(
                    f"The brain data shows a real difference between the two conditions "
                    f"at the time you predicted, and a known brain signal was detected. "
                    f"This is consistent with what your hypothesis predicted."
                )
            )
        elif has_effect and not has_component:
            verdict   = "Partial — effect present, no named component"
            colour    = "medium"
            rationale = self._phrase(
                r=(
                    f"An amplitude difference of {round(diff,3)}µV was observed in the window, "
                    f"but no standard ERP component was detected at the expected latency. "
                    f"The effect may be real but does not map onto a known component."
                ),
                c=(
                    f"There is a difference between conditions but it doesn't match "
                    f"a recognised brain signal pattern."
                ),
                g=(
                    f"The brain did respond differently to the two conditions, "
                    f"but the response doesn't match a well-known brain signal pattern. "
                    f"It could still be meaningful — it just needs more investigation."
                )
            )
        elif not has_effect:
            verdict   = "Inconclusive — no measurable difference"
            colour    = "medium"
            rationale = self._phrase(
                r=(
                    f"No meaningful amplitude difference was found between {self.cond_a} "
                    f"and {self.cond_b} in the {self.time_window[0]}–{self.time_window[1]}ms window. "
                    f"This may reflect a null effect, insufficient power, or a mismatch between "
                    f"the selected window and the actual response latency."
                ),
                c=(
                    f"No clear difference was detected between the two conditions "
                    f"in the selected time window."
                ),
                g=(
                    f"The brain responded similarly to both conditions in the time period "
                    f"you selected. This doesn't necessarily mean there is no effect — "
                    f"the time window or conditions may need adjusting."
                )
            )
        else:
            verdict   = "Inconclusive"
            colour    = "medium"
            rationale = self._phrase(
                r="Results are mixed and do not clearly support or contradict the hypothesis.",
                c="The results are unclear.",
                g="The data doesn't give a clear answer either way."
            )

        return {
            "verdict":   verdict,
            "colour":    colour,
            "rationale": rationale
        }

    # -----------------------------------------------------------------------
    # Limits
    # -----------------------------------------------------------------------

    def _generate_limits(self) -> List[str]:
        limits = []

        limits.append(self._phrase(
            r=(
                "Statistical significance was not assessed. "
                "The amplitude differences reported are descriptive — "
                "permutation testing or mixed-effects modelling would be needed "
                "to make inferential claims."
            ),
            c=(
                "These results have not been tested for statistical significance. "
                "Further analysis is needed before drawing clinical conclusions."
            ),
            g=(
                "We haven't run any statistical tests yet — the numbers shown are "
                "descriptions of the data, not proof that the effect is real."
            )
        ))

        limits.append(self._phrase(
            r=(
                f"The analysis is limited to the {self.time_window[0]}–{self.time_window[1]}ms window. "
                f"Effects outside this range are not captured in this report."
            ),
            c=(
                "Only the selected time window was examined. "
                "Effects at other latencies are not included."
            ),
            g=(
                "We only looked at a specific slice of time after the stimulus. "
                "Something interesting might be happening at other time points."
            )
        ))

        limits.append(self._phrase(
            r=(
                "Single-channel or averaged ERP analysis cannot localise the neural source. "
                "Source analysis (e.g. LORETA, beamforming) would be required for spatial claims."
            ),
            c=(
                "The brain region generating these signals cannot be identified "
                "from scalp EEG alone."
            ),
            g=(
                "EEG tells us when something happens in the brain but not exactly where. "
                "The signals we see are a mix from many brain areas."
            )
        ))

        limits.append(self._phrase(
            r=(
                "Condition differences in trial count may affect the reliability of the average. "
                f"{self.cond_a} had {self.erp_a.get('n_trials',0)} trials, "
                f"{self.cond_b} had {self.erp_b.get('n_trials',0)} trials."
            ),
            c=(
                "The two conditions had different numbers of trials, "
                "which may affect the reliability of the comparison."
            ),
            g=(
                "One condition had more trials than the other. "
                "More trials generally means a more reliable average."
            )
        ))

        return limits

    # -----------------------------------------------------------------------
    # Next steps
    # -----------------------------------------------------------------------

    def _generate_next_steps(
        self,
        erp: Dict[str, Any],
        band: Dict[str, Any]
    ) -> List[str]:
        steps = []
        direction = erp["direction"]
        comps_a   = erp["components_a"]

        if direction != "no difference":
            steps.append(self._phrase(
                r="Run permutation testing or paired t-tests on mean amplitude in the window of interest to assess statistical reliability.",
                c="Statistical testing should be performed to confirm these findings.",
                g="The next step is running a statistical test to check if this difference is real or just chance."
            ))

        if not comps_a:
            steps.append(self._phrase(
                r=f"Adjust the time window — the expected component may peak outside {self.time_window[0]}–{self.time_window[1]}ms for this paradigm.",
                c="Consider widening the time window to capture the full response.",
                g="Try looking at a wider time range — the brain response might be happening slightly earlier or later than expected."
            ))

        steps.append(self._phrase(
            r="Examine topographic distribution to identify which scalp regions are driving the effect.",
            c="Review the electrode distribution of the response to identify the most relevant brain regions.",
            g="Look at which sensors are showing the strongest response — this can help understand which part of the brain is involved."
        ))

        # If alpha suppression might be relevant
        alpha_a = self.bp_a.get("Alpha (8-12 Hz)", 0)
        alpha_b = self.bp_b.get("Alpha (8-12 Hz)", 0)
        if alpha_a != alpha_b and abs(alpha_a - alpha_b) / max(alpha_b, 1e-10) > 0.1:
            steps.append(self._phrase(
                r="Alpha power difference detected — consider time-frequency analysis (e.g. Morlet wavelet) for a more precise characterisation of oscillatory dynamics.",
                c="A difference in alpha brain waves was found — time-frequency analysis could give more detail.",
                g="The brain's alpha rhythm was different between conditions — this is worth exploring further."
            ))

        steps.append(self._phrase(
            r="Replicate with additional participants to establish reliability across individuals.",
            c="These findings should be confirmed with a larger sample.",
            g="This analysis is from one recording. Testing more people would confirm whether this is a consistent finding."
        ))

        return steps

    # -----------------------------------------------------------------------
    # Epoch methods paragraph
    # -----------------------------------------------------------------------

    def _generate_epoch_methods(self) -> str:
        tq     = self.trial_quality
        n      = tq.get("n_trials", 0)
        n_kept = tq.get("n_clean", n)
        thresh = tq.get("threshold_uv", 100)

        return (
            f"Epochs were created time-locked to stimulus onset for conditions "
            f"{self.cond_a} and {self.cond_b}, spanning "
            f"{self.time_window[0]}–{self.time_window[1]} ms post-stimulus. "
            f"Trials exceeding ±{thresh} µV peak-to-peak on any EEG channel were rejected "
            f"({n - n_kept} of {n} trials removed). "
            f"Baseline correction was applied using the pre-stimulus period. "
            f"ERPs were computed by averaging across retained trials per condition."
        )

    # -----------------------------------------------------------------------
    # Audience-aware phrase helper
    # -----------------------------------------------------------------------

    def _phrase(self, r: str, c: str, g: str) -> str:
        return {"researcher": r, "clinician": c, "general": g}.get(
            self.audience, r
        )
