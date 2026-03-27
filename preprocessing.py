import os
import json
from typing import Optional, Dict, Any, List

import numpy as np
import mne
from mne.preprocessing import ICA
from history_manager import HistoryManager


class EEGPreprocessor:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.raw: Optional[mne.io.BaseRaw] = None
        self.cleaned_raw: Optional[mne.io.BaseRaw] = None
        self.ica: Optional[ICA] = None
        self.history = HistoryManager()
        self.report: Dict[str, Any] = {
            "file_path": file_path,
            "loaded": False,
            "n_channels": None,
            "sfreq": None,
            "duration_sec": None,
            "bad_channels": [],
            "steps": [],
            "notes": [],
        }

    def load_set_file(self) -> None:
        """Load EEGLAB .set file."""
        self.raw = mne.io.read_raw_eeglab(self.file_path, preload=True)
        self.cleaned_raw = self.raw.copy()

        self.report["loaded"] = True
        self.report["n_channels"] = len(self.raw.ch_names)
        self.report["sfreq"] = float(self.raw.info["sfreq"])
        self.report["duration_sec"] = float(self.raw.times[-1])

        self.report["steps"].append("Loaded EEGLAB .set file")

        self.history.set_original_raw(self.raw)
        self.history.add_snapshot(
            label="Original Import",
            raw=self.raw,
            step_type="import",
            details={"file_path": self.file_path}
        )

    def get_basic_info(self) -> Dict[str, Any]:
        """Return file and signal summary."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded. Call load_set_file() first.")

        return {
            "file_name": os.path.basename(self.file_path),
            "n_channels": len(self.cleaned_raw.ch_names),
            "channel_names": self.cleaned_raw.ch_names,
            "sampling_rate_hz": self.cleaned_raw.info["sfreq"],
            "duration_sec": self.cleaned_raw.times[-1],
        }

    def apply_bandpass_filter(self, l_freq: float = 1.0, h_freq: float = 40.0) -> None:
        """Apply bandpass filter."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        self.cleaned_raw.filter(l_freq=l_freq, h_freq=h_freq, fir_design="firwin")
        self.report["steps"].append(f"Applied bandpass filter: {l_freq}-{h_freq} Hz")

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label=f"Bandpass {l_freq}-{h_freq} Hz",
            raw=self.cleaned_raw,
            step_type="filter",
            details={"l_freq": l_freq, "h_freq": h_freq})

    def apply_notch_filter(self, freqs=50.0) -> None:
        """Apply notch filter for line noise."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        self.cleaned_raw.notch_filter(freqs=freqs)
        self.report["steps"].append(f"Applied notch filter at {freqs} Hz")

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label=f"Notch Filter {freqs} Hz",
            raw=self.cleaned_raw,
            step_type="filter",
            details={"notch_freq": freqs}
        )

    def set_average_reference(self) -> None:
        """Set average EEG reference."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        self.cleaned_raw.set_eeg_reference("average", projection=False)
        self.report["steps"].append("Set EEG average reference")

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label="Average Reference",
            raw=self.cleaned_raw,
            step_type="reference",
            details={"reference": "average"}
        )

    def detect_bad_channels(
        self,
        variance_z_threshold: float = 3.0,
        flat_std_threshold: float = 1e-7
    ) -> List[str]:
        """
        Detect potentially bad channels using simple statistics:
        - very low std => flat channel
        - extreme variance z-score => unusually noisy channel
        """
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        data = self.cleaned_raw.get_data(picks="eeg")
        ch_names = self.cleaned_raw.copy().pick("eeg").ch_names

        channel_std = np.std(data, axis=1)
        channel_var = np.var(data, axis=1)

        var_mean = np.mean(channel_var)
        var_std = np.std(channel_var)

        if var_std == 0:
            var_z = np.zeros_like(channel_var)
        else:
            var_z = (channel_var - var_mean) / var_std

        bads = []
        for i, ch in enumerate(ch_names):
            if channel_std[i] < flat_std_threshold:
                bads.append(ch)
            elif abs(var_z[i]) > variance_z_threshold:
                bads.append(ch)

        bads = sorted(list(set(bads)))
        self.cleaned_raw.info["bads"] = bads
        self.report["bad_channels"] = bads
        self.report["steps"].append(
            f"Detected bad channels using std/variance rules: {', '.join(bads) if bads else 'None'}"
        )

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label=f"Bad Channel Detection ({len(bads)} found)",
            raw=self.cleaned_raw,
            step_type="bad_channels",
            details={
                "bad_channels": bads,
                "variance_z_threshold": variance_z_threshold,
                "flat_std_threshold": flat_std_threshold
            }
        )

        return bads

    def interpolate_bad_channels(self,bads) -> None:
        """Interpolate detected bad channels."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        if not self.cleaned_raw.info["bads"]:
            self.report["notes"].append("No bad channels to interpolate.")
            return

        self.interpolate = self.cleaned_raw.interpolate_bads(reset_bads=False)
        self.report["interpolation"] = {
            "performed": True,
            "channels": bads,
            "n_interpolated": len(bads)
        }
        self.report["steps"].append("Interpolated bad channels")

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label=f"Interpolated Bad Channels ({len(bads)})",
            raw=self.cleaned_raw,
            step_type="interpolation",
            details={"interpolated_channels": bads, "n_interpolated": len(bads)}
        )

    def fit_ica(
        self,
        n_components: Optional[int] = 20,
        random_state: int = 97,
        max_iter: str = "auto"
    ) -> None:
        """
        Fit ICA for later artifact inspection/removal.
        This does NOT automatically remove components.
        """
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        ica = ICA(
            n_components=n_components,
            random_state=random_state,
            max_iter=max_iter,
            method="fastica"
        )
        ica.fit(self.cleaned_raw)
        self.ica = ica

        self.history.set_pre_ica_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label="Baseline (pre-ICA)",
            raw=self.cleaned_raw.copy(),
            step_type="ica_version",
            details={"excluded_components": []}
        )   

        self.report["ica"] = {
        "fitted": True,
        "n_components_requested": n_components,
        "n_components_fitted": getattr(ica, "n_components_", None),
        "excluded_components": [],
        "applied": False
        }

        self.report["steps"].append(f"Fitted ICA with n_components={n_components}")

        # ica.plot_components()
        # ica.plot_sources(self.raw)

    def apply_ica_exclusion(self, exclude_components) -> None:
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")
        if self.ica is None:
            raise ValueError("ICA has not been fitted yet.")

        baseline = self.history.pre_ica_raw
        if baseline is None:
            raise ValueError("Pre-ICA baseline not found.")

        picks = [int(x.strip()) for x in exclude_components.split(",") if x.strip()]

        ica_copy = self.ica.copy()
        ica_copy.exclude = picks

        new_raw = baseline.copy()
        ica_copy.apply(new_raw)
        self.cleaned_raw = new_raw

        baseline_data = baseline.get_data()
        cleaned_data = self.cleaned_raw.get_data()

        print("Applied picks:", picks)
        print("Inside apply_ica_exclusion -> all equal:", np.array_equal(baseline_data, cleaned_data))
        print("Inside apply_ica_exclusion -> max abs diff:", np.max(np.abs(baseline_data - cleaned_data)))

        if "ica" not in self.report:
            self.report["ica"] = {}

        self.report["ica"]["excluded_components"] = picks
        self.report["ica"]["applied"] = True

        self.report["steps"].append(
            f"Applied ICA exclusion for components: {picks}"
        )

        self.history.add_snapshot(
            label=f"ICA Exclusion {picks}",
            raw=self.cleaned_raw,
            step_type="ica",
            details={"excluded_components": picks}
        )

    def compute_artifact_metrics(self) -> Dict[str, Any]:
        """Compute simple artifact/quality metrics."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        data = self.cleaned_raw.get_data(picks="eeg")

        peak_to_peak = np.ptp(data, axis=1)
        rms = np.sqrt(np.mean(data ** 2, axis=1))
        std = np.std(data, axis=1)

        metrics = {
            "mean_peak_to_peak": float(np.mean(peak_to_peak)),
            "max_peak_to_peak": float(np.max(peak_to_peak)),
            "mean_rms": float(np.mean(rms)),
            "max_rms": float(np.max(rms)),
            "mean_std": float(np.mean(std)),
            "max_std": float(np.max(std)),
        }

        self.report["artifact_metrics"] = metrics
        self.report["steps"].append("Computed artifact/quality metrics")

        self.history.update_current_raw(self.cleaned_raw)
        self.history.add_snapshot(
            label="Artifact Metrics Computed",
            raw=self.cleaned_raw,
            step_type="metrics",
            details=metrics
        )

        return metrics

    def auto_detect_ica_artifacts(self):
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")
        if self.ica is None:
            raise ValueError("ICA has not been fitted yet.")

        suggested = {
            "eog_indices": [],
            "ecg_indices": [],
            "suggested_exclude": []
        }

        # Try EOG detection
        try:
            eog_indices, eog_scores = self.ica.find_bads_eog(self.cleaned_raw)
            suggested["eog_indices"] = eog_indices
            suggested["eog_scores"] = [float(x) for x in eog_scores]
        except Exception as e:
            suggested["eog_error"] = str(e)

        # Try ECG detection
        try:
            ecg_indices, ecg_scores = self.ica.find_bads_ecg(self.cleaned_raw)
            suggested["ecg_indices"] = ecg_indices
            suggested["ecg_scores"] = [float(x) for x in ecg_scores]
        except Exception as e:
            suggested["ecg_error"] = str(e)

        combined = sorted(set(suggested["eog_indices"] + suggested["ecg_indices"]))
        suggested["suggested_exclude"] = combined

        self.report["ica_auto_detection"] = suggested
        self.report["steps"].append(
            f"Auto-detected ICA artifact components: {combined if combined else 'None'}"
        )

        return suggested

    def extract_eeg_channels(self,raw):
        eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        channel_names = [raw.ch_names[i] for i in eeg_picks]
        return channel_names
        pass


    def save_cleaned_file(self, output_path: str) -> None:
        """Save cleaned data as FIF."""
        if self.cleaned_raw is None:
            raise ValueError("No EEG data loaded.")

        self.cleaned_raw.save(output_path, overwrite=True)
        self.report["steps"].append(f"Saved cleaned file to {output_path}")

    def save_report_json(self, output_path: str) -> None:
        """Save preprocessing report as JSON."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2)

    def run_basic_pipeline(
        self,
        l_freq: float = 1.0,
        h_freq: float = 40.0,
        notch_freq: float = 50.0,
        interpolate_bads: bool = False,
        fit_ica: bool = False,
        run_bandpass: bool = True,
        run_notch: bool = True,
        run_reference: bool = True,
        run_bad_channels: bool = True,
        run_metrics: bool = True,
        n_ica_components: int = 20,
        progress_callback=None,
    ) -> Dict[str, Any]:
        """Convenience pipeline. Each step can be toggled via run_* flags.
        progress_callback(msg: str) is called before each step if provided."""

        def _progress(msg):
            if progress_callback:
                progress_callback(msg)

        _progress("Loading file...")
        self.load_set_file()

        if run_bandpass:
            _progress(f"Bandpass filter {l_freq}–{h_freq} Hz...")
            self.apply_bandpass_filter(l_freq=l_freq, h_freq=h_freq)

        if run_notch:
            _progress(f"Notch filter {notch_freq} Hz...")
            self.apply_notch_filter(freqs=notch_freq)

        if run_reference:
            _progress("Setting average reference...")
            self.set_average_reference()

        bads = []
        if run_bad_channels:
            _progress("Detecting bad channels...")
            bads = self.detect_bad_channels()

        if interpolate_bads and bads:
            _progress(f"Interpolating {len(bads)} bad channel(s)...")
            self.interpolate_bad_channels(bads)

        if fit_ica:
            _progress(f"Fitting ICA ({n_ica_components} components)...")
            self.fit_ica(n_components=n_ica_components)

        if run_metrics:
            _progress("Computing artifact metrics...")
            self.compute_artifact_metrics()

        return self.report