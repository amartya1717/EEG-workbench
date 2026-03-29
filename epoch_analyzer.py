"""
Epoch Analyzer
--------------
Handles all epoch-related processing on top of a cleaned MNE Raw object.
No GUI code here — pure signal processing and data structuring.

Pipeline:
    1. read_events()          → find what events exist in the recording
    2. create_epochs()        → epoch around a chosen event, define window + baseline
    3. get_trial_quality()    → per-trial amplitude QC, rejection flags
    4. get_erp()              → trial-averaged ERP per condition
    5. get_band_power()       → per-band power comparison across conditions
    6. get_epoch_report()     → structured summary for the interpretation layer
"""

from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import mne


class EpochAnalyzer:

    def __init__(self, raw: mne.io.BaseRaw):
        """
        Parameters
        ----------
        raw : mne.io.BaseRaw
            The cleaned raw object from EEGPreprocessor.cleaned_raw.
            Must have data loaded (preload=True).
        """
        self.raw     = raw
        self.epochs  = None          # mne.Epochs — set after create_epochs()
        self.events  = None          # np.ndarray shape (n_events, 3)
        self.event_id = {}           # dict mapping label → event code

        self.tmin    = -0.2
        self.tmax    =  0.8
        self.baseline = (-0.2, 0.0)
        self.reject_threshold_uv = 100.0  # µV — trials exceeding this are flagged

        self.report: Dict[str, Any] = {
            "events_found":    {},
            "epochs_created":  False,
            "n_trials_total":  0,
            "n_trials_kept":   0,
            "n_trials_rejected": 0,
            "rejection_threshold_uv": self.reject_threshold_uv,
            "conditions":      [],
            "steps":           []
        }

    # -----------------------------------------------------------------------
    # 1. Event reading
    # -----------------------------------------------------------------------

    def read_events(self) -> Dict[str, int]:
        """
        Read events from the Raw object and return a dict of
        {event_label: trial_count}.

        For EEGLAB .set files loaded via MNE, events are stored as
        annotations which MNE converts automatically.
        """
        try:
            events, event_id = mne.events_from_annotations(self.raw, verbose=False)
        except Exception as e:
            self.report["steps"].append(f"Event reading failed: {str(e)}")
            return {}

        self.events   = events
        self.event_id = event_id

        # Count trials per condition
        counts = {}
        for label, code in event_id.items():
            n = int(np.sum(events[:, 2] == code))
            counts[label] = n

        self.report["events_found"] = counts
        self.report["steps"].append(
            f"Found {len(events)} events across {len(event_id)} condition(s)"
        )

        return counts

    # -----------------------------------------------------------------------
    # 2. Epoch creation
    # -----------------------------------------------------------------------

    def create_epochs(
        self,
        event_labels: List[str],
        tmin: float = -0.2,
        tmax: float =  0.8,
        baseline: Tuple[float, float] = (-0.2, 0.0),
        reject_threshold_uv: float = 100.0
    ) -> mne.Epochs:
        """
        Create epochs around the specified event labels.

        Parameters
        ----------
        event_labels      : list of event label strings to epoch around
        tmin / tmax       : epoch window in seconds relative to event onset
        baseline          : baseline correction window (start, end) in seconds
        reject_threshold_uv : peak-to-peak amplitude threshold in µV for rejection
        """
        if self.events is None:
            self.read_events()

        if not self.events.size:
            raise ValueError("No events found in the recording.")

        self.tmin    = tmin
        self.tmax    = tmax
        self.baseline = baseline
        self.reject_threshold_uv = reject_threshold_uv

        # Build event_id dict for only the selected labels
        selected_id = {
            label: code
            for label, code in self.event_id.items()
            if label in event_labels
        }

        if not selected_id:
            raise ValueError(
                f"None of the selected labels {event_labels} were found in the recording. "
                f"Available: {list(self.event_id.keys())}"
            )

        reject_v = {"eeg": reject_threshold_uv * 1e-6}  # convert µV → V for MNE

        self.epochs = mne.Epochs(
            self.raw,
            self.events,
            event_id=selected_id,
            tmin=tmin,
            tmax=tmax,
            baseline=baseline,
            reject=reject_v,
            preload=True,
            verbose=False
        )

        n_total    = sum(self.report["events_found"].get(l, 0) for l in event_labels)
        n_kept     = len(self.epochs)
        n_rejected = n_total - n_kept

        self.report["epochs_created"]      = True
        self.report["n_trials_total"]      = n_total
        self.report["n_trials_kept"]       = n_kept
        self.report["n_trials_rejected"]   = n_rejected
        self.report["rejection_threshold_uv"] = reject_threshold_uv
        self.report["conditions"]          = event_labels
        self.report["tmin"]                = tmin
        self.report["tmax"]                = tmax
        self.report["baseline"]            = list(baseline)
        self.report["steps"].append(
            f"Created epochs: {n_kept}/{n_total} trials kept "
            f"(rejected {n_rejected} above {reject_threshold_uv} µV)"
        )

        return self.epochs

    # -----------------------------------------------------------------------
    # 3. Trial quality
    # -----------------------------------------------------------------------

    def get_trial_quality(self) -> Dict[str, Any]:
        """
        Per-trial quality metrics.
        Returns a dict with per-trial amplitude stats and rejection flags.
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Call create_epochs() first.")

        data = self.epochs.get_data(picks="eeg")  # shape: (n_trials, n_ch, n_times)

        ptp_per_trial = np.ptp(data, axis=2).max(axis=1) * 1e6  # µV, max across channels
        rms_per_trial = np.sqrt(np.mean(data ** 2, axis=(1, 2))) * 1e6

        threshold = self.reject_threshold_uv
        flags = ["clean" if p <= threshold else "rejected" for p in ptp_per_trial]

        trial_labels = []
        for i, ep in enumerate(self.epochs.events):
            code = ep[2]
            label = next(
                (k for k, v in self.epochs.event_id.items() if v == code),
                str(code)
            )
            trial_labels.append(label)

        return {
            "n_trials":         len(ptp_per_trial),
            "ptp_per_trial_uv": ptp_per_trial.tolist(),
            "rms_per_trial_uv": rms_per_trial.tolist(),
            "flags":            flags,
            "trial_labels":     trial_labels,
            "threshold_uv":     threshold,
            "times":            self.epochs.times.tolist()
        }

    # -----------------------------------------------------------------------
    # 4. ERP computation
    # -----------------------------------------------------------------------

    def get_erp(
        self,
        condition: Optional[str] = None,
        channel_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Compute the ERP (trial average) for a condition.

        Parameters
        ----------
        condition       : event label string. If None, averages all trials.
        channel_indices : list of channel indices to include. If None, uses all EEG.

        Returns dict with times, mean waveform, SEM, and detected peaks.
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Call create_epochs() first.")

        if condition and condition in self.epochs.event_id:
            subset = self.epochs[condition]
        else:
            subset = self.epochs

        data = subset.get_data(picks="eeg")  # (n_trials, n_ch, n_times)

        if channel_indices:
            data = data[:, channel_indices, :]

        times   = subset.times                          # (n_times,)
        mean    = data.mean(axis=0).mean(axis=0) * 1e6 # average across trials then channels, µV
        sem     = (data.mean(axis=1).std(axis=0) / np.sqrt(len(data))) * 1e6

        # Per-channel average for topomap (not averaged across channels)
        mean_per_channel = data.mean(axis=0) * 1e6     # (n_ch, n_times)

        peaks = self._detect_erp_components(times, mean)

        return {
            "condition":        condition or "all",
            "n_trials":         len(subset),
            "times":            times.tolist(),
            "mean_uv":          mean.tolist(),
            "sem_uv":           sem.tolist(),
            "mean_per_channel": mean_per_channel.tolist(),
            "peaks":            peaks
        }

    def get_erp_comparison(
        self,
        condition_a: str,
        condition_b: str
    ) -> Dict[str, Any]:
        """
        Compute ERPs for two conditions and their difference wave.
        """
        erp_a = self.get_erp(condition_a)
        erp_b = self.get_erp(condition_b)

        mean_a = np.array(erp_a["mean_uv"])
        mean_b = np.array(erp_b["mean_uv"])
        diff   = (mean_a - mean_b).tolist()

        return {
            "condition_a": erp_a,
            "condition_b": erp_b,
            "difference":  diff,
            "times":       erp_a["times"]
        }

    # -----------------------------------------------------------------------
    # 5. Band power per condition
    # -----------------------------------------------------------------------

    def get_band_power(self, condition: Optional[str] = None) -> Dict[str, float]:
        """
        Compute mean power in standard EEG bands for a condition.
        Returns dict of {band_name: mean_power}.
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Call create_epochs() first.")

        if condition and condition in self.epochs.event_id:
            subset = self.epochs[condition]
        else:
            subset = self.epochs

        data  = subset.get_data(picks="eeg")   # (n_trials, n_ch, n_times)
        sfreq = self.epochs.info["sfreq"]

        # Average across trials and channels before FFT
        mean_signal = data.mean(axis=0).mean(axis=0)  # (n_times,)
        n      = len(mean_signal)
        freqs  = np.fft.rfftfreq(n, d=1 / sfreq)
        fft    = np.fft.rfft(mean_signal)
        psd    = (np.abs(fft) ** 2) / n

        bands = {
            "Delta (1-4 Hz)":  (1,  4),
            "Theta (4-8 Hz)":  (4,  8),
            "Alpha (8-12 Hz)": (8,  12),
            "Beta (12-30 Hz)": (12, 30),
            "Gamma (30-45 Hz)":(30, 45),
        }

        result = {}
        for name, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            result[name] = float(psd[mask].mean()) if np.any(mask) else 0.0

        return result

    # -----------------------------------------------------------------------
    # 6. Trial grid data (for heatmap)
    # -----------------------------------------------------------------------

    def get_trial_grid(
        self,
        channel_index: int = 0,
        condition: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns per-trial amplitude matrix for one channel.
        Shape: (n_trials, n_times) — used to draw the trial heatmap.
        """
        if self.epochs is None:
            raise ValueError("No epochs created. Call create_epochs() first.")

        if condition and condition in self.epochs.event_id:
            subset = self.epochs[condition]
        else:
            subset = self.epochs

        data        = subset.get_data(picks="eeg")   # (n_trials, n_ch, n_times)
        channel_data = data[:, channel_index, :] * 1e6  # µV

        trial_labels = []
        for ep in subset.events:
            code  = ep[2]
            label = next(
                (k for k, v in subset.event_id.items() if v == code),
                str(code)
            )
            trial_labels.append(label)

        return {
            "matrix":       channel_data.tolist(),   # (n_trials, n_times)
            "times":        subset.times.tolist(),
            "trial_labels": trial_labels,
            "channel_name": self.epochs.info["ch_names"][channel_index],
            "n_trials":     len(subset)
        }

    # -----------------------------------------------------------------------
    # 7. Epoch report
    # -----------------------------------------------------------------------

    def get_epoch_report(self) -> Dict[str, Any]:
        """
        Structured summary of the epoch session.
        Passed to the interpretation layer.
        """
        return dict(self.report)

    # -----------------------------------------------------------------------
    # Internal — ERP component detection
    # -----------------------------------------------------------------------

    def _detect_erp_components(
        self,
        times: np.ndarray,
        mean_uv: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Simple peak-picking for standard ERP components.
        Searches predefined latency windows for local extrema.
        """
        COMPONENTS = [
            # name,   polarity,   window_ms
            ("P1",   "positive",  ( 80, 140)),
            ("N1",   "negative",  (100, 200)),
            ("N170", "negative",  (140, 200)),
            ("P2",   "positive",  (150, 250)),
            ("N2",   "negative",  (200, 350)),
            ("P300", "positive",  (250, 600)),
            ("N400", "negative",  (300, 500)),
        ]

        times_ms = times * 1000
        detected = []

        for name, polarity, (t_lo, t_hi) in COMPONENTS:
            mask = (times_ms >= t_lo) & (times_ms <= t_hi)
            if not np.any(mask):
                continue

            window_data = mean_uv[mask]
            window_times = times_ms[mask]

            if polarity == "positive":
                idx   = np.argmax(window_data)
                amp   = float(window_data[idx])
                found = amp > 0.5   # minimum amplitude to count as detected
            else:
                idx   = np.argmin(window_data)
                amp   = float(window_data[idx])
                found = amp < -0.5

            if found:
                detected.append({
                    "name":      name,
                    "polarity":  polarity,
                    "latency_ms": float(window_times[idx]),
                    "amplitude_uv": amp,
                    "window_ms": [t_lo, t_hi]
                })

        return detected
