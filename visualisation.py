from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import mne


class plot_manager:
    def __init__(self):
        pass
 

    def plot_raw_channel(self,raw = None):


        data, times = raw.get_data(picks=[0], return_times=True)  # first channel
        channel_name = raw.ch_names[0]

        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(times, data[0])
        ax.set_title(f"Raw EEG - {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (V)")
        ax.grid(True)

        return fig

    def plot_psd(self,raw):


        data = raw.get_data(picks="eeg")
        sfreq = raw.info["sfreq"]

        # Average PSD across channels using simple FFT
        n = data.shape[1]
        freqs = np.fft.rfftfreq(n, d=1/sfreq)
        fft_vals = np.fft.rfft(data, axis=1)
        psd = (np.abs(fft_vals) ** 2) / n
        mean_psd = psd.mean(axis=0)

        # Limit to 0-60 Hz for display
        mask = freqs <= 60

        fig = Figure(figsize=(7, 4), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs[mask], mean_psd[mask])
        ax.set_title("Average Power Spectral Density")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.grid(True)

        return fig

    def plot_ica_components(self,data):
       
        data.plot_components()
        
            # self.preprocessor.ica.plot_sources()
       

    def plot_ica_sources(self,data,c_raw):
      
        data.plot_sources(c_raw)
     

    def plot_ica_properties_from_entry(self,ica_entry,c_raw,data):

        text = ica_entry.get().strip()

        picks = [int(x.strip()) for x in text.split(",") if x.strip()]

        data.plot_properties(c_raw,picks)

    def plot_comparision_psd(self,raw1, raw2,label1,label2):

        print("raw1 id:", id(raw1))
        print("raw2 id:", id(raw2))
        print("raw1 is raw2:", raw1 is raw2)

        data1 = raw1.get_data(picks="eeg")
        data2 = raw2.get_data(picks="eeg")

        print("data1 == data2:", np.array_equal(data1, data2))
        print("max abs diff:", np.max(np.abs(data1 - data2)))

        sfreq1 = raw1.info["sfreq"]
        sfreq2 = raw2.info["sfreq"]

        if sfreq1 != sfreq2:
            raise ValueError("Sampling rates do not match.")

        n1 = data1.shape[1]
        n2 = data2.shape[1]

        freqs1 = np.fft.rfftfreq(n1, d=1 / sfreq1)
        freqs2 = np.fft.rfftfreq(n2, d=1 / sfreq2)

        fft1 = np.fft.rfft(data1, axis=1)
        fft2 = np.fft.rfft(data2, axis=1)

        psd1 = (np.abs(fft1) ** 2) / n1
        psd2 = (np.abs(fft2) ** 2) / n2

        mean_psd1 = psd1.mean(axis=0)
        mean_psd2 = psd2.mean(axis=0)

        mask1 = freqs1 <= 60
        mask2 = freqs2 <= 60

        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(freqs1[mask1], mean_psd1[mask1], label=label1)
        ax.plot(freqs2[mask2], mean_psd2[mask2], label=label2)
        ax.set_title("PSD Comparison")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Power")
        ax.legend()
        ax.grid(True)

        return fig

        

    def plot_raw_overlay_comparision(self, raw1, raw2,label1,label2, channel_index= 0):

        if channel_index < 0 or channel_index >= len(raw1.ch_names):
            raise ValueError("Invalid channel index.")

        if raw1.info["sfreq"] != raw2.info["sfreq"]:
            raise ValueError("Sampling rates do not match.")
        
        print("raw1 id:", id(raw1))
        print("raw2 id:", id(raw2))
        print("raw1 is raw2:", raw1 is raw2)

        data1, times1 = raw1.get_data(picks=[channel_index], return_times=True)
        data2, times2 = raw2.get_data(picks=[channel_index], return_times=True)

        print("data1 == data2:", np.array_equal(data1, data2))
        print("max abs diff:", np.max(np.abs(data1 - data2)))

        channel_name = raw1.ch_names[channel_index]

        min_len = min(len(times1), len(times2))
        times1 = times1[:min_len]
        data1 = data1[0][:min_len]
        data2 = data2[0][:min_len]

        print("Max abs diff:", np.max(np.abs(data1 - data2)))
        print("Mean abs diff:", np.mean(np.abs(data1 - data2)))
        print("All equal:", np.array_equal(data1, data2))

        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(times1, data1, label=label1,alpha =0.7)
        ax.plot(times1, data2, label=label2,alpha =0.7)
        ax.set_title(f"Raw Overlay Comparison - {channel_name}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend()
        ax.grid(True)

        return fig
        
    
    def plot_difference_signal(self,raw1,raw2, label1,label2,channel_index=0):
        if channel_index < 0 or channel_index >= len(raw1.ch_names):
            raise ValueError("Invalid channel index.")

        if raw1.info["sfreq"] != raw2.info["sfreq"]:
            raise ValueError("Sampling rates do not match.")
        
        print("raw1 id:", id(raw1))
        print("raw2 id:", id(raw2))
        print("raw1 is raw2:", raw1 is raw2)

        data1, times1 = raw1.get_data(picks=[channel_index], return_times=True)
        data2, times2 = raw2.get_data(picks=[channel_index], return_times=True)

        print("data1 == data2:", np.array_equal(data1, data2))
        print("max abs diff:", np.max(np.abs(data1 - data2)))

        channel_name = raw1.ch_names[channel_index]

        min_len = min(len(times1), len(times2))
        times = times1[:min_len]
        sig1 = data1[0][:min_len]
        sig2 = data2[0][:min_len]

        difference = (sig1 - sig2) *1e6

        print("Max abs diff:", np.max(np.abs(difference)))
        print("Mean abs diff:", np.mean(np.abs(difference)))
        print("All equal:", np.array_equal(sig1 *1e6, sig2*1e6))

        fig = Figure(figsize=(8, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.plot(times, difference)
        ax.set_title(f"Difference Signal - {channel_name},{label1} - {label2}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude Difference (µV)")
        ax.grid(True)

        return fig
        

    def plot_band_power_comparision(self,raw1,raw2,label1,label2):

        if raw1.info["sfreq"] != raw2.info["sfreq"]:
            raise ValueError("Sampling rates do not match.")
        
        print("raw1 id:", id(raw1))
        print("raw2 id:", id(raw2))
        print("raw1 is raw2:", raw1 is raw2)

        data1 = raw1.get_data(picks="eeg")
        data2 = raw2.get_data(picks="eeg")
        sfreq = raw1.info["sfreq"]

        print("data1 == data2:", np.array_equal(data1, data2))
        print("max abs diff:", np.max(np.abs(data1 - data2)))

        def compute_band_powers(data, sfreq):
            n = data.shape[1]
            freqs = np.fft.rfftfreq(n, d=1 / sfreq)
            fft_vals = np.fft.rfft(data, axis=1)
            psd = (np.abs(fft_vals) ** 2) / n
            mean_psd = psd.mean(axis=0)

            bands = {
                "Delta (1-4)": (1, 4),
                "Theta (4-8)": (4, 8),
                "Alpha (8-12)": (8, 12),
                "Beta (12-30)": (12, 30),
                "Gamma (30-45)": (30, 45),
            }

            band_powers = {}
            for band_name, (low, high) in bands.items():
                mask = (freqs >= low) & (freqs < high)
                band_powers[band_name] = mean_psd[mask].mean() if np.any(mask) else 0.0

            return band_powers

        bp1 = compute_band_powers(data1, sfreq)
        bp2 = compute_band_powers(data2, sfreq)

        labels = list(bp1.keys())
        vals1 = [bp1[label] for label in labels]
        vals2 = [bp2[label] for label in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig = Figure(figsize=(9, 5), dpi=100)
        ax = fig.add_subplot(111)
        ax.bar(x - width / 2, vals1, width=width, label=label1)
        ax.bar(x + width / 2, vals2, width=width, label=label2)
        ax.set_title("Band Power Comparison")
        ax.set_xlabel("Frequency Bands")
        ax.set_ylabel("Mean Power")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.legend()
        ax.grid(True, axis="y")

        return fig

    # -------------------------------------------------------------------
    # Epoch plots
    # -------------------------------------------------------------------

    def plot_erp(self, erp_data: dict, title: str = None) -> Figure:
        """
        Plot the ERP waveform with SEM shading and detected component labels.

        Parameters
        ----------
        erp_data : dict returned by EpochAnalyzer.get_erp()
        title    : optional override title
        """
        times   = np.array(erp_data["times"]) * 1000   # convert to ms
        mean    = np.array(erp_data["mean_uv"])
        sem     = np.array(erp_data["sem_uv"])
        peaks   = erp_data.get("peaks", [])
        cond    = erp_data.get("condition", "")
        n       = erp_data.get("n_trials", 0)

        fig = Figure(figsize=(9, 4), dpi=100)
        ax  = fig.add_subplot(111)

        # Baseline marker
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        # SEM shading
        ax.fill_between(times, mean - sem, mean + sem, alpha=0.2, color="#1565c0")

        # Mean waveform
        ax.plot(times, mean, color="#1565c0", linewidth=1.8, label=f"{cond} (n={n})")

        # Compute data range first — needed for annotation offsets and y-limits
        data_min = float((mean - sem).min())
        data_max = float((mean + sem).max())
        span = max(data_max - data_min, 0.5)

        # Component labels
        label_offset = span * 0.25
        for peak in peaks:
            lat = peak["latency_ms"]
            amp = peak["amplitude_uv"]
            direction = label_offset if amp > 0 else -label_offset
            ax.annotate(
                peak["name"],
                xy=(lat, amp),
                xytext=(lat + 15, amp + direction),
                fontsize=8,
                color="#b71c1c",
                arrowprops=dict(arrowstyle="->", color="#b71c1c", lw=0.8)
            )
            ax.plot(lat, amp, "r.", markersize=6)

        ax.set_title(title or f"ERP — {cond}")
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Center waveform with tight y-limits
        pad = span * 0.3
        ax.set_ylim(data_min - pad, data_max + pad)

        fig.tight_layout()
        return fig

    def plot_erp_comparison(self, comparison_data: dict) -> Figure:
        """
        Overlay two ERP waveforms and plot their difference wave below.

        Parameters
        ----------
        comparison_data : dict returned by EpochAnalyzer.get_erp_comparison()
        """
        times  = np.array(comparison_data["times"]) * 1000
        erp_a  = comparison_data["condition_a"]
        erp_b  = comparison_data["condition_b"]
        diff   = np.array(comparison_data["difference"])

        mean_a = np.array(erp_a["mean_uv"])
        mean_b = np.array(erp_b["mean_uv"])
        sem_a  = np.array(erp_a["sem_uv"])
        sem_b  = np.array(erp_b["sem_uv"])

        fig = Figure(figsize=(9, 6), dpi=100)
        ax1, ax2 = fig.subplots(2, 1, sharex=True)

        # Top — overlay
        for ax in [ax1, ax2]:
            ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.3)

        ax1.fill_between(times, mean_a - sem_a, mean_a + sem_a, alpha=0.15, color="#1565c0")
        ax1.fill_between(times, mean_b - sem_b, mean_b + sem_b, alpha=0.15, color="#b71c1c")
        ax1.plot(times, mean_a, color="#1565c0", linewidth=1.8,
                 label=f"{erp_a['condition']} (n={erp_a['n_trials']})")
        ax1.plot(times, mean_b, color="#b71c1c", linewidth=1.8,
                 label=f"{erp_b['condition']} (n={erp_b['n_trials']})")
        ax1.set_ylabel("Amplitude (µV)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_title("ERP Comparison")

        # Bottom — difference wave
        ax2.fill_between(times, diff, 0, where=np.array(diff) > 0,
                         alpha=0.3, color="#2e7d32", label="A > B")
        ax2.fill_between(times, diff, 0, where=np.array(diff) < 0,
                         alpha=0.3, color="#b71c1c", label="B > A")
        ax2.plot(times, diff, color="#333", linewidth=1.4)
        ax2.set_ylabel("Difference (µV)")
        ax2.set_xlabel("Time (ms)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.set_title(f"Difference: {erp_a['condition']} − {erp_b['condition']}")

        fig.tight_layout()
        return fig

    def plot_trial_grid(self, grid_data: dict) -> Figure:
        """
        Heatmap of amplitude across all trials × time.
        Rows = trials, columns = timepoints. Colour = amplitude in µV.

        Parameters
        ----------
        grid_data : dict returned by EpochAnalyzer.get_trial_grid()
        """
        matrix  = np.array(grid_data["matrix"])   # (n_trials, n_times)
        times   = np.array(grid_data["times"]) * 1000
        ch_name = grid_data.get("channel_name", "")
        n       = grid_data.get("n_trials", matrix.shape[0])

        # Symmetric colour scale
        vmax = np.percentile(np.abs(matrix), 95)
        vmin = -vmax

        fig = Figure(figsize=(9, max(3, n * 0.18 + 1.5)), dpi=100)
        ax  = fig.add_subplot(111)

        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            extent=[times[0], times[-1], n, 0],
            vmin=vmin, vmax=vmax,
            cmap="RdBu_r"
        )

        ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Trial")
        ax.set_title(f"Trial Grid — {ch_name}  ({n} trials)")

        fig.colorbar(im, ax=ax, label="Amplitude (µV)", shrink=0.8)
        fig.tight_layout()

        return fig

    def plot_epoch_band_power(
        self,
        band_power_a: dict,
        band_power_b: dict,
        label_a: str,
        label_b: str
    ) -> Figure:
        """
        Bar chart comparing band power between two epoch conditions.

        Parameters
        ----------
        band_power_a/b : dicts returned by EpochAnalyzer.get_band_power()
        """
        labels = list(band_power_a.keys())
        vals_a = [band_power_a[l] for l in labels]
        vals_b = [band_power_b[l] for l in labels]

        x     = np.arange(len(labels))
        width = 0.35

        fig = Figure(figsize=(9, 4), dpi=100)
        ax  = fig.add_subplot(111)

        ax.bar(x - width / 2, vals_a, width=width, label=label_a, color="#1565c0", alpha=0.85)
        ax.bar(x + width / 2, vals_b, width=width, label=label_b, color="#b71c1c", alpha=0.85)

        ax.set_title("Epoch Band Power Comparison")
        ax.set_xlabel("Frequency Band")
        ax.set_ylabel("Mean Power")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.4)
        fig.tight_layout()

        return fig