"""
EEG Tool Configuration Manager
-------------------------------
Manages all user-configurable settings in a single JSON file.
Provides defaults, load, save, and validation.

Config file is saved alongside the tool as eeg_config.json.
If it doesn't exist, defaults are used and written on first save.

Structure:
  - thresholds   : quality assessment boundaries
  - pipeline     : default preprocessing parameters
  - scoring      : point deductions for each quality issue
"""

import json
import os
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Default configuration
# All values here are overridable via the Settings panel or eeg_config.json
# ---------------------------------------------------------------------------

DEFAULTS: Dict[str, Any] = {

    # Quality thresholds — boundaries between good / medium / poor verdicts
    "thresholds": {

        # Bad channel ratio (proportion of total channels)
        "bad_channel_ratio_good":    0.05,   # <5%  → good
        "bad_channel_ratio_medium":  0.15,   # 5–15% → medium, >15% → poor

        # Mean RMS amplitude in microvolts
        "mean_rms_good_uv":          5.0,    # <5µV  → good
        "mean_rms_medium_uv":        15.0,   # 5–15µV → medium, >15µV → poor

        # Max peak-to-peak amplitude in microvolts
        "max_ptp_good_uv":           100.0,  # <100µV → good
        "max_ptp_medium_uv":         200.0,  # 100–200µV → medium, >200µV → poor

        # Recording duration in seconds
        "duration_good_sec":         120.0,  # >120s → good
        "duration_medium_sec":       60.0,   # 60–120s → medium, <60s → poor
    },

    # Score deductions — how many points each issue costs (out of 100)
    "scoring": {
        "bad_channel_high":   30,
        "bad_channel_medium": 15,
        "rms_high":           20,
        "rms_medium":          8,
        "ptp_medium":         15,
        "duration_high":      20,
        "duration_medium":     5,
    },

    # Grade boundaries — minimum score for each grade label
    "grades": {
        "good":         80,
        "acceptable":   60,
        "poor":         40,
        # below 40 → "Problematic — consider re-recording"
    },

    # Default pipeline parameters shown in the GUI
    "pipeline": {
        "l_freq":        1.0,
        "h_freq":        40.0,
        "notch_freq":    50.0,
        "n_ica_components": 20,
        "interpolate_bads": True,
        "fit_ica":          True,
        "variance_z_threshold":  3.0,
        "flat_std_threshold":    1e-7,
    },

    # UI preferences
    "ui": {
        "default_audience": "researcher",   # researcher / clinician / general
    }
}


# ---------------------------------------------------------------------------
# Config manager
# ---------------------------------------------------------------------------

CONFIG_FILENAME = "eeg_config.json"


class ConfigManager:
    """
    Singleton-style config manager.
    
    Usage:
        cfg = ConfigManager()          # loads from file or uses defaults
        val = cfg.get("thresholds", "mean_rms_good_uv")
        cfg.set("thresholds", "mean_rms_good_uv", 6.0)
        cfg.save()
    """

    def __init__(self, config_path: str = None):
        self._path   = config_path or self._default_path()
        self._config = self._deep_copy(DEFAULTS)
        self._load()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get(self, section: str, key: str) -> Any:
        """Get a config value. Falls back to default if missing."""
        try:
            return self._config[section][key]
        except KeyError:
            return DEFAULTS.get(section, {}).get(key)

    def get_section(self, section: str) -> Dict[str, Any]:
        """Return a full section as a dict."""
        return dict(self._config.get(section, DEFAULTS.get(section, {})))

    def set(self, section: str, key: str, value: Any) -> None:
        """Set a single value. Creates section if missing."""
        if section not in self._config:
            self._config[section] = {}
        self._config[section][key] = value

    def set_section(self, section: str, values: Dict[str, Any]) -> None:
        """Replace an entire section."""
        self._config[section] = values

    def save(self) -> None:
        """Persist current config to JSON file."""
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._config, f, indent=2)

    def reset_to_defaults(self) -> None:
        """Reset everything to factory defaults (does not save automatically)."""
        self._config = self._deep_copy(DEFAULTS)

    def reset_section(self, section: str) -> None:
        """Reset one section to defaults."""
        self._config[section] = self._deep_copy(DEFAULTS.get(section, {}))

    @property
    def path(self) -> str:
        return self._path

    def as_dict(self) -> Dict[str, Any]:
        return self._deep_copy(self._config)

    # -----------------------------------------------------------------------
    # Internal
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return  # use defaults, write on first save
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                saved = json.load(f)
            # Merge saved over defaults so new keys in DEFAULTS still appear
            for section, values in saved.items():
                if section in self._config and isinstance(values, dict):
                    self._config[section].update(values)
                else:
                    self._config[section] = values
        except (json.JSONDecodeError, OSError):
            pass  # corrupt file → use defaults silently

    @staticmethod
    def _default_path() -> str:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), CONFIG_FILENAME)

    @staticmethod
    def _deep_copy(d: Any) -> Any:
        return json.loads(json.dumps(d))


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------

config = ConfigManager()