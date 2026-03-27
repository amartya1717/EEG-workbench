# EEG Analysis Workbench

A desktop research tool for end-to-end EEG preprocessing, 
epoch analysis, and hypothesis-driven interpretation.



## What It Does

- Full preprocessing pipeline (bandpass, notch, ICA, bad channel detection)
- Epoch analysis with trial-level inspection and ERP component detection
- Hypothesis report — links ERP and frequency findings to a stated hypothesis
- Audience-aware interpretation (researcher / clinician / general)
- Config-driven quality thresholds — no hardcoded values

## Why I Built It

[One paragraph — the research problem from your gustometer/EEG work 
that no existing tool solved cleanly]

## Key Design Decisions

**Snapshot history system** — every preprocessing step saves a deep-copied 
raw object, so you can compare signal state at any point in the pipeline.

**ICA baseline freezing** — ICA exclusion always applies to a frozen 
pre-ICA snapshot, not accumulated state. Changing ICA after epoching 
triggers a stale warning and clears epoch results.

**Config-driven interpretation** — quality thresholds and scoring weights 
live in eeg_config.json, not in code. Different labs have different standards.

**Audience-aware output** — every finding renders in three versions 
(researcher / clinician / general) selected at runtime.

## Installation
```bash
pip install mne numpy matplotlib
python EEG_main.py
```

## Requirements

- Python 3.9+
- Compatible with EEGLAB .set files

## Status

Active development — core pipeline and analysis complete.
Planned: batch processing, HTML report export, LLM interpretation layer.