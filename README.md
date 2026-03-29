# EEG Workbench

A modular, research-grade EEG analysis tool built with PySide6 and MNE-Python. Designed for neuroscience researchers, students, and developers who need a flexible desktop GUI for EEG preprocessing, visualization, and analysis — without writing a pipeline from scratch.

![Preprocessing & ICA](screenshots/EEG1.png)
![Band Power & PSD](screenshots/EEG2.png)
![Raw Visualization](screenshots/EEG3.png)
![Epoch Analysis](screenshots/EEG4.png)
![Hypothesis Reporting](screenshots/EEG5.png)
---

## Features

### Preprocessing & ICA
- Load raw EEG files (FIF, EDF, BrainVision, and other MNE-supported formats)
- Filtering, re-referencing, and bad channel interpolation
- Independent Component Analysis (ICA) with interactive component inspection and rejection

### Visualization
- Raw signal viewer with channel selection and time navigation
- Power Spectral Density (PSD) plots
- Band power analysis (delta, theta, alpha, beta, gamma)
- Snapshot comparison across processing stages

### Epoch Analysis
- Event-based epoching with configurable time windows
- Epoch rejection by amplitude threshold
- Averaged evoked responses
- Epoch-level statistics and inspection panel

### Hypothesis Testing & Reporting
- Built-in hypothesis panel for defining and testing analytical questions
- Automated report generation from analysis results
- History manager for tracking analysis sessions

---

## Installation

**Requirements:** Python 3.10+

```bash
git clone https://github.com/amartya1717/EEG-workbench.git
cd EEG-workbench
pip install -r requirements.txt
python main.py
```

### Core Dependencies
- [MNE-Python](https://mne.tools/) — EEG/MEG data processing
- [PySide6](https://doc.qt.io/qtforpython/) — GUI framework
- NumPy, SciPy, Matplotlib

---

## Project Structure

```
EEG-workbench/
├── main.py                  # Entry point
├── EEG_app.py               # Application shell and layout
├── preprocessing.py         # Filtering, ICA, bad channel handling
├── visualisation.py         # Raw, PSD, band power plots
├── epoch_analyzer.py        # Epoching and evoked analysis
├── epoch_panel.py           # Epoch inspection UI
├── analysis_widget.py       # Core analysis widget
├── hypothesis_panel.py      # Hypothesis definition UI
├── hypothesis_reporter.py   # Report generation
├── interpreter.py           # Analysis interpreter layer
├── history_manager.py       # Session history tracking
├── settings_panel.py        # User settings
└── config.py                # Global configuration
```

---

## Usage

1. Launch with `python main.py`
2. Load an EEG file via the file browser
3. Run preprocessing (filter → re-reference → ICA)
4. Inspect and reject ICA components
5. Visualize band power, PSD, or raw signals
6. Define epochs around events of interest
7. Set up hypotheses and generate analysis reports

---

## Status

Active development. Core pipeline is functional. Contributions and feedback welcome — open an issue or submit a pull request.

---

## Author

**Amartya** — University of Sydney, Neuroscience & Medical Science  
Built as part of ongoing EEG research infrastructure development.

---

## License

MIT License — free to use, modify, and distribute with attribution.
