from copy import deepcopy
import numpy as np


class HistoryManager:
    def __init__(self):
        self.original_raw = None
        self.current_raw = None
        self.pre_ica_raw = None

        self.snapshots = []
        self.reports = []
        self.metadata = {}

    def _clone_raw(self, raw):
        clone = raw.copy()
        clone.load_data()
        clone._data = raw.get_data().copy()
        clone.info = deepcopy(raw.info)
        return clone

    def set_original_raw(self, raw):
        self.original_raw = self._clone_raw(raw)
        self.current_raw = self._clone_raw(raw)

    def set_pre_ica_raw(self, raw):
        self.pre_ica_raw = self._clone_raw(raw)

    def update_current_raw(self, raw):
        self.current_raw = self._clone_raw(raw)

    def add_snapshot(self, label, raw, step_type=None, details=None):
        snapshot = {
            "label": label,
            "raw": self._clone_raw(raw),
            "step_type": step_type,
            "details": deepcopy(details) if details else {}
        }
        print("Snapshot saved:", label, "step_type:", step_type)
        self.snapshots.append(snapshot)

    def add_report(self, report):
        self.reports.append(deepcopy(report))

    def set_metadata(self, metadata):
        self.metadata = deepcopy(metadata)

    def get_snapshot_labels(self):
        return [snap["label"] for snap in self.snapshots]

    def get_snapshot(self, index):
        return self.snapshots[index]

    def get_latest_snapshot(self):
        if not self.snapshots:
            return None
        return self.snapshots[-1]