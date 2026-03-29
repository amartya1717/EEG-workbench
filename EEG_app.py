from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import Qt
from analysis_widget import AnalysisWidget


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Analysis Tool")
        widget = AnalysisWidget()
        self.setCentralWidget(widget)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showMaximized()
        super().keyPressEvent(event)
