import sys
import matplotlib
matplotlib.use("QtAgg")

from PySide6.QtWidgets import QApplication
from EEG_app import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
