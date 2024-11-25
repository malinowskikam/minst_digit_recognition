from PyQt6.QtWidgets import QMainWindow, QWidget


class MainWindow(QMainWindow):
    title = "App"

    def __init__(self):
        super().__init__()

        self.setWindowTitle(self.title)
        self.setCentralWidget(QWidget())
