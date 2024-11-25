import sys

from PyQt6.QtWidgets import QApplication

from mnist_digit_recognition.app.main_window import MainWindow


def main():
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    app.exec()
    exit(0)


if __name__ == '__main__':
    main()
