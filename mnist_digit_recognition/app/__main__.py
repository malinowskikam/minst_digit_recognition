import sys

from PyQt6.QtWidgets import QApplication, QWidget


def main():
    app = QApplication(sys.argv)

    window = QWidget()
    window.show()

    app.exec()
    exit(0)


if __name__ == '__main__':
    main()
