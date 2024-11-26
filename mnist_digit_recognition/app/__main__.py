import logging
import sys

from PyQt6.QtWidgets import QApplication

from mnist_digit_recognition.app.main_window import MainWindow

logger = logging.getLogger("mnist_digit_recognition.app")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    logger.info("Starting app")
    app.exec()
    exit(0)

if __name__ == '__main__':
    main()
