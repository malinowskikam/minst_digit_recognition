import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPixmap, QPainter, QPen, QMouseEvent, QImage
from PyQt6.QtWidgets import QLabel


class Canvas(QLabel):
    """
    Drawable canvas
    """
    _pen: QPen

    _canvas_size = 280, 280
    _target_size = 28, 28
    _pen_size = 23

    _background_color = QColor("white")
    _foreground_color = QColor("black")

    _last_paint_position = None

    def __init__(self):
        super().__init__()
        # Set size to 10x input size
        self.setFixedSize(*self._canvas_size)

        # Set up pixmap
        pixmap = QPixmap(*self._canvas_size)
        pixmap.fill(self._background_color)
        self.setPixmap(pixmap)

        # Set up pen
        self._pen = QPen()
        self._pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        self._pen.setColor(self._foreground_color)
        self._pen.setWidth(self._pen_size)

    def clear(self) -> None:
        """
        Fill the canvas with background color
        """
        pixmap = self.pixmap()
        pixmap.fill(self._background_color)
        self.setPixmap(pixmap)

    def get_image(self) -> np.ndarray:
        pixmap = self.pixmap()
        qt_image = (pixmap.toImage()
                    .convertToFormat(QImage.Format.Format_Grayscale8)
                    .scaled(*self._target_size))
        qt_image.invertPixels()

        # Convert data into numpy array
        width = qt_image.width()
        height = qt_image.height()
        ptr = qt_image.bits()
        ptr.setsize(width * height)
        arr = np.array(ptr)

        return arr

    def mousePressEvent(self, e: QMouseEvent):
        pixmap = self.pixmap()
        painter = QPainter(pixmap)
        painter.setPen(self._pen)
        painter.drawPoint(e.position())
        painter.end()
        self._last_paint_position = e.position()
        self.setPixmap(pixmap)

    def mouseMoveEvent(self, e: QMouseEvent):
        # This should always be true if the hover mouse event is not enabled
        if self._last_paint_position:
            pixmap = self.pixmap()
            painter = QPainter(pixmap)
            painter.setPen(self._pen)
            painter.drawLine(self._last_paint_position, e.position())
            painter.end()

            self._last_paint_position = e.position()
            self.setPixmap(pixmap)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._last_paint_position = None
