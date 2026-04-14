from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

class RussellGraph(QWidget):
    """ A custom PySide6 widget to natively draw Russell's Circumplex Model """
    def __init__(self):
        super().__init__()
        self.setMinimumSize(300, 300)
        self.valence = 0.0
        self.arousal = 0.0

    def update_point(self, v, a):
        self.valence = float(v)
        self.arousal = float(a)
        self.update() # Triggers a repaint

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w = self.width()
        h = self.height()
        center_x = w / 2
        center_y = h / 2

        # Draw Background Circle
        radius = min(w, h) / 2 - 20
        painter.setPen(QPen(QColor(200, 200, 200), 2))
        painter.setBrush(QBrush(QColor(245, 245, 245)))
        painter.drawEllipse(QRectF(center_x - radius, center_y - radius, radius * 2, radius * 2))

        # Draw Axes
        painter.setPen(QPen(QColor(150, 150, 150), 2, Qt.DashLine))
        painter.drawLine(center_x, center_y - radius, center_x, center_y + radius) # Arousal (Y)
        painter.drawLine(center_x - radius, center_y, center_x + radius, center_y) # Valence (X)

        # Draw Labels
        painter.setPen(QPen(Qt.black))
        font = QFont("Arial", 8, QFont.Bold)
        painter.setFont(font)
        painter.drawText(center_x - 15, center_y - radius - 5, "High Arousal")
        painter.drawText(center_x - 15, center_y + radius + 15, "Low Arousal")
        painter.drawText(center_x + radius + 5, center_y + 5, "Positive")
        painter.drawText(center_x - radius - 50, center_y + 5, "Negative")

        # Map Valence/Arousal (-1.0 to 1.0) to pixel coordinates
        # Note: Y-axis is inverted in GUI drawing (0 is top)
        dot_x = center_x + (self.valence * radius)
        dot_y = center_y - (self.arousal * radius)

        # Draw the Data Point
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(255, 50, 50))) # Red dot
        painter.drawEllipse(QRectF(dot_x - 8, dot_y - 8, 16, 16))