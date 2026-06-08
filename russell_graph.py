#russell_graph
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
        self.target_valence = 0.0
        self.target_arousal = 0.0

    def update_point(self, cv, ca, tv, ta):
        """ Updates both current and target coordinates simultaneously """
        self.valence = float(cv)
        self.arousal = float(ca)
        self.target_valence = float(tv)
        self.target_arousal = float(ta)
        self.update() # Triggers paintEvent natively

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


        # Remap standard [-1, 1] grid into inverted UI space coordinates
        curr_x = center_x + (self.valence * radius)
        curr_y = center_y - (self.arousal * radius)
        
        targ_x = center_x + (self.target_valence * radius)
        targ_y = center_y - (self.target_arousal * radius) 

        # Draw a clean vector line indicating intended emotional trajectory
        path_pen = QPen(QColor(0, 85, 164, 180), 2, Qt.SolidLine) # Blue vector trail
        painter.setPen(path_pen)
        painter.drawLine(QPointF(curr_x, curr_y), QPointF(targ_x, targ_y))

        # --- 5. DRAW TRACKING MARKERS ---
        # Draw target destination marker (Hollow Blue Circle)
        painter.setPen(QPen(QColor(0, 85, 164), 2))
        painter.setBrush(QBrush(QColor(0, 85, 164, 40))) # Translucent blue filling
        painter.drawEllipse(QRectF(targ_x - 6, targ_y - 6, 12, 12))

        # Draw current driver state marker (Solid Red Circle)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(239, 68, 68))) 
        painter.drawEllipse(QRectF(curr_x - 6, curr_y - 6, 12, 12))