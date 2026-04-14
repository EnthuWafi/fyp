# main.py
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

# Import the background thread from your worker file
from system_pipeline import SystemPipelineThread 
from russell_graph import RussellGraph

class App(QWidget):
    def __init__(self):
        super().__init__()
        
        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            pass # Failsafe if style.qss is missing

        self.setWindowTitle("Driver Music Regulation System")
        self.resize(1000, 600)

        # --- MASTER LAYOUT ---
        master_layout = QHBoxLayout()

        # --- LEFT PANEL: CAMERA ---
        left_panel = QVBoxLayout()
        title = QLabel("Live Driver Feed")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setStyleSheet("border: 2px solid gray;")
        
        left_panel.addWidget(title)
        left_panel.addWidget(self.image_label)

        # --- RIGHT PANEL: TELEMETRY & GRAPH ---
        right_panel = QVBoxLayout()
        
        telemetry_title = QLabel("System Telemetry")
        telemetry_title.setFont(QFont("Arial", 18, QFont.Bold))
        telemetry_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(telemetry_title)

        # 1. The Russell Graph
        self.russell_graph = RussellGraph()
        right_panel.addWidget(self.russell_graph, alignment=Qt.AlignCenter)

        # 2. V/A Data Readouts
        va_layout = QHBoxLayout()
        self.valence_label = QLabel("Valence: 0.00")
        self.arousal_label = QLabel("Arousal: 0.00")
        self.valence_label.setFont(QFont("Arial", 14))
        self.arousal_label.setFont(QFont("Arial", 14))
        va_layout.addWidget(self.valence_label)
        va_layout.addWidget(self.arousal_label)
        right_panel.addLayout(va_layout)

        # 3. Emotion State
        self.emotion_label = QLabel("State: Neutral")
        self.emotion_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.emotion_label.setStyleSheet("color: #333333;")
        right_panel.addWidget(self.emotion_label)

        # Divider Line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        right_panel.addWidget(line)

        # 4. Active Protocol
        self.protocol_label = QLabel("Protocol: Initializing...")
        self.protocol_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.protocol_label.setStyleSheet("color: #0055A4;") # Blue emphasis
        right_panel.addWidget(self.protocol_label)

        # 5. Current Track
        self.track_label = QLabel("Track: None")
        self.track_label.setFont(QFont("Arial", 12))
        self.track_label.setWordWrap(True)
        right_panel.addWidget(self.track_label)

        right_panel.addStretch() # Pushes everything to the top neatly

        # --- COMPILE LAYOUTS ---
        master_layout.addLayout(left_panel, stretch=2) # Camera gets 2/3 of screen
        master_layout.addLayout(right_panel, stretch=1) # Telemetry gets 1/3 of screen
        
        self.setLayout(master_layout)

        # --- Start the Background Thread ---
        self.thread = SystemPipelineThread()
        self.thread.update_ui_signal.connect(self.update_gui)
        self.thread.start()
        
    def update_gui(self, cv_img, valence, arousal, track, protocol, emotion):
        # Update Text Labels
        self.valence_label.setText(f"Valence: {valence:.2f}")
        self.arousal_label.setText(f"Arousal: {arousal:.2f}")
        self.track_label.setText(f"Track: {track}")
        self.protocol_label.setText(f"Protocol: {protocol}")
        self.emotion_label.setText(f"State: {emotion}")

        # Update the live Graph!
        self.russell_graph.update_point(valence, arousal)

        # Update Video Feed
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        scaled_pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
        self.image_label.setPixmap(scaled_pixmap)

    def closeEvent(self, event):
        self.thread.terminate()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())