# main.py
import os
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Qt, QRectF

from system_pipeline import SystemPipelineThread 
from russell_graph import RussellGraph
from setup import setup_database

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

        # --- MASTER LAYOUT ---
        master_layout = QHBoxLayout()
        master_layout.setContentsMargins(20, 20, 20, 20)
        master_layout.setSpacing(25)

        # --- LEFT PANEL: CAMERA ---
        left_panel = QVBoxLayout()
        left_panel.setSpacing(10)
        
        title = QLabel("Live Driver Feed")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        
        self.image_label = QLabel()
        self.image_label.setObjectName("videoLabel") # Maps directly to QSS #videoLabel
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        
        left_panel.addWidget(title)
        left_panel.addWidget(self.image_label)

        # --- RIGHT PANEL: TELEMETRY & GRAPH ---
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        telemetry_title = QLabel("Russell Graph")
        telemetry_title.setFont(QFont("Arial", 16, QFont.Bold))
        telemetry_title.setAlignment(Qt.AlignCenter)
        right_panel.addWidget(telemetry_title)

        # 1. The Russell Graph
        self.russell_graph = RussellGraph()
        right_panel.addWidget(self.russell_graph, alignment=Qt.AlignCenter)

        # 2. V/A Data Readouts
        va_layout = QHBoxLayout()
        self.valence_label = QLabel("Valence: 0.00")
        self.arousal_label = QLabel("Arousal: 0.00")
        self.valence_label.setFont(QFont("Arial", 13))
        self.arousal_label.setFont(QFont("Arial", 13))
        va_layout.addWidget(self.valence_label)
        va_layout.addWidget(self.arousal_label)
        right_panel.addLayout(va_layout)

        # Music Vector Readout
        music_va_layout = QHBoxLayout()
        self.music_v_label = QLabel("Music Valence: 0.00")
        self.music_a_label = QLabel("Music Arousal: 0.00")
        self.music_v_label.setObjectName("musicValenceLabel")
        self.music_a_label.setObjectName("musicArousalLabel")
        music_va_layout.addWidget(self.music_v_label)
        music_va_layout.addWidget(self.music_a_label)
        right_panel.addLayout(music_va_layout)

        # 3. Emotion State
        self.emotion_label = QLabel("State: Neutral")
        self.emotion_label.setObjectName("emotionLabel")
        right_panel.addWidget(self.emotion_label)

        # Divider Line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        line.setStyleSheet("background-color: #333333;")
        right_panel.addWidget(line)

        # 4. Active Protocol
        self.protocol_label = QLabel("Protocol: Initializing...")
        self.protocol_label.setObjectName("protocolLabel")
        right_panel.addWidget(self.protocol_label)

        # 5. Current Track
        self.track_label = QLabel("Track: None")
        self.track_label.setObjectName("trackLabel")
        self.track_label.setWordWrap(True)
        right_panel.addWidget(self.track_label)

        # Control Operations
        self.skip_button = QPushButton("Skip Track ⏭")
        self.skip_button.setObjectName("skipButton")
        self.skip_button.setMinimumHeight(40)
        right_panel.addWidget(self.skip_button)

        right_panel.addStretch()

        # --- COMPILE LAYOUTS ---
        master_layout.addLayout(left_panel, stretch=2)
        master_layout.addLayout(right_panel, stretch=1)
        
        self.setLayout(master_layout)

        # --- Start the Background Thread ---
        self.thread = SystemPipelineThread()
        self.thread.update_ui_signal.connect(self.update_gui)
        self.thread.start()

        self.skip_button.clicked.connect(self.thread.request_skip)
        
    def update_gui(self, cv_img, valence, arousal, target_v, target_a, music_v, music_a, track, protocol, emotion):
        # Update Text Labels
        self.valence_label.setText(f"Valence: {valence:.2f}")
        self.arousal_label.setText(f"Arousal: {arousal:.2f}")
        self.track_label.setText(f"Track: {track}")
        self.protocol_label.setText(f"Protocol: {protocol}")
        self.emotion_label.setText(f"State: {emotion}")

        self.music_v_label.setText(f"Music Valence: {music_v:.2f}")
        self.music_a_label.setText(f"Music Arousal: {music_a:.2f}")

        # Update the live Graph
        self.russell_graph.update_point(valence, arousal, target_v, target_a)

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

def main():
    db_file = "music_system.db"
    ann_file = "music_vectors.ann"

    if not os.path.exists(db_file) or not os.path.exists(ann_file):
        print("[FIRST-RUN] Application assets missing. Beginning automated environment compilation...")
        setup_database(db_path=db_file, ann_path=ann_file)
        print("[FIRST-RUN] Environment compiled successfully.")

    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()