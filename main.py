# main.py
import sys
import cv2
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap, QFont
from PySide6.QtCore import Qt

# Import the background thread from your worker file
from system_pipeline import SystemPipelineThread 

class App(QWidget):
    def __init__(self):
        super().__init__()

        with open("style.qss", "r") as f:
                    self.setStyleSheet(f.read())


        self.setWindowTitle("Driver Music Regulation System")
     
        self.resize(800, 600)

        # --- UI Layout setup ---
        main_layout = QVBoxLayout()

        title = QLabel("Intelligent Driver Cabin")
        title.setFont(QFont("Arial", 24, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setObjectName("videoLabel")
        main_layout.addWidget(self.image_label)

        dash_layout = QHBoxLayout()
        
        self.valence_label = QLabel("Valence: 0.0")
        self.valence_label.setFont(QFont("Arial", 14))
        
        self.arousal_label = QLabel("Arousal: 0.0")
        self.arousal_label.setFont(QFont("Arial", 14))
        
        self.track_label = QLabel("Track: None")
        self.track_label.setFont(QFont("Arial", 16, QFont.Bold))
        self.track_label.setObjectName("trackLabel")

        dash_layout.addWidget(self.valence_label)
        dash_layout.addWidget(self.arousal_label)
        dash_layout.addStretch()
        dash_layout.addWidget(self.track_label)

        main_layout.addLayout(dash_layout)
        self.setLayout(main_layout)

        # --- Start the Background Thread ---
        self.thread = SystemPipelineThread()
        self.thread.update_ui_signal.connect(self.update_gui)
        self.thread.start()

    def update_gui(self, cv_img, valence, arousal, track):
        self.valence_label.setText(f"Valence: {valence}")
        self.arousal_label.setText(f"Arousal: {arousal}")
        self.track_label.setText(f"Now Playing: {track}")

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