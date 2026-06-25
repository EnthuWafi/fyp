# main.py
import os
import sys
import cv2
import sqlite3
import pandas as pd
from PySide6.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QFrame, QPushButton, QSlider
from PySide6.QtGui import QImage, QPixmap, QFont, QPainter, QPen, QBrush, QColor
from PySide6.QtCore import Slot, Qt, QRectF, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput

from system_pipeline import SystemPipelineThread 
from russell_graph import RussellGraph
from setup import setup_database
from export_telemetry import get_telemetry
from dotenv import load_dotenv

load_dotenv()

class App(QWidget):
    def __init__(self, db_path="music_system.db", annoy_index_path='music_vectors.ann'):
        super().__init__()
        
        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except FileNotFoundError:
            pass

        self.setWindowTitle("Driver Music Regulation System")
        self.resize(1000, 600)
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

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

        # NEW: Media Playback Progress Slider
        progress_layout = QHBoxLayout()
        self.time_elapsed_label = QLabel("00:00")
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.setObjectName("progressSlider")
        self.progress_slider.setRange(0, 100)
        self.progress_slider.setEnabled(False) # Track progress as read-only to protect pipeline state
        self.time_total_label = QLabel("00:00")
        
        progress_layout.addWidget(self.time_elapsed_label)
        progress_layout.addWidget(self.progress_slider)
        progress_layout.addWidget(self.time_total_label)
        right_panel.addLayout(progress_layout)

        # NEW: Volume Controller Slider
        volume_layout = QHBoxLayout()
        volume_icon = QLabel("🔊")
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setObjectName("volumeSlider")
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.setMinimumWidth(50)
        self.volume_slider.setMaximumWidth(100)
        
        volume_layout.addStretch() 
        volume_layout.addWidget(volume_icon)
        volume_layout.addWidget(self.volume_slider)
        right_panel.addLayout(volume_layout)

        # Control Operations
        button_row_layout = QHBoxLayout()
        button_row_layout.setSpacing(10)
        
        self.skip_button = QPushButton("Skip Track ⏭")
        self.skip_button.setObjectName("skipButton")
        self.skip_button.setMinimumHeight(40)
        
        self.export_button = QPushButton("Export Data")
        self.export_button.setObjectName("exportButton")
        self.export_button.setMinimumHeight(40)
        
        button_row_layout.addWidget(self.skip_button)
        button_row_layout.addWidget(self.export_button)
        right_panel.addLayout(button_row_layout)

        right_panel.addStretch()

        # --- COMPILE LAYOUTS ---
        master_layout.addLayout(left_panel, stretch=2)
        master_layout.addLayout(right_panel, stretch=1)
        
        self.setLayout(master_layout)

        # --- Start the Background Thread ---
        self.thread = SystemPipelineThread(db_path, annoy_index_path, self.player, self.audio_output)
        self.thread.update_ui_signal.connect(self.update_gui)

        self.thread.progress_signal.connect(self.update_playback_progress)
        self.volume_slider.valueChanged.connect(self.thread.set_volume)

        self.thread.pipeline_request_play_signal.connect(self.execute_main_thread_play)

        self.thread.start()

        self.skip_button.clicked.connect(self.thread.request_skip)
        self.export_button.clicked.connect(self.export_telemetry)
        
        
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

    def format_time(self, seconds):
        """Converts raw numerical seconds into standard MM:SS readout strings."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def update_playback_progress(self, elapsed, total):
        """Updates the media tracking sliders based on internal VLC playback metrics."""
        self.time_elapsed_label.setText(self.format_time(elapsed))
        self.time_total_label.setText(self.format_time(total))
        
        if total > 0:
            percentage = int((elapsed / total) * 100)
            self.progress_slider.setValue(percentage)

    def export_telemetry(self):
        """Extracts runtime logs from SQLite and compile a standardized evaluation CSV."""
        
        # Generate the destination file path right inside the current dataset folder
        db_source = self.thread.db_path
        output_dir = os.path.dirname(db_source)
        output_csv = os.path.join(output_dir, "usability_test_results.csv")
        
        print(f"[UI] Initializing telemetry data dump from {db_source}...")
        conn = sqlite3.connect(db_source)

        try:
            df = get_telemetry(conn)
            df.to_csv(output_csv, index=False)
            
            self.export_button.setText(f"Export Complete at {output_csv}")
            self.export_button.setStyleSheet("background-color: #059669; color: white;")
            print(f"[SUCCESS] Telemetry compiled cleanly: {output_csv}")
        except Exception as e:
            print(f"[ERROR] Native export compilation broke: {e}")
            self.export_button.setText("Export Failed!")
            self.export_button.setStyleSheet("background-color: #dc2626; color: white;")
        finally:
            conn.close()

    @Slot(str)
    def execute_main_thread_play(self, local_file_path):
        abs_path = os.path.abspath(local_file_path)
        
        self.player.setSource(QUrl.fromLocalFile(abs_path))
        self.player.play()

        current_vol = self.volume_slider.value()
        self.thread.audio_player.change_volume(current_vol)
        
        print(f"[MAIN PLAYBACK] Core event loop executing tracking file: {abs_path}")

       
    
    def closeEvent(self, event):
        self.thread.terminate()
        event.accept()

def main():
    env_db = os.getenv("SQLITE_DATABASE", "music_system.db")
    env_ann = os.getenv("ANNOY_INDEX", "music_vectors.ann")
    
    db_file = os.path.abspath(env_db)
    ann_file = os.path.abspath(env_ann)

    if not os.path.exists(db_file) or not os.path.exists(ann_file):
        print("[FIRST-RUN] Application assets missing. Beginning automated environment compilation...")
        setup_database(db_path=db_file, ann_path=ann_file)
        print("[FIRST-RUN] Environment compiled successfully.")

    app = QApplication(sys.argv)
    window = App(db_file, ann_file)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()