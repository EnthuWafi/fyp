# system_pipeline.py
import cv2
import time
import numpy as np
from PySide6.QtCore import Signal, QThread

# Import your cleanly separated components
from emotion_models import FaceChannelEmotionModel
from music_regulator import IsoPrincipleRegulator

class SystemPipelineThread(QThread):
    update_ui_signal = Signal(np.ndarray, float, float, str)

    def run(self):
        cap = cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize your components
        emotion_model = FaceChannelEmotionModel()
        music_regulator = IsoPrincipleRegulator()
        
        last_music_update = time.time()
        current_track = "Initializing..."

        while True:
            ret, frame = cap.read()
            if not ret: continue

            # 1. FIND THE FACE (Handled by the pipeline)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            val, aro = 0.0, 0.0

            if len(faces) > 0:
                # Grab the first face
                x, y, w, h = faces[0]
                
                # Draw the bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Crop the face and pass it to the model
                cropped_face = frame[y:y+h, x:x+w]
                emotion_data = emotion_model.predict(cropped_face)
                
                val = emotion_data['valence']
                aro = emotion_data['arousal']

                # 2. REGULATE THE MUSIC
                if time.time() - last_music_update > 3:
                    current_track = music_regulator.select_track(val, aro)
                    last_music_update = time.time()

            # 3. UPDATE THE GUI
            self.update_ui_signal.emit(frame, val, aro, current_track)