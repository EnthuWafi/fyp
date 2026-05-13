# system_pipeline.py
import cv2
import time
import numpy as np
from collections import deque
from PySide6.QtCore import Signal, QThread
from youtube_player import YouTubeQueuePlayer

from emotion_models import EmotionModel
from music_regulator import IsoPrincipleRegulator

class SystemPipelineThread(QThread):
    # Update signal to accept both protocol and track strings
    update_ui_signal = Signal(np.ndarray, float, float, str, str, str) 

    def run(self):

        camera_index = 1

        cap = cv2.VideoCapture(camera_index)

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        emotion_model = EmotionModel()
        music_regulator = IsoPrincipleRegulator()
        audio_player = YouTubeQueuePlayer()

        # 60-second Smoothing Buffer
        v_buffer = deque(maxlen=60)
        a_buffer = deque(maxlen=60)
        
        last_sample_time = time.time()
        last_music_update = time.time()
        last_va_time = time.time() #for va
        
        current_protocol = "Initializing..."
        current_emotion = "Neutral"

        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            # --- 1. UI LAYER: FAST FACE TRACKING ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
            # --- 1 Hz SAMPLING RATE ---
            # Limits capturing the face only once per second instead of many times
            current_time = time.time()
            if current_time - last_sample_time >= 1.0:
                last_sample_time = current_time

                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    # Draw bounding box for the GUI
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Crop the COLOR frame to send to ONNX
                    cropped_face = frame[y:y+h, x:x+w]
                    
                    # 2. PREDICT
                    emotion_data = emotion_model.predict(cropped_face)
                    
                    # Add to buffer
                    v_buffer.append(emotion_data['valence'])
                    a_buffer.append(emotion_data['arousal'])
            
            # --- REGULATION LOGIC ---

            # Check how much time is left on the current song
            time_left = audio_player.get_time_remaining()
            
            # STATE 1: START (Nothing is playing yet)
            if audio_player.current_track is None and len(v_buffer) >= 1:
                
                # Step A: Initiate the download (Only if we aren't already downloading!)
                if not audio_player.is_fetching and audio_player.next_stream_url is None:
                    avg_v = sum(v_buffer) / len(v_buffer)
                    avg_a = sum(a_buffer) / len(a_buffer)
                    
                    current_protocol, initial_track = music_regulator.select_track(avg_v, avg_a)
                    print(f"[SYSTEM] Cold Start: Downloading '{initial_track}'...")
                    audio_player.prefetch_song(initial_track)
                
                # Step B: Play it ONLY after the background download finishes
                elif not audio_player.is_fetching and audio_player.next_stream_url is not None:
                    audio_player.play_next_in_queue()
                    print(f"[SYSTEM] Now Playing: {audio_player.current_track}")
                
            # STATE 2: RUNNING STATE (A song is currently playing)
            elif len(v_buffer) >= 5:
                avg_v = sum(v_buffer) / len(v_buffer)
                avg_a = sum(a_buffer) / len(a_buffer)
                
                # Condition 1: Buffer the NEXT song (45+ seconds left)
                if time_left < 45 and not audio_player.is_fetching and audio_player.next_stream_url is None:

                    music_regulator.evaluate_feedback(avg_a)

                    current_protocol, next_track_string = music_regulator.select_track(avg_v, avg_a)
                    print(f"[SYSTEM] Queueing up next track: {next_track_string}...")
                    audio_player.prefetch_song(next_track_string)

                # Condition 2: The song is ending. Transition!
                if time_left <= 1 and not audio_player.is_fetching and audio_player.next_stream_url is not None: 
                    audio_player.play_next_in_queue()
                    print(f"[SYSTEM] Now Playing: {audio_player.current_track}")

            # Update GUI
            
            raw_v = round(v_buffer[-1], 2) if len(v_buffer) > 0 else 0.0
            raw_a = round(a_buffer[-1], 2) if len(a_buffer) > 0 else 0.0

            current_emotion = self.va_to_emotion(raw_a, raw_v)
            
            
            self.update_ui_signal.emit(frame, raw_v, raw_a, audio_player.current_track, current_protocol, current_emotion)

    def va_to_emotion(self, arousal, valence):
        distance_from_middle = 0.1

        if abs(arousal) <= distance_from_middle and abs(valence) <= distance_from_middle:
            return "Neutral"
        elif arousal > distance_from_middle and valence > distance_from_middle:
            return "Happy / Excited"
        elif arousal < -distance_from_middle and valence > distance_from_middle:
            return "Calm / Relaxed"
        elif arousal > distance_from_middle and valence < -distance_from_middle:
            return "Angry / Stressed"
        elif arousal < -distance_from_middle and valence < -distance_from_middle:
            return "Sad / Fatigued"
        else:
            return "Transitioning..."

            