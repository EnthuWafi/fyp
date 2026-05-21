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
    update_ui_signal = Signal(np.ndarray, float, float, str, str, str) 

    def __init__(self):
        super().__init__()
        self.skip_flag = False

    def run(self):
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        emotion_model = EmotionModel()
        music_regulator = IsoPrincipleRegulator()
        audio_player = YouTubeQueuePlayer()

        # 60-second Macro Buffers
        v_buffer = deque(maxlen=60)
        a_buffer = deque(maxlen=60)
        
        # --- THE DECOUPLED TIMERS ---
        last_ml_time = 0
        last_macro_time = time.time()
        
        current_protocol = "Initializing..."
        current_emotion = "Neutral"
        
        raw_v, raw_a = 0.0, 0.0
        engine_status = "Waiting for face..."
    
        while True:
            ret, frame = cap.read()
            if not ret: continue
            
            current_time = time.time()
            
            # --- 1. FAST CLOCK: FACE TRACKING (Runs at 30 FPS for the GUI) ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                margin = int(h * 0.1)
                x_m, y_m = max(0, x - margin), max(0, y - margin)
                w_m, h_m = min(frame.shape[1] - x_m, w + 2*margin), min(frame.shape[0] - y_m, h + 2*margin)

                # Instantly draw the bounding box so the video feed feels perfectly smooth
                cv2.rectangle(frame, (x_m, y_m), (x_m+w_m, y_m+h_m), (0, 255, 0), 2)
                
                # --- 2. MEDIUM CLOCK: ML INFERENCE (Locked to 10 FPS max) ---
                # Only runs if a face is present AND 0.1 seconds have passed
                if (current_time - last_ml_time) >= 0.1: 
                    cropped_face = frame[y_m:y_m+h_m, x_m:x_m+w_m]
                    if cropped_face.size > 0:
                        emotion_data = emotion_model.predict(cropped_face)
                        raw_v = emotion_data['valence']
                        raw_a = emotion_data['arousal']
                        engine_status = emotion_data['engine']
                        current_emotion = self.va_to_emotion(raw_a, raw_v)
                    last_ml_time = current_time

            # --- 3. SLOW CLOCK: MACRO BUFFER (Locked to 1 Hz) ---
            if (current_time - last_macro_time) >= 1.0:
                v_buffer.append(raw_v)
                a_buffer.append(raw_a)
                print(f"[MODEL] Engine: {engine_status} | V: {raw_v:.2f} | A: {raw_a:.2f}")
                last_macro_time = current_time

            if self.skip_flag:
                self.skip_flag = False
                print("[SYSTEM] Skip triggered by user!")
                
                audio_player.player.stop() # Immediately halt VLC
                
                if audio_player.next_stream_url is not None:
                    # Scenario A: We already prefetched the next song. Play it instantly!
                    audio_player.play_next_in_queue()
                    print(f"[SYSTEM] Now Playing: {audio_player.current_track}")
                else:
                    # Scenario B: We don't have a song queued yet. 
                    # Wipe the current track so the system falls back into 'Cold Start' mode
                    # and fetches a fresh song based on your current mood.
                    audio_player.current_track = None
                    audio_player.start_time = 0
            
            # --- 4. REGULATION LOGIC ---
            time_left = audio_player.get_time_remaining()
            
            if audio_player.current_track is None and len(v_buffer) >= 1:
                if not audio_player.is_fetching and audio_player.next_stream_url is None:
                    avg_v = sum(v_buffer) / len(v_buffer)
                    avg_a = sum(a_buffer) / len(a_buffer)
                    current_protocol, initial_track = music_regulator.select_track(avg_v, avg_a)
                    print(f"[SYSTEM] Cold Start: Fetching '{initial_track}'...")
                    audio_player.prefetch_song(initial_track)
                elif not audio_player.is_fetching and audio_player.next_stream_url is not None:
                    audio_player.play_next_in_queue()
                    print(f"[SYSTEM] Now Playing: {audio_player.current_track}")
                
            elif len(v_buffer) >= 5:
                avg_v = sum(v_buffer) / len(v_buffer)
                avg_a = sum(a_buffer) / len(a_buffer)
                if time_left > 0 and time_left < 45 and not audio_player.is_fetching and audio_player.next_stream_url is None:
                    music_regulator.evaluate_feedback(avg_a)
                    current_protocol, next_track_string = music_regulator.select_track(avg_v, avg_a)
                    print(f"[SYSTEM] Queueing up next track: {next_track_string}...")
                    audio_player.prefetch_song(next_track_string)
                if time_left <= 0 and not audio_player.is_fetching and audio_player.next_stream_url is not None: 
                    audio_player.play_next_in_queue()
                    print(f"[SYSTEM] Now Playing: {audio_player.current_track}")

            # --- 5. GUI UPDATE (Runs at 30 FPS) ---
            self.update_ui_signal.emit(frame, raw_v, raw_a, str(audio_player.current_track), current_protocol, current_emotion)
    def va_to_emotion(self, arousal, valence):
        distance_from_middle = 0.05

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
    
    def request_skip(self):
        """Called by the PyQt GUI to safely request a track skip."""
        self.skip_flag = True

            