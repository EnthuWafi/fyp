# system_pipeline.py
import cv2
import time
import numpy as np
import threading
from collections import deque

from PySide6.QtCore import Signal, QThread
from youtube_player import YouTubeQueuePlayer

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from emotion_models import EmotionModel
from music_regulator import IsoPrincipleRegulator

class SystemPipelineThread(QThread):
    update_ui_signal = Signal(np.ndarray, float, float, float, float, float, float, str, str, str) 

    def __init__(self):
        super().__init__()
        self.mediapipe_path = './models/blaze_face_full_range_sparse.tflite'

        self.skip_flag = False

        self.lock = threading.Lock()
        self.latest_bbox = None 

        self.anger_start_time = None
        self.emergency_engaged = False
    
    def mp_callback(self, result: vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        """This function runs automatically on a background thread when MediaPipe finds a face."""
        if result.detections:
            bbox = result.detections[0].bounding_box
            # Save the coordinates safely
            with self.lock:
                self.latest_bbox = [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height]
        else:
            with self.lock:
                self.latest_bbox = None


    def run(self):
        camera_index = 1
        cap = cv2.VideoCapture(camera_index)
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # make face detector
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        emotion_model = EmotionModel()
        music_regulator = IsoPrincipleRegulator()
        audio_player = YouTubeQueuePlayer()

        start_time = time.time()
        last_ml_time = 0
        last_print_time = time.time()
        
        current_protocol = "Initializing..."
        current_emotion = "Neutral"
        music_exact_va_data = [0.0, 0.0]
        
        raw_v, raw_a = 0.0, 0.0
        engine_status = "Waiting for face..."
        
    

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.mediapipe_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.mp_callback)

        with FaceDetector.create_from_options(options) as detector:            
            while True:
                ret, frame = cap.read()
                if not ret: continue
                
                current_time = time.time()
                
                # --- FACE TRACKING  ---
                # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
                timestamp_ms = int((current_time - start_time) * 1000)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detector.detect_async(mp_image, timestamp_ms)

                with self.lock:
                    local_bbox = self.latest_bbox

                if local_bbox:
                    x, y, w, h = local_bbox
                    # margin = int(h * 0.1)
                    # x_m, y_m = max(0, x - margin), max(0, y - margin)
                    # w_m, h_m = min(frame.shape[1] - x_m, w + 2*margin), min(frame.shape[0] - y_m, h + 2*margin)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # --- ML INFERENCE ---
                    # Only runs if a face is present AND 0.1 seconds have passed
                    if (current_time - last_ml_time) >= 0.1: 
                        cropped_face = frame[y:y+h, x:x+w]
                        if cropped_face.size > 0:
                            emotion_data = emotion_model.predict(cropped_face)
                            raw_v = emotion_data['valence']
                            raw_a = emotion_data['arousal']
                            engine_status = emotion_data['engine']
                            current_emotion = self.va_to_emotion(raw_a, raw_v)
                        last_ml_time = current_time

                # --- PRINT---
                if (current_time - last_print_time) >= 1.0:
                    print(f"[MODEL] Engine: {engine_status} | V: {raw_v:.2f} | A: {raw_a:.2f}")
                    last_print_time = current_time

                if self.skip_flag:
                    self.skip_flag = False
                    print("[SYSTEM] Skip triggered by user!")
                    
                    total_duration = audio_player.current_duration
                    time_played = audio_player.get_elapsed_time()

                    music_regulator.evaluate_feedback(raw_v, raw_a)
                    music_regulator.db.log_playback(music_regulator.current_track_id, time_played, total_duration, explicit_skip=True)
                    
                    audio_player.player.stop()

                    if audio_player.play_next_in_queue():
                        # We already prefetched the next song. Play it instantly
                        print(f"[SYSTEM] Now Playing: {audio_player.current_track}")   
                    else:
                        # We don't have a song queued yet. 
                        print(f"[SYSTEM] No song queued yet!")  
                
                # REGULATION LOGIC
                time_left = audio_player.get_time_remaining()
                
                # Emergency Override
                if current_emotion == "Angry / Stressed":
                    if self.anger_start_time is None:
                        self.anger_start_time = current_time
                        
                    # Evaluate explicit temporal threshold constraint
                    if (current_time - self.anger_start_time) >= 10.0 and not self.emergency_engaged:
                        print("[SAFETY INTERRUPT] Continuous driver distress detected for 10s. Engaging Emergency Calm Protocol.")
                        self.emergency_engaged = True
                        
                        self.fade_out(audio_player)
                        current_protocol, next_track_string, music_exact_va_data = music_regulator.select_track(raw_v, raw_a, force_calm=True)
                        print(f"[SYSTEM] Now Fetching '{next_track_string}'...")
                        
                        audio_player.current_track = None 
                        audio_player.prefetch_song(next_track_string)
                else:
                    # Reset variables
                    self.anger_start_time = None
                    self.emergency_engaged = False
                

                if audio_player.current_track is None:

                    if not audio_player.is_fetching and audio_player.next_stream_url is None:
                        current_protocol, initial_track, music_exact_va_data = music_regulator.select_track(raw_v, raw_a)

                        print(f"[SYSTEM] Cold Start: Fetching '{initial_track}'...")
                        audio_player.prefetch_song(initial_track)

                    elif not audio_player.is_fetching:
                        if audio_player.play_next_in_queue():
                            # We already prefetched the next song. Play it instantly
                            print(f"[SYSTEM] Now Playing: {audio_player.current_track}")      
                        else:
                            # We don't have a song queued yet. 
                            print(f"[SYSTEM] No song queued yet!")  
                    
                else:

                    if time_left > 0 and time_left < 45 and not audio_player.is_fetching and audio_player.next_stream_url is None:

                        music_regulator.evaluate_feedback(raw_v, raw_a)
                        current_protocol, next_track_string, music_exact_va_data = music_regulator.select_track(raw_v, raw_a)

                        print(f"[SYSTEM] Queueing up next track: {next_track_string}...")
                        audio_player.prefetch_song(next_track_string)

                    if time_left <= 0 and not audio_player.is_fetching and audio_player.next_stream_url is not None: 

                        total_duration = audio_player.current_duration
                        time_played = audio_player.current_duration

                        music_regulator.evaluate_feedback(raw_v, raw_a)
                        music_regulator.db.log_playback(music_regulator.current_track_id, time_played, total_duration, explicit_skip=False)
                    

                        if audio_player.play_next_in_queue():
                            # We already prefetched the next song. Play it instantly
                            print(f"[SYSTEM] Now Playing: {audio_player.current_track}")   
                        else:
                            # We don't have a song queued yet. 
                            print(f"[SYSTEM] No song queued yet!") 

                target_v = music_regulator.target_v if music_regulator.target_v is not None else 0.0
                target_a = music_regulator.target_a if music_regulator.target_a is not None else 0.0

                music_v = music_exact_va_data[0]
                music_a = music_exact_va_data[1]

                self.update_ui_signal.emit(
                    frame, raw_v, raw_a, target_v, target_a, music_v, music_a,
                    str(audio_player.current_track), 
                    current_protocol, current_emotion)

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

    def fade_out(self, audio_player: YouTubeQueuePlayer):
        # audio fade-out sequence using VLC volume adjustments
        initial_vol = audio_player.player.audio_get_volume()
        for vol in range(initial_vol, -1, -5):
            audio_player.player.audio_set_volume(vol)
            # Progressively drops audio levels over ~1.5 seconds
            time.sleep(0.05) 
            
            
        audio_player.player.stop()
        audio_player.player.audio_set_volume(initial_vol)

    def request_skip(self):
        """Called by main to request a track skip."""
        self.skip_flag = True

            