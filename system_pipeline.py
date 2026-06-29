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
    progress_signal = Signal(float, float)

    pipeline_request_play_signal = Signal(str)

    def __init__(self, db_path, annoy_index_path, qt_player, qt_audio):
        super().__init__()
        self.mediapipe_path = './models/blaze_face_full_range_sparse.tflite'

        self.skip_flag = False

        self.lock = threading.Lock()
        self.latest_face_data = None 
        
        self.anger_start_time = None
        self.emergency_engaged = False

        self.db_path = db_path
        self.annoy_index_path = annoy_index_path

        self.qt_player = qt_player
        self.qt_audio = qt_audio

        self.audio_player = YouTubeQueuePlayer(self.qt_player, self.qt_audio, parent=self)
        self.audio_player.request_play_signal.connect(self.pipeline_request_play_signal)

        self.pending_volume = 100

        self.is_running = True

    def stop_pipeline(self):
        """Safely updates the loop state from the main thread interface."""
        with self.lock:
            self.is_running = False

    def set_volume(self, value):
        """Thread-safe setter called directly by the UI layout."""
        with self.lock:
            self.pending_volume = value
    
    def mp_callback(self, result: vision.FaceDetectorResult, output_image: mp.Image, timestamp_ms: int):
        """This function runs automatically on a background thread when MediaPipe finds a face."""
        # if result.detections:
        #     bbox = result.detections[0].bounding_box
        #     # Save the coordinates safely
        #     with self.lock:
        #         self.latest_face_data = [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height]
        # else:
        #     with self.lock:
        #         self.latest_face_data = None
        if result.detections:
            detection = result.detections[0]
            bbox = detection.bounding_box
            kps = detection.keypoints if hasattr(detection, 'keypoints') else []
            
            with self.lock:
                self.latest_face_data = {
                    'bbox': [bbox.origin_x, bbox.origin_y, bbox.width, bbox.height],
                    'keypoints': [(kp.x, kp.y) for kp in kps]
                }
        else:
            with self.lock:
                self.latest_face_data = None


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
        music_regulator = IsoPrincipleRegulator(self.db_path, self.annoy_index_path)

        start_time = time.time()
        last_ml_time = 0
        last_print_time = time.time()
        
        current_protocol = "Initializing..."
        current_emotion = "Neutral"
        music_exact_va_data = [0.0, 0.0]
        
        raw_v, raw_a = 0.0, 0.0
        engine_status = "Waiting for face..."
        
        current_track_db_id = None
        next_track_db_id = None
    

        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=self.mediapipe_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.mp_callback)

        with FaceDetector.create_from_options(options) as detector:            
            while self.is_running:
                ret, frame = cap.read()
                if not ret: continue
                
                current_time = time.time()
                
                timestamp_ms = int((current_time - start_time) * 1000)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                detector.detect_async(mp_image, timestamp_ms)

                with self.lock:
                    local_face = self.latest_face_data

                if local_face:
                    x, y, w, h = local_face['bbox']
                    kps = local_face['keypoints']
                    # margin = int(h * 0.1)
                    # x_m, y_m = max(0, x - margin), max(0, y - margin)
                    # w_m, h_m = min(frame.shape[1] - x_m, w + 2*margin), min(frame.shape[0] - y_m, h + 2*margin)
                    h_max, w_max, _ = frame.shape
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, w_max - x), min(h, h_max - y)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # --- ML INFERENCE ---
                    # Only runs if a face is present AND 0.1 seconds have passed
                    if (current_time - last_ml_time) >= 0.1: 
                        if len(kps) >= 2:
                            # Map normalized keypoint floats to absolute pixel locations
                            p1 = np.array([kps[0][0] * w_max, kps[0][1] * h_max])
                            p2 = np.array([kps[1][0] * w_max, kps[1][1] * h_max])
                            
                            # Calculate the rotational angle between the left and right eyes
                            dx = p2[0] - p1[0]
                            dy = p2[1] - p1[1]
                            angle = np.degrees(np.arctan2(dy, dx))
                            
                            # Standardize direction to maintain an upright head orientation
                            if abs(angle) > 45:
                                angle = 0.0
                            
                            # Compute the transformation center at the midpoint of the eyes
                            eye_center = (float((p1[0] + p2[0]) / 2.0), float((p1[1] + p2[1]) / 2.0))
                            rot_matrix = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
                            
                            # Warp the entire frame to flatten the eye plane horizontally
                            aligned_frame = cv2.warpAffine(frame, rot_matrix, (w_max, h_max))
                            cropped_face = aligned_frame[y:y+h, x:x+w]
                        else:
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
                    
                    total_duration = self.audio_player.current_duration
                    time_played = self.audio_player.get_elapsed_time()

                    music_regulator.evaluate_feedback(raw_v, raw_a)
                    music_regulator.db.log_playback(music_regulator.current_track_id, time_played, total_duration, explicit_skip=True)
                    
                    self.audio_player.qt_player.stop()

                    if self.audio_player.play_next_in_queue():
                        # We already prefetched the next song. Play it instantly
                        print(f"[SYSTEM] Now Playing: {self.audio_player.current_track}")   
                    else:
                        # We don't have a song queued yet. 
                        print(f"[SYSTEM] No song queued yet!")  
                
                # --- AUDIO VOLUME INTERCEPT ---
                with self.lock:
                    if self.pending_volume is not None:
                        self.audio_player.change_volume(self.pending_volume)
                        self.pending_volume = None

                # REGULATION LOGIC
                time_left = self.audio_player.get_time_remaining()
                
                # Emergency Override
                if current_emotion == "Angry / Stressed":
                    if self.anger_start_time is None:
                        self.anger_start_time = current_time
                        
                    # Evaluate explicit temporal threshold constraint
                    if (current_time - self.anger_start_time) >= 10.0 and not self.emergency_engaged:
                        print("[SAFETY INTERRUPT] Continuous driver distress detected for 10s. Engaging Emergency Calm Protocol.")
                        self.emergency_engaged = True
                        
                        self.audio_player.fade_out()
                        next_track_db_id, current_protocol, next_track_string, music_exact_va_data = music_regulator.select_track(raw_v, raw_a, force_calm=True)
                        print(f"[SYSTEM] Now Fetching '{next_track_string}'...")
                        
                        self.audio_player.current_track = None 
                        current_track_db_id = next_track_db_id
                        self.audio_player.prefetch_song(next_track_string)
                else:
                    # Reset variables
                    self.anger_start_time = None
                    self.emergency_engaged = False
                

                # No music yet
                if self.audio_player.current_track is None:

                    if not self.audio_player.is_fetching and self.audio_player.next_stream_url is None:
                        next_track_db_id, current_protocol, initial_track, music_exact_va_data = music_regulator.select_track(raw_v, raw_a)

                        print(f"[SYSTEM] Cold Start: Fetching '{initial_track}'...")
                        self.audio_player.prefetch_song(initial_track)

                    elif not self.audio_player.is_fetching:
                        if self.audio_player.play_next_in_queue():
                            current_track_db_id = next_track_db_id
                            # We already prefetched the next song. Play it instantly
                            print(f"[SYSTEM] Now Playing: {self.audio_player.current_track}")      
                        else:
                            # We don't have a song queued yet. 
                            print(f"[SYSTEM] No song queued yet!")  
                    
                else:

                    if time_left > 0 and time_left < 45 and not self.audio_player.is_fetching and self.audio_player.next_stream_url is None:

                        music_regulator.evaluate_feedback(raw_v, raw_a)
                        next_track_db_id, current_protocol, next_track_string, music_exact_va_data = music_regulator.select_track(raw_v, raw_a)

                        print(f"[SYSTEM] Queueing up next track: {next_track_string}...")
                        self.audio_player.prefetch_song(next_track_string)

                    if time_left <= 0 and not self.audio_player.is_fetching and self.audio_player.next_stream_url is not None: 

                        total_duration = self.audio_player.current_duration
                        time_played = self.audio_player.current_duration

                        music_regulator.evaluate_feedback(raw_v, raw_a)
                        music_regulator.db.log_playback(current_track_db_id, time_played, total_duration, explicit_skip=False)
                    

                        if self.audio_player.play_next_in_queue():
                            # We already prefetched the next song. Play it instantly
                            current_track_db_id = next_track_db_id
                            print(f"[SYSTEM] Now Playing: {self.audio_player.current_track}")   
                        else:
                            # We don't have a song queued yet. 
                            print(f"[SYSTEM] No song queued yet!") 

                target_v = music_regulator.target_v if music_regulator.target_v is not None else 0.0
                target_a = music_regulator.target_a if music_regulator.target_a is not None else 0.0

                music_v = music_exact_va_data[0]
                music_a = music_exact_va_data[1]

                # --- EMIT PROGRESS METRICS ---
                if self.audio_player.current_track is not None:
                    elapsed = self.audio_player.get_elapsed_time()
                    total = float(self.audio_player.current_duration)
                    self.progress_signal.emit(elapsed, total)
                else:
                    self.progress_signal.emit(0.0, 1.0)

                self.update_ui_signal.emit(
                    frame, raw_v, raw_a, target_v, target_a, music_v, music_a,
                    str(self.audio_player.current_track), 
                    current_protocol, current_emotion)
                
            print("[SHUTDOWN] Releasing camera hardware interface...")
            cap.release()
            print("[SHUTDOWN] Closing active SQLite connection registries...")
            music_regulator.close()
            print("[SHUTDOWN] Pipeline thread terminated cleanly.")

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

    def request_skip(self):
        """Called by main to request a track skip."""
        self.skip_flag = True

            