import cv2
import numpy as np
import time
from ai_edge_litert.interpreter import Interpreter
from collections import deque
from pathlib import Path

class EmotionModel:
    def __init__(self, 
                 enet_path='enet_b0_8_best_vgaf_float32.tflite', 
                 gru_path='edge_gru_modelv3.tflite', 
                 slp_path='enet_slp_modelv3.tflite'):
        
        print("[INFO] Loading TFLite Edge Models (ENet + GRU + MLP)...")

        base_dir = Path(__file__).resolve().parent
        model_dir = base_dir / "models"
        
        abs_enet = model_dir / enet_path
        abs_gru  = model_dir / gru_path
        abs_slp  = model_dir / slp_path

        self.enet_interp = Interpreter(model_path=abs_enet)
        self.gru_interp  = Interpreter(model_path=abs_gru)
        self.slp_interp  = Interpreter(model_path=abs_slp)
        
        self.enet_interp.allocate_tensors()
        self.gru_interp.allocate_tensors()
        self.slp_interp.allocate_tensors()
        
        # I/O Pointers
        self.enet_in  = self.enet_interp.get_input_details()[0]
        self.enet_out = self.enet_interp.get_output_details()[0]['index']
        
        self.gru_in  = self.gru_interp.get_input_details()[0]['index']
        self.gru_out = self.gru_interp.get_output_details()[0]['index']
        
        self.slp_in  = self.slp_interp.get_input_details()[0]['index']
        self.slp_out = self.slp_interp.get_output_details()[0]['index']
        
        # The Temporal Memory
        self.history_queue = deque(maxlen=15)
        self.prediction_buffer = deque(maxlen=15)
        self.max_time_gap = 3.0

        self.confidence_threshold = 0.65 
        
        # Cache the last valid prediction to maintain system stability during accumulation
        self.last_valid_prediction = {"valence": 0.0, "arousal": 0.0, "engine": "Initializing"}

    def predict(self, raw_color_crop):
        """ PyTorch-style preprocessing and dynamic temporal parsing """
        
        # Prep: Resize and RGB
        face_resized = cv2.resize(raw_color_crop, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # PyTorch ImageNet Normalization (Required for Savchenko ENet)
        # Formula: image = (image - mean) / std
        face_norm = face_rgb.astype(np.float32) / 255.0
        face_norm = (face_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        face_input = np.expand_dims(face_norm, axis=0).astype(np.float32)
        
        #Extract the Raw Logits
        self.enet_interp.set_tensor(self.enet_in['index'], face_input)
        self.enet_interp.invoke()
        raw_logits = self.enet_interp.get_tensor(self.enet_out)[0]
        
        # Temporal Queue Update 
        current_time = time.time()
        if len(self.history_queue) > 0:
            time_since_last_frame = current_time - self.history_queue[-1][0]
            if time_since_last_frame > self.max_time_gap:
                self.history_queue.clear()
                self.prediction_buffer.clear()
                print("[WARN] Face tracking lost. Flushing temporal memory.")

        self.history_queue.append((current_time, raw_logits))
        
        current_l = len(self.history_queue)

        e_x = np.exp(raw_logits - np.max(raw_logits))
        current_probs = e_x / e_x.sum(axis=0)
        confidence = np.max(current_probs)

        # confidence should be checked first for early exit
        if confidence < self.confidence_threshold and current_l == 15:    
            # Maximum adjusted frame rate reached. Run the heavy temporal model.
            temporal_sequence = np.array([item[1] for item in self.history_queue], dtype=np.float32)
            gru_input = np.expand_dims(temporal_sequence, axis=0) 
            
            self.gru_interp.set_tensor(self.gru_in, gru_input)
            self.gru_interp.invoke()
            v_a_prediction = self.gru_interp.get_tensor(self.gru_out)[0]
            
            active_engine = "GRU (L=15)"

            self.last_valid_prediction = {
                "valence": round(float(v_a_prediction[0]), 2), 
                "arousal": round(float(v_a_prediction[1]), 2),
                "engine": active_engine
            }

            return self.last_valid_prediction
        else:
            # Early Exit. 
            # This will run if the confidence exceed a certain level, or if it does not have enough
            # frame to run the GRU model

            slp_input = np.expand_dims(raw_logits, axis=0).astype(np.float32)
            self.slp_interp.set_tensor(self.slp_in, slp_input)
            self.slp_interp.invoke()
            v_a_prediction = self.slp_interp.get_tensor(self.slp_out)[0]

            active_engine = f"SLP Early Exit (Conf: {confidence:.2f}, L={current_l})"

            self.prediction_buffer.append((float(v_a_prediction[0]), float(v_a_prediction[1])))
        
            # calc mathematical mean across the prediction history window
            smoothed_v = np.mean([pred[0] for pred in self.prediction_buffer])
            smoothed_a = np.mean([pred[1] for pred in self.prediction_buffer])

            self.last_valid_prediction = {
                "valence": round(float(smoothed_v), 2), 
                "arousal": round(float(smoothed_a), 2),
                "engine": active_engine
            }

            return self.last_valid_prediction
