import cv2
import numpy as np
import os
import time
from ai_edge_litert.interpreter import Interpreter
from collections import deque

class EmotionModel:
    def __init__(self, 
                 enet_path='models/enet_b0_8_best_vgaf_float32.tflite', 
                 gru_path='models/edge_gru_modelv3.tflite', 
                 slp_path='models/enet_slp_modelv3.tflite'):
        
        print("[INFO] Loading TFLite Edge Models (ENet + GRU + MLP)...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        abs_enet = os.path.join(base_dir, enet_path)
        abs_gru  = os.path.join(base_dir, gru_path)
        abs_slp  = os.path.join(base_dir, slp_path)

        self.enet_interp = Interpreter(model_path=enet_path)
        self.gru_interp  = Interpreter(model_path=gru_path)
        self.slp_interp  = Interpreter(model_path=slp_path)
        
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
        
        expected_shape = self.enet_in['shape']
        face_input = np.expand_dims(face_norm, axis=0).astype(np.float32)
        
        #Extract the Raw Logits
        self.enet_interp.set_tensor(self.enet_in['index'], face_input)
        self.enet_interp.invoke()
        raw_logits = self.enet_interp.get_tensor(self.enet_out)[0]
        
        # Temporal Queue Update 
        current_time = time.time()
        if len(self.history_queue) > 0:
            time_since_last_frame = current_time - self.history_queue[-1][0]
            if time_since_last_frame > 2.0: # 2 seconds of missing face
                self.history_queue.clear()
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
        else:
            # Early Exit. 
            # This will run if the confidence exceed a certain level, or if it does not have enough
            # frame to run the GRU model

            slp_input = np.expand_dims(raw_logits, axis=0).astype(np.float32)
            self.slp_interp.set_tensor(self.slp_in, slp_input)
            self.slp_interp.invoke()
            v_a_prediction = self.slp_interp.get_tensor(self.slp_out)[0]

            active_engine = f"SLP Early Exit (Conf: {confidence:.2f}, L={current_l}"

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
