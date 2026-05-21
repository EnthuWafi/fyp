import cv2
import numpy as np
import os
import time
from ai_edge_litert.interpreter import Interpreter
from collections import deque

class EmotionModel:
    def __init__(self, 
                 enet_path='models/enet_b0_8_best_vgaf_float32.tflite', 
                 gru_path='models/edge_gru_model_hybrid.tflite', 
                 mlp_path='models/enet_mlp_model.tflite'):
        
        print("[INFO] Loading TFLite Edge Models (ENet + GRU + MLP)...")

        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        abs_enet = os.path.join(base_dir, enet_path)
        abs_gru  = os.path.join(base_dir, gru_path)
        abs_mlp  = os.path.join(base_dir, mlp_path)

        self.enet_interp = Interpreter(model_path=enet_path)
        self.gru_interp  = Interpreter(model_path=gru_path)
        self.mlp_interp  = Interpreter(model_path=mlp_path)
        
        self.enet_interp.allocate_tensors()
        self.gru_interp.allocate_tensors()
        self.mlp_interp.allocate_tensors()
        
        # I/O Pointers
        self.enet_in  = self.enet_interp.get_input_details()[0]
        self.enet_out = self.enet_interp.get_output_details()[0]['index']
        
        self.gru_in  = self.gru_interp.get_input_details()[0]['index']
        self.gru_out = self.gru_interp.get_output_details()[0]['index']
        
        self.mlp_in  = self.mlp_interp.get_input_details()[0]['index']
        self.mlp_out = self.mlp_interp.get_output_details()[0]['index']
        
        # The Temporal Memory
        self.history_queue = deque(maxlen=15)
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
        
        # 1. Extract the Raw Logits
        self.enet_interp.set_tensor(self.enet_in['index'], face_input)
        self.enet_interp.invoke()
        raw_logits = self.enet_interp.get_tensor(self.enet_out)[0]
        
        # 2. CONVERT TO SOFTMAX PROBABILITIES 
        e_x = np.exp(raw_logits - np.max(raw_logits))
        scores = e_x / e_x.sum(axis=0)

        # Temporal Queue Update
        current_time = time.time()

        while len(self.history_queue) > 0 and (current_time - self.history_queue[0][0]) > self.max_time_gap:
            self.history_queue.popleft()
                
        self.history_queue.append((current_time, scores))
        
        pooled_scores = np.mean([item[1] for item in self.history_queue], axis=0)
        
        # Calculate the maximal inter-class confidence score
        confidence = np.max(pooled_scores)
        current_l = len(self.history_queue)

            
        if confidence >= self.confidence_threshold:
            # Early Exit. Confidence is high enough to bypass temporal smoothing.
            
            mlp_input = np.expand_dims(pooled_scores, axis=0).astype(np.float32)
            self.mlp_interp.set_tensor(self.mlp_in, mlp_input)
            self.mlp_interp.invoke()
            v_a_prediction = self.mlp_interp.get_tensor(self.mlp_out)[0]
            
            self.last_valid_prediction = {
                "valence": round(float(v_a_prediction[0]), 2), 
                "arousal": round(float(v_a_prediction[1]), 2),
                "engine": f"MLP Early Exit (Conf: {confidence:.2f})"
            }
        elif current_l == 15:    
            # Maximum adjusted frame rate reached. Run the heavy temporal model.
            temporal_sequence = np.array([item[1] for item in self.history_queue], dtype=np.float32)
            gru_input = np.expand_dims(temporal_sequence, axis=0) 
            
            self.gru_interp.set_tensor(self.gru_in, gru_input)
            self.gru_interp.invoke()
            v_a_prediction = self.gru_interp.get_tensor(self.gru_out)[0]
            
            self.last_valid_prediction = {
                "valence": round(float(v_a_prediction[0]), 2), 
                "arousal": round(float(v_a_prediction[1]), 2),
                "engine": "GRU (L=15)"
            }
        else:
            # NO EXIT: Confidence is too low. Continue accumulating frames.
            # Return the last known valid coordinate so the music system doesn't crash.
            self.last_valid_prediction = {
                "valence": self.last_valid_prediction["valence"],
                "arousal": self.last_valid_prediction["arousal"],
                "engine": f"Accumulating (L={current_l}, Conf: {confidence:.2f})"
            }
        return self.last_valid_prediction
