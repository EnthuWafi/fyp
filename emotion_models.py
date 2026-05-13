import cv2
import numpy as np
import os
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
        # Load Interpreters
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
        
        # The Temporal Memory (Rolls over automatically)
        self.history_queue = deque(maxlen=15)

    def predict(self, raw_color_crop):
        """ PyTorch-style preprocessing and dynamic temporal parsing """
        
        # Prep: Resize and RGB
        face_resized = cv2.resize(raw_color_crop, (224, 224))
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        
        # PyTorch ImageNet Normalization (Strictly required for Savchenko ENet)
        # Formula: image = (image - mean) / std
        face_norm = face_rgb.astype(np.float32) / 255.0
        face_norm = (face_norm - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        
        # Bulletproof Shape Check:
        # onnx2tf usually converts to NHWC (1, 224, 224, 3), but if it kept
        # the original PyTorch NCHW (1, 3, 224, 224), we adapt dynamically.
        expected_shape = self.enet_in['shape']
        if expected_shape[1] == 3: 
            print("NCHW PyTorch Style")
            face_norm = np.transpose(face_norm, (2, 0, 1)) # NHWC -> NCHW
            
        face_input = np.expand_dims(face_norm, axis=0).astype(np.float32)
        
        # 3. ENet Inference (Extract the 8 Scores)
        self.enet_interp.set_tensor(self.enet_in['index'], face_input)
        self.enet_interp.invoke()
        scores = self.enet_interp.get_tensor(self.enet_out)[0]
        
        # 4. Temporal Queue Update
        self.history_queue.append(scores)
        
        # 5. Dynamic Engine Execution (GRU vs MLP)
        if len(self.history_queue) == 15:
            # We have momentum. Use the Temporal Engine.
            gru_input = np.array(self.history_queue, dtype=np.float32)
            gru_input = np.expand_dims(gru_input, axis=0) 
            
            self.gru_interp.set_tensor(self.gru_in, gru_input)
            self.gru_interp.invoke()
            v_a_prediction = self.gru_interp.get_tensor(self.gru_out)[0]
            engine_used = "GRU"
            
        else:
            # Booting up. Use the Adaptive Fallback Engine.
            mlp_input = np.expand_dims(scores, axis=0).astype(np.float32)
            self.mlp_interp.set_tensor(self.mlp_in, mlp_input)
            self.mlp_interp.invoke()
            v_a_prediction = self.mlp_interp.get_tensor(self.mlp_out)[0]
            engine_used = "MLP"
            
        return {
            "valence": round(float(v_a_prediction[0]), 2), 
            "arousal": round(float(v_a_prediction[1]), 2),
            "engine": engine_used
        }