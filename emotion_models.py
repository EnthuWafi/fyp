# emotion_models.py
import cv2
import numpy as np
import onnxruntime as ort
from keras.applications.mobilenet_v2 import preprocess_input

# --- MODEL 1: FACECHANNEL ---
class FaceChannelEmotionModel:
    def __init__(self, model_path='facechannel.onnx'):
        print("[INFO] Loading FaceChannel via ONNX Runtime Engine...")
        # Force CPU execution to bypass any residual GPU conflicts
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        
        # Dynamically check what shape the ONNX engine expects
        self.input_shape = self.session.get_inputs()[0].shape
        print(f"[INFO] ONNX Engine expecting input shape: {self.input_shape}")

    def predict(self, raw_color_crop):
        """ Fully Robust ONNX Preprocessing """
        
        # 1. Defensive Grayscale Check
        if len(raw_color_crop.shape) == 3 and raw_color_crop.shape[2] == 3:
            gray_face = cv2.cvtColor(raw_color_crop, cv2.COLOR_BGR2GRAY)
        else:
            gray_face = raw_color_crop
            
        # 2. Resize and Normalize
        face_resized = cv2.resize(gray_face, (64, 64))
        face_normalized = face_resized.astype('float32') / 255.0
        
        # 3. Adapt to NCHW format
        if self.input_shape[1] == 1:
            face_ready = np.reshape(face_normalized, (1, 1, 64, 64))
        else:
            face_ready = np.reshape(face_normalized, (1, 64, 64, 1))
            
        # 4. Predict instantly via ONNX
        predictions = self.session.run(None, {self.input_name: face_ready})
        
        # Parse the ONNX output list: [array([[arousal]]), array([[valence]])]
        try:
            arousal = float(predictions[0][0][0])
            valence = float(predictions[1][0][0])
        except Exception as e:
            print(f"[WARNING] Failed to parse ONNX output: {e}")
            arousal, valence = 0.0, 0.0

        return {"valence": round(valence, 2), "arousal": round(arousal, 2)}

# --- MODEL 2: YOUR FUTURE PROJECT ---
class MobileNetV2EmotionModel:
    def __init__(self, model_path='my_final_year_project_model.h5'):
        from keras.models import load_model
        self.model = load_model(model_path)

    def predict(self, raw_color_crop):
        """ MobileNetV2 specific preprocessing """
        # 1. MobileNet needs RGB (Color), so we skip grayscale conversion!
        # MobileNet usually needs 224x224
        face_resized = cv2.resize(raw_color_crop, (224, 224))
        
        # 2. Use MobileNet's highly specific mathematical normalization (-1 to 1)
        face_ready = preprocess_input(np.expand_dims(face_resized, axis=0))
        
        # 3. Predict
        prediction = self.model.predict(face_ready)
        return {"valence": float(prediction[0][0]), "arousal": float(prediction[0][1])}