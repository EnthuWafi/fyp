# emotion_models.py
import cv2
import numpy as np
from FaceChannel.FaceChannelV1.FaceChannelV1 import FaceChannelV1
from keras.applications.mobilenet_v2 import preprocess_input

# --- MODEL 1: FACECHANNEL ---
class FaceChannelEmotionModel:
    def __init__(self):
        self.model = FaceChannelV1(type='Dim', loadModel=True)

    def predict(self, raw_color_crop):
        """ FaceChannel specific preprocessing """
        gray_face = cv2.cvtColor(raw_color_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. FaceChannel needs 64x64
        face_resized = cv2.resize(gray_face, (64, 64))
        
        # 3. Simple 0-1 Normalization
        face_normalized = face_resized.astype('float32') / 255.0
        face_ready = np.reshape(face_normalized, (64, 64, 1))
        
        # 4. Predict
        prediction = self.model.predict([face_ready], preprocess=False)
        return {"valence": float(prediction[0][1]), "arousal": float(prediction[0][0])}

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