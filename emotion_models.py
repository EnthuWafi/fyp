# emotion_models.py
import cv2
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input

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