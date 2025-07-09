import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

def load_age_model(model_path="models/weights.28-3.73.hdf5"):
    model = load_model(model_path, compile=False)  # ✅ Skip optimizer
    return model

# Detect face and crop
def detect_face(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = img[y:y+h, x:x+w]
    face = cv2.resize(face, (64, 64))
    return face

# Predict age
def predict_age(image_path, model):
    face = detect_face(image_path)
    if face is None:
        return "No face detected"

    face = np.array(face).astype('float32') / 255.0
    face = np.expand_dims(face, axis=0)
    age = model.predict(face)[0][0]
    return f"{int(age)} years"
