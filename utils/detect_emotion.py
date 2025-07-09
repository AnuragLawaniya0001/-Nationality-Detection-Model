import numpy as np
from keras.models import load_model
from PIL import Image

# Emotion classes
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load model
def load_emotion_model(model_path="models/emotion_detection_model.keras"):
    return load_model(model_path)

# Predict
def predict_emotion(image_path, model):
    img = Image.open(image_path).convert("L").resize((48, 48))
    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 48, 48, 1)

    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)
    return emotion_labels[predicted_index]
