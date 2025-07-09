# utils/detect_dress_color.py

import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

def detect_dress_region(image_path):
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Assume first face detected
    (x, y, fw, fh) = faces[0]

    # Define dress region: below the face, height of 1.5x face height
    dress_y_start = y + fh
    dress_y_end = min(h, dress_y_start + int(1.5 * fh))
    dress_region = img[dress_y_start:dress_y_end, x:x + fw]

    return dress_region

def get_dominant_color(image, k=3):
    if image is None or image.size == 0:
        return "No dress region"

    img = cv2.resize(image, (100, 100))
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(img)

    counts = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_
    dominant_color = center_colors[counts.most_common(1)[0][0]]
    b, g, r = map(int, dominant_color)

    return f"RGB({r}, {g}, {b})"

def predict_dress_color(image_path):
    dress_region = detect_dress_region(image_path)
    return get_dominant_color(dress_region)


