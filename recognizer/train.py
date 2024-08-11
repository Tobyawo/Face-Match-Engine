import cv2 as cv
import numpy as np
import base64
import json
from io import BytesIO
from PIL import Image
import os

def preprocess_image(img):
    return cv.resize(img, (200, 200))

def face_detector(img):
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None
    x, y, w, h = faces[0]
    return img[y:y+h, x:x+w], (x, y, w, h)

def train_model(images, email, model_path, label_map_path):
    recognizer = cv.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}

    # Load existing model if it exists
    if os.path.exists(model_path):
        recognizer.read(model_path)
        if os.path.exists(label_map_path):
            with open(label_map_path, 'rb') as f:
                label_map = np.load(f, allow_pickle=True).item()

    label_counter = max(label_map.keys(), default=-1) + 1

    for img_str in images:
        img_data = base64.b64decode(img_str)
        img = Image.open(BytesIO(img_data))
        img_gray = cv.cvtColor(np.array(img), cv.COLOR_BGR2GRAY)
        
        face, coord = face_detector(img_gray)
        if face is not None:
            face = preprocess_image(face)
            faces.append(face)
            labels.append(label_counter)

    label_map[label_counter] = email

    if faces and labels:
        recognizer.update(faces, np.array(labels))
    else:
        recognizer.train(faces, np.array(labels))

    recognizer.save(model_path)
    with open(label_map_path, 'wb') as f:
        np.save(f, label_map)
