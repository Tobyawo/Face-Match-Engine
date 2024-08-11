import cv2
import numpy as np

def recognize_face(img, model_path, label_map_path):
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    with open(label_map_path, 'rb') as f:
        label_map = np.load(f, allow_pickle=True).item()

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    label, confidence = recognizer.predict(gray_img)

    return label_map.get(label, "Unknown"), confidence
