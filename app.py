from flask import Flask, request, jsonify
import json
import cv2 as cv
import numpy as np
import os
import base64

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './faces/users/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_image(image):
    # Resize the image to 400x400
    image = cv.resize(image, (600, 600))
    return image

# Function to register a user
def register_user(email, password):
    if not email or not password:
        return {"message": "Email and password are required."}, 400

    users = read_users()
    if any(user['email'] == email for user in users):
        return {"message": "User already exists."}, 400

    users.append({'email': email, 'password': password})
    write_users(users)

    return {"message": "User registered successfully!"}, 201

def face_detector(image):
    # Add your face detection logic here
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return image[y:y+h, x:x+w], (x, y, w, h)
    return None, (0, 0, 0, 0)

@app.route('/train', methods=['POST'])
def train():
    data = request.json
    email = data.get('email')
    password = data.get('password')
    images = data.get('images')

    # Validate email and password
    if not email or not password:
        return jsonify({"error": "Email or password missing"}), 400

    # Validate provided images
    if not images:
        return jsonify({"error": "No images provided"}), 400

    # Ensure upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)

    count = 0
    faceFound = False

    for img_str in images:
        # Remove data URL scheme and adjust padding for base64 string
        if img_str.startswith('data:image/png;base64,'):
            img_str = img_str.replace('data:image/png;base64,', '')
        padding = len(img_str) % 4
        if padding:
            img_str += '=' * (4 - padding)

        try:
            nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
            img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        except Exception as e:
            return jsonify({"error": f"Error decoding image: {str(e)}"}), 400

        if img is None:
            return jsonify({"error": "Error decoding image"}), 400

        # Convert image to grayscale and detect faces
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face, coord = face_detector(gray_img)
        if face is not None:
            face = preprocess_image(face)
            count += 1
            faceFound = True
            image_name_path = os.path.join(UPLOAD_FOLDER, f"{email}_{count}.jpg")
            cv.imwrite(image_name_path, face)
        else:
            count += 1
            print(f"No face detected in image {count}")

    # Ensure at least one face was detected
    if not faceFound:
        return jsonify({"error": "No faces detected"}), 400

    data_path = UPLOAD_FOLDER
    onlyfiles = [f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))]

    training_data, labels = [], []
    labels_to_name = {}

    for i, file in enumerate(onlyfiles):
        print(f"Processing file: {file}")
        image_path = os.path.join(data_path, file)
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        if img is None:
            continue
        # Resize images to a uniform size
        if img.shape != (600, 600):
            img = cv.resize(img, (600, 600))
        training_data.append(np.asarray(img, dtype=np.uint8))
        labels.append(i)
        file_email, _ = file.split("_")
        labels_to_name[i] = {"email": file_email, "password": password}

    # Debug statements for training data
    print("Training data before conversion to numpy array:", [td.shape for td in training_data])
    print("Labels before conversion to numpy array:", labels)

    try:
        if count == 0:
            return jsonify({"error": "No faces detected"}), 400
        labels = np.asarray(labels, dtype=np.int32)
        training_data = np.asarray(training_data, dtype=np.uint8)
    except Exception as e:
        return jsonify({"error": f"Error converting to numpy array: {str(e)}"}), 400

    try:
        # Train the face recognition model
        model = cv.face.LBPHFaceRecognizer_create()
        model.train(training_data, labels)
        model.write("trainer/model.xml")

        # Save label-to-name mapping with error handling
        try:
            with open("labels_to_name.json", "w") as write_file:
                json.dump(labels_to_name, write_file)
                # registration_response = register_user(file_email, password)
                # return jsonify({"message": "Model trained successfully"})
        except Exception as e:
            return jsonify({"error": f"Error writing to JSON file: {str(e)}"}), 500
    
        # Register the user after training
        registration_response, status_code = register_user(email, password)
        if status_code == 201:
            return jsonify({
                "message": "Model trained successfully and user registered!",
                "registration_message": registration_response['message']
            })
        else:
            return jsonify({
                "message": "Model trained successfully, but registration failed.",
                "registration_error": registration_response['message']
            }), 400
    except Exception as e:
        return jsonify({"error": f"Error during model training: {str(e)}"}), 500



@app.route('/match-faces', methods=['POST'])
def matchFace():
    images = request.json['images']  # Accept multiple images
    success_response = None

    with open("labels_to_name.json", "r") as read_file:
        labels_to_name = json.load(read_file)

    model = cv.face.LBPHFaceRecognizer_create()
    model.read("trainer/model.xml")

    for img_str in images:
        nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        if img is None:
            continue  # Skip to the next image if decoding fails

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        face, coord = face_detector(gray_img)
        if face is None:
            continue  # Skip to the next image if no face is detected

        face = preprocess_image(face)
        results = model.predict(face)
        confidence = int(100 * (1 - (results[1] / 300)))

        if results[1] < 500:
            matched_id = str(results[0])
            email = labels_to_name[matched_id]["email"]
            password = labels_to_name[matched_id]["password"]
            success_response = {
                "email": email,
                "confidence": confidence,
                "password": password
            }
            break  # Stop the loop once a match is found

    if success_response:
        return jsonify(success_response)
    else:
        return jsonify({"error": "No face recognized in any of the provided images"}), 400
    


@app.route('/recognize', methods=['POST'])
def recognize():
    img_str = request.json['image']
    nparr = np.frombuffer(base64.b64decode(img_str), np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Error decoding image"}), 400

    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    face, coord = face_detector(gray_img)
    if face is None:
        return jsonify({"error": "No face detected"}), 400

    with open("labels_to_name.json", "r") as read_file:
        labels_to_name = json.load(read_file)

    face = preprocess_image(face)
    model = cv.face.LBPHFaceRecognizer_create()
    model.read("trainer/model.xml")

    results = model.predict(face)
    confidence = int(100 * (1 - (results[1] / 300)))
    if results[1] < 500:
        matched_id = str(results[0])
        email = labels_to_name[matched_id]["email"]
        password = labels_to_name[matched_id]["password"]
        print("result is " + str(confidence))
        return jsonify({"email": email, "confidence": confidence, "password": password})
    


USERS_FILE_PATH = './users/users.json'


def read_users():
    # Check if the file exists and is not empty
    if not os.path.exists(USERS_FILE_PATH) or os.path.getsize(USERS_FILE_PATH) == 0:
        return []  # Return an empty list if the file does not exist or is empty

    try:
        with open(USERS_FILE_PATH, 'r') as file:
            return json.load(file)  # Try loading the JSON data
    except json.JSONDecodeError:
        # Handle JSON decoding errors, such as when the file is empty or malformed
        print("Error: Failed to read users.json, returning empty user list.")
        return []  # Return an empty list if there's a JSON decoding error

# Helper function to write users to the file
def write_users(users):
    with open(USERS_FILE_PATH, 'w') as file:
        json.dump(users, file, indent=4)

# Endpoint to register a new user
@app.route('/register', methods=['POST'])
def register():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required.'}), 400

    users = read_users()
    if any(user['email'] == email for user in users):
        return jsonify({'message': 'User already exists.'}), 400

    users.append({'email': email, 'password': password})
    write_users(users)

    return jsonify({'message': 'User registered successfully!'}), 201




# Endpoint to login a user
@app.route('/login', methods=['POST'])
def login():
    data = request.json
    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'message': 'Email and password are required.'}), 400

    users = read_users()
    user = next((user for user in users if user['email'] == email and user['password'] == password), None)

    if user:
        return jsonify({'message': 'Login successful!'}), 200
    else:
        return jsonify({'message': 'Invalid email or password.'}), 401



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port)
    # app.run(host='0.0.0.0', port=5001, debug=True)
    # 
    
