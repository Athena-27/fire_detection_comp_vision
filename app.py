import os
from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import img_to_array
import joblib
import time
import logging
import winsound
import threading

# Set up basic logging
logging.basicConfig(level=logging.DEBUG)

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Load the pre-trained VGG16 model for feature extraction
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Load the pre-trained SVM model
model_load_path = r'C:\Users\eshwa\OneDrive\Documents\svm_fire_detection_model____3.pkl'
if not os.path.exists(model_load_path):
    raise FileNotFoundError(f"The model file does not exist: {model_load_path}")

svm_model = joblib.load(model_load_path)

app = Flask(__name__)

fire_detected = False
fire_probability = 0.0
alarm_ringing = False

def preprocess_frame(frame):
    """Preprocess the frame using OpenCV."""
    frame = cv2.resize(frame, (224, 224))
    frame = img_to_array(frame)
    frame = np.expand_dims(frame, axis=0)
    frame = preprocess_input(frame)
    return frame

def extract_features(frame):
    """Extract features using VGG16."""
    features = vgg16_model.predict(frame)
    features = features.flatten()
    return features

def classify_fire(features):
    """Classify the features using SVM."""
    prediction = svm_model.predict([features])
    probability = svm_model.predict_proba([features])
    return prediction, probability

def trigger_alarm():
    global alarm_ringing

    if not alarm_ringing:
        alarm_ringing = True

        def play_alarm():
            duration = 800  # milliseconds (adjust as needed)
            frequency = 1500  # Hz (adjust as needed)
            while alarm_ringing:
                winsound.Beep(frequency, duration)

        threading.Thread(target=play_alarm, daemon=True).start()

def stop_alarm():
    global alarm_ringing
    alarm_ringing = False

    # Optionally add a short pause after stopping the alarm
    time.sleep(0.1)  # Adjust as needed

def generate_frames():
    global fire_detected, fire_probability

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60)
    frame_count = 0
    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % 30 == 0:
            preprocessed_frame = preprocess_frame(frame)
            features = extract_features(preprocessed_frame)
            prediction, probability = classify_fire(features)
            fire_detected = (prediction == 1)
            fire_probability = float(probability[0][1] if fire_detected else 0.0)
            logging.debug(f"Fire detected: {fire_detected}, Probability: {fire_probability}")

            if fire_detected:
                trigger_alarm()
            else:
                stop_alarm()

        status_text = f"Fire Detected: {fire_probability:.2f}%" if fire_detected else "No Fire Detected"
        status_color = (0, 0, 255) if fire_detected else (0, 255, 0)

        cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

        if fire_detected:
            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv_frame, (0, 50, 50), (10, 255, 255))
            contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        frame_count += 1
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        elapsed_time = time.time() - start_time
        sleep_time = max(1./24 - elapsed_time, 0)
        time.sleep(sleep_time)

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fire_status')
def fire_status():
    global fire_detected, fire_probability
    fire_status = {
        'fire_detected': bool(fire_detected),
        'fire_probability': float(fire_probability)
    }
    logging.debug(f"Returning fire status: {fire_status}")
    return jsonify(fire_status)

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
